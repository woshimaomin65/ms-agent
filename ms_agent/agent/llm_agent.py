# Copyright (c) ModelScope Contributors. All rights reserved.
"""
llm_agent.py — LLMAgent 核心实现模块

本模块实现了 ms-agent 框架的核心执行单元 `LLMAgent`，继承自 `Agent` 基类，
提供基于大语言模型（LLM）的智能体完整生命周期管理。

调用链概览
----------
用户代码
  └─► LLMAgent.run(messages)          # 对外入口，支持 stream 参数
        └─► LLMAgent.run_loop(messages) # 主循环，调度所有组件
              ├─► prepare_llm()         # 初始化 LLM 实例
              ├─► prepare_runtime()     # 初始化运行时上下文
              ├─► prepare_tools()       # 初始化并连接 ToolManager
              ├─► load_memory()         # 加载记忆工具
              ├─► prepare_rag()         # 加载 RAG 组件
              ├─► read_history()        # 从磁盘恢复历史（可选）
              ├─► create_messages()     # 标准化消息列表
              ├─► do_skill(messages)    # 技能路由（AutoSkills，优先级最高）
              ├─► do_rag(messages)      # RAG 增强用户查询
              └─► step(messages)        # 单步 LLM+工具调用（循环执行）
                    ├─► condense_memory()       # 内存压缩/摘要
                    ├─► llm.generate()          # LLM 推理（流式或非流式）
                    ├─► parallel_tool_call()    # 并行工具调用
                    └─► Token 统计更新

核心能力
--------
- **多轮对话**：支持最大轮数限制（DEFAULT_MAX_CHAT_ROUND），超限后自动截断
- **流式输出**：`generation_config.stream=True` 时启用，逐 token 推送到 stdout
- **推理内容展示**：`show_reasoning=True` 时打印模型思考过程（reasoning_content）
- **工具调用**：通过 ToolManager 并行执行多工具，支持 MCP 协议服务
- **内存管理**：支持步骤后（add_after_step）和任务后（add_after_task）写入记忆
- **RAG 增强**：对用户查询进行向量检索增强
- **AutoSkills**：延迟初始化，将特定查询路由到预定义技能 DAG 执行
- **历史持久化**：对话历史与运行时状态序列化到磁盘，支持断点续跑
- **回调钩子**：在任务开始/结束、LLM 生成前后、工具调用前后触发外部回调
- **进程级 Token 统计**：跨实例共享，使用 asyncio.Lock 保证并发安全
"""
import asyncio
import importlib
import inspect
import os.path
import sys
import threading
import uuid
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import json
from ms_agent.agent.runtime import Runtime
from ms_agent.callbacks import Callback, callbacks_mapping
from ms_agent.llm.llm import LLM
from ms_agent.llm.utils import Message, ToolResult
from ms_agent.memory import Memory, get_memory_meta_safe, memory_mapping
from ms_agent.memory.memory_manager import SharedMemoryManager
from ms_agent.rag.base import RAG
from ms_agent.rag.utils import rag_mapping
from ms_agent.tools import ToolManager
from ms_agent.utils import async_retry, read_history, save_history
from ms_agent.utils.constants import DEFAULT_TAG, DEFAULT_USER
from ms_agent.utils.logger import get_logger
from omegaconf import DictConfig, OmegaConf

from ..config.config import Config, ConfigLifecycleHandler
from .base import Agent

logger = get_logger()


class LLMAgent(Agent):
    """
    An agent designed to run LLM-based tasks with support for tools, memory,
    planning, callbacks, and automatic skill execution.

    This class provides a full lifecycle for running an LLM agent, including:
    - Prompt preparation
    - Chat history management
    - External tool calling
    - Memory retrieval and updating
    - Planning logic
    - Stream or non-stream response generation
    - Callback hooks at various stages of execution
    - Automatic skill detection and execution (AutoSkills integration)

    Args:
        config (DictConfig): Pre-loaded configuration object.
        tag (str): The name of this class defined by the user.
        trust_remote_code (bool): Whether to trust remote code if any.
        **kwargs: Additional keyword arguments passed to the parent Agent constructor.

    Skills Configuration (in config.skills):
        path: Path(s) to skill directories.
        enable_retrieve: Whether to use retriever (None=auto based on skill count).
        retrieve_args: Arguments for HybridRetriever (top_k, min_score).
        max_candidate_skills: Maximum candidate skills to consider.
        max_retries: Maximum retry attempts for skill execution.
        work_dir: Working directory for skill execution.
        use_sandbox: Whether to use Docker sandbox.
        auto_execute: Whether to auto-execute skills after retrieval.

    Example:
        ```python
        config = DictConfig({
            'llm': {...},
            'skills': {
                'path': '/path/to/skills',
                'auto_execute': True,
                'work_dir': '/path/to/workspace'
            }
        })
        agent = LLMAgent(config, tag='my-agent')
        result = await agent.run('Generate a PDF report for Q4 sales of Apple')
        ```
    """

    AGENT_NAME = 'LLMAgent'  # Agent 类型标识符，用于日志与注册

    DEFAULT_SYSTEM = 'You are a helpful assistant.'  # 未配置 system prompt 时的默认值

    DEFAULT_MAX_CHAT_ROUND = 20  # 默认最大对话轮数，超过后强制截断并标记结束

    # ── 进程级 Token 统计（跨实例共享，由 TOKEN_LOCK 保护并发写入） ──
    TOTAL_PROMPT_TOKENS = 0               # 累计输入 Token 数
    TOTAL_COMPLETION_TOKENS = 0           # 累计输出 Token 数
    TOTAL_CACHED_TOKENS = 0               # 累计命中缓存的 Token 数（减少计费）
    TOTAL_CACHE_CREATION_INPUT_TOKENS = 0 # 累计写入缓存消耗的 Token 数
    TOKEN_LOCK = asyncio.Lock()           # asyncio 锁，保护上述四个计数器的原子更新

    def __init__(self,
                 config: DictConfig = DictConfig({}),
                 tag: str = DEFAULT_TAG,
                 trust_remote_code: bool = False,
                 **kwargs):
        # 若配置中缺少 llm 节点，则从默认的 agent.yaml 合并补全
        if not hasattr(config, 'llm'):
            default_yaml = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'agent.yaml')
            llm_config = Config.from_task(default_yaml)
            config = OmegaConf.merge(llm_config, config)
        super().__init__(config, tag, trust_remote_code)

        # ── 各核心组件（在 run_loop 中延迟初始化） ──
        self.callbacks: List[Callback] = []          # 已注册的生命周期回调列表
        self.tool_manager: Optional[ToolManager] = None  # 工具调用管理器
        self.memory_tools: List[Memory] = []         # 内存工具实例列表（支持多个）
        self.rag: Optional[RAG] = None               # RAG（检索增强生成）组件
        self.llm: Optional[LLM] = None               # 底层 LLM 实例
        self.runtime: Optional[Runtime] = None       # 运行时上下文（round、should_stop 等）
        self.max_chat_round: int = 0                 # 当前任务允许的最大轮数（从 config 读取）

        # ── 缓存续跑相关 ──
        self.load_cache = kwargs.get('load_cache', False)   # 是否从磁盘恢复上次对话历史
        self.config.load_cache = self.load_cache

        # ── MCP（Model Context Protocol）服务器配置 ──
        self.mcp_server_file = kwargs.get('mcp_server_file', None)  # MCP 配置文件路径
        self.mcp_config: Dict[str, Any] = self.parse_mcp_servers(
            kwargs.get('mcp_config', {}))            # 解析后的 MCP 服务器字典
        self.mcp_client = kwargs.get('mcp_client', None)  # 外部传入的 MCP 客户端实例

        # 加载外部 ConfigLifecycleHandler（task_begin/task_end 配置钩子）
        self.config_handler = self.register_config_handler()

        # ── AutoSkills 延迟初始化状态变量 ──
        self._auto_skills = None               # AutoSkills 实例（首次使用时初始化）
        self._auto_skills_initialized = False  # 是否已尝试初始化（避免重复）
        self._last_skill_result = None         # 最近一次技能执行结果（调试用）
        self._skill_mode_active = False        # 当前是否处于技能执行模式

    def _get_skills_config(self) -> Optional[DictConfig]:
        """从 Agent 配置中提取 skills 节点。

        Returns:
            DictConfig | None: 技能配置对象；若未配置则返回 None。
        """
        if hasattr(self.config, 'skills') and self.config.skills:
            return self.config.skills
        return None

    def _ensure_auto_skills(self) -> bool:
        """惰性初始化 AutoSkills（延迟初始化模式）。

        首次调用时依据配置构建 AutoSkills 实例；后续调用直接返回缓存结果。
        若 Docker 未运行，会自动禁用沙箱模式（use_sandbox=False）。

        Returns:
            bool: True 表示 AutoSkills 已成功初始化且可用，False 表示不可用。
        """
        if self._auto_skills_initialized:
            return self._auto_skills is not None

        skills_config = self._get_skills_config()
        if not skills_config:
            self._auto_skills_initialized = True
            return False

        skills_path = getattr(skills_config, 'path', None)
        if not skills_path:
            logger.debug('No skills path configured')
            self._auto_skills_initialized = True
            return False

        # 确保 LLM 已初始化（AutoSkills 需要使用 LLM 进行 query 分析）
        if self.llm is None:
            self.prepare_llm()

        try:
            from ms_agent.skill.auto_skills import AutoSkills

            # 检查 Docker 可用性，不可用时自动关闭沙箱
            use_sandbox = getattr(skills_config, 'use_sandbox', True)
            if use_sandbox:
                from ms_agent.utils.docker_utils import is_docker_daemon_running
                if not is_docker_daemon_running():
                    logger.warning(
                        'Docker not running, disabling sandbox for skills')
                    use_sandbox = False

            # 构建检索参数（top_k、min_score 等）
            retrieve_args = {}
            if hasattr(skills_config, 'retrieve_args'):
                retrieve_args = OmegaConf.to_container(
                    skills_config.retrieve_args)

            self._auto_skills = AutoSkills(
                skills=skills_path,
                llm=self.llm,
                enable_retrieve=getattr(skills_config, 'enable_retrieve',
                                        None),
                retrieve_args=retrieve_args,
                max_candidate_skills=getattr(skills_config,
                                             'max_candidate_skills', 10),
                max_retries=getattr(skills_config, 'max_retries', 3),
                work_dir=getattr(skills_config, 'work_dir', None),
                use_sandbox=use_sandbox,
            )
            logger.info(
                f'AutoSkills initialized with {len(self._auto_skills.all_skills)} skills'
            )
            self._auto_skills_initialized = True
            return True

        except Exception as e:
            logger.warning(f'Failed to initialize AutoSkills: {e}')
            self._auto_skills_initialized = True
            return False

    @property
    def skills_available(self) -> bool:
        """检查 AutoSkills 是否可用（已成功初始化）。"""
        return self._ensure_auto_skills()

    @property
    def auto_skills(self):
        """获取 AutoSkills 实例；若未配置则返回 None。"""
        self._ensure_auto_skills()
        return self._auto_skills

    async def should_use_skills(self, query: str) -> bool:
        """判断当前查询是否需要走技能（Skill）路径。

        利用 AutoSkills 内置的 LLM 分析（`_analyze_query`）来判断 query
        是否属于已注册技能的覆盖范围。

        Args:
            query: 用户的原始查询字符串。

        Returns:
            bool: True 表示应使用技能路径，False 表示回退到普通 LLM 对话。
        """
        if not self._ensure_auto_skills():
            return False

        skills_config = self._get_skills_config()
        if not skills_config:
            return False
        skills_path = getattr(skills_config, 'path', None)
        if not skills_path:
            return False

        # 调用 AutoSkills 内置分析器，由 LLM 决策是否需要技能
        try:
            needs_skills, _, _, _ = self._auto_skills._analyze_query(query)
            return needs_skills
        except Exception as e:
            logger.error(f'Skill analysis error: {e}')
            return False

    async def get_skill_dag(self, query: str):
        """仅构建技能执行 DAG，不实际执行技能。

        适合在 `auto_execute=False` 时向用户展示执行计划。

        Args:
            query: 用户的查询字符串。

        Returns:
            SkillDAGResult: 包含技能选择与执行顺序的 DAG 结果；AutoSkills 不可用时返回 None。
        """
        if not self._ensure_auto_skills():
            return None
        return await self._auto_skills.get_skill_dag(query)

    async def execute_skills(self, query: str, execution_input=None):
        """执行技能 DAG 并返回运行结果。

        Args:
            query: 用户的查询字符串。
            execution_input: 可选的技能初始输入数据（传递给第一个技能节点）。

        Returns:
            SkillDAGResult: 含执行结果的 DAG 对象；AutoSkills 不可用时返回 None。
            执行结果同时缓存到 `self._last_skill_result` 供调试使用。
        """
        if not self._ensure_auto_skills():
            return None

        skills_config = self._get_skills_config()
        stop_on_failure = getattr(skills_config, 'stop_on_failure',
                                  True) if skills_config else True

        result = await self._auto_skills.run(
            query=query,
            execution_input=execution_input,
            stop_on_failure=stop_on_failure)
        self._last_skill_result = result
        return result

    def _format_skill_result_as_messages(self, dag_result) -> List[Message]:
        """将技能执行结果格式化为 Agent 消息历史中的 Message 对象列表。

        处理三种情形：
        1. `chat_response` 存在：直接作为 assistant 消息返回（纯对话回应）。
        2. DAG 不完整（`is_complete=False`）：返回澄清消息。
        3. 有执行结果（`execution_result`）：汇总各技能的输出、文件、耗时等信息。

        Args:
            dag_result: AutoSkills 返回的 SkillDAGResult 对象。

        Returns:
            List[Message]: 格式化后的消息列表，用于追加到对话历史。
        """
        messages = []

        # 情形 1：技能判定为纯对话，直接返回模型回复
        if dag_result.chat_response:
            messages.append(
                Message(role='assistant', content=dag_result.chat_response))
            return messages

        # 情形 2：未找到合适的技能，返回提示或澄清内容
        if not dag_result.is_complete:
            content = "I couldn't find suitable skills for this task."
            if dag_result.clarification:
                content += f'\n\n{dag_result.clarification}'
            messages.append(Message(role='assistant', content=content))
            return messages

        # 情形 3：有实际执行结果，汇总各技能输出
        if dag_result.execution_result:
            exec_result = dag_result.execution_result
            skill_names = list(dag_result.selected_skills.keys())

            if exec_result.success:
                content = f"Successfully executed {len(skill_names)} skill(s): {', '.join(skill_names)}\n\n"

                # 遍历每个技能的输出，仅展示前 1000 字符防止过长
                for skill_id, result in exec_result.results.items():
                    if result.success and result.output:
                        output = result.output
                        if output.stdout:
                            stdout_preview = output.stdout[:1000]
                            if len(output.stdout) > 1000:
                                stdout_preview += '...'
                            content += f'**{skill_id} output:**\n{stdout_preview}\n\n'
                        if output.output_files:
                            content += f'**Generated files:** {list(output.output_files.values())}\n\n'

                content += f'Total execution time: {exec_result.total_duration_ms:.2f}ms'
            else:
                content = 'Skill execution completed with errors.\n\n'
                for skill_id, result in exec_result.results.items():
                    if not result.success:
                        content += f'**{skill_id} failed:** {result.error}\n'

            messages.append(Message(role='assistant', content=content))
        else:
            # 仅有 DAG 计划，未执行（auto_execute=False 时走此分支）
            skill_names = list(dag_result.selected_skills.keys())
            content = f'Found {len(skill_names)} relevant skill(s) for your task:\n'
            for skill_id, skill in dag_result.selected_skills.items():
                desc_preview = skill.description[:100]
                if len(skill.description) > 100:
                    desc_preview += '...'
                content += f'- **{skill.name}** ({skill_id}): {desc_preview}\n'
            content += f'\nExecution order: {dag_result.execution_order}'

            messages.append(Message(role='assistant', content=content))

        return messages

    def register_callback(self, callback: Callback):
        """注册单个生命周期回调实例。

        回调将在任务开始、LLM 生成、工具调用、任务结束等各个钩子点被触发。

        Args:
            callback (Callback): 实现了 `Callback` 接口的回调实例。
        """
        self.callbacks.append(callback)

    def parse_mcp_servers(self, mcp_config: Dict[str, Any]) -> Dict[str, Any]:
        """解析 MCP（Model Context Protocol）服务器配置。

        支持两种来源：
        1. 外部 JSON 文件（`mcp_server_file` 路径）：适合团队共享配置。
        2. 直接传入的字典（`mcp_config` 参数）：适合程序化配置。

        若同时提供，字典参数会覆盖文件中同名字段（merge 语义）。

        Args:
            mcp_config (Dict[str, Any]): 直接传入的 MCP 配置字典，可为空。

        Returns:
            Dict[str, Any]: 合并后的最终 MCP 配置字典。
        """
        mcp_config = mcp_config or {}
        if self.mcp_server_file is not None and os.path.isfile(
                self.mcp_server_file):
            with open(self.mcp_server_file, 'r') as f:
                config = json.load(f)
                config.update(mcp_config)
                return config
        return mcp_config

    @contextmanager
    def config_context(self):
        """配置生命周期上下文管理器。

        进入时调用 `ConfigLifecycleHandler.task_begin` 对配置进行预处理；
        退出时调用 `task_end` 进行后处理（例如恢复修改、写入日志等）。

        若未配置 `config_handler`，则透明传递，无副作用。
        """
        if self.config_handler is not None:
            self.config = self.config_handler.task_begin(self.config, self.tag)
        yield
        if self.config_handler is not None:
            self.config = self.config_handler.task_end(self.config, self.tag)

    def register_config_handler(self) -> Optional[ConfigLifecycleHandler]:
        """从配置中动态加载外部 ConfigLifecycleHandler。

        通过 `config.handler` 字段指定模块名，动态 import 并实例化其中第一个
        直接继承 `ConfigLifecycleHandler` 的类。

        安全限制：
        - 需要 `trust_remote_code=True`，否则抛出 AssertionError。
        - 需要 `config.local_dir` 存在，否则无法定位外部文件。

        Returns:
            ConfigLifecycleHandler | None: 成功加载返回实例，未配置则返回 None。

        Raises:
            AssertionError: trust_remote_code 为 False，或找不到合法的 handler 类。
        """
        handler_file = getattr(self.config, 'handler', None)
        if handler_file is not None:
            local_dir = self.config.local_dir
            assert self.config.trust_remote_code, (
                f'[External Code]A Config Lifecycle handler '
                f'registered in the config: {handler_file}. '
                f'\nThis is external code, if you trust this workflow, '
                f'please specify `--trust_remote_code true`')
            assert local_dir is not None, 'Using external py files, but local_dir cannot be found.'
            if local_dir not in sys.path:
                sys.path.insert(0, local_dir)

            handler_module = importlib.import_module(handler_file)
            module_classes = {
                name: cls
                for name, cls in inspect.getmembers(handler_module,
                                                    inspect.isclass)
            }
            handler = None
            for name, handler_cls in module_classes.items():
                if handler_cls.__bases__[
                        0] is ConfigLifecycleHandler and handler_cls.__module__ == handler_file:
                    handler = handler_cls()
            assert handler is not None, f'Config Lifecycle handler class cannot be found in {handler_file}'
            return handler
        return None

    def register_callback_from_config(self):
        """从配置中动态加载并实例化回调（Callback）列表。

        支持两类回调来源：
        1. **内置回调**：名称在 `callbacks_mapping` 中，直接实例化。
        2. **外部回调**：名称不在映射中，从 `local_dir` 动态 import 外部 .py 文件，
           找到继承自 `Callback` 的类并实例化。

        安全限制：外部回调需要 `trust_remote_code=True`，否则抛出 AssertionError。

        Raises:
            AssertionError: 未授权的外部代码，或 local_dir 未配置。
        """
        local_dir = self.config.local_dir if hasattr(self.config,
                                                     'local_dir') else None
        if hasattr(self.config, 'callbacks'):
            callbacks = self.config.callbacks or []
            for _callback in callbacks:
                subdir = os.path.dirname(_callback)
                assert local_dir is not None, 'Using external py files, but local_dir cannot be found.'
                if subdir:
                    subdir = os.path.join(local_dir, str(subdir))
                _callback = os.path.basename(_callback)
                if _callback not in callbacks_mapping:
                    # 外部回调：动态 import，需要 trust_remote_code
                    if not self.trust_remote_code:
                        raise AssertionError(
                            '[External Code Found] Your config file contains external code, '
                            'instantiate the code may be UNSAFE, if you trust the code, '
                            'please pass `trust_remote_code=True` or `--trust_remote_code true`'
                        )
                    if local_dir not in sys.path:
                        sys.path.insert(0, local_dir)
                    if subdir and subdir not in sys.path:
                        sys.path.insert(0, subdir)
                    if _callback.endswith('.py'):
                        _callback = _callback[:-3]
                    callback_file = importlib.import_module(_callback)
                    module_classes = {
                        name: cls
                        for name, cls in inspect.getmembers(
                            callback_file, inspect.isclass)
                    }
                    for name, cls in module_classes.items():
                        # 查找直接继承自 Callback 的类（排除导入的基类）
                        if issubclass(
                                cls, Callback) and cls.__module__ == _callback:
                            self.callbacks.append(cls(self.config))  # noqa
                else:
                    # 内置回调：直接从 callbacks_mapping 实例化
                    self.callbacks.append(callbacks_mapping[_callback](
                        self.config))

    async def on_task_begin(self, messages: List[Message]):
        """触发"任务开始"钩子，并打印日志。"""
        self.log_output(f'Agent {self.tag} task beginning.')
        await self.loop_callback('on_task_begin', messages)

    async def on_task_end(self, messages: List[Message]):
        """触发"任务结束"钩子，并打印日志。"""
        self.log_output(f'Agent {self.tag} task finished.')
        await self.loop_callback('on_task_end', messages)

    async def on_generate_response(self, messages: List[Message]):
        """触发"LLM 生成前"钩子（在每次调用 LLM 之前执行）。"""
        await self.loop_callback('on_generate_response', messages)

    async def on_tool_call(self, messages: List[Message]):
        """触发"工具调用前"钩子（在工具调用列表确认后、实际执行前触发）。"""
        await self.loop_callback('on_tool_call', messages)

    async def after_tool_call(self, messages: List[Message]):
        """触发"工具调用后"钩子，并判断是否需要停止循环。

        若最新消息是 assistant 且没有 tool_calls，说明 LLM 已给出最终回复，
        设置 `runtime.should_stop = True` 终止主循环。
        """
        if messages[-1].role == 'assistant' and not messages[-1].tool_calls:
            self.runtime.should_stop = True
        await self.loop_callback('after_tool_call', messages)

    async def loop_callback(self, point, messages: List[Message]):
        """遍历所有已注册回调，依次触发指定钩子点的方法。

        Args:
            point (str): 要触发的回调方法名（如 'on_task_begin'、'after_tool_call'）。
            messages (List[Message]): 当前对话历史，传递给每个回调。
        """
        for callback in self.callbacks:
            await getattr(callback, point)(self.runtime, messages)

    async def parallel_tool_call(self,
                                 messages: List[Message]) -> List[Message]:
        """并行执行消息中的所有工具调用，并将结果追加到对话历史。

        取最新 assistant 消息中的 `tool_calls` 列表，批量交给 ToolManager
        并行调度执行，然后将每个工具的响应包装为 `role='tool'` 消息追加。

        若某个 tool_call 的 id 为 None，会自动生成随机 UUID 补全，
        以满足部分 LLM 对 tool_call_id 的强制要求。

        Args:
            messages (List[Message]): 当前对话历史，最后一条应为含 tool_calls 的 assistant 消息。

        Returns:
            List[Message]: 追加了所有工具响应消息后的完整对话历史。
        """
        tool_call_result = await self.tool_manager.parallel_call_tool(
            messages[-1].tool_calls)
        assert len(tool_call_result) == len(messages[-1].tool_calls)
        for tool_call_result, tool_call_query in zip(tool_call_result,
                                                     messages[-1].tool_calls):
            tool_call_result_format = ToolResult.from_raw(tool_call_result)
            _new_message = Message(
                role='tool',
                content=tool_call_result_format.text,
                tool_call_id=tool_call_query['id'],
                name=tool_call_query['tool_name'],
                resources=tool_call_result_format.resources)

            if _new_message.tool_call_id is None:
                # tool_call_id 为 None 时补充随机 ID，避免下游报错
                _new_message.tool_call_id = str(uuid.uuid4())[:8]
                tool_call_query['id'] = _new_message.tool_call_id
            messages.append(_new_message)
            self.log_output(_new_message.content)
        return messages

    async def prepare_tools(self):
        """初始化 ToolManager 并建立与所有工具（含 MCP 服务）的连接。"""
        self.tool_manager = ToolManager(
            self.config,
            self.mcp_config,
            self.mcp_client,
            trust_remote_code=self.trust_remote_code)
        await self.tool_manager.connect()

    async def cleanup_tools(self):
        """释放 ToolManager 持有的所有工具资源（如关闭 MCP 连接）。"""
        await self.tool_manager.cleanup()

    @property
    def stream(self):
        """是否启用流式输出（逐 token 推送到 stdout）。

        从 `generation_config.stream` 读取，默认为 False（非流式）。
        """
        generation_config = getattr(self.config, 'generation_config',
                                    DictConfig({}))
        return getattr(generation_config, 'stream', False)

    @property
    def show_reasoning(self) -> bool:
        """是否在流式模式下打印模型的思考过程（reasoning_content）。

        注意：
        - 仅影响本地控制台输出，不影响 API 响应结构。
        - 实际推理内容由 `Message.reasoning_content` 字段承载（需后端支持）。
        - 输出目标由 `reasoning_output` 属性决定（stdout 或 stderr）。
        """
        generation_config = getattr(self.config, 'generation_config',
                                    DictConfig({}))
        return bool(getattr(generation_config, 'show_reasoning', False))

    @property
    def reasoning_output(self) -> str:
        """推理内容的输出目标（`show_reasoning=True` 时有效）。

        可选值：
        - ``"stdout"``：推理内容输出到标准输出，与 assistant 内容交织。
        - ``"stderr"``（默认）：输出到标准错误，保持 stdout 干净，便于管道处理。
        """
        generation_config = getattr(self.config, 'generation_config',
                                    DictConfig({}))
        return str(getattr(generation_config, 'reasoning_output', 'stdout'))

    def _write_reasoning(self, text: str):
        """将推理文本写入目标输出流（stdout 或 stderr）。

        Args:
            text (str): 要写入的推理内容片段。
        """
        if not text:
            return
        if self.reasoning_output.lower() == 'stdout':
            sys.stdout.write(text)
            sys.stdout.flush()
        else:
            # 默认写入 stderr，避免污染管道中的 stdout
            sys.stderr.write(text)
            sys.stderr.flush()

    @property
    def system(self):
        """当前任务的系统提示词（System Prompt）。

        从 `config.prompt.system` 读取；若未配置则返回 None（由 create_messages 兜底）。
        """
        return getattr(
            getattr(self.config, 'prompt', DictConfig({})), 'system', None)

    @property
    def query(self):
        """用户查询内容。

        从 `config.prompt.query` 读取；若为空，则从标准输入（stdin）交互获取。
        """
        query = getattr(
            getattr(self.config, 'prompt', DictConfig({})), 'query', None)
        if not query:
            query = input('>>>')
        return query

    async def create_messages(
            self, messages: Union[List[Message], str]) -> List[Message]:
        """将输入标准化为统一的消息列表格式。

        处理两种输入：
        - **字符串**：构造 ``[system_message, user_message]`` 两条消息。
        - **消息列表**：若配置中有 system prompt 且与当前第一条不同，则替换 system 消息内容。

        Args:
            messages (Union[List[Message], str]): 字符串形式的用户提问，或已有的消息列表。

        Returns:
            List[Message]: 标准化后的消息列表。

        Raises:
            AssertionError: 输入既不是 str 也不是 list 时抛出。
        """
        if isinstance(messages, list):
            system = self.system
            if system is not None and messages[
                    0].role == 'system' and system != messages[0].content:
                # 配置中的 system prompt 与已有消息不同，替换之
                messages[0].content = system
        else:
            assert isinstance(
                messages, str
            ), f'inputs can be either a list or a string, but current is {type(messages)}'
            # 字符串输入：组装标准的 [system, user] 两条消息
            messages = [
                Message(
                    role='system',
                    content=self.system or LLMAgent.DEFAULT_SYSTEM),
                Message(role='user', content=messages or self.query),
            ]
        return messages

    async def do_rag(self, messages: List[Message]):
        """用 RAG 检索结果增强用户查询（就地修改 messages[1].content）。

        若未配置 RAG 组件，此方法为空操作（no-op）。
        增强后的内容将包含相关文档片段，帮助 LLM 给出更精准的回答。
        """
        if self.rag is not None:
            messages[1].content = await self.rag.query(messages[1].content)

    async def do_skill(self,
                       messages: List[Message]) -> Optional[List[Message]]:
        """检测用户查询是否需要走技能路径，并在需要时执行技能 DAG。

        执行流程：
        1. 提取 messages[1] 中的用户查询文本。
        2. 调用 `should_use_skills()` 由 LLM 判断是否需要技能。
        3. 若需要：
           - `auto_execute=True`（默认）→ 执行完整 DAG（`execute_skills`）。
           - `auto_execute=False` → 仅构建执行计划（`get_skill_dag`），不实际运行。
        4. 将技能结果格式化后追加到消息列表并返回。
        5. 若技能执行失败或 DAG 为空，回退到标准 LLM 对话路径。

        Args:
            messages: 已标准化的消息列表（index 0 为 system，index 1 为 user）。

        Returns:
            List[Message] | None:
              - 成功执行技能：返回含技能结果消息的更新列表。
              - 无需技能或执行失败：返回 None，由调用方继续标准对话流程。
        """
        # 提取用户查询（index 1 应为 user 消息）
        query = (
            messages[1].content
            if len(messages) > 1 and messages[1].role == 'user' else None)

        if not query:
            return None

        # LLM 分析是否走技能路径
        if not await self.should_use_skills(query):
            return None

        logger.info('Query detected as skill-related, using skill processing.')
        self._skill_mode_active = True

        try:
            skills_config = self._get_skills_config()
            auto_execute = getattr(skills_config, 'auto_execute',
                                   True) if skills_config else True

            if auto_execute:
                dag_result = await self.execute_skills(query)
            else:
                dag_result = await self.get_skill_dag(query)

            if dag_result:
                skill_messages = self._format_skill_result_as_messages(
                    dag_result)
                for msg in skill_messages:
                    messages.append(msg)
                return messages

            # DAG 为空，回退到标准对话路径
            self._skill_mode_active = False
            return None

        except Exception as e:
            logger.warning(
                f'Skill execution failed: {e}, falling back to standard agent')
            self._skill_mode_active = False
            return None

    async def load_memory(self):
        """根据配置初始化所有内存工具实例，并追加到 `self.memory_tools`。

        支持通过 `SharedMemoryManager` 实现跨实例共享内存（如 mem0、faiss 等）。
        内存类型由 `config.memory` 的键名决定，必须在 `memory_mapping` 中注册。

        Raises:
            AssertionError: 配置中指定的内存类型不存在于 `memory_mapping` 时抛出。
        """
        self.config: DictConfig
        if hasattr(self.config, 'memory'):
            for mem_instance_type, _memory in self.config.memory.items():
                assert mem_instance_type in memory_mapping, (
                    f'{mem_instance_type} not in memory_mapping, '
                    f'which supports: {list(memory_mapping.keys())}')

                shared_memory = await SharedMemoryManager.get_shared_memory(
                    self.config, mem_instance_type)
                self.memory_tools.append(shared_memory)

    async def prepare_rag(self):
        """从配置加载并初始化 RAG（检索增强生成）组件。

        RAG 类型由 `config.rag.name` 决定，必须在 `rag_mapping` 中注册。
        """
        if hasattr(self.config, 'rag'):
            rag = self.config.rag
            if rag is not None:
                assert rag.name in rag_mapping, (
                    f'{rag.name} not in rag_mapping, '
                    f'which supports: {list(rag_mapping.keys())}')
                self.rag: RAG = rag_mapping(rag.name)(self.config)

    async def condense_memory(self, messages: List[Message]) -> List[Message]:
        """用内存工具对当前对话历史进行压缩或摘要更新。

        依次将消息历史传入每个内存工具（`memory_tool.run(messages)`），
        内存工具可能会截断、摘要或注入相关历史片段，以控制 context 长度。

        Args:
            messages (List[Message]): 当前完整对话历史。

        Returns:
            List[Message]: 经过所有内存工具处理后的（可能更短的）对话历史。
        """
        for memory_tool in self.memory_tools:
            messages = await memory_tool.run(messages)
        return messages

    def log_output(self, content: str):
        """将内容格式化后写入日志（自动添加 tag 前缀）。

        若内容超过 1024 字符，则截取首尾各 512 字符并在中间插入省略号，
        避免日志文件膨胀。

        Args:
            content (str): 要记录的文本内容。
        """
        if len(content) > 1024:
            content = content[:512] + '\n...\n' + content[-512:]
        for line in content.split('\n'):
            for _line in line.split('\\n'):
                logger.info(f'[{self.tag}] {_line}')

    def handle_new_response(self, messages: List[Message],
                            response_message: Message):
        """处理 LLM 生成的新响应消息，并将其规范化后追加到对话历史。

        主要职责：
        1. 断言响应消息不为 None。
        2. 若响应含 tool_calls，格式化后打印工具调用详情（JSON 缩进）。
        3. 若响应消息尚未在列表末尾，则追加。
        4. 若 assistant 消息内容为空但有 tool_calls，补充占位文本
           （部分 LLM 不允许 content 为空的 assistant 消息）。

        Args:
            messages (List[Message]): 当前对话历史。
            response_message (Message): LLM 返回的最新响应消息。
        """
        assert response_message is not None, 'No response message generated from LLM.'
        if response_message.tool_calls:
            self.log_output('[tool_calling]:')
            for tool_call in response_message.tool_calls:
                tool_call = deepcopy(tool_call)
                if isinstance(tool_call['arguments'], str):
                    try:
                        # 尝试将字符串参数解析为 dict，方便打印和后续处理
                        tool_call['arguments'] = json.loads(
                            tool_call['arguments'])
                    except json.decoder.JSONDecodeError:
                        pass
                self.log_output(
                    json.dumps(tool_call, ensure_ascii=False, indent=4))

        if messages[-1] is not response_message:
            messages.append(response_message)

        if messages[-1].role == 'assistant' and not messages[
                -1].content and response_message.tool_calls:
            # 部分 LLM 要求 assistant 消息 content 不为空，补充占位文本
            messages[-1].content = 'Let me do a tool calling.'

    @async_retry(max_attempts=Agent.retry_count, delay=1.0)
    async def step(
        self, messages: List[Message]
    ) -> AsyncGenerator[List[Message], Any]:  # type: ignore
        """执行 Agent 交互循环中的单步操作。

        每一步包含以下子操作（顺序执行）：

        1. 深拷贝消息历史，避免多次 yield 时发生数据污染。
        2. 调用内存工具压缩/更新对话历史（condense_memory）。
        3. 触发 on_generate_response 钩子（LLM 调用前）。
        4. 调用 LLM 生成响应（流式或非流式）：
           - **流式**：逐 token yield 到 stdout，同时可选打印 reasoning_content。
           - **非流式**：等待完整响应后一次性打印。
        5. 处理新响应（handle_new_response）：追加到消息列表，补全空 content。
        6. 触发 on_tool_call 钩子。
        7. 若响应含 tool_calls，并行执行所有工具（parallel_tool_call）。
        8. 触发 after_tool_call 钩子（若无工具调用则设 should_stop=True）。
        9. 原子更新进程级 Token 统计（TOKEN_LOCK 保护）。
        10. 通过 `yield` 将更新后的消息列表返回给调用方。

        若步骤失败，`@async_retry` 装饰器会重试最多 `Agent.retry_count` 次。

        **缓存续跑逻辑**：若 `load_cache=True` 且最新消息已是 assistant 回复，
        则跳过 LLM 调用，直接使用缓存的 assistant 消息，并将 `load_cache` 重置为 False。

        Args:
            messages (List[Message]): 当前对话历史。

        Yields:
            List[Message]: 每次 LLM 生成 token（流式）或完成整步后更新的消息历史。
        """
        messages = deepcopy(messages)
        if (not self.load_cache) or messages[-1].role != 'assistant':
            messages = await self.condense_memory(messages)
            await self.on_generate_response(messages)
            tools = await self.tool_manager.get_tools()

            if self.stream:
                self.log_output('[assistant]:')
                _content = ''         # 已累积的 assistant 文本内容（用于增量推送）
                _reasoning = ''       # 已累积的推理内容（用于增量计算新片段）
                is_first = True       # 标记首个 chunk，用于将消息追加到列表
                _response_message = None
                _printed_reasoning_header = False  # 是否已打印 [thinking]: 头部
                for _response_message in self.llm.generate(
                        messages, tools=tools):
                    if is_first:
                        # 首个 chunk：将消息对象加入列表（后续 chunk 就地更新）
                        messages.append(_response_message)
                        is_first = False

                    # 可选：流式打印模型推理内容（thinking/reasoning）
                    if self.show_reasoning:
                        reasoning_text = getattr(_response_message,
                                                 'reasoning_content', '') or ''
                        # 部分 provider 在不同 chunk 中可能重置内容长度，需检测回退
                        if len(reasoning_text) < len(_reasoning):
                            _reasoning = ''
                        new_reasoning = reasoning_text[len(_reasoning):]
                        if new_reasoning:
                            if not _printed_reasoning_header:
                                self._write_reasoning('[thinking]:\n')
                                _printed_reasoning_header = True
                            self._write_reasoning(new_reasoning)
                            _reasoning = reasoning_text

                    # 增量推送 assistant 文本到 stdout
                    new_content = _response_message.content[len(_content):]
                    sys.stdout.write(new_content)
                    sys.stdout.flush()
                    _content = _response_message.content
                    messages[-1] = _response_message  # 更新列表末尾为最新 chunk
                    yield messages
                if self.show_reasoning and _printed_reasoning_header:
                    self._write_reasoning('\n')  # 推理内容结束后换行
                sys.stdout.write('\n')
            else:
                # 非流式：等待完整响应
                _response_message = self.llm.generate(messages, tools=tools)
                if self.show_reasoning:
                    reasoning_text = getattr(_response_message,
                                             'reasoning_content', '') or ''
                    if reasoning_text:
                        self._write_reasoning('[thinking]:\n')
                        self._write_reasoning(reasoning_text)
                        self._write_reasoning('\n')
                if _response_message.content:
                    self.log_output('[assistant]:')
                    self.log_output(_response_message.content)

            # LLM 响应生成完毕，规范化并追加到消息历史
            self.handle_new_response(messages, _response_message)
            await self.on_tool_call(messages)
        else:
            # 缓存续跑分支：最新消息已是 assistant，跳过 LLM 调用
            # 重置 load_cache，避免影响后续子任务的正常执行
            self.load_cache = False
            # 直接复用缓存的 assistant 消息，防止同一任务生成不同回复
            _response_message = messages[-1]
        self.save_history(messages)

        if _response_message.tool_calls:
            # 有工具调用时，并行执行并将结果追加到消息历史
            messages = await self.parallel_tool_call(messages)

        await self.after_tool_call(messages)

        # ── Token 统计更新 ──
        prompt_tokens = _response_message.prompt_tokens
        completion_tokens = _response_message.completion_tokens
        cached_tokens = getattr(_response_message, 'cached_tokens', 0) or 0
        cache_creation_input_tokens = getattr(
            _response_message, 'cache_creation_input_tokens', 0) or 0

        # 使用异步锁保证跨协程/实例的原子性累加
        async with LLMAgent.TOKEN_LOCK:
            LLMAgent.TOTAL_PROMPT_TOKENS += prompt_tokens
            LLMAgent.TOTAL_COMPLETION_TOKENS += completion_tokens
            LLMAgent.TOTAL_CACHED_TOKENS += cached_tokens
            LLMAgent.TOTAL_CACHE_CREATION_INPUT_TOKENS += cache_creation_input_tokens

        # 打印当前步骤的 Token 消耗
        self.log_output(
            f'[usage] prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}'
        )
        if cached_tokens or cache_creation_input_tokens:
            self.log_output(
                f'[usage_cache] cache_hit: {cached_tokens}, cache_created: {cache_creation_input_tokens}'
            )
        # 打印进程累计 Token（跨所有实例、所有 step）
        self.log_output(
            f'[usage_total] total_prompt_tokens: {LLMAgent.TOTAL_PROMPT_TOKENS}, '
            f'total_completion_tokens: {LLMAgent.TOTAL_COMPLETION_TOKENS}')
        if LLMAgent.TOTAL_CACHED_TOKENS or LLMAgent.TOTAL_CACHE_CREATION_INPUT_TOKENS:
            self.log_output(
                f'[usage_cache_total] total_cache_hit: {LLMAgent.TOTAL_CACHED_TOKENS}, '
                f'total_cache_created: {LLMAgent.TOTAL_CACHE_CREATION_INPUT_TOKENS}'
            )

        yield messages

    def prepare_llm(self):
        """根据配置初始化底层 LLM 实例（通过 `LLM.from_config` 工厂方法）。"""
        self.llm: LLM = LLM.from_config(self.config)

    def prepare_runtime(self):
        """初始化运行时上下文（Runtime），记录 round、should_stop 等状态。"""
        self.runtime: Runtime = Runtime(llm=self.llm)

    def read_history(self, messages: List[Message],
                     **kwargs) -> Tuple[DictConfig, Runtime, List[Message]]:
        """从磁盘加载上一次会话的对话历史（断点续跑）。

        仅在 `load_cache=True` 时生效。若磁盘有历史记录，会恢复：
        - 配置快照（config）
        - 运行时状态（Runtime：round、should_stop 等）
        - 完整消息列表（messages）

        边界处理：若历史末尾是 `role='tool'` 消息，则丢弃该消息重跑，
        因为工具响应可能因为上次执行出错而处于不一致状态。

        Args:
            messages (List[Message]): 当前输入消息（续跑时作为 fallback）。

        Returns:
            Tuple[DictConfig, Runtime, List[Message]]:
              - 恢复的配置对象
              - 恢复的运行时对象
              - 恢复的消息历史（若无缓存则返回原始 messages）
        """
        if isinstance(messages, str):
            query = messages
        else:
            query = messages[1].content
        if not query or not self.load_cache:
            return self.config, self.runtime, messages

        config, _messages = read_history(self.output_dir, self.tag)
        if config is not None and _messages is not None:
            if hasattr(config, 'runtime'):
                runtime = Runtime(llm=self.llm)
                runtime.from_dict(config.runtime)
                delattr(config, 'runtime')
            else:
                runtime = self.runtime
            if _messages[-1].role == 'tool':
                # 丢弃末尾的 tool 响应消息，强制重新执行
                # 因为上次中断可能发生在工具调用处理过程中，状态可能不一致
                _messages = _messages[:-1]
            return config, runtime, _messages
        else:
            return self.config, self.runtime, messages

    def get_user_id(self, default_user_id=DEFAULT_USER) -> Optional[str]:
        """从内存配置中提取 user_id（用于内存读写的用户标识）。

        遍历所有内存配置项，返回第一个非空的 user_id；
        若均未配置，则返回默认值 `DEFAULT_USER`。

        Args:
            default_user_id: 未配置时的默认用户 ID。

        Returns:
            str | None: 用户 ID 字符串。
        """
        if hasattr(self.config, 'memory') and self.config.memory:
            for memory_config in self.config.memory:
                if hasattr(memory_config, 'user_id') and memory_config.user_id:
                    user_id = memory_config.user_id
                    break
        return user_id

    def _get_step_memory_info(self, memory_config: DictConfig):
        """从单条内存配置中提取"步骤后写入"所需的元数据。

        检查 `add_after_step` 钩子配置，提取 user_id、agent_id、run_id、memory_type。
        若四个字段均为 None，则返回全 None（表示该内存实例不需要步骤后写入）。

        Args:
            memory_config: 单个内存工具的配置对象。

        Returns:
            Tuple[user_id, agent_id, run_id, memory_type]
        """
        user_id, agent_id, run_id, memory_type = get_memory_meta_safe(
            memory_config, 'add_after_step')
        if all(value is None
               for value in [user_id, agent_id, run_id, memory_type]):
            return None, None, None, None
        user_id = user_id or getattr(memory_config, 'user_id', None)
        return user_id, agent_id, run_id, memory_type

    def _get_run_memory_info(self, memory_config: DictConfig):
        """从单条内存配置中提取"任务后写入"所需的元数据。

        检查 `add_after_task` 钩子配置，默认将 agent_id 设置为 self.tag。
        若四个字段均为 None，则返回全 None。

        Args:
            memory_config: 单个内存工具的配置对象。

        Returns:
            Tuple[user_id, agent_id, run_id, memory_type]
        """
        user_id, agent_id, run_id, memory_type = get_memory_meta_safe(
            memory_config,
            'add_after_task',
            default_user_id=getattr(memory_config, 'user_id', None))
        if all(value is None
               for value in [user_id, agent_id, run_id, memory_type]):
            return None, None, None, None
        user_id = user_id or getattr(memory_config, 'user_id', None)
        agent_id = agent_id or self.tag
        memory_type = memory_type or None
        return user_id, agent_id, run_id, memory_type

    async def add_memory(self, messages: List[Message], add_type, **kwargs):
        """将当前对话历史写入内存工具（步骤后或任务后）。

        根据 `add_type` 参数选择元数据提取策略：
        - ``'add_after_step'``：每一步结束后调用，适合短期记忆。
        - ``'add_after_task'``：整个任务完成后在后台线程调用，适合长期记忆。

        仅当元数据（user_id/agent_id/run_id/memory_type）至少有一个非 None 时，
        才实际调用 `memory_tool.add()`。

        Args:
            messages (List[Message]): 当前完整对话历史。
            add_type (str): 写入时机，'add_after_step' 或 'add_after_task'。
            **kwargs: 透传给内存工具的额外参数。
        """
        if hasattr(self.config, 'memory') and self.config.memory:
            tools_num = len(self.memory_tools) if self.memory_tools else 0

            for idx, (mem_instance_type,
                      memory_config) in enumerate(self.config.memory.items()):
                if add_type == 'add_after_task':
                    user_id, agent_id, run_id, memory_type = self._get_run_memory_info(
                        memory_config)
                else:
                    user_id, agent_id, run_id, memory_type = self._get_step_memory_info(
                        memory_config)

                if idx < tools_num:
                    if any(v is not None
                           for v in [user_id, agent_id, run_id, memory_type]):
                        await self.memory_tools[idx].add(
                            messages,
                            user_id=user_id,
                            agent_id=agent_id,
                            run_id=run_id,
                            memory_type=memory_type)

    def save_history(self, messages: List[Message], **kwargs):
        """将当前对话历史持久化到磁盘，以支持断点续跑。

        若配置中 `save_history=False` 或用户查询为空，则跳过保存。
        保存内容包括：消息列表 + 运行时状态（config.runtime 快照）。

        Args:
            messages (List[Message]): 当前对话历史。
        """
        query = None
        if len(messages) > 1 and messages[1].role == 'user':
            query = messages[1].content
        elif messages:
            query = messages[0].content
        if not query:
            return

        if not getattr(self.config, 'save_history', True):
            return

        config: DictConfig = deepcopy(self.config)
        config.runtime = self.runtime.to_dict()
        save_history(
            self.output_dir, task=self.tag, config=config, messages=messages)

    async def run_loop(self, messages: Union[List[Message], str],
                       **kwargs) -> AsyncGenerator[Any, Any]:
        """Agent 主执行循环：调度所有组件并驱动多轮 LLM + 工具调用。

        整体执行流程
        -----------
        **初始化阶段**（每次调用均执行）：
          1. 从配置读取最大轮数（max_chat_round）。
          2. 注册回调（register_callback_from_config）。
          3. 初始化 LLM、Runtime、ToolManager、Memory、RAG。

        **首轮（round == 0）阶段**：
          4. 标准化输入消息（create_messages）。
          5. **技能路由**（do_skill）：若匹配技能，直接执行并返回，跳过后续循环。
          6. RAG 增强（do_rag）：将检索结果注入用户查询。
          7. 触发 on_task_begin 回调。

        **续跑阶段（round > 0）**：
          直接进入对话循环（跳过初始化和技能路由）。

        **对话循环**（while not should_stop）：
          8. 执行单步 step(messages)，yield 每个 chunk（流式）或最终结果。
          9. round 计数 +1。
          10. 步骤后内存写入（add_after_step）。
          11. 持久化历史（save_history）。
          12. 若 round 超过 max_chat_round，追加截断消息并强制停止。

        **收尾阶段**：
          13. 触发 on_task_end 回调，清理工具资源（cleanup_tools）。
          14. 在线程池中异步执行任务后内存写入（add_after_task），不阻塞主流程。

        Args:
            messages (Union[List[Message], str]): 用户输入（字符串或消息列表）。

        Yields:
            List[Message]: 每个 step 产生的最新消息列表（流式为多次，非流式为一次）。

        Raises:
            Exception: 运行时异常向上传播，并打印 traceback 日志。
        """
        try:
            # ── 步骤 1：初始化所有组件 ──
            self.max_chat_round = getattr(self.config, 'max_chat_round',
                                          LLMAgent.DEFAULT_MAX_CHAT_ROUND)
            self.register_callback_from_config()
            self.prepare_llm()
            self.prepare_runtime()
            await self.prepare_tools()
            await self.load_memory()
            await self.prepare_rag()
            self.runtime.tag = self.tag

            if messages is None:
                messages = self.query  # 从配置或标准输入获取查询

            # ── 步骤 2：断点续跑 - 从磁盘恢复历史状态 ──
            self.config, self.runtime, messages = self.read_history(messages)

            if self.runtime.round == 0:
                # ── 步骤 3：首轮初始化 ──
                # 3a. 标准化消息格式
                messages = await self.create_messages(messages)

                # 3b. 技能路由（优先级最高）：命中则直接返回，不进入对话循环
                skill_result = await self.do_skill(messages)
                if skill_result is not None:
                    await self.on_task_begin(skill_result)
                    yield skill_result
                    await self.on_task_end(skill_result)
                    await self.cleanup_tools()
                    return

                # 3c. RAG 增强：将检索到的相关文档注入用户查询
                await self.do_rag(messages)
                await self.on_task_begin(messages)

            # 打印当前所有非 system 消息（方便调试）
            for message in messages:
                if message.role != 'system':
                    self.log_output('[' + message.role + ']:')
                    self.log_output(message.content)

            # ── 步骤 4：对话循环 ──
            while not self.runtime.should_stop:
                async for messages in self.step(messages):
                    yield messages
                self.runtime.round += 1
                # 步骤后内存写入（短期记忆）和历史持久化
                await self.add_memory(
                    messages, add_type='add_after_step', **kwargs)
                self.save_history(messages)

                # +1：下一轮 assistant 可能给出最终结论，因此阈值 +1
                if self.runtime.round >= self.max_chat_round + 1:
                    if not self.runtime.should_stop:
                        # 追加截断提示消息
                        messages.append(
                            Message(
                                role='assistant',
                                content=
                                f'Task {messages[1].content} was cutted off, because '
                                f'max round({self.max_chat_round}) exceeded.'))
                    self.runtime.should_stop = True
                    yield messages

            # ── 步骤 5：收尾 ──
            await self.on_task_end(messages)
            await self.cleanup_tools()
            yield messages

            # 任务后内存写入在独立线程中执行，不阻塞主协程
            def _add_memory():
                asyncio.run(
                    self.add_memory(
                        messages, add_type='add_after_task', **kwargs))

            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, _add_memory)
        except Exception as e:
            import traceback
            logger.warning(traceback.format_exc())
            if hasattr(self.config, 'help'):
                logger.error(
                    f'[{self.tag}] Runtime error, please follow the instructions:\n\n {self.config.help}'
                )
            raise e

    async def run(
            self, messages: Union[List[Message], str], **kwargs
    ) -> Union[List[Message], AsyncGenerator[List[Message], Any]]:
        """对外公开的 Agent 执行入口。

        根据 `stream` 参数决定返回模式：
        - **非流式**（默认）：等待全部生成完成，返回最终消息列表 `List[Message]`。
        - **流式**：立即返回一个异步生成器，调用方可 `async for` 逐步接收消息列表。

        运行于 `config_context()` 上下文管理器中，确保 task_begin/task_end
        配置钩子在任务前后被调用。

        Args:
            messages (Union[List[Message], str]): 用户输入，字符串或消息列表。
            **kwargs: 额外参数，支持 `stream=True` 切换为流式模式。

        Returns:
            List[Message]: 非流式模式下的最终消息列表。
            AsyncGenerator[List[Message], Any]: 流式模式下的异步生成器。
        """
        stream = kwargs.get('stream', False)
        with self.config_context():
            if stream:
                # 流式模式：将 generation_config.stream 置为 True，返回异步生成器
                OmegaConf.update(
                    self.config, 'generation_config.stream', True, merge=True)

                async def stream_generator():
                    async for _chunk in self.run_loop(
                            messages=messages, **kwargs):
                        yield _chunk

                return stream_generator()
            else:
                # 非流式模式：消费所有 chunk，仅保留最后一个（最终结果）
                res = None
                async for chunk in self.run_loop(messages=messages, **kwargs):
                    res = chunk
                return res
