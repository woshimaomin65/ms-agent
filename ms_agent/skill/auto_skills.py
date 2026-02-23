# flake8: noqa
# isort: skip_file
# yapf: disable
"""
AutoSkills —— 自动技能检索与 DAG 执行引擎。

【整体架构】

用户 query
   │
   ▼
AutoSkills.run(query)
   │
   ├─① get_skill_dag(query)          # 检索并规划
   │     │
   │     ├─ 模式A: enable_retrieve=False
   │     │     └─ _direct_select_skills()   # 将全部 Skill 放入 LLM 上下文，一次性选择
   │     │
   │     └─ 模式B: enable_retrieve=True（Skill 数量 > 10 时自动开启）
   │           ├─ _analyze_query()           # LLM 判断是否需要 Skill（还是纯聊天）
   │           ├─ _async_retrieve_skills()   # HybridRetriever 并行检索候选 Skill
   │           ├─ _filter_skills('fast')     # LLM 快速过滤（名称+描述）
   │           ├─ _filter_skills('deep')     # LLM 深度过滤（含 Skill 内容）
   │           └─ _build_dag()              # LLM 构建依赖 DAG + 执行顺序
   │
   └─② execute_dag(dag_result)       # 按 DAG 执行
         └─ DAGExecutor.execute()
               │
               ├─ 串行节点: _execute_single_skill()
               └─ 并行节点: _execute_parallel_group() → asyncio.gather()
                     │
                     └─ _execute_with_progressive_analysis()
                           ├─ Phase 1: SkillAnalyzer.analyze_skill_plan()    # LLM 制定执行计划
                           ├─ Phase 2: SkillAnalyzer.load_skill_resources()  # 按计划加载资源
                           ├─ Phase 3: SkillAnalyzer.generate_execution_commands() # LLM 生成命令
                           └─ 执行命令（含自我反思重试 _execute_command_with_retry）

【关键数据结构】

- SkillSchema:       单个 Skill 的元数据（id/name/description/scripts/content 等）
- SkillContext:      执行上下文（含懒加载的 scripts/references/resources）
- SkillExecutionPlan: LLM 分析后的执行计划（需要哪些脚本/包/步骤）
- DAG:               Dict[skill_id, List[skill_id]]，邻接表，dag[A]=[B] 表示 A 依赖 B
- execution_order:   List[skill_id | List[skill_id]]，子列表表示可并行执行的组

【核心技术点】

1. HybridRetriever:     稠密向量（Dense）+ 稀疏词频（Sparse）混合检索，召回相关 Skill
2. 渐进式分析（Progressive Analysis）:
     不一次性加载所有资源，而是先让 LLM 制定计划，再按需加载，减少 Token 消耗
3. 自我反思（Self-Reflection）:
     执行失败后，让 LLM 分析错误原因并修复代码，最多重试 max_retries 次
4. 上下游数据链路:
     上游 Skill 的 stdout/output_files 通过环境变量 UPSTREAM_OUTPUTS 注入下游 Skill
5. 拓扑排序:
     _topological_sort_dag() 对 DAG 做 Kahn 算法拓扑排序，保证依赖先于被依赖者执行
"""
import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import json
from ms_agent.llm import LLM
from ms_agent.llm.utils import Message
from ms_agent.retriever.hybrid_retriever import HybridRetriever
from ms_agent.skill.container import (ExecutionInput, ExecutionOutput,
                                      ExecutorType, SkillContainer)
from ms_agent.skill.loader import load_skills
from ms_agent.skill.prompts import (PROMPT_ANALYZE_EXECUTION_ERROR,
                                    PROMPT_ANALYZE_QUERY_FOR_SKILLS,
                                    PROMPT_BUILD_SKILLS_DAG,
                                    PROMPT_DIRECT_SELECT_SKILLS,
                                    PROMPT_FILTER_SKILLS_DEEP,
                                    PROMPT_FILTER_SKILLS_FAST,
                                    PROMPT_SKILL_ANALYSIS_PLAN,
                                    PROMPT_SKILL_EXECUTION_COMMAND)
from ms_agent.skill.schema import SkillContext, SkillExecutionPlan, SkillSchema
from ms_agent.utils.logger import get_logger

logger = get_logger()


def _configure_logger_to_dir(log_dir: Path) -> None:
    """
    将 logger 的输出重定向到指定目录下的文件（ms_agent.log）。

    若该路径的 FileHandler 已存在则不重复添加，避免重复写日志。
    会先移除旧的 FileHandler，再添加新的，保证同时只有一个文件输出目标。

    Args:
        log_dir: 日志文件所在目录，不存在时会自动创建。
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'ms_agent.log'

    # Check if file handler for this path already exists
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            if Path(handler.baseFilename).resolve() == log_file.resolve():
                return  # Already configured

    # Remove existing file handlers and add new one
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)

    file_handler = logging.FileHandler(str(log_file), mode='a')
    file_handler.setFormatter(logging.Formatter('[%(levelname)s:%(name)s] %(message)s'))
    file_handler.setLevel(logger.level)
    logger.addHandler(file_handler)
    logger.info(f'Logger configured to output to: {log_file}')


@dataclass
class SkillExecutionResult:
    """
    单个 Skill 的执行结果。

    Attributes:
        skill_id: 被执行的 Skill 标识符。
        success:  执行是否成功（exit_code == 0 时为 True）。
        output:   容器返回的 ExecutionOutput，含 stdout/stderr/output_files 等。
        error:    执行失败时的错误描述；成功时为 None。
    """
    skill_id: str
    success: bool = False
    output: Optional[ExecutionOutput] = None
    error: Optional[str] = None


@dataclass
class DAGExecutionResult:
    """
    整个 Skill DAG 的执行汇总结果。

    Attributes:
        success:          所有 Skill 均执行成功时为 True；任一失败则为 False。
        results:          Dict，key 为 skill_id，value 为对应的 SkillExecutionResult。
        execution_order:  实际执行顺序（含并行组），与计划顺序一致或因失败提前终止而更短。
        total_duration_ms: 从第一个 Skill 开始到最后一个 Skill 结束的总耗时（毫秒）。
    """
    success: bool = False
    results: Dict[str, SkillExecutionResult] = field(default_factory=dict)
    execution_order: List[Union[str, List[str]]] = field(default_factory=list)
    total_duration_ms: float = 0.0

    def get_skill_output(self, skill_id: str) -> Optional[ExecutionOutput]:
        """获取指定 Skill 的执行输出；若该 Skill 未执行则返回 None。"""
        result = self.results.get(skill_id)
        return result.output if result else None


class SkillAnalyzer:
    """
    渐进式 Skill 分析器：分阶段加载上下文，避免一次性将所有资源塞给 LLM。

    执行分为三个阶段：
    1. Plan  阶段：仅读取 Skill 元数据 + SKILL.md 概览，让 LLM 制定执行计划
    2. Load  阶段：根据计划按需加载脚本/参考文档/资源，而非全量加载
    3. Exec  阶段：将加载的内容传给 LLM，生成具体可执行命令

    设计动机：Skill 包可能包含大量文件，全量加载会消耗大量 Token 并干扰 LLM 判断。
    渐进式加载只取"LLM 认为需要的"部分，显著降低 Token 消耗。
    """

    def __init__(self, llm: 'LLM'):
        """
        初始化 SkillAnalyzer。

        Args:
            llm: 用于计划制定和命令生成的 LLM 实例。
        """
        self.llm = llm

    def _llm_generate(self, prompt: str) -> str:
        """将 prompt 封装成 user message 发给 LLM，返回文本响应。"""
        messages = [Message(role='user', content=prompt)]
        logger.debug(f'Input msg to LLM in SkillAnalyzer: {messages}')
        response = self.llm.generate(messages=messages)
        res = response.content if hasattr(response,
                                           'content') else str(response)
        logger.debug(f'LLM response in SkillAnalyzer: {res}')
        return res

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        从 LLM 响应中鲁棒地解析 JSON。

        LLM 有时会在 JSON 前后包裹 Markdown 代码块（```json ... ```），
        或在 JSON 前后添加说明文字。此方法按以下顺序尝试解析：
        1. 去掉 Markdown 代码块后直接 json.loads
        2. 找最外层 {} 括号对，提取后解析
        3. 用正则 \\{[\\s\\S]*\\} 匹配，提取后解析
        全部失败则返回空 dict 并记录警告。
        """
        # Remove markdown code blocks if present
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*$', '', response)
        response = response.strip()

        # Try direct parsing first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON object from response
        try:
            # Find the outermost JSON object
            start = response.find('{')
            if start != -1:
                # Find matching closing brace
                depth = 0
                for i, char in enumerate(response[start:], start):
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            json_str = response[start:i + 1]
                            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Try regex extraction as fallback
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        logger.warning(f'Failed to parse JSON: {response[:500]}...')
        return {}

    def analyze_skill_plan(self,
                           skill: SkillSchema,
                           query: str,
                           root_path: Path = None) -> SkillContext:
        """
        Phase 1：分析 Skill 并生成执行计划（只读元数据，不加载脚本资源）。

        向 LLM 提供 Skill 的名称、描述、SKILL.md 概览（最多 4000 字符），
        以及可用脚本/参考/资源的文件名列表（不含内容），
        让 LLM 判断该 Skill 能否处理 query，并规划所需的脚本、包、步骤。

        Args:
            skill:     待分析的 SkillSchema 对象。
            query:     用户的原始任务描述。
            root_path: Skill 上下文的根路径，默认为 skill 文件所在目录的父目录。

        Returns:
            SkillContext，其中 plan 字段已填充执行计划；资源尚未加载（懒加载）。
        """
        # Create context with lazy loading
        context = SkillContext(
            skill=skill,
            query=query,
            root_path=root_path or skill.skill_path.parent)

        # Build prompt with skill overview (not full content)
        prompt = PROMPT_SKILL_ANALYSIS_PLAN.format(
            query=query,
            skill_id=skill.skill_id,
            skill_name=skill.name,
            skill_description=skill.description,
            skill_content=skill.content[:4000] if skill.content else '',
            scripts_list=', '.join(context.get_scripts_list()) or 'None',
            references_list=', '.join(context.get_references_list()) or 'None',
            resources_list=', '.join(context.get_resources_list()) or 'None')

        response = self._llm_generate(prompt)
        parsed = self._parse_json_response(response)

        # Build execution plan
        plan = SkillExecutionPlan(
            can_handle=parsed.get('can_handle', False),
            plan_summary=parsed.get('plan_summary', ''),
            steps=parsed.get('steps', []),
            required_scripts=parsed.get('required_scripts', []),
            required_references=parsed.get('required_references', []),
            required_resources=parsed.get('required_resources', []),
            required_packages=parsed.get('required_packages', []),
            parameters=parsed.get('parameters', {}),
            reasoning=parsed.get('reasoning', ''))

        context.plan = plan
        context.spec.plan = plan.plan_summary

        logger.info(
            f'Skill analysis plan: can_handle={plan.can_handle}, '
            f'scripts={plan.required_scripts}, refs={plan.required_references}, '
            f'packages={plan.required_packages}'
        )

        return context

    def load_skill_resources(self, context: SkillContext) -> SkillContext:
        """
        Phase 2：根据 Phase 1 生成的执行计划，按需加载 Skill 资源。

        仅加载计划中声明要使用的脚本/参考文档/资源，而非全量加载。
        若 plan 为空或 can_handle=False，跳过加载直接返回。

        Args:
            context: 包含 Phase 1 计划的 SkillContext。

        Returns:
            加载了所需资源后的 SkillContext（scripts/references/resources 已填充）。
        """
        if not context.plan or not context.plan.can_handle:
            logger.warning('No valid plan, skipping resource loading')
            return context

        context.load_from_plan()
        logger.info(
            f'Loaded resources: scripts={len(context.scripts)}, '
            f'refs={len(context.references)}, res={len(context.resources)}')

        return context

    def generate_execution_commands(
            self, context: SkillContext) -> List[Dict[str, Any]]:
        """
        Phase 3：根据已加载的资源，生成具体可执行命令列表。

        将执行计划 + 脚本内容 + 参考文档发送给 LLM，
        让 LLM 输出一组结构化命令（python_code / python_script / shell / javascript）。

        若 LLM 未生成任何命令（空列表），则回退策略：
        - 尝试加载 Skill 目录中所有脚本
        - 将 .py 文件构造为 python_code 命令，.sh 文件构造为 shell 命令

        Args:
            context: 已加载资源的 SkillContext。

        Returns:
            命令字典列表，每个 dict 含 type/code/path/requirements 等字段。
        """
        if not context.plan:
            return []

        prompt = PROMPT_SKILL_EXECUTION_COMMAND.format(
            query=context.query,
            skill_id=context.skill.skill_id,
            execution_plan=json.dumps(
                {
                    'plan_summary': context.plan.plan_summary,
                    'steps': context.plan.steps,
                    'parameters': context.plan.parameters,
                },
                indent=2),
            scripts_content=context.get_loaded_scripts_content(),
            references_content=context.get_loaded_references_content()[:2000],
            resources_content=context.get_loaded_resources_content()[:2000])

        response = self._llm_generate(prompt)
        parsed = self._parse_json_response(response)

        commands = parsed.get('commands', [])

        # Fallback: if no commands generated, try to use loaded scripts directly
        if not commands:
            # If no scripts loaded yet, try to load all available scripts
            if not context.scripts and context.skill.scripts:
                logger.info(
                    f'Loading all scripts as fallback: {[s.name for s in context.skill.scripts]}')
                context.load_scripts()  # Load all scripts

            if context.scripts:
                logger.warning(
                    f'No commands generated, using {len(context.scripts)} loaded scripts as fallback')
                # context.scripts is List[Dict] with keys: name, file, path, abs_path, content
                for script_info in context.scripts:
                    script_name = script_info.get('name', '')
                    script_content = script_info.get('content', '')
                    if script_name.endswith('.py') and script_content:
                        commands.append({
                            'type': 'python_code',
                            'code': script_content,
                            'requirements': context.plan.required_packages if context.plan else []
                        })
                    elif script_name.endswith('.sh') and script_content:
                        commands.append({
                            'type': 'shell',
                            'code': script_content
                        })

        context.spec.tasks = json.dumps(commands, indent=2)

        return commands

    async def analyze_and_prepare(
            self,
            skill: SkillSchema,
            query: str,
            root_path: Path = None
    ) -> Tuple[SkillContext, List[Dict[str, Any]]]:
        """
        渐进式分析的完整入口：依次执行 Plan → Load → GenerateCommands 三阶段。

        三个阶段均通过 asyncio.to_thread 在线程池中运行（避免阻塞事件循环），
        支持在异步环境下并发执行多个 Skill 的分析。

        若 Phase 1 判断 can_handle=False，立即返回 (context, [])，不执行后续阶段。

        Args:
            skill:     待分析的 SkillSchema。
            query:     用户任务描述。
            root_path: Skill 上下文根路径。

        Returns:
            (SkillContext, 命令列表)。若 Skill 无法处理 query，命令列表为空。
        """
        # Phase 1: Create plan
        context = await asyncio.to_thread(self.analyze_skill_plan, skill,
                                          query, root_path)

        if not context.plan or not context.plan.can_handle:
            return context, []

        # Phase 2: Load resources
        await asyncio.to_thread(self.load_skill_resources, context)

        # Phase 3: Generate commands
        commands = await asyncio.to_thread(self.generate_execution_commands,
                                           context)

        return context, commands


@dataclass
class SkillDAGResult:
    """
    AutoSkills 一次运行的完整结果，包含 Skill 选择、DAG 结构和执行结果。

    Attributes:
        dag:              邻接表，dag[A]=[B,C] 表示 A 依赖 B 和 C。
        execution_order:  拓扑排序后的执行顺序；子列表中的 Skill 可并行执行。
        selected_skills:  最终选定的 Skill 字典（skill_id → SkillSchema）。
        is_complete:      当前选出的 Skill 是否足以完成任务。
        clarification:    若 Skill 不足，包含向用户请求补充信息的提示语。
        chat_response:    纯聊天模式下（无需 Skill）的直接回复文本。
        execution_result: DAG 执行完毕后填充，包含每个 Skill 的执行结果汇总。
    """
    dag: Dict[str, List[str]] = field(default_factory=dict)
    execution_order: List[Union[str, List[str]]] = field(default_factory=list)
    selected_skills: Dict[str, SkillSchema] = field(default_factory=dict)
    is_complete: bool = False
    clarification: Optional[str] = None
    chat_response: Optional[str] = None
    execution_result: Optional[DAGExecutionResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """将 SkillDAGResult 序列化为普通字典，便于 JSON 输出或日志记录。"""
        return {
            'dag':
            self.dag,
            'execution_order':
            self.execution_order,
            'selected_skills':
            {k: v.__dict__
             for k, v in self.selected_skills.items()},
            'is_complete':
            self.is_complete,
            'clarification':
            self.clarification,
            'chat_response':
            self.chat_response,
            'execution_result':
            self.execution_result.__dict__ if self.execution_result else None,
        }


class DAGExecutor:
    """
    Skill DAG 的执行引擎，支持依赖感知的并行执行。

    核心职责：
    - 按 execution_order 依次（或并行）执行各 Skill
    - 将上游 Skill 的输出通过环境变量注入下游 Skill（数据链路）
    - 支持渐进式分析（Progressive Analysis）：Plan → Load → Execute
    - 支持自我反思重试（Self-Reflection）：失败后让 LLM 修复代码再重试

    并行策略：
      execution_order 中的子列表（List[str]）表示可并行的 Skill 组，
      通过 asyncio.gather 并发执行；单个字符串则串行执行。
    """

    def __init__(self,
                 container: SkillContainer,
                 skills: Dict[str, SkillSchema],
                 workspace_dir: Optional[Path] = None,
                 llm: 'LLM' = None,
                 enable_progressive_analysis: bool = True,
                 enable_self_reflection: bool = True,
                 max_retries: int = 3):
        """
        初始化 DAGExecutor。

        Args:
            container:                   执行 Skill 的沙箱容器（Docker 或本地）。
            skills:                      全量 Skill 字典（skill_id → SkillSchema）。
            workspace_dir:               Skill 执行的工作目录，默认使用 container.workspace_dir。
            llm:                         用于渐进式分析和自我反思的 LLM 实例；为 None 时两者均禁用。
            enable_progressive_analysis: 是否启用渐进式分析（需要 llm 不为 None）。
            enable_self_reflection:      是否启用自我反思重试（需要 llm 不为 None）。
            max_retries:                 每个 Skill 的最大重试次数（含首次执行）。

        内部状态：
            _outputs:            已执行 Skill 的输出缓存，供下游 Skill 引用。
            _contexts:           各 Skill 的 SkillContext，用于记录分析结果。
            _execution_attempts: 各 Skill 的已尝试执行次数，用于日志。
        """
        self.container = container
        self.skills = skills
        self.workspace_dir = workspace_dir or container.workspace_dir
        self.llm = llm
        self.enable_progressive_analysis = enable_progressive_analysis and llm is not None
        self.enable_self_reflection = enable_self_reflection and llm is not None
        self.max_retries = max_retries

        # Skill analyzer for progressive analysis
        self._analyzer: Optional[SkillAnalyzer] = None
        if self.enable_progressive_analysis:
            self._analyzer = SkillAnalyzer(llm)

        # Execution state: stores outputs keyed by skill_id
        self._outputs: Dict[str, ExecutionOutput] = {}

        # Skill contexts from progressive analysis
        self._contexts: Dict[str, SkillContext] = {}

        # Track execution attempts for retry logging
        self._execution_attempts: Dict[str, int] = {}

    def _get_skill_dependencies(self, skill_id: str,
                                dag: Dict[str, List[str]]) -> List[str]:
        """
        从 DAG 中获取指定 Skill 的直接前置依赖列表。

        dag[A] = [B, C] 表示 A 依赖 B 和 C，故返回 [B, C]。
        若 skill_id 不在 DAG 中（无依赖），返回空列表。
        """
        return dag.get(skill_id, [])

    def _build_execution_input(
            self,
            skill_id: str,
            dag: Dict[str, List[str]],
            execution_input: Optional[ExecutionInput] = None) -> ExecutionInput:
        """
        为指定 Skill 构建执行输入，将上游依赖的输出注入为环境变量。

        以 base_input 为基础，查找所有上游依赖 Skill 的执行输出，
        将其 stdout/stderr/exit_code/output_files 序列化为 JSON，
        写入环境变量 UPSTREAM_OUTPUTS 传给下游进程。

        Args:
            skill_id:        当前要执行的 Skill ID。
            dag:             技能依赖 DAG（邻接表）。
            execution_input: 用户提供的原始输入（可为 None）。

        Returns:
            注入了上游输出的 ExecutionInput。
        """
        base_input = execution_input or ExecutionInput()

        # Get outputs from upstream dependencies
        dependencies = self._get_skill_dependencies(skill_id, dag)
        upstream_data: Dict[str, Any] = {}

        for dep_id in dependencies:
            if dep_id in self._outputs:
                dep_output = self._outputs[dep_id]
                # Pass stdout/return_value as upstream data
                upstream_data[dep_id] = {
                    'stdout': dep_output.stdout,
                    'stderr': dep_output.stderr,
                    'return_value': dep_output.return_value,
                    'exit_code': dep_output.exit_code,
                    'output_files':
                    {k: str(v)
                     for k, v in dep_output.output_files.items()},
                }

        # ── 上游输出注入 ──────────────────────────────────────────────
        # DAG 中，下游 Skill 需要感知上游 Skill 的产出。
        # 这里的做法是：把所有上游输出序列化成 JSON，
        # 通过环境变量 UPSTREAM_OUTPUTS 传给下游进程。
        # 下游脚本可用 os.environ['UPSTREAM_OUTPUTS'] 读取。
        # 对于较长的 stdout，还单独提供 UPSTREAM_{ID}_STDOUT 便于快速访问。
        # ─────────────────────────────────────────────────────────────
        # Inject upstream data into environment variables as JSON
        env_vars = base_input.env_vars.copy()
        if upstream_data:
            env_vars['UPSTREAM_OUTPUTS'] = json.dumps(upstream_data)
            # Also provide individual upstream references
            for dep_id, data in upstream_data.items():
                safe_key = dep_id.replace('-', '_').replace('.', '_').replace('@', '_').replace('/', '_').upper()
                if data.get('stdout'):
                    env_vars[f'UPSTREAM_{safe_key}_STDOUT'] = data[
                        'stdout'][:4096]

        return ExecutionInput(
            args=base_input.args,
            kwargs=base_input.kwargs,
            env_vars=env_vars,
            input_files=base_input.input_files,
            stdin=base_input.stdin,
            working_dir=base_input.working_dir,
            requirements=base_input.requirements,
        )

    def _determine_executor_type(self, skill: SkillSchema) -> ExecutorType:
        """
        根据 Skill 的主脚本文件扩展名确定执行器类型。

        判断规则（取第一个脚本的类型）：
        - .py              → PYTHON_SCRIPT
        - .sh / .bash      → SHELL
        - .js / .mjs       → JAVASCRIPT
        - 无脚本或未知扩展名 → PYTHON_CODE（直接执行 skill.content 字符串）
        """
        if not skill.scripts:
            return ExecutorType.PYTHON_CODE

        # Check first script's extension
        primary_script = skill.scripts[0]
        ext = primary_script.type.lower()

        if ext in ['.py']:
            return ExecutorType.PYTHON_SCRIPT
        elif ext in ['.sh', '.bash']:
            return ExecutorType.SHELL
        elif ext in ['.js', '.mjs']:
            return ExecutorType.JAVASCRIPT
        else:
            return ExecutorType.PYTHON_CODE

    async def _execute_single_skill(
            self,
            skill_id: str,
            dag: Dict[str, List[str]],
            execution_input: Optional[ExecutionInput] = None,
            query: str = '') -> SkillExecutionResult:
        """
        执行单个 Skill，自动处理上游依赖输入注入和执行模式选择。

        执行流程：
        1. 从 DAG 中收集上游依赖的输出，构建 ExecutionInput
        2. 若启用渐进式分析 → 调用 _execute_with_progressive_analysis
           否则 → 调用 _execute_direct（直接运行脚本文件）
        3. 捕获所有异常，确保单个 Skill 失败不会崩溃整个 DAG

        Args:
            skill_id:        要执行的 Skill 标识符。
            dag:             技能依赖 DAG。
            execution_input: 用户提供的初始输入（可为 None）。
            query:           用户原始 query，供渐进式分析使用。

        Returns:
            SkillExecutionResult，无论成功或失败均返回（不抛出异常）。
        """
        skill = self.skills.get(skill_id)
        if not skill:
            return SkillExecutionResult(
                skill_id=skill_id,
                success=False,
                error=f'Skill not found: {skill_id}')

        try:
            # Build base input with upstream outputs
            exec_input = self._build_execution_input(skill_id, dag, execution_input)

            # Use progressive analysis if enabled
            if self.enable_progressive_analysis and self._analyzer:
                return await self._execute_with_progressive_analysis(
                    skill, skill_id, exec_input, query)

            # Fallback: direct execution without progressive analysis
            return await self._execute_direct(skill, skill_id, exec_input)

        except Exception as e:
            logger.error(f'Skill execution failed for {skill_id}: {e}')
            return SkillExecutionResult(
                skill_id=skill_id, success=False, error=str(e))

    async def _execute_with_progressive_analysis(
            self, skill: SkillSchema, skill_id: str,
            exec_input: ExecutionInput, query: str) -> SkillExecutionResult:
        """
        Execute skill using progressive analysis.

        Args:
            skill: SkillSchema to execute.
            skill_id: Skill identifier.
            exec_input: Execution input with upstream data.
            query: User query for context.

        Returns:
            SkillExecutionResult with execution outcome.

        【渐进式分析（Progressive Analysis）的设计意图】

        朴素做法：把 Skill 的所有脚本、资源、文档一股脑塞给 LLM，让它生成执行命令。
        问题：Token 浪费严重，且 Skill 包内容越多越容易让 LLM 迷失。

        渐进式做法（三阶段）：
          Phase 1 — 制定计划（analyze_skill_plan）
            仅提供 Skill 的元数据（名称/描述/SKILL.md 概览），
            让 LLM 判断该 Skill 能否处理当前 query，
            并列出执行所需的具体脚本、参考文档、Python 包。
            → 输出：SkillExecutionPlan（can_handle / required_scripts / ...）

          Phase 2 — 按需加载（load_skill_resources）
            根据 Phase 1 的计划，只加载"声明要用到"的资源，而非全部。
            → 减少无关内容对 LLM 的干扰，降低 Token 消耗。

          Phase 3 — 生成命令（generate_execution_commands）
            把已加载的脚本内容 + 执行计划传给 LLM，
            让它生成具体的执行命令（python_code / python_script / shell / javascript）。
            → 输出：List[Dict]，每个 dict 是一条可执行的命令。

        若 Phase 1 判断 can_handle=False，直接返回失败，不执行后续阶段。
        若 Phase 3 未生成任何命令，回退到直接执行 Skill 目录中的脚本文件。
        """
        # Phase 1 & 2: Analyze and load resources
        # Use skill's directory as root_path for proper file resolution
        context, commands = await self._analyzer.analyze_and_prepare(
            skill, query, skill.skill_path)

        # Store context for reference
        self._contexts[skill_id] = context

        # Mount skill directory in container for sandbox access
        self.container.mount_skill_directory(skill_id, skill.skill_path)

        if not context.plan or not context.plan.can_handle:
            return SkillExecutionResult(
                skill_id=skill_id,
                success=False,
                error=
                f'Skill cannot handle query: {context.plan.reasoning if context.plan else "No plan"}'
            )

        if not commands:
            return SkillExecutionResult(
                skill_id=skill_id,
                success=False,
                error='No execution commands generated')

        # Phase 3: Execute commands with retry support for all types
        outputs: List[ExecutionOutput] = []
        for cmd in commands:
            cmd_type = cmd.get('type', 'python_code')

            # Use retry mechanism for all command types
            if self.enable_self_reflection:
                output = await self._execute_command_with_retry(
                    cmd=cmd,
                    cmd_type=cmd_type,
                    skill_id=skill_id,
                    exec_input=exec_input,
                    context=context,
                    skill=skill,
                    query=query)
            else:
                # Self-reflection disabled - execute without retry
                output = await self._execute_command(cmd, cmd_type, skill_id,
                                                     exec_input, context)
            outputs.append(output)

            if output.exit_code != 0:
                # Stop on first failure (after retries exhausted)
                break

        # Merge outputs
        final_output = self._merge_outputs(outputs)

        # Store output for downstream skills
        self._outputs[skill_id] = final_output
        self.container.spec.link_upstream(skill_id, final_output)

        return SkillExecutionResult(
            skill_id=skill_id,
            success=(final_output.exit_code == 0),
            output=final_output,
            error=final_output.stderr if final_output.exit_code != 0 else None)

    async def _execute_direct(
            self, skill: SkillSchema, skill_id: str,
            exec_input: ExecutionInput) -> SkillExecutionResult:
        """
        不经过渐进式分析，直接执行 Skill 的脚本文件（降级模式）。

        适用场景：enable_progressive_analysis=False，或 LLM 不可用时。
        执行逻辑：
        - 若 Skill 有脚本文件 → 执行第一个脚本（按扩展名决定执行器类型）
        - 若无脚本 → 将 skill.content 作为 Python 代码字符串执行

        执行完成后将输出写入 _outputs 并通知容器（用于上下游数据链路）。
        """
        # Mount skill directory for sandbox access
        self.container.mount_skill_directory(skill_id, skill.skill_path)

        executor_type = self._determine_executor_type(skill)

        if skill.scripts:
            script_path = skill.scripts[0].path
            output = await self.container.execute(
                executor_type=executor_type,
                skill_id=skill_id,
                script_path=script_path,
                input_spec=exec_input)
        else:
            output = await self.container.execute_python_code(
                code=skill.content or '# No executable content',
                skill_id=skill_id,
                input_spec=exec_input)

        self._outputs[skill_id] = output
        self.container.spec.link_upstream(skill_id, output)

        return SkillExecutionResult(
            skill_id=skill_id,
            success=(output.exit_code == 0),
            output=output,
            error=output.stderr if output.exit_code != 0 else None)

    async def _execute_command(self, cmd: Dict[str, Any], cmd_type: str,
                               skill_id: str, exec_input: ExecutionInput,
                               context: SkillContext) -> ExecutionOutput:
        """
        执行渐进式分析生成的单条命令。

        根据 cmd_type 路由到不同的容器执行方法：
        - python_script: 执行指定路径的 .py 文件（先在 skill_dir 找，找不到再在 root_path 找）
        - python_code:   直接执行代码字符串
        - shell:         执行 shell 命令字符串
        - javascript:    执行 JS 代码字符串
        - 其他/未知:     降级为 python_code

        执行前会合并三处来源的 Python 包依赖（plan/cmd/exec_input），去重后传给容器。
        命令参数（parameters）会同时注入 args、kwargs 和环境变量（大写 key）。

        Args:
            cmd:        命令字典（含 type/code/path/parameters/requirements）。
            cmd_type:   命令类型字符串。
            skill_id:   当前 Skill 标识符。
            exec_input: 基础执行输入（含上游注入的环境变量）。
            context:    已加载资源的 SkillContext。

        Returns:
            ExecutionOutput（含 stdout/stderr/exit_code/output_files）。
        """
        # Merge parameters into input
        params = cmd.get('parameters', {})
        # Use skill directory as working directory for proper file access
        working_dir = exec_input.working_dir or context.skill_dir

        # Collect all requirements: from plan, command, and input
        all_requirements = []
        if context.plan and context.plan.required_packages:
            all_requirements.extend(context.plan.required_packages)
        all_requirements.extend(cmd.get('requirements', []))
        all_requirements.extend(exec_input.requirements)
        # Deduplicate while preserving order
        seen = set()
        unique_requirements = []
        for req in all_requirements:
            if req not in seen:
                seen.add(req)
                unique_requirements.append(req)

        merged_input = ExecutionInput(
            args=exec_input.args + list(params.values()),
            kwargs={
                **exec_input.kwargs,
                **params
            },
            env_vars={
                **exec_input.env_vars,
                'SKILL_DIR': str(context.skill_dir),
                **{k.upper(): str(v)
                   for k, v in params.items()}
            },
            input_files=exec_input.input_files,
            stdin=exec_input.stdin,
            working_dir=working_dir,
            requirements=unique_requirements)

        if cmd_type == 'python_script':
            script_path = cmd.get('path')
            if script_path:
                # Resolve path relative to skill directory
                full_path = context.skill_dir / script_path
                if not full_path.exists():
                    full_path = context.root_path / script_path
                return await self.container.execute_python_script(
                    script_path=full_path,
                    skill_id=skill_id,
                    input_spec=merged_input)
            else:
                code = cmd.get('code', '')
                return await self.container.execute_python_code(
                    code=code, skill_id=skill_id, input_spec=merged_input)

        elif cmd_type == 'python_code':
            code = cmd.get('code', '')
            return await self.container.execute_python_code(
                code=code, skill_id=skill_id, input_spec=merged_input)

        elif cmd_type == 'shell':
            command = cmd.get('code') or cmd.get('command', '')
            return await self.container.execute_shell(
                command=command, skill_id=skill_id, input_spec=merged_input)

        elif cmd_type == 'javascript':
            code = cmd.get('code', '')
            return await self.container.execute_javascript(
                code=code, skill_id=skill_id, input_spec=merged_input)

        else:
            # Default to python code
            code = cmd.get('code', '')
            return await self.container.execute_python_code(
                code=code, skill_id=skill_id, input_spec=merged_input)

    async def _execute_command_with_retry(
            self, cmd: Dict[str, Any], cmd_type: str,
            skill_id: str, exec_input: ExecutionInput,
            context: SkillContext, skill: SkillSchema,
            query: str) -> ExecutionOutput:
        """
        Execute a command with retry logic for all execution types.

        Always retries up to max_retries times. Uses LLM analysis to improve
        the fix between retries when self-reflection is enabled.

        Args:
            cmd: Command dictionary.
            cmd_type: Type of command.
            skill_id: Skill identifier.
            exec_input: Base execution input.
            context: SkillContext.
            skill: SkillSchema for error analysis.
            query: User query for context.

        Returns:
            ExecutionOutput from command execution.

        【自我反思（Self-Reflection）重试机制】

        普通重试：失败后原封不动重试，对「代码本身有 bug」的情况无效。

        自我反思重试（Self-Reflection）：
          每次执行失败后，将失败代码 + stderr + stdout 传给 LLM，
          让 LLM 分析错误原因并给出修复后的代码，再用修复版重试。

        流程：
          for attempt in 1..max_retries:
              执行当前 cmd
              if 成功 → return
              if 不是最后一次 且 cmd_type 是 Python 代码:
                  _analyze_execution_error() → LLM 返回 {is_fixable, fixed_code, additional_requirements}
                  if is_fixable → 替换 cmd['code'] 为 fixed_code（下一轮用修复版）
                  if 需要额外包 → 更新 exec_input.requirements
          return 最后一次执行结果（即使失败）

        局限性：
        - 自我反思只对 python_code / python_script 类型生效（shell/js 直接重试）
        - LLM 不一定能修复所有错误（如缺少外部服务、权限不足等）
        - max_retries 默认为 3，避免无限循环消耗 Token
        """
        current_cmd = cmd.copy()
        last_output = None

        for attempt in range(1, self.max_retries + 1):
            self._execution_attempts[skill_id] = attempt
            logger.info(f'[{skill_id}] Execution attempt {attempt}/{self.max_retries}')

            # Execute the command
            output = await self._execute_command(
                current_cmd, cmd_type, skill_id, exec_input, context)
            last_output = output

            # Check if successful
            if output.exit_code == 0:
                if attempt > 1:
                    logger.info(
                        f'[{skill_id}] Execution succeeded after {attempt} attempts')
                return output

            # Collect error info
            error_msg = output.stderr[:500] if output.stderr else 'Unknown error'
            logger.warning(f'[{skill_id}] Attempt {attempt} failed: {error_msg[:200]}')

            # Last attempt - no need to analyze
            if attempt >= self.max_retries:
                logger.warning(
                    f'[{skill_id}] Max retries ({self.max_retries}) reached')
                continue

            # Try to analyze and fix if self-reflection is enabled
            if self.enable_self_reflection and cmd_type in ('python_code', 'python_script'):
                code = current_cmd.get('code', '')
                if code:
                    logger.info(f'[{skill_id}] Analyzing error for retry...')
                    analysis = self._analyze_execution_error(
                        skill=skill,
                        failed_code=code,
                        output=output,
                        query=query,
                        attempt=attempt)

                    error_info = analysis.get('error_analysis', {})
                    is_fixable = error_info.get('is_fixable', False)
                    fixed_code = analysis.get('fixed_code')
                    additional_reqs = analysis.get('additional_requirements', [])

                    logger.info(
                        f'[{skill_id}] Error analysis: type={error_info.get("error_type")}, '
                        f'fixable={is_fixable}')

                    # Apply fix if available
                    if is_fixable and fixed_code:
                        current_cmd = current_cmd.copy()
                        current_cmd['code'] = fixed_code
                        logger.info(f'[{skill_id}] Applying fix')

                    # Add additional requirements
                    if additional_reqs:
                        logger.info(f'[{skill_id}] Adding requirements: {additional_reqs}')
                        exec_input = ExecutionInput(
                            args=exec_input.args,
                            kwargs=exec_input.kwargs,
                            env_vars=exec_input.env_vars,
                            input_files=exec_input.input_files,
                            working_dir=exec_input.working_dir,
                            requirements=list(set(exec_input.requirements + additional_reqs)))
            else:
                logger.info(f'[{skill_id}] Retrying without code modification')

        logger.error(f'[{skill_id}] All {self.max_retries} attempts failed')
        return last_output

    def _merge_outputs(self,
                       outputs: List[ExecutionOutput]) -> ExecutionOutput:
        """
        将多条命令的输出合并为一个 ExecutionOutput。

        合并规则：
        - stdout/stderr: 用换行符拼接所有非空输出
        - exit_code:     取第一个非 0 的 exit_code；全部为 0 则结果为 0
        - duration_ms:   所有命令耗时之和
        - output_files:  合并所有命令产生的输出文件（后者覆盖同名文件）
        """
        if not outputs:
            return ExecutionOutput()
        if len(outputs) == 1:
            return outputs[0]

        # Merge all outputs
        merged_stdout = '\n'.join(o.stdout for o in outputs if o.stdout)
        merged_stderr = '\n'.join(o.stderr for o in outputs if o.stderr)
        final_exit_code = next(
            (o.exit_code for o in outputs if o.exit_code != 0), 0)
        total_duration = sum(o.duration_ms for o in outputs)

        # Merge output files
        merged_files = {}
        for o in outputs:
            merged_files.update(o.output_files)

        return ExecutionOutput(
            stdout=merged_stdout,
            stderr=merged_stderr,
            exit_code=final_exit_code,
            output_files=merged_files,
            duration_ms=total_duration)

    def _analyze_execution_error(
            self,
            skill: SkillSchema,
            failed_code: str,
            output: ExecutionOutput,
            query: str,
            attempt: int) -> Dict[str, Any]:
        """
        对执行失败的代码进行 LLM 错误分析，返回修复方案。

        将失败代码、stderr（最多 3000 字符）、stdout（最多 1000 字符）
        以及当前重试次数发送给 LLM，请求其分析错误原因并给出修复后的代码。

        返回字典结构：
        {
            "error_analysis": {
                "error_type":  错误类型（如 ImportError / TypeError 等）,
                "is_fixable":  是否可通过修改代码修复（bool）,
                "reason":      错误原因描述
            },
            "fixed_code":            修复后的完整代码字符串（可为 None）,
            "additional_requirements": 需要额外安装的 Python 包列表
        }

        若 LLM 不可用或解析失败，返回 is_fixable=False 的兜底结果。
        """
        if not self.llm:
            return {'error_analysis': {'is_fixable': False},
                    'fixed_code': None}

        prompt = PROMPT_ANALYZE_EXECUTION_ERROR.format(
            query=query,
            skill_id=skill.skill_id,
            skill_name=skill.name,
            failed_code=failed_code[:8000],  # Limit code length
            stderr=output.stderr[:3000] if output.stderr else '',
            stdout=output.stdout[:1000] if output.stdout else '',
            attempt=attempt,
            max_attempts=self.max_retries)

        try:
            response = self.llm.generate(
                messages=[Message(role='user', content=prompt)])
            # Parse JSON response - handle different response formats
            response_text = (response.content if hasattr(response, 'content')
                             else str(response)).strip()
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning(f'Error analyzing execution failure: {e}')

        return {'error_analysis': {'is_fixable': False}, 'fixed_code': None}

    async def _execute_parallel_group(
            self,
            skill_ids: List[str],
            dag: Dict[str, List[str]],
            execution_input: Optional[ExecutionInput] = None,
            query: str = '') -> List[SkillExecutionResult]:
        """
        并行执行一组相互独立的 Skill（它们之间没有依赖关系）。

        使用 asyncio.gather 并发启动所有 Skill 的执行任务，
        等待全部完成后返回结果列表（顺序与 skill_ids 一致）。

        Args:
            skill_ids:       要并行执行的 Skill ID 列表。
            dag:             技能依赖 DAG（用于注入各自的上游输出）。
            execution_input: 用户提供的初始输入，所有并行 Skill 共享。
            query:           用户 query，供渐进式分析使用。

        Returns:
            每个 Skill 的 SkillExecutionResult 列表。
        """
        tasks = [
            self._execute_single_skill(sid, dag, execution_input, query)
            for sid in skill_ids
        ]
        return await asyncio.gather(*tasks)

    async def execute(self,
                      dag: Dict[str, List[str]],
                      execution_order: List[Union[str, List[str]]],
                      execution_input: Optional[ExecutionInput] = None,
                      stop_on_failure: bool = True,
                      query: str = '') -> DAGExecutionResult:
        """
        Execute the skill DAG according to execution order.

        Execution order format: [skill1, skill2, [skill3, skill4], skill5, ...]
        - Single string items are executed sequentially
        - List items (sublists) are executed in parallel

        Args:
            dag: Skill dependency DAG (adjacency list).
            execution_order: Ordered list with parallel groups as sublists.
            execution_input: Optional initial input for all skills.
            stop_on_failure: Whether to stop execution on first failure.
            query: User query for progressive skill analysis.

        Returns:
            DAGExecutionResult with all execution outcomes.
        """
        import time
        start_time = time.time()

        results: Dict[str, SkillExecutionResult] = {}
        actual_order: List[Union[str, List[str]]] = []
        all_success = True

        for item in execution_order:
            if isinstance(item, list):
                # Parallel execution group
                group_results = await self._execute_parallel_group(
                    item, dag, execution_input, query)
                for res in group_results:
                    results[res.skill_id] = res
                    if not res.success:
                        all_success = False
                actual_order.append(item)

                if not all_success and stop_on_failure:
                    logger.warning(
                        f'Stopping DAG execution due to failure in parallel group: {item}'
                    )
                    break
            else:
                # Sequential execution
                result = await self._execute_single_skill(
                    item, dag, execution_input, query)
                results[result.skill_id] = result
                actual_order.append(item)

                if not result.success:
                    all_success = False
                    if stop_on_failure:
                        logger.warning(
                            f'Stopping DAG execution due to failure: {item}')
                        break

        total_duration = (time.time() - start_time) * 1000

        return DAGExecutionResult(
            success=all_success,
            results=results,
            execution_order=actual_order,
            total_duration_ms=total_duration)

    def get_skill_context(self, skill_id: str) -> Optional[SkillContext]:
        """Get the skill context from progressive analysis."""
        return self._contexts.get(skill_id)

    def get_all_contexts(self) -> Dict[str, SkillContext]:
        """Get all skill contexts from progressive analysis."""
        return self._contexts.copy()

    def get_executed_skill_ids(self) -> List[str]:
        """Get list of skill_ids that have been executed with contexts."""
        return list(self._contexts.keys())


class AutoSkills:
    """
    Automatic skill retrieval and DAG construction for user queries.

    Uses hybrid retrieval (dense + sparse) to find relevant skills,
    with LLM-based analysis and reflection loop for completeness checking.
    Supports DAG-based skill execution with dependency management.
    """

    def __init__(self,
                 skills: Union[str, List[str], List[SkillSchema]],
                 llm: LLM,
                 enable_retrieve: Union[bool, None] = None,
                 retrieve_args: Dict[str, Any] = None,
                 max_candidate_skills: int = 10,
                 max_retries: int = 3,
                 work_dir: Optional[Union[str, Path]] = None,
                 use_sandbox: bool = True,
                 **kwargs):
        """
        Initialize AutoSkills with skills corpus and retriever.

        Args:
            skills: Path(s) to skill directories or list of SkillSchema.
                Alternatively, single repo_id or list of repo_ids from ModelScope.
                e.g. skills='ms-agent/claude_skills', refer to `https://modelscope.cn/models/ms-agent/claude_skills`
            llm: LLM instance for query analysis and evaluation.
            enable_retrieve: If True, use HybridRetriever for skill search.
                If False, put all skills into LLM context for direct selection.
                If None, enable search only if skills > 10 automatically.
            retrieve_args: Additional arguments for HybridRetriever.
                Attributes:
                    top_k: Number of top results to retrieve per query.
                    min_score: Minimum score threshold for retrieval.
            max_candidate_skills: Maximum number of candidate skills to consider.
            max_retries: Maximum retry attempts for failed executions for each skill.
            work_dir: Working directory for skill execution.
            use_sandbox: Whether to use Docker sandbox for execution.

        Examples:
            >>> from omegaconf import DictConfig
            >>> from ms_agent.llm.openai_llm import OpenAI
            >>> from ms_agent.skill.auto_skills import SkillDAGResult
            >>> config = DictConfig(
                {
                    'llm': {
                        'service': 'openai',
                        'model': 'gpt-4',
                        'openai_api_key': 'your-api-key',
                        'openai_base_url': 'your-base-url'
                        }
                    }
            >>> )
            >>> llm_instance = OpenAI.from_config(config)
            >>> auto_skills = AutoSkills(
                skills='/path/to/skills',
                llm=llm_instance,
                )
            >>> async def main():
                    result: SkillDAGResult = await auto_skills.run(query='Analyze sales data and generate mock report for Nvidia Q4 2025 in PDF format.')
                    print(result.execution_result)
            >>> import asyncio
            >>> asyncio.run(main())
        """
        # Dict of <skill_id, SkillSchema>
        self.all_skills: Dict[str, SkillSchema] = load_skills(skills=skills)
        logger.info(f'Loaded {len(self.all_skills)} skills from {skills}')

        self.llm = llm
        self.enable_retrieve = len(
            self.all_skills) > 10 if enable_retrieve is None else enable_retrieve
        retrieve_args = retrieve_args or {}
        self.top_k = retrieve_args.get('top_k', 3)
        self.min_score = retrieve_args.get('min_score', 0.8)
        self.max_candidate_skills = max_candidate_skills
        self.max_retries = max_retries
        self.work_dir = Path(work_dir) if work_dir else None
        self.use_sandbox = use_sandbox
        self.kwargs = kwargs

        if self.use_sandbox:
            from ms_agent.utils.docker_utils import is_docker_daemon_running
            if not is_docker_daemon_running():
                raise RuntimeError(
                    'Docker daemon is not running. Please start Docker to use sandbox mode.'
                )

        # Configure logger to output to work_dir/logs if work_dir is specified
        if self.work_dir:
            _configure_logger_to_dir(self.work_dir / 'logs')

        # Build corpus and skill_id mapping
        self.corpus: List[str] = []
        self.corpus_to_skill_id: Dict[str, str] = {}
        self._build_corpus()

        # Initialize retriever only if search is enabled
        self.retriever: Optional[HybridRetriever] = None
        if self.enable_retrieve and self.corpus:
            self.retriever = HybridRetriever(corpus=self.corpus, **kwargs)

        # Container and executor (lazy initialization)
        self._container: Optional[SkillContainer] = None
        self._executor: Optional[DAGExecutor] = None

    def _build_corpus(self):
        """Build corpus from skills for retriever indexing."""
        for skill_id, skill in self.all_skills.items():
            # Concatenate skill_id, name, description as corpus document
            doc = f'[{skill_id}] {skill.name}: {skill.description}'
            self.corpus.append(doc)
            self.corpus_to_skill_id[doc] = skill_id

    def _extract_skill_id_from_doc(self, doc: str) -> Optional[str]:
        """Extract skill_id from corpus document string."""
        # First try direct lookup
        if doc in self.corpus_to_skill_id:
            return self.corpus_to_skill_id[doc]
        # Fallback: extract from [skill_id] pattern
        match = re.match(r'\[([^\]]+)\]', doc)
        return match.group(1) if match else None

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response with robust extraction."""
        # Remove markdown code blocks if present
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*$', '', response)
        response = response.strip()

        # Try direct parsing first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON object from response
        try:
            # Find the outermost JSON object
            start = response.find('{')
            if start != -1:
                # Find matching closing brace
                depth = 0
                for i, char in enumerate(response[start:], start):
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            json_str = response[start:i + 1]
                            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Try regex extraction as fallback
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        logger.warning(f'Failed to parse JSON response: {response[:300]}...')
        return {}

    def _get_skills_overview(self, limit: int = 20) -> str:
        """Generate a brief overview of all available skills."""
        lines = []
        for skill_id, skill in self.all_skills.items():
            lines.append(
                f'- [{skill_id}] {skill.name}: {skill.description[:200]}')
        return '\n'.join(lines[:limit])  # Limit to avoid token overflow

    def _get_all_skills_context(self) -> str:
        """Generate full context of all skills for direct LLM selection."""
        lines = []
        for skill_id, skill in self.all_skills.items():
            lines.append(f'- [{skill_id}] {skill.name}\n  {skill.description}')
        return '\n'.join(lines)

    def _format_retrieved_skills(self, skill_ids: Set[str]) -> str:
        """Format retrieved skills for LLM prompt."""
        lines = []
        for skill_id in skill_ids:
            if skill_id in self.all_skills:
                skill = self.all_skills[skill_id]
                lines.append(
                    f'- [{skill_id}] {skill.name}\n  {skill.description}\n Main Content: {skill.content[:3000]}')
        return '\n'.join(lines)

    def _llm_generate(self, prompt: str) -> str:
        """Generate LLM response from prompt."""
        messages = [Message(role='user', content=prompt)]
        logger.debug(f'Input msg to LLM: {messages}')       # set env `LOG_LEVEL=DEBUG`
        response = self.llm.generate(messages=messages)
        res = response.content if hasattr(response,
                                           'content') else str(response)
        logger.debug('LLM response: {}'.format(res))
        return res

    async def _async_llm_generate(self, prompt: str) -> str:
        """Async wrapper for LLM generation."""
        return await asyncio.to_thread(self._llm_generate, prompt)

    def _analyze_query(
        self,
        query: str,
    ) -> Tuple[bool, str, List[str], Optional[str]]:
        """
        Analyze user query to determine if skills are needed.

        Args:
            query: User's original query.

        Returns:
            Tuple of (needs_skills, intent_summary, skill_queries, chat_response).
        """
        prompt = PROMPT_ANALYZE_QUERY_FOR_SKILLS.format(
            query=query, skills_overview=self._get_skills_overview())
        response = self._llm_generate(prompt)
        parsed = self._parse_json_response(response)

        needs_skills = parsed.get('needs_skills', True)
        intent = parsed.get('intent_summary', query)
        queries = parsed.get('skill_queries', [query])
        chat_response = parsed.get('chat_response')
        return needs_skills, intent, queries if queries else [query
                                                              ], chat_response

    async def _async_retrieve_skills(self, queries: List[str]) -> Set[str]:
        """
        Retrieve skills for multiple queries in parallel.

        Args:
            queries: List of search queries.

        Returns:
            Set of unique skill_ids from all queries.
        """
        if not self.retriever:
            return set()

        # Run parallel async searches
        tasks = [
            self.retriever.async_search(
                query=q, top_k=self.top_k, min_score=self.min_score)
            for q in queries
        ]
        results = await asyncio.gather(*tasks)

        # Collect unique skill_ids
        skill_ids = set()
        for result_list in results:
            for doc, score in result_list:
                skill_id = self._extract_skill_id_from_doc(doc)
                if skill_id:
                    skill_ids.add(skill_id)
        return skill_ids

    def _filter_skills(
            self,
            query: str,
            skill_ids: Set[str],
            mode: Literal['fast', 'deep'] = 'fast'
    ) -> Set[str]:
        """
        Filter skills based on relevance to the query.

        Args:
            query: User's query.
            skill_ids: Set of candidate skill_ids.
            mode: 'fast' for name+description only, 'deep' for full content analysis.

        Returns:
            Set of filtered skill_ids that are relevant.
        """
        if len(skill_ids) <= 1:
            return skill_ids

        # Format candidate skills based on mode
        if mode == 'deep':
            # Include name, description, and content (truncated)
            skill_entries = []
            for sid in skill_ids:
                if sid not in self.all_skills:
                    continue
                skill = self.all_skills[sid]
                content = skill.content[:3000] if skill.content else ''
                entry = (
                    f'### [{sid}] {skill.name}\n'
                    f'**Description**: {skill.description}\n'
                    f'**Content**: {content}'
                )
                skill_entries.append(entry)
            candidate_skills_text = '\n\n'.join(skill_entries)
            prompt = PROMPT_FILTER_SKILLS_DEEP.format(
                query=query,
                candidate_skills=candidate_skills_text)
        else:
            # Fast mode: name and description only
            candidate_skills_text = '\n'.join([
                f'- [{sid}] {self.all_skills[sid].name}: {self.all_skills[sid].description}'
                for sid in skill_ids if sid in self.all_skills
            ])
            prompt = PROMPT_FILTER_SKILLS_FAST.format(
                query=query,
                candidate_skills=candidate_skills_text)

        response = self._llm_generate(prompt)
        parsed = self._parse_json_response(response)

        filtered_ids = parsed.get('filtered_skill_ids', list(skill_ids))

        # For deep mode, also check skill_analysis for can_execute
        if mode == 'deep':
            skill_analysis = parsed.get('skill_analysis', {})
            final_ids = []
            for sid in filtered_ids:
                analysis = skill_analysis.get(sid, {})
                # Keep skill if can_execute is True or not specified
                if analysis.get('can_execute', True):
                    final_ids.append(sid)
                else:
                    logger.info(
                        f'Removing skill [{sid}]: cannot execute - '
                        f'{analysis.get("reason", "")[:200]}'
                    )
            filtered_ids = final_ids

        logger.info(
            f'Filter ({mode}): {len(skill_ids)} -> {len(filtered_ids)} skills. '
            f'Reason: {parsed.get("reasoning", "")[:1000]}'
        )

        return set(filtered_ids)

    def _build_dag(self, query: str, skill_ids: Set[str]) -> Dict[str, Any]:
        """
        Filter skills and build execution DAG.

        Performs deep filtering and DAG construction in one LLM call.

        Args:
            query: Original user query.
            skill_ids: Set of candidate skill_ids.

        Returns:
            Dict containing 'filtered_skill_ids', 'dag', and 'execution_order'.

        【DAG 结构说明】
        LLM 返回的 DAG 是邻接表格式：
            dag[A] = [B, C]  表示 Skill A 依赖于 B 和 C（B、C 必须先于 A 执行）

        execution_order 是拓扑排序结果，格式：
            ["skill_b", "skill_c", ["skill_d", "skill_e"], "skill_a"]
            - 字符串：串行执行
            - 列表：该组内 Skill 可并行执行

        如果 LLM 未给出 execution_order，则由 _topological_sort_dag() 自动推导。

        校验逻辑：
        - LLM 可能幻觉出不存在的 skill_id，这里过滤掉非候选集中的 id
        - DAG 依赖项也只保留候选集内的合法 id
        - 若 LLM 过滤后为空，保守地保留全部候选（避免误删）
        """
        skills_info = self._format_retrieved_skills(skill_ids)
        prompt = PROMPT_BUILD_SKILLS_DAG.format(
            query=query, selected_skills=skills_info)
        response = self._llm_generate(prompt)
        parsed = self._parse_json_response(response)

        # Get filtered skills and validate they exist in input
        raw_filtered = parsed.get('filtered_skill_ids', list(skill_ids))
        filtered_ids = set(sid for sid in raw_filtered if sid in skill_ids)

        # If no valid IDs returned, keep all input skills
        if not filtered_ids:
            logger.warning('No valid skill IDs in LLM response, keeping all input skills')
            filtered_ids = skill_ids

        logger.info(f'DAG filter: {len(skill_ids)} -> {len(filtered_ids)} skills')

        # Validate and clean DAG - only keep valid skill IDs
        raw_dag = parsed.get('dag', {})
        dag = {}
        for sid, deps in raw_dag.items():
            if sid in filtered_ids:
                # Filter dependencies to only valid skill IDs
                valid_deps = [d for d in deps if d in filtered_ids]
                dag[sid] = valid_deps

        # Ensure all filtered skills are in DAG
        for sid in filtered_ids:
            if sid not in dag:
                dag[sid] = []

        # Validate execution_order - only keep valid skill IDs
        raw_order = parsed.get('execution_order', [])
        order = self._validate_execution_order(raw_order, filtered_ids)

        # Fallback: derive execution_order from DAG using topological sort
        if not order and filtered_ids:
            order = self._topological_sort_dag(dag)
            logger.info(f'Derived execution_order from DAG: {order}')

        return {
            'filtered_skill_ids': filtered_ids,
            'dag': dag,
            'execution_order': order
        }

    def _validate_execution_order(
            self,
            raw_order: List[Union[str, List[str]]],
            valid_ids: Set[str]
    ) -> List[Union[str, List[str]]]:
        """
        Validate execution order, keeping only valid skill IDs.

        Args:
            raw_order: Raw execution order from LLM.
            valid_ids: Set of valid skill IDs.

        Returns:
            Validated execution order with only valid skill IDs.
        """
        result = []
        for item in raw_order:
            if isinstance(item, list):
                valid_group = [sid for sid in item if sid in valid_ids]
                if valid_group:
                    if len(valid_group) == 1:
                        result.append(valid_group[0])
                    else:
                        result.append(valid_group)
            elif item in valid_ids:
                result.append(item)
        return result

    def _topological_sort_dag(self, dag: Dict[str, List[str]]) -> List[str]:
        """
        Perform topological sort on DAG to get execution order.

        Args:
            dag: Adjacency list where dag[A] = [B, C] means A depends on B, C.

        Returns:
            Topologically sorted list of skill IDs (dependencies first).

        【算法：Kahn 算法（BFS 拓扑排序）】
        约定：dag[A] = [B] 表示 A 依赖 B，即 B 必须先于 A 执行。
        入度（in_degree）定义：节点 A 的入度 = A 依赖的节点数量（即 len(dag[A])）。
        入度为 0 的节点没有前置依赖，可以最先执行。

        步骤：
        1. 计算每个节点的入度（= 其依赖列表长度）
        2. 将所有入度为 0 的节点加入队列
        3. 每次从队列取出一个节点，加入结果序列
        4. 将所有"依赖了该节点"的后续节点入度 -1；若入度降为 0，加入队列
        5. 重复直到队列为空

        注意：本实现中前面有一段被覆盖的 in_degree 初始化代码（第一个循环），
        属于死代码——它计算的 in_degree 会被后面的赋值语句 in_degree[node] = len(deps) 覆盖。
        实际生效的逻辑从 `in_degree = {node: 0 for node in dag}` 重新开始。

        若存在环或孤立节点，result 不包含所有节点，剩余节点会追加到末尾并记录警告。
        """
        if not dag:
            return []

        # ── 注意：下面这段 in_degree 初始化是死代码 ──────────────────
        # 这里的 pass 什么都没做，紧接着的循环也只是检查 dep 是否在 in_degree 里，
        # 但这个 in_degree 字典会在后面被完全重新赋值，所以这段逻辑无实际作用。
        # ──────────────────────────────────────────────────────────────
        # Calculate in-degree for each node
        in_degree = {node: 0 for node in dag}
        for node, deps in dag.items():
            for dep in deps:
                if dep in in_degree:
                    pass  # dep is a dependency, node depends on it
            # Count how many nodes depend on this node
        for node, deps in dag.items():
            for dep in deps:
                if dep not in in_degree:
                    in_degree[dep] = 0

        # ── 实际生效的入度计算从这里开始 ─────────────────────────────
        # 入度 = 该节点依赖的前置节点数（即 dag[node] 的长度）
        # 入度为 0 → 无前置依赖 → 可以最先执行
        # ──────────────────────────────────────────────────────────────
        # Recalculate: in dag[A] = [B], A depends on B, so B must come before A
        # We need to build reverse mapping
        in_degree = {node: 0 for node in dag}
        for dep in set(d for deps in dag.values() for d in deps):
            if dep not in in_degree:
                in_degree[dep] = 0

        for node, deps in dag.items():
            in_degree[node] = len(deps)

        # Start with nodes that have no dependencies
        queue = [node for node, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            # Sort for deterministic order
            queue.sort()
            node = queue.pop(0)
            result.append(node)

            # Reduce in-degree for nodes that depend on this node
            for other_node, deps in dag.items():
                if node in deps and other_node in in_degree:
                    in_degree[other_node] -= 1
                    if in_degree[other_node] == 0:
                        queue.append(other_node)

        # If not all nodes processed, there might be a cycle or disconnected nodes
        remaining = set(dag.keys()) - set(result)
        if remaining:
            logger.warning(f'Topological sort incomplete, adding remaining: {remaining}')
            result.extend(sorted(remaining))

        return result

    def _filter_execution_order(
            self,
            execution_order: List[Union[str, List[str]]],
            valid_skill_ids: Set[str]
    ) -> List[Union[str, List[str]]]:
        """
        Filter execution order to only include valid skill_ids.

        Args:
            execution_order: Original execution order (may contain parallel groups).
            valid_skill_ids: Set of skill_ids that should be kept.

        Returns:
            Filtered execution order with only valid skills.
        """
        filtered = []
        for item in execution_order:
            if isinstance(item, list):
                # Parallel group: filter and keep if any remain
                filtered_group = [sid for sid in item if sid in valid_skill_ids]
                if filtered_group:
                    if len(filtered_group) == 1:
                        filtered.append(filtered_group[0])
                    else:
                        filtered.append(filtered_group)
            elif item in valid_skill_ids:
                filtered.append(item)
        return filtered

    def _direct_select_skills(self, query: str) -> SkillDAGResult:
        """
        Directly select skills using LLM with all skills in context.

        Used when enable_retrieve=False. Puts all skills into LLM context
        and lets LLM select relevant skills and build DAG in one call.

        Args:
            query: User's task query.

        Returns:
            SkillDAGResult containing the skill execution DAG.
        """
        prompt = PROMPT_DIRECT_SELECT_SKILLS.format(
            query=query, all_skills=self._get_all_skills_context())
        response = self._llm_generate(prompt)
        parsed = self._parse_json_response(response)

        # Handle chat-only response
        needs_skills = parsed.get('needs_skills', True)
        chat_response = parsed.get('chat_response')

        if not needs_skills:
            logger.info('Chat-only query, no skills needed')
            if chat_response:
                print(f'\n[Chat Response]\n{chat_response}\n')
            return SkillDAGResult(
                is_complete=True, chat_response=chat_response)

        # Extract selected skills and DAG
        selected_ids = parsed.get('selected_skill_ids', [])
        dag = parsed.get('dag', {})
        order = parsed.get('execution_order', [])

        # Validate skill_ids exist
        valid_ids = {sid for sid in selected_ids if sid in self.all_skills}
        selected = {sid: self.all_skills[sid] for sid in valid_ids}

        logger.info(f'Direct selection: {valid_ids}')

        return SkillDAGResult(
            dag=dag,
            execution_order=order,
            selected_skills=selected,
            is_complete=bool(valid_ids),
            clarification=None if valid_ids else 'No relevant skills found.')

    async def get_skill_dag(self, query: str) -> SkillDAGResult:
        """
        Run the autonomous skill retrieval and DAG construction loop.

        Iteratively retrieves skills, evaluates completeness with reflection,
        and builds execution DAG. Loop terminates when:
        - Query is chat-only (no skills needed)
        - Max iterations reached
        - Skills are deemed complete for the task
        - Clarification from user is needed

        Args:
            query: User's task query.

        Returns:
            SkillDAGResult containing the skill execution DAG.

        【检索流水线详解】
        Step 1  _analyze_query()
                  LLM 判断 query 是否需要 Skill（区分「纯聊天」vs「需要执行」），
                  同时将原始 query 拆解为多个更精准的 skill_queries（子查询），
                  提升后续检索的召回率。

        Step 2  _async_retrieve_skills()
                  对每个子查询并行调用 HybridRetriever，合并去重后得到候选 Skill 集合。
                  HybridRetriever = BM25（稀疏）+ 向量相似度（稠密）混合打分。

        Step 3  _filter_skills('fast') + _filter_skills('deep')
                  两轮 LLM 过滤：
                  - fast: 只看名称+描述，快速淘汰明显不相关的 Skill
                  - deep: 看 Skill 完整内容，精细判断能否执行当前任务
                  两轮设计的目的：先快速缩小候选集，再对少量候选做精细分析，
                  在准确率和 Token 消耗之间取得平衡。

        Step 4  _build_dag()
                  LLM 在过滤后的 Skill 集合上：
                  a) 再次过滤（最终确认）
                  b) 分析 Skill 间依赖关系，构建 DAG
                  c) 给出拓扑排序后的执行顺序（含并行分组）
        """
        if not self.all_skills:
            logger.warning('No skills loaded, returning empty result')
            return SkillDAGResult()

        # Direct selection mode: put all skills into LLM context
        if not self.enable_retrieve:
            logger.info('Direct selection mode (enable_retrieve=False)')
            return self._direct_select_skills(query)

        # Search mode: use HybridRetriever
        if not self.retriever:
            logger.warning('Retriever not initialized, returning empty result')
            return SkillDAGResult()

        # Step 1: Analyze query to determine if skills are needed
        needs_skills, intent, skill_queries, chat_response = self._analyze_query(
            query)
        logger.info(f'Needs skills: {needs_skills}, Intent: {intent}')

        # If chat-only, return empty DAG with chat response
        if not needs_skills:
            logger.info('Chat-only query, no skills needed')
            if chat_response:
                print(f'\n[Chat Response]\n{chat_response}\n')
            return SkillDAGResult(
                is_complete=True, chat_response=chat_response)

        clarification: Optional[str] = None

        # Step 2: Retrieve skills
        collected_skills = await self._async_retrieve_skills(skill_queries)
        logger.info(f'Retrieved skills: {collected_skills}')

        if not collected_skills:
            clarification = 'No relevant skills found. Please provide more details.'
            return SkillDAGResult(
                is_complete=False, clarification=clarification)

        # Limit candidate skills to max_candidate_skills
        if len(collected_skills) > self.max_candidate_skills:
            logger.warning(
                f'Too many candidate skills ({len(collected_skills)}), '
                f'limiting to {self.max_candidate_skills}'
            )
            collected_skills = set(list(collected_skills)[:self.max_candidate_skills])

        # Step 3: Fast filter by name/description
        collected_skills = self._filter_skills(query, collected_skills, mode='fast')
        logger.info(f'After fast filter: {collected_skills}')

        if len(collected_skills) > 1:
            collected_skills = self._filter_skills(query, collected_skills, mode='deep')
            logger.info(f'After deep filter: {collected_skills}')

        if not collected_skills:
            clarification = 'No relevant skills found after filtering. Please refine your query.'
            return SkillDAGResult(
                is_complete=False, clarification=clarification)

        # Step 4: Build DAG with integrated deep filtering
        dag_result = self._build_dag(query, collected_skills)

        filtered_ids = dag_result.get('filtered_skill_ids', collected_skills)
        skills_dag: Dict[str, Any] = dag_result.get('dag', {})
        execution_order: List[str] = dag_result.get('execution_order', [])

        if not filtered_ids:
            clarification = 'No relevant skills found after filtering. Please refine your query.'
            return SkillDAGResult(
                is_complete=False, clarification=clarification)

        # Build selected skills dict from filtered results
        selected = {
            sid: self.all_skills[sid]
            for sid in filtered_ids if sid in self.all_skills
        }

        logger.info(
            f'Final DAG built with skills: {skills_dag}, execution order: {execution_order}'
        )

        return SkillDAGResult(
            dag=skills_dag,
            execution_order=execution_order,
            selected_skills=selected,
            is_complete=(clarification is None),
            clarification=clarification)

    def _get_container(self) -> SkillContainer:
        """Get or create SkillContainer instance."""
        if self._container is None:
            self._container = SkillContainer(
                workspace_dir=self.work_dir,
                use_sandbox=self.use_sandbox,
                **{
                    k: v
                    for k, v in self.kwargs.items() if k in [
                        'timeout', 'image', 'memory_limit',
                        'enable_security_check', 'network_enabled'
                    ]
                })
        return self._container

    def _get_executor(self) -> DAGExecutor:
        """Get or create DAGExecutor instance."""
        if self._executor is None:
            container = self._get_container()
            self._executor = DAGExecutor(
                container=container,
                skills=self.all_skills,
                workspace_dir=self.work_dir,
                llm=self.llm,
                enable_progressive_analysis=True,
                max_retries=self.max_retries)
        return self._executor

    async def execute_dag(self,
                          dag_result: SkillDAGResult,
                          execution_input: Optional[ExecutionInput] = None,
                          stop_on_failure: bool = True,
                          query: str = '') -> DAGExecutionResult:
        """
        Execute the skill DAG from a SkillDAGResult.

        Executes skills according to the execution_order, handling:
        - Sequential execution for single skill items
        - Parallel execution for skill groups (sublists)
        - Input/output linking between dependent skills
        - Progressive skill analysis (plan -> load -> execute)

        Args:
            dag_result: SkillDAGResult containing DAG and execution order.
            execution_input: Optional initial input for skills.
            stop_on_failure: Whether to stop on first failure.
            query: User query for progressive skill analysis.

        Returns:
            DAGExecutionResult with all execution outcomes.
        """
        if not dag_result.is_complete:
            logger.warning('DAG is not complete, execution may fail')

        if not dag_result.execution_order:
            logger.warning('Empty execution order, nothing to execute')
            return DAGExecutionResult(success=True)

        executor = self._get_executor()
        result = await executor.execute(
            dag=dag_result.dag,
            execution_order=dag_result.execution_order,
            execution_input=execution_input,
            stop_on_failure=stop_on_failure,
            query=query)

        # Attach result to dag_result for convenience
        dag_result.execution_result = result

        logger.info(f'DAG execution completed: success={result.success}, '
                    f'duration={result.total_duration_ms:.2f}ms')

        return result

    def get_execution_spec(self) -> Optional[str]:
        """Get the execution spec log as markdown string."""
        if self._container:
            return self._container.get_spec_log()
        return None

    def save_execution_spec(self,
                            output_path: Optional[Union[str, Path]] = None):
        """Save the execution spec to a markdown file."""
        if self._container:
            self._container.save_spec_log(output_path)

    def cleanup(self, keep_spec: bool = True):
        """Clean up container workspace."""
        if self._container:
            self._container.cleanup(keep_spec=keep_spec)

    def get_skill_context(self, skill_id: str) -> Optional[SkillContext]:
        """
        Get the skill context for an executed skill.

        Args:
            skill_id: The skill identifier (e.g., 'pdf@latest').

        Returns:
            SkillContext if the skill was executed, None otherwise.
        """
        if self._executor:
            return self._executor.get_skill_context(skill_id)
        return None

    def get_all_skill_contexts(self) -> Dict[str, SkillContext]:
        """
        Get all skill contexts from executed skills.

        Returns:
            Dict mapping skill_id to SkillContext.
        """
        if self._executor:
            return self._executor.get_all_contexts()
        return {}

    def get_executed_skill_ids(self) -> List[str]:
        """
        Get list of skill_ids that were executed.

        Returns:
            List of skill_ids with available contexts.
        """
        if self._executor:
            return self._executor.get_executed_skill_ids()
        return []

    async def run(
            self,
            query: str,
            execution_input: Optional[ExecutionInput] = None,
            stop_on_failure: bool = True
    ) -> SkillDAGResult:
        """
        Run skill retrieval and execute the resulting DAG in one call.

        Combines get_skill_dag() and execute_dag().
        Uses progressive skill analysis for each skill execution.

        Args:
            query: User's task query.
            execution_input: Optional initial input for skills.
            stop_on_failure: Whether to stop on first failure.

        Returns:
            SkillDAGResult with execution_result populated.
        """
        dag_result = await self.get_skill_dag(query)

        # Skip execution for chat-only results
        if dag_result.chat_response:
            logger.info('Chat-only response, skipping execution')
            return dag_result

        # Skip if skills are incomplete
        if not dag_result.is_complete:
            logger.warning(f'Skills incomplete: {dag_result.clarification}')
            return dag_result

        # Execute the DAG
        if dag_result.execution_order:
            await self.execute_dag(
                dag_result, execution_input, stop_on_failure, query=query)

        return dag_result
