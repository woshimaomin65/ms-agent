# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Agent 构建器（工厂类）模块。

该模块提供 `AgentLoader` 类，负责根据配置文件或参数动态构建 Agent 实例。
支持以下三种 Agent 类型：
  1. `LLMAgent`：标准 LLM 驱动的智能体（默认）；
  2. `CodeAgent`：代码生成专用智能体；
  3. 外部自定义 Agent：通过 `code_file` 参数加载用户提供的 Python 文件中的 Agent 子类。

调用链（由 run.py 触发）：
    RunCMD._execute_with_config()
      └─ AgentLoader.build(config_dir_or_id, config, ...)
           ├─ Config.from_task(config_dir_or_id) → 加载配置
           ├─ [外部代码] _load_external_code(config, code_file)
           ├─ [LLMAgent]  LLMAgent(config, tag, trust_remote_code)
           └─ [CodeAgent] CodeAgent(config, tag, trust_remote_code)
"""

import importlib
import inspect
import logging
import os
import sys
from typing import Dict, Optional

from ms_agent.config.config import Config
from ms_agent.utils.constants import DEFAULT_AGENT_FILE, DEFAULT_TAG
from ms_agent.utils.logger import get_logger
from omegaconf import DictConfig, OmegaConf

from .base import Agent

logger = get_logger(__name__)


class AgentLoader:
    """Agent 工厂类，负责根据配置动态创建 Agent 实例。

    该类提供静态/类方法，不需要实例化，直接通过 AgentLoader.build(...) 调用。
    支持从配置目录、ModelScope Hub ID 或已加载的 DictConfig 对象构建 Agent。
    """

    @classmethod
    def build(cls,
              config_dir_or_id: Optional[str] = None,
              config: Optional[DictConfig] = None,
              env: Optional[Dict[str, str]] = None,
              tag: Optional[str] = None,
              trust_remote_code: bool = False,
              **kwargs) -> Agent:
        """根据配置构建并返回 Agent 实例。

        构建流程：
        1. 若提供了 config_dir_or_id，从磁盘或 ModelScope Hub 加载配置；
        2. 若同时提供了 config 参数，将其与文件配置合并（文件配置优先）；
        3. 确定 Agent 标签（tag），写入配置；
        4. 确定 Agent 类型（type 字段）和外部代码文件（code_file 字段）；
        5. 若存在 code_file，加载外部 Python 文件中的 Agent 子类；
        6. 否则根据 agent_type 实例化内置 LLMAgent 或 CodeAgent。

        参数：
            config_dir_or_id (Optional[str]): 配置文件目录路径或 ModelScope Hub 仓库 ID。
                若路径不存在，会尝试从 ModelScope 下载。
            config (Optional[DictConfig]): 已加载的配置对象，与文件配置合并使用。
            env (Optional[Dict[str, str]]): 额外的环境变量，传入 Config.from_task。
            tag (Optional[str]): Agent 运行标签；若为 None，使用配置中的 tag 或默认值。
            trust_remote_code (bool): 是否允许加载外部 Python 代码，默认 False。
            **kwargs: 额外参数，如 load_cache、mcp_server_file、code_file 等。

        返回：
            Agent: 构建好的 Agent 实例（LLMAgent、CodeAgent 或外部自定义 Agent）。

        异常：
            ValueError: 当 agent_type 不是已知类型时抛出。
        """
        agent_config: Optional[DictConfig] = None

        logger.info(f'[AgentLoader] build() called with config_dir_or_id={config_dir_or_id}, tag={tag}')
        logger.info(f'[AgentLoader] kwargs: {list(kwargs.keys())}')

        # 步骤 1：从路径或 Hub ID 加载配置文件
        if config_dir_or_id is not None:
            logger.info(f'[AgentLoader] Loading config from {config_dir_or_id}')
            if not os.path.exists(config_dir_or_id):
                # 本地路径不存在，尝试作为 ModelScope Hub 仓库 ID 下载
                logger.info(f'[AgentLoader] Path does not exist, attempting ModelScope download')
                from modelscope import snapshot_download
                config_dir_or_id = snapshot_download(config_dir_or_id)
            agent_config: DictConfig = Config.from_task(config_dir_or_id, env)
            logger.info(f'[AgentLoader] Config loaded: {list(agent_config.keys()) if agent_config else "None"}')

        # 步骤 2：将传入的 config 参数与文件配置合并
        if config is not None:
            logger.info(f'[AgentLoader] Merging with provided config')
            if agent_config is not None:
                # 已有文件配置，将传入配置叠加合并（传入配置覆盖文件配置）
                agent_config = OmegaConf.merge(agent_config, config)
            else:
                # 没有文件配置，直接使用传入配置
                agent_config = config

        # 步骤 3：确定 Agent 标签（tag）
        if tag is None:
            # 从配置中读取 tag，若无则使用默认标签 DEFAULT_TAG
            agent_tag = getattr(agent_config, 'tag', None) or DEFAULT_TAG
        else:
            agent_tag = tag
        agent_config.tag = agent_tag
        agent_config.trust_remote_code = trust_remote_code
        logger.info(f'[AgentLoader] Agent tag set to: {agent_tag}')

        # 步骤 4：确保 local_dir 字段存在（供外部代码加载使用）
        if getattr(agent_config, 'local_dir',
                   None) is None and config_dir_or_id is not None:
            agent_config.local_dir = config_dir_or_id

        # 导入内置 Agent 类型
        from .llm_agent import LLMAgent
        from .code_agent import CodeAgent

        # 默认 Agent 类型为 LLMAgent
        agent_type = LLMAgent.AGENT_NAME

        # 步骤 5：确定 code_file 和 agent_type
        if 'code_file' in kwargs:
            # 优先使用 kwargs 中显式传入的 code_file
            code_file = kwargs.pop('code_file')
            logger.info(f'[AgentLoader] code_file from kwargs: {code_file}')
        elif agent_config is not None:
            # 从配置中读取 agent 类型和外部代码文件路径
            agent_type = getattr(agent_config, 'type',
                                 '').lower() or agent_type.lower()
            code_file = getattr(agent_config, 'code_file', None)
            logger.info(f'[AgentLoader] agent_type from config: {agent_type}, code_file: {code_file}')
        else:
            # 兜底：要求 local_dir 存在，使用默认 agent 文件名
            assert getattr(agent_config, 'local_dir', None) is not None
            code_file = os.path.join(
                getattr(agent_config, 'local_dir', ''), DEFAULT_AGENT_FILE)
            logger.info(f'[AgentLoader] Using default code_file: {code_file}')

        # 步骤 6：根据是否有外部代码文件选择构建方式
        if code_file is not None:
            # 从外部 Python 文件动态加载自定义 Agent 类
            logger.info(f'[AgentLoader] Loading external code from {code_file}')
            agent_instance = cls._load_external_code(agent_config, code_file,
                                                     **kwargs)
        else:
            assert agent_config is not None
            if agent_type == LLMAgent.AGENT_NAME.lower():
                # 构建标准 LLM 智能体
                logger.info(f'[AgentLoader] Creating LLMAgent instance')
                agent_instance = LLMAgent(agent_config, agent_tag,
                                          trust_remote_code, **kwargs)
            elif agent_type == CodeAgent.AGENT_NAME.lower():
                # 构建代码生成专用智能体
                logger.info(f'[AgentLoader] Creating CodeAgent instance')
                agent_instance = CodeAgent(agent_config, agent_tag,
                                           trust_remote_code, **kwargs)
            else:
                raise ValueError(f'Unknown agent type: {agent_type}')
        logger.info(f'[AgentLoader] Agent {agent_tag} created successfully, type={type(agent_instance).__name__}')
        return agent_instance

    @classmethod
    def _load_external_code(cls, config, code_file, **kwargs) -> 'Agent':
        """从外部 Python 文件中动态加载 Agent 子类并实例化。

        安全说明：
            此方法会执行用户提供的任意 Python 代码，存在安全风险。
            因此必须在 config.trust_remote_code=True 时才允许执行。

        加载流程：
        1. 将配置目录（local_dir）加入 sys.path；
        2. 若 code_file 位于子目录，也将子目录加入 sys.path；
        3. 使用 importlib 动态导入模块；
        4. 通过反射（inspect.getmembers）找到继承自 Agent 的类；
        5. 实例化该类并返回。

        参数：
            config (DictConfig): 已加载的配置对象（必须包含 local_dir）。
            code_file (str): 外部 Agent Python 文件的路径（相对或绝对路径）。
            **kwargs: 传递给 Agent 构造函数的额外参数。

        返回：
            Agent: 外部自定义 Agent 的实例。

        异常：
            AssertionError: 若 code_file 为 None、未启用 trust_remote_code、
                或文件中找不到合法的 Agent 子类时抛出。
        """
        assert code_file is not None, 'Code file cannot be None'
        # 安全检查：必须显式开启 trust_remote_code
        assert config.trust_remote_code, (
            f'[External Code]A code file is required to run in the LLMAgent: {code_file}'
            f'\nThis is external code, if you trust this code file, '
            f'please specify `--trust_remote_code true`')

        # 分离子目录和文件名
        subdir = os.path.dirname(code_file)
        code_file = os.path.basename(code_file)
        local_dir = config.local_dir
        assert local_dir is not None, 'Using external py files, but local_dir cannot be found.'

        # 若 code_file 位于子目录，拼接完整子目录路径
        if subdir:
            subdir = os.path.join(local_dir, subdir)  # noqa

        # 将配置目录加入 Python 模块搜索路径
        if local_dir not in sys.path:
            sys.path.insert(0, local_dir)

        # 若存在子目录，也加入搜索路径
        subdir_inserted = False
        if subdir and subdir not in sys.path:
            sys.path.insert(0, subdir)
            subdir_inserted = True

        # 去掉 .py 后缀，得到模块名
        if code_file.endswith('.py'):
            code_file = code_file[:-3]

        # 若模块已缓存，先清除以确保使用最新代码
        if code_file in sys.modules:
            del sys.modules[code_file]

        # 动态导入外部模块
        code_module = importlib.import_module(code_file)

        # 反射获取模块中所有类
        module_classes = {
            name: agent_cls
            for name, agent_cls in inspect.getmembers(code_module,
                                                      inspect.isclass)
        }

        # 在模块中找到第一个继承自 Agent 的类（直接子类，且定义在该模块中）
        agent_instance = None
        for name, agent_cls in module_classes.items():
            if Agent in agent_cls.__mro__[
                    1:] and agent_cls.__module__ == code_file:
                agent_instance = agent_cls(
                    config,
                    config.tag,
                    trust_remote_code=config.trust_remote_code,
                    **kwargs)
                break

        assert agent_instance is not None, (
            f'Cannot find a proper agent class in the external code file: {code_file}')

        # 清理：将临时加入的子目录从 sys.path 中移除
        if subdir_inserted:
            sys.path.pop(0)
        return agent_instance
