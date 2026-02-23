# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Agent 抽象基类模块。

定义了所有智能体（Agent）必须实现的基础接口 `Agent`，
包括：
  - 核心执行方法 `run`（异步抽象方法）；
  - 历史记录的读取与保存（read_history / save_history）；
  - 工作流中的下一个 Agent 决策（next_flow）。

所有具体 Agent 实现（如 LLMAgent、CodeAgent）均继承此类。
"""

import os
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, List, Tuple, Union

from ms_agent.llm import Message
from ms_agent.utils import read_history, save_history
from ms_agent.utils.constants import DEFAULT_OUTPUT_DIR, DEFAULT_RETRY_COUNT
from omegaconf import DictConfig


class Agent(ABC):
    """所有 Agent 的抽象基类。

    提供以下核心能力：
    - 配置对象（DictConfig）管理；
    - 可信远程代码（trust_remote_code）安全控制；
    - 异步执行接口（run）由子类实现；
    - 对话历史的磁盘持久化（read_history / save_history）；
    - 工作流中的流向控制（next_flow）。

    使用示例：
        class MyAgent(Agent):
            async def run(self, inputs, **kwargs):
                ...  # 具体推理逻辑

    参数：
        config (DictConfig): 已加载的配置对象，通常由 Config.from_task 生成。
        tag (str): 用于标识本次 Agent 运行的标签，影响日志和历史文件命名。
        trust_remote_code (bool): 是否允许加载外部（第三方）Python 代码，默认 False。
    """

    # 重试次数：从环境变量 AGENT_RETRY_COUNT 读取，默认值来自常量 DEFAULT_RETRY_COUNT
    retry_count = int(os.environ.get('AGENT_RETRY_COUNT', DEFAULT_RETRY_COUNT))

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        """初始化 Agent 基类。

        参数：
            config (DictConfig): 已加载的配置对象。
            tag (str): 标识此次运行的自定义标签，用于日志和历史文件命名。
            trust_remote_code (bool): 是否信任外部代码（如自定义 handler）。
                若配置中包含外部 Python 文件，必须显式设为 True。
        """
        # 保存配置对象（DictConfig 格式，由 omegaconf 管理）
        self.config = config
        # 运行标签，用于区分不同的 Agent 实例或任务
        self.tag = tag
        # 是否允许加载外部（用户自定义）Python 代码
        self.trust_remote_code = trust_remote_code
        # 将 tag 和 trust_remote_code 写回配置，便于下游组件读取
        self.config.tag = tag
        self.config.trust_remote_code = trust_remote_code
        # 输出目录：从配置中获取，若未设置则使用默认路径
        self.output_dir = getattr(self.config, 'output_dir',
                                  DEFAULT_OUTPUT_DIR)

    @abstractmethod
    async def run(
            self, inputs: Union[str, List[Message]], **kwargs
    ) -> Union[List[Message], AsyncGenerator[List[Message], Any]]:
        """Agent 的主执行方法（抽象接口，子类必须实现）。

        定义了 Agent 如何处理输入并生成输出消息。

        参数：
            inputs (Union[str, List[Message]]): Agent 的输入数据，可以是：
                - str：用户的原始提问字符串；
                - List[Message]：完整的历史对话消息列表（用于多轮对话）。

        返回：
            Union[List[Message], AsyncGenerator[List[Message], Any]]:
                - List[Message]：非流式模式下返回完整消息列表；
                - AsyncGenerator：流式模式下返回异步生成器，逐步产出消息列表。

        异常：
            NotImplementedError: 若子类未实现此方法则抛出。
        """
        raise NotImplementedError()

    def read_history(self, messages: Any,
                     **kwargs) -> Tuple[DictConfig, List[Message]]:
        """从磁盘读取之前保存的对话历史记录。

        委托给 ms_agent.utils.read_history 工具函数，
        根据 output_dir 和 tag 定位历史文件。

        参数：
            messages: 当前输入（用于确定是否需要恢复历史）。

        返回：
            Tuple[DictConfig, List[Message]]:
                (config, messages) 历史配置和消息列表，
                若无历史记录则返回 (None, None)。
        """
        return read_history(self.output_dir, self.tag)

    def save_history(self, messages: Any, **kwargs):
        """将当前对话历史记录保存到磁盘。

        若配置中 save_history=False，则跳过保存。
        委托给 ms_agent.utils.save_history 工具函数。

        参数：
            messages: 当前的消息列表，将被序列化并写入磁盘。
        """
        # 检查配置中是否允许保存历史（默认允许）
        if not getattr(self.config, 'save_history', True):
            return
        save_history(self.output_dir, self.tag, self.config, messages)

    def next_flow(self, idx: int) -> int:
        """在工作流（Workflow）中决定下一个执行的 Agent 索引。

        默认实现是顺序执行（返回 idx + 1）。
        子类可以重写此方法实现条件跳转等复杂工作流逻辑。

        参数：
            idx (int): 当前 Agent 在工作流中的索引。

        返回：
            int: 下一个要执行的 Agent 的索引。
        """
        return idx + 1
