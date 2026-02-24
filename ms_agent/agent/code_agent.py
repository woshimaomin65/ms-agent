# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, List, Union

from ms_agent.llm import Message
from omegaconf import DictConfig

from .base import Agent


class CodeAgent(Agent):
    """代码代理类，可在工作流中执行外部代码"""

    AGENT_NAME = 'CodeAgent'

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        """初始化代码代理

        参数：
            config: 配置对象
            tag: 代理标签
            trust_remote_code: 是否信任远程代码
            **kwargs: 其他关键字参数
        """
        super().__init__(config, tag, trust_remote_code, **kwargs)
        self.load_cache = kwargs.get('load_cache', False)

    async def run(self, inputs: Union[str, List[Message]],
                  **kwargs) -> List[Message]:
        """运行外部代码。默认实现不执行任何操作

        参数：
            inputs: 输入可以是提示字符串，
                或来自上一个代理的消息列表

        返回：
            输出给下一个代理的消息
        """
        _config = None
        _messages = None
        if self.load_cache:
            # 如果启用了缓存，尝试读取历史记录
            _config, _messages = self.read_history(inputs)
        if _config is not None and _messages is not None:
            # 如果缓存命中，直接使用缓存的配置和消息
            self.config = _config
            return _messages
        # 缓存未命中，执行代码并保存历史
        messages = await self.execute_code(inputs, **kwargs)
        self.save_history(messages, **kwargs)
        return messages

    async def execute_code(self, inputs: Union[str, List[Message]],
                           **kwargs) -> List[Message]:
        """执行代码的抽象方法，由子类实现具体逻辑

        参数：
            inputs: 输入可以是提示字符串，或消息列表

        返回：
            执行结果消息列表
        """
        return inputs
