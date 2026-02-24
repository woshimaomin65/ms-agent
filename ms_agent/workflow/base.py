# Copyright (c) ModelScope Contributors. All rights reserved.
from abc import ABC, abstractmethod
from typing import Dict, Optional

from ms_agent.config import Config
from omegaconf import DictConfig


class Workflow(ABC):
    """工作流基类，定义基于代理的处理步骤序列。

    工作流管理多个代理的执行流程，每个代理负责整体流程中的特定任务。
    子类需要实现 `run` 方法。

    Args:
        config_dir_or_id (str): 工作流配置所在的目录路径或ID。
        config (DictConfig): 工作流的直接配置字典。
        env (Dict[str, str]): 加载配置时使用的环境变量。
        trust_remote_code (bool): 是否允许加载远程代码。默认为False。
        **kwargs: 其他配置选项，包括：
            - load_cache (bool): 是否使用之前运行的缓存结果。默认为True。
            - mcp_server_file (Optional[str]): 如果需要，MCP服务器文件的路径。默认为None。
    """

    def __init__(self,
                 config_dir_or_id: Optional[str] = None,
                 config: Optional[DictConfig] = None,
                 env: Optional[Dict[str, str]] = None,
                 trust_remote_code: bool = False,
                 **kwargs):
        # 初始化工作流配置
        # 如果提供了配置目录或ID，则从该位置加载配置；否则使用传入的config参数
        if config_dir_or_id is None:
            self.config = config
        else:
            self.config = Config.from_task(config_dir_or_id, env)
        
        # 保存配置目录或ID
        self.config_dir_or_id = config_dir_or_id
        
        # 是否信任远程代码
        self.trust_remote_code = trust_remote_code
        
        # 是否启用缓存功能，默认为False（注意：原注释说默认为True，但代码中是False）
        self.load_cache = kwargs.get('load_cache', False)
        
        # MCP服务器文件路径
        self.mcp_server_file = kwargs.get('mcp_server_file', None)
        
        # 环境变量
        self.env = env
        
        # 工作流链列表，用于存储工作流中的各个处理步骤
        self.workflow_chains = []
        
        # 构建工作流，此方法由子类实现
        self.build_workflow()

    @abstractmethod
    def build_workflow(self):
        """根据配置构建执行链。
        
        此抽象方法必须由子类实现，用于根据配置创建工作流的执行链。
        """
        pass

    @abstractmethod
    async def run(self, inputs, **kwargs):
        """异步执行工作流。
        
        此抽象方法必须由子类实现，用于执行工作流并返回结果。
        
        Args:
            inputs: 工作流的输入数据
            **kwargs: 其他可选参数
        
        Returns:
            执行结果
        """
        pass
