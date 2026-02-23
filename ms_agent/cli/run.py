# Copyright (c) ModelScope Contributors. All rights reserved.
"""
`ms-agent run` 子命令模块。

该模块实现了 CLI 中最核心的 `run` 子命令，用于启动一次 Agent 推理/任务执行。
支持两种运行模式：
  1. **交互模式**：不传 --query，程序进入循环，逐轮接收用户输入；
  2. **单次模式**：传入 --query，直接将该 query 发送给 Agent 并结束。

同时支持通过 --config 指定配置文件目录，或通过 --project 指定内置项目名称。
若配置中定义了 workflow 结构，则走工作流引擎；否则走单 Agent 引擎。

调用链：
    run_cmd() → RunCMD.define_args() → RunCMD(args).execute()
      └─ Config.from_task(config)
           ├─ [workflow] WorkflowLoader.build() → engine.run(query)
           └─ [agent]    AgentLoader.build()    → engine.run(query)
"""

import argparse
import asyncio
import os
from importlib import resources as importlib_resources

from ms_agent.config import Config
from ms_agent.utils import get_logger, strtobool
from ms_agent.utils.constants import AGENT_CONFIG_FILE, MS_AGENT_ASCII

from .base import CLICommand

# 获取本模块专用的日志记录器
logger = get_logger()


def subparser_func(args):
    """子命令工厂函数，由 argparse 通过 set_defaults(func=...) 调用。

    当用户执行 `ms-agent run ...` 时，argparse 将解析后的 args 传入此函数，
    返回 RunCMD 实例，随后 cli.py 调用 cmd.execute() 触发实际执行逻辑。

    参数：
        args: argparse.Namespace，包含已解析的命令行参数。

    返回：
        RunCMD 实例。
    """
    return RunCMD(args)


def list_builtin_projects():
    """列出 ms_agent/projects 目录下所有内置项目的名称。

    使用 importlib.resources 访问包内资源，以确保在以 zip/wheel 方式安装时
    也能正确枚举内置项目目录。

    返回：
        按字母顺序排列的项目名称列表（字符串）。
        若无法访问资源目录，返回空列表并打印警告。
    """
    try:
        # 定位 ms_agent 包内的 projects 子目录
        root = importlib_resources.files('ms_agent').joinpath('projects')
        if not root.exists():
            return []
        # 枚举目录下的所有子目录（每个子目录代表一个内置项目）
        return sorted([p.name for p in root.iterdir() if p.is_dir()])
    except Exception as e:
        # 发生任何异常时不让帮助信息崩溃，仅打印警告
        logger.warning(f'Could not list built-in projects: {e}')
        return []


def project_help_text():
    """生成 --project 参数的帮助文本，动态包含当前可用的内置项目列表。

    返回：
        包含可用项目名称的帮助字符串。
    """
    projects = list_builtin_projects()
    if projects:
        return (
            'Built-in bundled project name under package ms_agent/projects. '
            f'Available: {", ".join(projects)}')
    return 'Built-in bundled project name under package ms_agent/projects.'


class RunCMD(CLICommand):
    """实现 `ms-agent run` 子命令的命令类。

    该类负责：
    1. 解析并校验命令行参数（define_args）；
    2. 根据参数加载配置（Config.from_task）；
    3. 构建 Agent 引擎或 Workflow 引擎；
    4. 以异步方式运行推理（asyncio.run(engine.run(query))）。
    """

    # 子命令名称，对应 `ms-agent run`
    name = 'run'

    def __init__(self, args):
        """初始化 RunCMD。

        参数：
            args: argparse.Namespace，包含所有已解析的命令行参数。
        """
        self.args = args

    @staticmethod
    def define_args(parsers: argparse.ArgumentParser):
        """向父解析器注册 `run` 子命令的所有参数。

        支持的命令行参数：
          --query              : 要发送给 LLM 的提问/指令；不传则进入交互模式
          --config             : 配置文件目录路径或 ModelScope 仓库 ID
          --project            : 内置项目名称（与 --config 互斥）
          --trust_remote_code  : 是否信任配置文件中包含的远程代码（默认 False）
          --load_cache         : 是否从缓存中加载历史步骤（默认 False）
          --mcp_config         : 额外的 MCP 服务器配置（JSON 字符串或路径）
          --mcp_server_file    : 额外的 MCP 服务器脚本文件
          --openai_api_key     : OpenAI 兼容服务的 API 密钥
          --modelscope_api_key : ModelScope 推理服务的 API 密钥
          --animation_mode     : 视频生成项目的动画模式（auto/human）

        参数：
            parsers: argparse 子解析器组，由 cli.py 中的 add_subparsers() 创建。
        """
        # 列出当前可用的内置项目，用于 --project 的合法值校验
        projects = list_builtin_projects()

        # 在子解析器组中注册 'run' 子命令
        parser: argparse.ArgumentParser = parsers.add_parser(RunCMD.name)

        # --query：用户要发送的问题或指令；可选，不传时进入交互对话模式
        parser.add_argument(
            '--query',
            required=False,
            type=str,
            help=
            'The query or prompt to send to the LLM. If not set, will enter an interactive mode.'
        )

        # --config：Agent 配置文件所在目录路径，或 ModelScope Hub 的仓库 ID
        parser.add_argument(
            '--config',
            required=False,
            type=str,
            default=None,
            help='The directory or the repo id of the config file')

        # --project：内置项目名称（与 --config 互斥），框架会自动定位项目目录
        parser.add_argument(
            '--project',
            required=False,
            type=str,
            default=None,
            choices=projects,          # 仅允许已知的内置项目名称
            help=project_help_text(),
        )

        # --trust_remote_code：是否信任配置中包含的远程代码，默认不信任
        parser.add_argument(
            '--trust_remote_code',
            required=False,
            type=str,
            default='false',
            help='Trust the code belongs to the config file, default False')

        # --load_cache：是否从上次中断的缓存恢复执行历史，适合重试失败的任务
        parser.add_argument(
            '--load_cache',
            required=False,
            type=str,
            default='false',
            help=
            'Load previous step histories from cache, this is useful when a query fails and retry'
        )

        # --mcp_config：额外的 MCP（Model Calling Protocol）服务器配置
        parser.add_argument(
            '--mcp_config',
            required=False,
            type=str,
            default=None,
            help='The extra mcp server config')

        # --mcp_server_file：额外的 MCP 服务器脚本文件路径
        parser.add_argument(
            '--mcp_server_file',
            required=False,
            type=str,
            default=None,
            help='An extra mcp server file.')

        # --openai_api_key：访问 OpenAI 兼容服务所需的 API 密钥
        parser.add_argument(
            '--openai_api_key',
            required=False,
            type=str,
            default=None,
            help='API key for accessing an OpenAI-compatible service.')

        # --modelscope_api_key：访问 ModelScope 推理服务所需的 API 密钥
        parser.add_argument(
            '--modelscope_api_key',
            required=False,
            type=str,
            default=None,
            help='API key for accessing ModelScope api-inference services.')

        # --animation_mode：视频生成项目专用，控制动画驱动模式（auto/human）
        parser.add_argument(
            '--animation_mode',
            required=False,
            type=str,
            choices=['auto', 'human'],
            default=None,
            help=
            'Animation mode for video_generate project: auto (default) or human.'
        )

        # 设置默认的子命令工厂函数，argparse 解析后会调用 subparser_func(args)
        parser.set_defaults(func=subparser_func)

    def execute(self):
        """执行 `ms-agent run` 命令的主逻辑。

        优先处理 --project 参数（内置项目模式），
        将项目目录路径写入 self.args.config，再走统一的配置加载流程。
        若未指定 --project，则直接进入配置加载流程。
        """
        if getattr(self.args, 'project', None):
            # --project 和 --config 不能同时指定
            if self.args.config:
                raise ValueError(
                    'Please specify only one of --config or --project')

            project = self.args.project
            # 使用 importlib.resources 定位包内的项目目录（兼容 zip 安装方式）
            project_trav = importlib_resources.files('ms_agent').joinpath(
                'projects', project)

            if not project_trav.exists():
                # 项目不存在时，列出可用项目供用户参考
                projects_root = importlib_resources.files('ms_agent').joinpath(
                    'projects')
                available = []
                if projects_root.exists():
                    available = [
                        p.name for p in projects_root.iterdir() if p.is_dir()
                    ]
                raise ValueError(
                    f'Unknown project: {project}. Available: {available}')

            # as_file 确保即使安装为 zip 也能获取真实的文件系统路径
            with importlib_resources.as_file(project_trav) as project_dir:
                self.args.config = str(project_dir)
                return self._execute_with_config()

        # 未指定 --project，直接走配置加载流程
        return self._execute_with_config()

    def _execute_with_config(self):
        """根据已确定的配置路径加载配置，构建引擎并运行。

        执行步骤：
        1. 若未指定 --config，尝试从当前工作目录查找 agent.yaml；
        2. 若 --config 指向不存在的路径，尝试从 ModelScope Hub 下载；
        3. 将字符串类型的布尔参数转换为实际布尔值；
        4. 若指定了 --animation_mode，写入环境变量供下游使用；
        5. 打印 ASCII 艺术 banner 及工作流贡献者信息；
        6. 使用 Config.from_task 加载配置；
        7. 根据配置类型选择 WorkflowLoader 或 AgentLoader 构建引擎；
        8. 以 asyncio.run 异步方式执行 engine.run(query)。
        """
        # 若未指定 --config，检查当前目录是否存在默认配置文件（agent.yaml）
        if not self.args.config:
            current_dir = os.getcwd()
            if os.path.exists(os.path.join(current_dir, AGENT_CONFIG_FILE)):
                self.args.config = os.path.join(current_dir, AGENT_CONFIG_FILE)
        elif not os.path.exists(self.args.config):
            # 路径不存在时，尝试作为 ModelScope Hub 仓库 ID 下载到本地
            from modelscope import snapshot_download
            self.args.config = snapshot_download(self.args.config)

        # 将 'true'/'false' 字符串转换为 Python bool 值
        self.args.trust_remote_code = strtobool(
            self.args.trust_remote_code)  # noqa
        self.args.load_cache = strtobool(self.args.load_cache)

        # 将动画模式写入环境变量，供视频生成等下游 Agent 读取
        if getattr(self.args, 'animation_mode', None):
            os.environ['MS_ANIMATION_MODE'] = self.args.animation_mode

        # 统一处理配置路径：文件路径取绝对路径，目录路径直接使用
        if os.path.isfile(self.args.config):
            config_path = os.path.abspath(self.args.config)
        else:
            config_path = self.args.config

        # 读取项目贡献者信息（author.txt），用于在 banner 下方展示
        author_file = os.path.join(config_path, 'author.txt')
        author = ''
        if os.path.exists(author_file):
            with open(author_file, 'r') as f:
                author = f.read()

        # 打印蓝色 ASCII 艺术 banner（MS_AGENT_ASCII）
        blue_color_prefix = '\033[34m'
        blue_color_suffix = '\033[0m'
        print(
            blue_color_prefix + MS_AGENT_ASCII + blue_color_suffix, flush=True)

        # 若存在贡献者信息，在 banner 下方以装饰边框展示
        line_start = '═════════════════════════Workflow Contributed By════════════════════════════'
        line_end = '════════════════════════════════════════════════════════════════════════════'
        if author:
            print(
                blue_color_prefix + line_start + blue_color_suffix, flush=True)
            print(
                blue_color_prefix + author.strip() + blue_color_suffix,
                flush=True)
            print(blue_color_prefix + line_end + blue_color_suffix, flush=True)

        # 使用 Config.from_task 从配置目录/文件中构建配置对象
        config = Config.from_task(self.args.config)

        if Config.is_workflow(config):
            # 配置中包含 workflow 定义时，使用工作流引擎（支持多 Agent 编排）
            from ms_agent.workflow.loader import WorkflowLoader
            engine = WorkflowLoader.build(
                config_dir_or_id=self.args.config,
                config=config,
                mcp_server_file=self.args.mcp_server_file,
                load_cache=self.args.load_cache,
                trust_remote_code=self.args.trust_remote_code)
        else:
            # 普通单 Agent 配置，使用 AgentLoader 构建单个 Agent 引擎
            from ms_agent.agent.loader import AgentLoader
            engine = AgentLoader.build(
                config_dir_or_id=self.args.config,
                config=config,
                mcp_server_file=self.args.mcp_server_file,
                load_cache=self.args.load_cache,
                trust_remote_code=self.args.trust_remote_code)

        # 以异步方式运行引擎；engine.run(query) 接受 None 表示进入交互模式
        asyncio.run(engine.run(self.args.query))
