# Copyright (c) ModelScope Contributors. All rights reserved.
"""
`ms-agent app` 子命令模块。

该模块实现了启动内置 Web 应用的 `app` 子命令。
目前支持两种内置应用类型：
  - `doc_research`：文档研究应用（基于 Gradio）
  - `fin_research`：金融研究应用（基于 Gradio）

用户通过以下方式启动应用：
    ms-agent app --app_type doc_research [--server_name 0.0.0.0] [--server_port 7860] [--share]

调用链：
    run_cmd() → AppCMD.define_args() → AppCMD(args).execute()
      ├─ [doc_research] ms_agent.app.doc_research.launch_server(...)
      └─ [fin_research]  ms_agent.app.fin_research.launch_server(...)
"""

import argparse

from .base import CLICommand


def subparser_func(args):
    """子命令工厂函数，由 argparse 通过 set_defaults(func=...) 调用。

    当用户执行 `ms-agent app ...` 时，argparse 将解析后的 args 传入此函数，
    返回 AppCMD 实例，随后 cli.py 调用 cmd.execute() 触发实际执行逻辑。

    参数：
        args: argparse.Namespace，包含所有已解析的命令行参数。

    返回：
        AppCMD 实例。
    """
    return AppCMD(args)


class AppCMD(CLICommand):
    """实现 `ms-agent app` 子命令的命令类。

    该类负责：
    1. 解析并校验 `app` 子命令相关参数（define_args）；
    2. 根据 --app_type 参数选择并启动对应的 Gradio Web 应用（execute）。
    """

    # 子命令名称，对应 `ms-agent app`
    name = 'app'

    def __init__(self, args):
        """初始化 AppCMD。

        参数：
            args: argparse.Namespace，包含所有已解析的命令行参数。
        """
        self.args = args

    @staticmethod
    def define_args(parsers: argparse.ArgumentParser):
        """向父解析器注册 `app` 子命令的所有参数。

        支持的命令行参数：
          --app_type    : 应用类型，必选，支持 'doc_research' 或 'fin_research'
          --server_name : Gradio 服务绑定的主机名（默认 '0.0.0.0'）
          --server_port : Gradio 服务绑定的端口号（默认 7860）
          --share       : 是否生成公开分享链接（Gradio share 功能）

        参数：
            parsers: argparse 子解析器组，由 cli.py 中的 add_subparsers() 创建。
        """
        # 在子解析器组中注册 'app' 子命令
        parser: argparse.ArgumentParser = parsers.add_parser(AppCMD.name)

        # 使用互斥组确保 --app_type 是必须指定的参数之一
        group = parser.add_mutually_exclusive_group(required=True)

        # --app_type：指定要启动的应用类型（doc_research / fin_research）
        group.add_argument(
            '--app_type',
            type=str,
            default='doc_research',
            help=
            'The app type, supported values: `doc_research`, `fin_research`')

        # --server_name：Gradio 服务器绑定的主机名，默认监听所有网络接口
        parser.add_argument(
            '--server_name',
            type=str,
            default='0.0.0.0',
            help='The gradio server name to bind to.')

        # --server_port：Gradio 服务器监听端口，默认 7860
        parser.add_argument(
            '--server_port',
            type=int,
            default=7860,
            help='The gradio server port to bind to.')

        # --share：若指定此标志，Gradio 将生成一个可公开访问的临时链接
        parser.add_argument(
            '--share',
            action='store_true',
            help='Whether to share the gradio app publicly.')

        # 设置默认的子命令工厂函数，argparse 解析后会调用 subparser_func(args)
        parser.set_defaults(func=subparser_func)

    def execute(self):
        """执行 `ms-agent app` 命令的主逻辑。

        根据 --app_type 参数选择对应应用的启动函数：
        - 'doc_research'：启动文档研究 Gradio 应用；
        - 'fin_research' ：启动金融研究 Gradio 应用；
        - 其他值：抛出 ValueError 异常。

        Gradio 应用启动后会阻塞当前进程，监听指定的服务器地址和端口。
        """
        if self.args.app_type == 'doc_research':
            # 延迟导入，仅在需要时加载文档研究相关依赖（如 Gradio 等）
            from ms_agent.app.doc_research import launch_server as launch_doc_research
            launch_doc_research(
                server_name=self.args.server_name,
                server_port=self.args.server_port,
                share=self.args.share)
        elif self.args.app_type == 'fin_research':
            # 延迟导入，仅在需要时加载金融研究相关依赖
            from ms_agent.app.fin_research import launch_server as launch_fin_research
            launch_fin_research(
                server_name=self.args.server_name,
                server_port=self.args.server_port,
                share=self.args.share)
        else:
            # 不支持的应用类型，抛出明确的错误信息
            raise ValueError(f'Unsupported app type: {self.args.app_type}')
