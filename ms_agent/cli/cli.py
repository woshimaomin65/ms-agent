"""
ms-agent 命令行工具的总入口模块。

该模块定义了 `run_cmd` 函数，是整个 CLI 工具的顶层入口。
通过 argparse 将不同的子命令（run、app、ui）聚合在一起，
用户执行 `ms-agent <command> [args]` 时最终调用此函数。

调用链：
    setup.py entry_point: ms-agent → ms_agent.cli.cli:run_cmd
    run_cmd()
      ├─ RunCMD  (ms-agent run ...)   → 运行 Agent 任务
      ├─ AppCMD  (ms-agent app ...)   → 启动 WebUI 应用
      └─ UICMD   (ms-agent ui ...)    → UI 相关命令
"""

import argparse

# 导入三个子命令类：app、run、ui
from ms_agent.cli.app import AppCMD
from ms_agent.cli.run import RunCMD
from ms_agent.cli.ui import UICMD


def run_cmd():
    """CLI 命令的总入口函数，负责聚合所有子命令。

    该函数通过 argparse 构建一个父解析器，并将所有子命令（run/app/ui）
    注册到其子解析器（subparsers）中，使用户可以通过
    `ms-agent run`、`ms-agent app`、`ms-agent ui` 等方式分别调用不同功能。

    执行流程：
    1. 创建父 ArgumentParser；
    2. 为 run/app/ui 分别注册子命令（define_args）；
    3. 解析命令行参数（允许未知参数，留给 config.py 处理）；
    4. 若没有指定子命令，打印帮助信息并退出；
    5. 通过 args.func(args) 实例化对应的子命令对象，再调用 execute()。
    """
    # 创建顶层命令解析器，程序名称为 'ModelScope-agent Command Line tool'
    parser = argparse.ArgumentParser(
        'ModelScope-agent Command Line tool',
        usage='ms-agent <command> [<args>]')

    # 添加子命令解析器组，用于承载 run / app / ui 等子命令
    subparsers = parser.add_subparsers(
        help='ModelScope-agent commands helpers')

    # 向子命令解析器组中注册各子命令及其参数定义
    RunCMD.define_args(subparsers)   # 注册 `ms-agent run` 子命令
    AppCMD.define_args(subparsers)   # 注册 `ms-agent app` 子命令
    UICMD.define_args(subparsers)    # 注册 `ms-agent ui`  子命令

    # 解析命令行参数；未知参数（_）将由 Config 类在后续处理
    args, _ = parser.parse_known_args()

    # 如果用户没有输入子命令，args 中不会包含 'func' 属性
    # 此时打印帮助信息并以错误码退出
    if not hasattr(args, 'func'):
        parser.print_help()
        exit(1)

    # args.func 由各子命令的 parser.set_defaults(func=subparser_func) 设置
    # 调用对应的工厂函数创建子命令实例，再执行其 execute() 方法
    cmd = args.func(args)
    cmd.execute()


if __name__ == '__main__':
    # 允许直接以 `python cli.py` 方式运行，同样调用 run_cmd 入口
    run_cmd()
