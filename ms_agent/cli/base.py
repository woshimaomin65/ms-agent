"""
CLI 命令基类模块。

定义了所有 CLI 子命令必须遵循的抽象接口 `CLICommand`。
每个子命令（RunCMD、AppCMD、UICMD 等）都继承此类，
并实现 `define_args`（参数定义）和 `execute`（命令执行）两个抽象方法。
"""

from abc import ABC, abstractmethod
from argparse import ArgumentParser


class CLICommand(ABC):
    """CLI 子命令的抽象基类。

    所有命令行子命令类（如 RunCMD、AppCMD、UICMD）均继承此类，
    并必须实现以下两个抽象方法：

    - `define_args`：向 argparse 子解析器注册该命令所需的参数；
    - `execute`：在解析完命令行参数后执行具体的业务逻辑。

    使用方式：
        class MyCMD(CLICommand):
            @staticmethod
            def define_args(parsers):
                parser = parsers.add_parser('my-cmd')
                parser.add_argument('--foo', ...)
                parser.set_defaults(func=lambda args: MyCMD(args))

            def execute(self):
                # 执行业务逻辑
                ...
    """

    @staticmethod
    @abstractmethod
    def define_args(parsers: ArgumentParser):
        """向父解析器注册子命令及其参数。

        参数：
            parsers: argparse 的子解析器组（由 parent_parser.add_subparsers() 返回），
                     各子命令调用 parsers.add_parser(name) 在其下创建专属解析器。

        子类实现时必须在最后调用 parser.set_defaults(func=subparser_func)，
        以便 cli.py 通过 args.func(args) 创建对应的命令对象。
        """
        raise NotImplementedError()

    @abstractmethod
    def execute(self):
        """执行该子命令的核心业务逻辑。

        该方法在 `cli.py:run_cmd` 中通过 `cmd.execute()` 调用，
        子类在此方法中实现具体的功能（如启动 Agent、启动 WebUI 等）。
        """
        raise NotImplementedError()
