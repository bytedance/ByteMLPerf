import os

from .assert_cli import AssertCLI
from ..utils.subprocess_tools import execute_shell

RUN_MODE_KEY = "RUN_MODE"
RUN_MODE_STRICT = "strict"


class CheckCli(AssertCLI):

    def __init__(self, *args, **kwargs):
        super(CheckCli, self).__init__(*args, **kwargs)
        self.args = None

    def command_name(self):
        return "check"

    def predefine_args(self):
        self.parser.add_argument("--check_mode", type=str, default="no",
                                 choices=["all", "strict", "nonstrict", "no"],
                                 help="which running mode needs to be checked")
        self.parser.add_argument("--nonstrict_mode_args", type=str, default="",
                                 help="the arguments are used with nonstric testing")
        super(CheckCli, self).predefine_args()

    def parse_args(self, *args, **kwargs):
        if self.args is None:
            args = super(CheckCli, self).parse_args(*args, **kwargs)
            args.use_predefined_parser_rules = True
            args.nonstrict_mode_args = args.nonstrict_mode_args.split(" ")

            if not self.is_strict_testing():
                args.run_script.extend(args.nonstrict_mode_args)

            if args.check_mode == "all":
                args.check_mode = self.current_running_mode()

            self.args = args
        return self.args

    def run(self):
        args = self.parse_args()
        if args.check_mode == self.current_running_mode():
            return super(CheckCli, self).run()
        else:
            res = execute_shell(args.run_script)
            exit(res.returncode)

    def current_running_mode(self):
        return os.environ.get(RUN_MODE_KEY, RUN_MODE_STRICT)

    def is_strict_testing(self):
        return self.current_running_mode() == RUN_MODE_STRICT


