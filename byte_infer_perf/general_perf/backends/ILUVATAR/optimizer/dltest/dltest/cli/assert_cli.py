# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.

import os
from typing import List, Iterable, Optional

from dltest.cli.log_parser_cli import LogParserCLI
from dltest.log_parser import LogParser
from dltest.model_compare_config import get_compare_config_with_full_path
from dltest.utils.misc import get_full_path
from dltest.utils.subprocess_tools import get_output
from dltest.model_compare_config import ComparatorConfig


FRAMEWORKS = list(ComparatorConfig.get_frameworks())

REMAINDER = '...'

assertion_expr_factory = dict(
    eq = "a == b",
    ne = "a != b",
    ge = "a >= b",
    le = "a <= b",
    gt = "a > b",
    lt = "a < b",
)


class AssertCLI(LogParserCLI):

    def command_name(self):
        return "assert"

    def predefine_args(self):
        super(AssertCLI, self).predefine_args()
        self.parser.add_argument('-b', '--assertion_second_value', type=float, default=None,
                                 help='It is used in assertion expression.')
        self.parser.add_argument('--print_result', action="store_true", default=False,
                                 help='Whether print result')
        self.parser.add_argument('--capture_output', type=str, default='pipe', choices=['pipe', 'tempfile'],
                                 help='The method of capture output')
        # FIXME: Using store_action to replase it
        self.parser.add_argument('--only_last', type=int, default=0,
                                 help='Whether use the last result to compare')
        self.parser.add_argument('--expr', type=str, default="ge",
                                 help=f"Assertion expression, option keys: {', '.join(assertion_expr_factory.keys())}" +
                                 ", or a executable code, such as `a > b`, `a > 1`, ...")
        self.parser.add_argument('--use_predefined_parser_rules', action="store_true", default=False,
                                 help='Whether use predefined args of parser.')
        self.parser.add_argument('--log', type=str, default=None, help="Log path")
        self.parser.add_argument("--run_script", default=[], nargs=REMAINDER)

    def parse_args(self, *args, **kwargs):
        args = super(AssertCLI, self).parse_args()
        args.only_last = args.only_last > 0
        if len(args.run_script) == 0 and args.log is None:
            raise ValueError("The one of `--run_script` or `--log` must be given.")

        if args.assertion_second_value is None:
            if args.expr is None:
                raise ValueError("The one of `--assertion_second_value` or `--expr` must be given.")

            if args.expr in assertion_expr_factory:
                raise ValueError(
                    "The comparison operators depend on the argument `assertion_second_value`."
                )

        return args

    def create_parser(self, args):
        if args.use_predefined_parser_rules:
            script_path = self._get_script_path(args.run_script)
            config = get_compare_config_with_full_path(script_path, to_dict=False)

            return LogParser(
                patterns=config.patterns, pattern_names=config.pattern_names,
                use_re=config.use_re, nearest_distance=config.nearest_distance,
                start_line_pattern_flag=config.start_line_pattern_flag,
                end_line_pattern_flag=config.end_line_pattern_flag,
                split_pattern=config.split_pattern,
                split_sep=config.split_sep,
                split_idx=config.split_idx
            )

        return LogParser(
            patterns=args.patterns, pattern_names=args.pattern_names,
            use_re=args.use_re, nearest_distance=args.nearest_distance,
            start_line_pattern_flag=args.start_flag,
            end_line_pattern_flag=args.end_flag,
            split_pattern=args.split_pattern,
            split_sep=args.split_sep,
            split_idx=args.split_idx
        )

    def run(self):
        args = self.parse_args()
        parser = self.create_parser(args)

        if args.print_result:
            print(args)

        output = self.get_log(args)
        parsed_logs = self.parser_log(parser, output, args)
        self.check_logs(parsed_logs, args)

    def get_log(self, args):
        if len(args.run_script) == 0:
            try:
                with open(args.log) as f:
                    return f.readlines()
            except:
                print(f"ERROR: Read log fail in {args.log}")
                exit(1)
        else:
            return get_output(args.run_script, capture_output_method=args.capture_output)

    def parser_log(self, parser, output, args) -> List[float]:
        results = parser.parse(output)
        if args.only_last:
            results = results[-1:]

        if len(results) == 0:
            raise ValueError("The parsed results is empty, please check patterns.")
        if isinstance(results[0], dict):
            if len(results[0]) == 0:
                raise ValueError("The parsed results is empty, please check patterns.")
            key = list(results[0].keys())[0]
            results = [result[key] for result in results]

        if isinstance(results[0], Iterable):
            results = [result[0] for result in results]

        return results

    def check_logs(self, parsed_logs, args):
        if args.print_result:
            print("Parsed result:", parsed_logs)

        assertion_expr = assertion_expr_factory.get(args.expr, args.expr)

        assert_results = []
        b = args.assertion_second_value
        for a in parsed_logs:
            assert_results.append(eval(assertion_expr))

        if args.print_result:
            print("The result of assertion expression:", assert_results)

        if any(assert_results):
            print("SUCCESS")
            exit(0)
        print("FAIL")
        exit(1)

    def _get_script_path(self, run_script: List[str]):
        # Find shell script by current run_script
        def _find_real_shell_script(cmd: List[str]):
            for i, field in enumerate(cmd):
                if field.endswith('.sh') and self._get_framework(field) in FRAMEWORKS:
                    return field

        real_shell_script = _find_real_shell_script(run_script)

        # Find shell script by parent process
        if real_shell_script is None:
            ppid = os.getppid()
            import psutil
            pproc = psutil.Process(ppid)
            pproc_cmd = pproc.cmdline()
            real_shell_script = _find_real_shell_script(pproc_cmd)

        if real_shell_script is not None:
            real_shell_script = self._get_script_abs_path(real_shell_script)
            return real_shell_script

        raise RuntimeError("The script is not named correctly, " + \
                           "please use a script name ending with the framework, " + \
                           f"got `{' '.join(run_script)}`, " + \
                           "e.g. train_resnet50_torch.sh")

    def _get_framework(self, shell_script: str) -> Optional[str]:
        try:
            return shell_script.split('.')[-2].split('_')[-1]
        except:
            return None

    def _get_script_abs_path(self, run_script):
        real_run_script = os.path.realpath(run_script)
        if os.path.exists(real_run_script):
            return real_run_script

        if "MODEL_DIR" in os.environ:
            return os.path.join(os.environ["MODEL_DIR"], run_script)

        if "OLDPWD" in os.environ:
            real_run_script = os.path.join(os.environ["OLDPWD"], run_script)
            if os.path.exists(real_run_script):
                return real_run_script

        raise FileNotFoundError("Not found running script path, " + \
                                "please set environment variable `MODEL_DIR`, " + \
                                "e.g /path/to/deeplearningsamples/executables/resnet.")

