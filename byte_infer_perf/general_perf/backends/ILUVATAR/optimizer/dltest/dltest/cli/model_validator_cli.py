# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.


import json
import os
import os.path as ospath
from pprint import pprint
from typing import List, Union

from dltest.utils.base_cli import BaseCLI
from dltest.utils.get_env import get_gpu_type
from dltest.utils.misc import get_full_path
from dltest.model_compare_config import get_compare_config_with_full_path
from dltest.log_comparator import compare_logs_with_paths
from dltest.utils.subprocess_tools import get_output


REMAINDER = '...'


class ModelValidatorCLI(BaseCLI):

    def command_name(self):
        return "validate"

    def predefine_args(self):
        super(ModelValidatorCLI, self).predefine_args()
        self.parser.add_argument('-l', '--compare_log', type=str, default=None, help="Compare log")
        self.parser.add_argument('--saved', type=str, default=None, help='Save to path')
        self.parser.add_argument('--with_exit_code', type=int, default=1, help="Add exit code for the result of compared")
        self.parser.add_argument('--print_result', action="store_true", default=False, help='Whether print result')
        self.parser.add_argument('--capture_output', type=str, default='pipe', choices=['pipe', 'tempfile'], help='The method of capture output')
        self.parser.add_argument("run_script", nargs=REMAINDER)

    def parse_args(self, *args, **kwargs):
        args = super(ModelValidatorCLI, self).parse_args()
        if len(args.run_script) == 0:
            print("ERROR: Invalid run_script")
            exit(1)

        return args

    def run(self):
        args = self.parse_args()
        output = self._run_script(args.run_script, capture_output_method=args.capture_output)
        self.compare_logs(
            output, args.compare_log, args.run_script,
            args.saved, args.with_exit_code,
            args.print_result
        )

    def compare_logs(self, output: List, compare_log: str,
                     run_script: List[str], saved: str=None,
                     with_exit_code: int=1, print_result=False):
        script_path = self._get_script_path(run_script)
        script_path = get_full_path(script_path)
        compare_args = get_compare_config_with_full_path(script_path)

        if compare_log is None:
            epoch = self._get_epoch(run_script)
            script_name = ospath.basename(script_path)
            dist_tag = self._get_dist_tag(script_name)
            compare_log = self._find_comparable_log(script_path, epoch, dist_tag)

            if not ospath.exists(compare_log):
                print(f"ERROR: {compare_log} not exist. Or please use argument `l` to locate log.")
                exit(1)

        compare_args['log1'] = output
        compare_args['log2'] = compare_log

        satisfied, results = compare_logs_with_paths(**compare_args)

        if print_result:
            pprint(results)

        if satisfied:
            print("SUCCESS")
        else:
            print("FAIL")

        if saved is not None:
            with open(saved, 'w') as f:
                json.dump(results, f)

        if with_exit_code:
            if satisfied:
                exit(0)
            else:
                exit(1)

    def _run_script(self, command: List, capture_output_method: str='tempfile'):
        return get_output(command, capture_output_method=capture_output_method)

    def _get_script_path(self, run_script: List[str]):
        for i, field in enumerate(run_script):
            if field.endswith('.py') or field.endswith('.sh'):
                return field

        raise RuntimeError("Not found the name of script, " +
                           "only support python or `sh` script, but got {}.".format(run_script))

    def _find_comparable_log(self, script_path: str, epoch: Union[str, int], dist_tag: str):
        gpu_type = get_gpu_type().lower()

        # Get the platform of trained log
        if gpu_type == "nv":
            gpu_type = 'bi'
        else:
            gpu_type = 'nv'

        script_path = get_full_path(script_path)
        project_dir = self._get_project_dir(script_path)
        script_name = ospath.basename(script_path)

        log_path = f"{project_dir}/runing_logs/{gpu_type}/{gpu_type}-{script_name}.epoch_{epoch}{dist_tag}.log"

        return log_path


    def _get_epoch(self, run_script: List[str]):
        for i, field in enumerate(run_script):
            if "--epoch" in field:
                if "=" in field:
                    return field.split("=")[1]
                else:
                    return run_script[i + 1]

        return 'default'

    def _get_dist_tag(self, script_name: str):
        try:
            import torch
            num_gpus = torch.cuda.device_count()
        except:
            num_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "all")

        if '_dist_' in script_name or '_multigpu_' in script_name:
            return f".{num_gpus}card"
        return ""

    def _get_project_dir(self, abs_path):
        abs_path = ospath.abspath(abs_path)
        script_dir = ospath.dirname(abs_path)
        executables_dir = ospath.dirname(script_dir)
        return ospath.dirname(executables_dir)
