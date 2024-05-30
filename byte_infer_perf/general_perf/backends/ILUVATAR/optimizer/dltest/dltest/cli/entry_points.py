# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.


from dltest.cli.assert_cli import AssertCLI
from dltest.cli.log_comparator_cli import LogComparatorCLI
from dltest.cli.model_validator_cli import ModelValidatorCLI
from dltest.cli.fetch_log_cli import FetchLog
from dltest.cli.check_cli import CheckCli


#log_comparator_cli = LogComparatorCLI()
#model_validator_cli = ModelValidatorCLI()
fetch_log_cli = FetchLog()
#assert_cli = AssertCLI()
#check_cli = CheckCli()


def make_execute_path():
    preffix = "dltest.cli.entry_points"
    clis = []
    for cli_var in globals():
        if cli_var.endswith('_cli'):
            cmd_name = globals()[cli_var].command_name()
            clis.append(f"ixdltest-{cmd_name}={preffix}:{cli_var}")

    return clis


