# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.
import copy
import os


def get_full_path(fname):
    pwd = os.getcwd()
    if fname.startswith('/'):
        return fname
    return os.path.join(pwd, fname)


def is_main_proc(rank):
    return str(rank) in ["None", "-1", "0"]


def main_proc_print(*args, **kwargs):
    if "RANK" in os.environ:
        if is_main_proc(os.environ["RANK"]):
            print(*args, **kwargs)
            return

    if "LOCAL_RANK" in os.environ:
        if is_main_proc(os.environ["LOCAL_RANK"]):
            print(*args, **kwargs)
            return

    print(*args, **kwargs)


def create_subproc_env():
    env = copy.copy(os.environ)
    env["USE_DLTEST"] = "1"
    return env