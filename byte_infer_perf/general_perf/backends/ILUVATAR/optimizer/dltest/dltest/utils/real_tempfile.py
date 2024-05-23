# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.


import os
import os.path as ospath
from pathlib import Path
import tempfile


class TemporaryFile:

    def __init__(self, with_open=False, mode='r'):
        self.name = None
        self.with_open = with_open
        self.mode = mode

        self.file = None

    def create(self):
        self.name = tempfile.mktemp()
        file_path = Path(self.name)
        file_path.touch()

    def delete(self):
        if self.name is not None and ospath.exists(self.name):
            os.unlink(self.name)

    def read(self):
        self._check_file_status()
        return self.file.read()

    def readlines(self):
        self._check_file_status()
        return self.file.readlines()

    def _check_file_status(self):
        if self.file is None:
            raise RuntimeError("File is closed, please reopen it.")

    def __enter__(self):
        self.create()
        if self.with_open:
            self.file = open(self.name, mode=self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.with_open:
            self.file.close()
        self.delete()








