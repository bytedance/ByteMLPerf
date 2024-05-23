# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.


from argparse import ArgumentParser
from abc import abstractmethod


class BaseCLI:

    def __init__(self, parser=None, *args, **kwargs):
        if parser is None:
            self.parser = ArgumentParser(description=self.description ,*args, **kwargs)

    def __call__(self):
        self.run()

    @property
    def description(self):
        return None

    @abstractmethod
    def command_name(self):
        pass

    def predefine_args(self):
        pass

    def parse_args(self, *args, **kwargs):
        self.predefine_args()
        return self.parser.parse_args(*args, **kwargs)

    @abstractmethod
    def run(self):
        pass



