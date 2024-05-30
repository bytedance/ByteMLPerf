# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.


import json
from typing import Mapping

from dltest.log_parser import LogParser, DEFAULT_NEAREST_MATCH_CHARS
from dltest.utils.base_cli import BaseCLI


class LogParserCLI(BaseCLI):

    def predefine_args(self):
        self.parser.add_argument('-p', '--patterns', nargs="*", type=str, default=None, help='Fetched patterns')
        self.parser.add_argument('-pn', '--pattern_names', nargs="*", type=str, default=None, help='The name of pattern')
        self.parser.add_argument('--use_re', action="store_true", default=False, help='Whether use regular expression')
        self.parser.add_argument('-d', '--nearest_distance', type=int, default=DEFAULT_NEAREST_MATCH_CHARS, help='The nearest distance of matched pattern')
        self.parser.add_argument('--start_flag', type=str, default=None, help='The flag of start to record log')
        self.parser.add_argument('--end_flag', type=str, default=None, help='The flag of stop to record log')
        self.parser.add_argument('--split_pattern', type=str, default=None, help='The pattern is used to match line')
        self.parser.add_argument('--split_sep', nargs="*", type=str, default=None, help='The seperator is used to split line')
        self.parser.add_argument('--split_idx', nargs="*", type=int, default=None, help='The index of split line')

    def parse_args(self, *args, **kwargs):
        args = super(LogParserCLI, self).parse_args(*args, **kwargs)

        return args

