# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.


import json
import sys
from typing import Mapping
from os.path import basename, join, exists, expanduser, dirname

from dltest.log_parser import LogParser
from dltest.cli.log_parser_cli import LogParserCLI
from dltest.utils.iluvatar import get_iluvatar_card_type, IluvatarGPU




def parse_target(target):
    result = {}
    targets = target.split(",")
    for i in targets:
        item = i.split(":")
        assert len(item) == 2
        key, value = item
        result[key] = float(value)
    return result
        

def load_json(file):
    file_path = expanduser(file)
    # 检查文件是否存在
    if exists(file_path):
        # 加载json文件
        with open(file_path, 'r') as file:
            data = json.load(file)
    else:
        # 创建一个空的json文件
        data = {}

    return data

def process_results(results):
    result = dict()
    for i in results["results"]:
        for k, v in i.items():
            result[k] = v[0]
    return result

class FetchLog(LogParserCLI):

    def command_name(self):
        return "fetch"

    def predefine_args(self):
        super(FetchLog, self).predefine_args()
        self.parser.add_argument('log', nargs='?', type=str, help="Log path")
        self.parser.add_argument('--saved', type=str, default=None, help='Save to path')
        self.parser.add_argument('--saved_entry', type=str, default=None, help='Save to path')
        self.parser.add_argument('-t_bi150','--target_bi150', type=str, default=-1.)
        self.parser.add_argument('-t_mr100','--target_mr100', type=str, default=-1.)
        self.parser.add_argument('-t_mr50','--target_mr50', type=str, default=-1.)

    def run(self):
        args = self.parse_args()
        parser = LogParser(
            patterns=args.patterns, pattern_names=args.pattern_names,
            use_re=args.use_re, nearest_distance=args.nearest_distance,
            start_line_pattern_flag=args.start_flag,
            end_line_pattern_flag=args.end_flag,
            split_pattern=args.split_pattern,
            split_sep=args.split_sep,
            split_idx=args.split_idx
        )

        results = parser.parse(args.log)
        if not isinstance(results, Mapping):
            results = dict(results=results)
        results = process_results(results)
        print(results)

        if args.saved is not None:
            saved = load_json(args.saved)
            if not args.saved_entry:
                raise Exception("You need to use --saved_entry to specify entry name of the result")

            saved[args.saved_entry] = results
            with open(args.saved, 'w') as f:
                json.dump(saved, f, indent=4)
        self.compare_results(args, results)


    def compare_results(self, args, results):
        card = get_iluvatar_card_type()
        if card == IluvatarGPU.UNKNOWN:
            print("Not known which card is used, can you use ixsmi in the environment?")
            return
        user_target = getattr(args, 'target_'+card.name.lower(), "")
        user_target = parse_target(user_target)

        is_expected = True
        for key, target in user_target.items():
            if key not in results:
                continue
            if results[key]<target:
                is_expected = False
                print(f"- Check {key} on {card.name} failed (result vs target): {results[key]}<{target}")
            else:
                print(f"- Check {key} on {card.name} passed (result vs target): {results[key]}>={target}")
        if not is_expected:
            sys.exit(1)
