# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.


from typing import List, Optional, Union, Mapping
import re
import sys


DEFAULT_NEAREST_MATCH_CHARS = 10


def read_file(file):
    with open(file, 'r') as f:
        return f.readlines()

def read_pipe():
    result = []
    for line in sys.stdin:
        result.append(line)
    return result

def postprocess_search_result(results: List[str]) -> List[float]:
    if len(results) != 0:
        results = list(map(float, results))
    return results


def extract_nearest_value_by_key_inline(content: str, key: str,
                                        nearest_distance: int=DEFAULT_NEAREST_MATCH_CHARS) -> List[float]:
    pattern = "%s[\s\S]{0,%d}?(\d+(?:\.\d+)?)" % (key, nearest_distance)
    return extract_value_by_pattern_inline(content, pattern)


def extract_value_by_pattern_inline(content: str, pattern: str) -> List[float]:
    results = re.findall(pattern, content)
    return postprocess_search_result(results)


def extract_value(content: str, pattern: str,
                  inline=True, use_re=False,
                  nearest_distance: int=DEFAULT_NEAREST_MATCH_CHARS) -> List[float]:
    if inline:
        if use_re:
            return extract_value_by_pattern_inline(content, pattern)
        else:
            return extract_nearest_value_by_key_inline(content, pattern, nearest_distance)
    else:
        raise NotImplementedError()


class LogParser:

    def __init__(self,
                 patterns: List[str]=None,
                 pattern_names: List[str]=None,
                 use_re: bool=False,
                 nearest_distance: int=DEFAULT_NEAREST_MATCH_CHARS,
                 start_line_pattern_flag: str=None,
                 end_line_pattern_flag: str=None,
                 split_pattern: Union[str, List]=None,
                 split_sep: List[str]=None,
                 split_idx: List[int]=None):
        if patterns is None and split_sep is None:
            raise ValueError("The one of argument `patterns` or `split_sep` must be given.")

        if pattern_names is not None:
            if isinstance(patterns, (tuple, list)) and patterns is not None and len(patterns) != len(pattern_names):
                raise ValueError("The length of `pattern_names` argument not equal to `patterns`.")
            if isinstance(split_sep, (tuple, list)) and split_sep is not None and len(split_sep) != len(pattern_names):
                raise ValueError("The length of `pattern_names` argument not equal to `split_sep`.")

        if split_sep is not None and (split_idx is None or not isinstance(split_idx, (int, tuple, list))):
            raise ValueError("Invalid index to split text, got {}.".format(split_idx))

        if split_sep is not None and split_pattern is None:
            raise ValueError("Invalid pattern to split text, got {}.".format(split_pattern))

        self.patterns = patterns
        self.use_re = use_re
        self.nearest_distance = nearest_distance
        self.start_line_pattern_flag = start_line_pattern_flag
        self.end_line_pattern_flag = end_line_pattern_flag

        if not isinstance(split_sep, (tuple, list)) and split_sep is not None:
            split_sep = [split_sep]

            if not isinstance(split_idx, (tuple, list)):
                split_idx = [split_idx]

        self.split_sep = split_sep
        self.split_idx = split_idx

        if pattern_names is None:
            if patterns is None:
                pattern_names = split_idx
            else:
                pattern_names = patterns
        self.pattern_names = pattern_names

        if not isinstance(split_pattern, (tuple, list)) and split_sep is not None:
            split_pattern = [split_pattern] * len(split_sep)
        self.split_pattern = split_pattern

        self.start_record = start_line_pattern_flag is None

    def parse(self, path_or_logs: Union[str, List]) -> List[dict]:
        """
        : return: [{matric_name: value}, ...]
        """

        
        if path_or_logs:
            path_or_logs = read_file(path_or_logs)
        else:
            path_or_logs = read_pipe()

        ret = []
        for line in path_or_logs:
            result = self.parse_inline(line)
            if len(result) == 0:
                continue
            ret.append(result)
        return ret

    def parse_inline(self, line) -> dict:
        if not self.can_record(line):
            return {}

        if self.split_sep is None:
            return self._parse_inline_by_match(line)
        return self._parse_inline_by_split(line)

    def _parse_inline_by_match(self, line: str):
        ret = {}
        for name, pattern in zip(self.pattern_names, self.patterns):
            result = extract_value(
                line, pattern, inline=True, use_re=self.use_re,
                nearest_distance=self.nearest_distance
            )
            if len(result) == 0:
                continue
            ret[name] = result
        return ret

    def _parse_inline_by_split(self, line: str, to_type=float):
        ret = {}
        for name, sep, idx, pattern in zip(self.pattern_names,
                                  self.split_sep,
                                  self.split_idx,
                                  self.split_pattern):
            if not self.can_matched(line, pattern):
                continue
            if '\t' in sep:
                segs = line.strip().split(sep)
            else:
                segs = line.strip().replace('\t', ' ').split(sep)
            segs = list(filter(lambda kv: kv.strip() not in ["", " ", None], segs))
            if len(segs) <= idx:
                continue
            ret[name] = to_type(segs[idx])
        return ret

    def can_record(self, line: str):
        if self.start_line_pattern_flag is None:
            self.start_record = True
        elif not self.start_record:
            self.start_record = self.can_matched(line, self.start_line_pattern_flag)

        if self.start_record:
            if self.end_line_pattern_flag is not None and self.can_matched(line, self.end_line_pattern_flag):
                self.start_record = False

        return self.start_record

    def can_matched(self, content: str, pattern: str):
        result = re.findall(pattern, content)
        return len(result) != 0

