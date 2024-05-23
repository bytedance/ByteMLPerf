# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.


from typing import List, Mapping, Union, Tuple
from .log_parser import LogParser, DEFAULT_NEAREST_MATCH_CHARS

LogLines = List[Mapping]
CompareResult = Tuple[bool, Union[List, Mapping]]


def _compute_errors(value1: Mapping, value2: Mapping, threshold: Mapping, allow_greater_than=False) -> CompareResult:
    if not isinstance(threshold, Mapping):
        _thds = dict()
        for key in value1.keys():
            _thds[key] = threshold
        threshold = _thds

    result = dict()
    satisfied = True
    for key, _thd in threshold.items():
        v1, v2 = value1[key], value2[key]
        origin_value_type = list
        if not isinstance(v1, (tuple, list)):
            origin_value_type = float
            v1 = [v1]
            v2 = [v2]

        real_errors = []
        for v1_i, v2_i in zip(v1, v2):
            real_error = v1_i - v2_i
            real_errors.append(real_error)
            if satisfied and abs(real_error) > _thd:
                if allow_greater_than and real_error > 0:
                    continue
                satisfied = False

        if origin_value_type is float and len(real_errors) > 0:
            real_errors = real_errors[0]

        result[key] = real_errors

    return satisfied, result


def compare_logs(log1: LogLines, log2: LogLines, threshold: Union[float, Mapping], allow_greater_than=False) -> CompareResult:
    total_lines = len(log1[0])
    real_errors = []
    satisfied = True
    for line_idx in range(total_lines):
        _satisfied, _error = _compute_errors(log1[line_idx], log2[line_idx], threshold, allow_greater_than=allow_greater_than)
        real_errors.append(_error)
        if satisfied and not _satisfied:
            satisfied = False

    return satisfied, real_errors


def compare_logs_by_last_result(log1: LogLines, log2: LogLines, threshold: Union[float, Mapping], allow_greater_than=False) -> CompareResult:
    if len(log1) == 0 or len(log2) == 0:
        return False, []
    return _compute_errors(log1[-1], log2[-1], threshold, allow_greater_than=allow_greater_than)


def compare_logs_with_paths(log1, log2, threshold: Union[float, Mapping],
                            patterns: List[str],
                            pattern_names: List[str] = None,
                            use_re: bool = False,
                            nearest_distance: int = DEFAULT_NEAREST_MATCH_CHARS,
                            start_line_pattern_flag: str = None,
                            end_line_pattern_flag: str = None,
                            only_last: bool=True,
                            split_pattern: Union[str, List] = None,
                            split_sep: List = None,
                            split_idx: List = None,
                            allow_greater_than: bool = False):
    parser = LogParser(
        patterns=patterns, pattern_names=pattern_names,
        use_re=use_re, nearest_distance=nearest_distance,
        start_line_pattern_flag=start_line_pattern_flag,
        end_line_pattern_flag=end_line_pattern_flag,
        split_pattern=split_pattern,
        split_sep=split_sep,
        split_idx=split_idx
    )

    log1 = parser.parse(log1)
    log2 = parser.parse(log2)

    if only_last:
        compare_result = compare_logs_by_last_result(log1, log2, threshold, allow_greater_than=allow_greater_than)
    else:
        compare_result = compare_logs(log1, log2, threshold, allow_greater_than=allow_greater_than)

    return compare_result[0], dict(log1=log1, log2=log2, errors=compare_result[-1])
