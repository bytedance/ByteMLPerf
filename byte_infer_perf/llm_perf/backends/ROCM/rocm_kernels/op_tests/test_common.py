# Copyright (c) 2024 Advanced Micro Devices, Inc.  All rights reserved.
#
# -*- coding:utf-8 -*-
# @Script: test_common.py
# @Author: valarLip
# @Email: lingpeng.jin@amd.com
# @Create At: 2024-11-03 15:53:32
# @Last Modified By: valarLip
# @Last Modified At: 2024-11-09 19:41:56
# @Description: This is description.

import torch
import numpy as np
num_iters = 100
num_warmup = 20


def perftest(name=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            latencies = []
            for i in range(num_iters+num_warmup):
                start_event.record()
                data = func(*args, **kwargs)
                end_event.record()
                end_event.synchronize()
                latencies.append(start_event.elapsed_time(end_event))
            avg = np.mean(latencies[num_warmup:]) * 1000
            return data, avg
        return wrapper
    return decorator


def checkAllclose(a, b, rtol=1e-2, atol=1e-2):
    isClose = torch.isclose(a, b, rtol=rtol, atol=atol)
    mask = ~isClose
    if isClose.all():
        print('[passed~]')
    else:
        percent = (a[mask]).numel()/a.numel()
        if percent > 0.01:
            print(f'''[failed!] {a.shape=}\n{b.shape=}\n        a:
            {a[mask]}
        b:
            {b[mask]}
    dtlta:
            {(a-b)[mask]}''')
        else:
            print(f'''[waring!] a and b results are not close\nmax delta:{(a-b).max()}
        delta details: {percent:.1%} ({(a[mask]).numel()} of {a.numel()}) elements {atol=} {rtol=}''')
