# Copyright (c) 2024 Advanced Micro Devices, Inc.  All rights reserved.
#
# -*- coding:utf-8 -*-
# @Script: test_common.py
# @Author: valarLip
# @Email: lingpeng.jin@amd.com
# @Create At: 2024-11-03 15:53:32
# @Last Modified By: valarLip
# @Last Modified At: 2024-11-03 16:20:14
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
    assert torch.allclose(
        a, b, rtol, atol), f'''torch and ck results are not close\ntorch: {a.shape}\n{a}\nck: {b.shape}\n{b}\nmax delta:{(a-b).max()}
    delta details: {(a[(a-b)>atol]).numel()/a.numel():.1%} ({(a[(a-b)>atol]).numel()} of {a.numel()}) elements are bigger than atol: {atol}
          a: 
            {a[(a-b)>atol]}
          b: 
            {b[(a-b)>atol]}
      dtlta: 
            {(a-b)[(a-b)>atol]}'''
