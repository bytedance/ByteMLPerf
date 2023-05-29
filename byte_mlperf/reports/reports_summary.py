# Copyright 2023 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

'''
labels : x轴坐标标签序列
datas : 数据集, 二维列表, 要求列表每个元素的长度必须与labels的长度一致
tick_step : 默认x轴刻度步长为1, 通过tick_step可调整x轴刻度步长。
group_gap : 柱子组与组之间的间隙，最好为正值，否则组与组之间重叠
bar_gap : 每组柱子之间的空隙, 默认为0, 每组柱子紧挨，正值每组柱子之间有间隙，负值每组柱子之间重叠
'''
def create_multi_bars(summary):  
    tick_step = 8
    group_gap = 5
    bar_gap = 0
    labels = []
    datas = []
    backends = []

    for name in summary.keys():
        labels.append(name)

    backends = list(summary[labels[0]].keys())

    for item in summary.values():
        for idx, qps in enumerate(item.values()):
            if idx == len(datas):
                datas.append([qps])
            else:
                datas[idx].append(qps)

    # ticks为x轴刻度
    ticks = np.arange(len(labels)) * tick_step
    # group_num为数据的组数，即每组柱子的柱子个数
    group_num = len(datas)
    # group_width为每组柱子的总宽度，group_gap 为柱子组与组之间的间隙。
    group_width = tick_step - group_gap
    # bar_span为每组柱子之间在x轴上的距离，即柱子宽度和间隙的总和
    bar_span = group_width / group_num
    # bar_width为每个柱子的实际宽度
    bar_width = bar_span - bar_gap
    # baseline_x为每组柱子第一个柱子的基准x轴位置，随后的柱子依次递增bar_span即可
    baseline_x = ticks - (group_width - bar_span) / 2
    
    def autolabel(backend, rects):
        """在*rects*中的每个柱状条上方附加一个文本标签，显示其高度"""
        for rect in rects:
            height = rect.get_height()
            plt.annotate('{}:{}'.format(backend,height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom')
        
    plt.figure(figsize=(22, 15))
    for index, y in enumerate(datas):
        rects = plt.bar(baseline_x + index*bar_span, y, bar_width, label=backends[index])
        autolabel(backends[index], rects)

    # x轴刻度标签位置与x轴刻度一致
    plt.xticks(ticks, labels, rotation=330)

    plt.legend()
    plt.xlabel('Backends')
    plt.ylabel('Model QPS')
    plt.yscale('log')
    plt.title('Reports Summary(QPS)')
    
    plt.savefig("byte_mlperf/reports/reports_summary.png", dpi=100)

def get_best_qps(backend, report_name):
    if not os.path.exists('byte_mlperf/reports/' + backend + '/' + report_name + "/result.json"):
        return 0
        
    with open('byte_mlperf/reports/' + backend + '/' + report_name + "/result.json",  'r') as f:
        report_info = json.load(f)
        all_qps= report_info['Performance']
        best_qps = 0
        for qps in all_qps:
            if qps['QPS'] > best_qps:
                best_qps = qps['QPS']
        return int(best_qps)

def reports_summary():
    all_backends = []
    for file in os.listdir('byte_mlperf/reports'):
        if os.path.isdir(os.path.join('byte_mlperf/reports', file)):
            all_backends.append(file)

    all_reports_names = []
    for backend in all_backends:
        for report_name in os.listdir('byte_mlperf/reports/' + backend):
            if report_name not in all_reports_names:
                all_reports_names.append(report_name)

    summary = {}
    for name in all_reports_names:
        summary[name] = {key : 0 for key in all_backends}
    
    for report_name in summary.items():
        for backend in report_name[1].keys():
            best_qps = get_best_qps(backend, report_name[0])
            summary[report_name[0]][backend] = best_qps
    
    create_multi_bars(summary)


if __name__ == "__main__":
    reports_summary()
