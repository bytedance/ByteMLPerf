# Copyright 2023 Graphcore Ltd.
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

import time

import numpy as np

INPUT_IDS = "input_ids"
POSITION_IDS = "position_ids"
SEGMENT_IDS = "segment_ids"
INPUT_MASK = "input_mask"


class DataSpec:
    def __init__(self, id, row, col, length, positions=None, flush=False, sender=0):
        self.id = id
        self.row = row
        self.col = col
        self.l = length
        self.positions = positions
        self.flush = flush
        self.sender = sender

    def __str__(self):
        return f"{self.id},{self.row},{self.col},{self.l}"

    def __repr__(self):
        return f"{self.id},{self.row},{self.col},{self.l}"

    def debug(self):
        return f"{self.id},{self.row},{self.col},{self.l}"

    def shift(self, row_offset):
        return DataSpec(self.id, self.row - row_offset, self.col, self.l)


class DataTransfer:
    def __init__(self, count, spec, data, last=False):
        self.count = count
        self.specs = spec
        self.data = data
        self.last = last

    def combine(self, data):
        new_specs = self.specs + data.specs
        new_data = dict()
        new_data[INPUT_IDS] = np.concatenate(
            (self.data[INPUT_IDS], data.data[INPUT_IDS]), axis=0
        )
        new_data[POSITION_IDS] = np.concatenate(
            (self.data[POSITION_IDS], data.data[POSITION_IDS]), axis=0
        )
        new_data[SEGMENT_IDS] = np.concatenate(
            (self.data[SEGMENT_IDS], data.data[SEGMENT_IDS]), axis=0
        )
        new_data[INPUT_MASK] = np.concatenate(
            (self.data[INPUT_MASK], data.data[INPUT_MASK]), axis=0
        )

        return DataTransfer(self.count, new_specs, new_data)

    def update(self, data):
        return DataTransfer(self.count, self.specs, data, self.last)

    def flush(self):
        return DataTransfer(self.count, self.specs, self.data, True)

    def split(self, groups, total_rows):
        size_of_group = int(total_rows / groups)
        new_specs = [[] for x in range(groups)]

        for spec in self.specs:
            gr = int(spec.row / size_of_group)
            new_specs[gr].append(spec.shift(gr * size_of_group))

        transfers = []
        for x in range(groups):
            new_data = dict()
            new_data[INPUT_IDS] = self.data[INPUT_IDS][
                x * size_of_group : (x + 1) * size_of_group
            ]
            new_data[POSITION_IDS] = self.data[POSITION_IDS][
                x * size_of_group : (x + 1) * size_of_group
            ]
            new_data[SEGMENT_IDS] = self.data[SEGMENT_IDS][
                x * size_of_group : (x + 1) * size_of_group
            ]
            new_data[INPUT_MASK] = self.data[INPUT_MASK][
                x * size_of_group : (x + 1) * size_of_group
            ]
            transfers.append(DataTransfer(self.count, new_specs[x], new_data))
        return transfers

    def debug(self):
        return f"{self.count} {len(self.specs)}"


def insert(input_data, dl, row, col, mask, input_ids, positions, segment_ids):
    input_data[INPUT_IDS][row, col : col + dl] = input_ids[:dl]
    input_data[POSITION_IDS][row, col : col + dl] = positions
    input_data[SEGMENT_IDS][row, col : col + dl] = segment_ids[:dl]
    input_data[INPUT_MASK][row, col : col + dl] = mask * np.ones(dl, dtype=np.uint32)


def create_input_data(b, s):
    input_data = dict()
    input_data[INPUT_IDS] = np.zeros((b, s), dtype=np.uint32)
    input_data[POSITION_IDS] = np.zeros((b, s), dtype=np.uint32)
    input_data[SEGMENT_IDS] = np.zeros((b, s), dtype=np.uint32)
    input_data[INPUT_MASK] = np.ones((b, s), dtype=np.uint32)
    return input_data


def find_row_greedy(b, s, data_len, col_idx):
    for x in range(b):
        if col_idx[x] + data_len <= s:
            return True, x
    return False, x


def find_row_full(b, s, data_len, col_idx):
    found = False
    min_idx = None
    min_err = 1000
    for x in range(b):
        err = s - data_len - col_idx[x]
        if err >= 0:
            if min_idx is None or (err > 0 and err < min_err):
                min_idx = x
                min_err = err
                found = True
            if err < 32:
                break
    return found, min_idx


def pack_data(query_samples, idx, batch_size, seq_len, greedy=False):
    input_data = create_input_data(batch_size, seq_len)

    spec = []
    col_idx = [0 for x in range(batch_size)]
    mask_idx = [0 for x in range(batch_size)]

    time.time()
    while idx < len(query_samples):
        eval_features = query_samples[idx]
        data_len = int(np.count_nonzero(eval_features.input_ids))
        input_ids = np.asarray(eval_features.input_ids[:data_len], dtype=np.uint32)
        positions = np.arange(data_len, dtype=np.uint32)
        segment_ids = np.asarray(eval_features.segment_ids[:data_len], dtype=np.uint32)

        if greedy:
            found, min_idx = find_row_greedy(batch_size, seq_len, data_len, col_idx)
        else:
            found, min_idx = find_row_full(batch_size, seq_len, data_len, col_idx)

        if found or col_idx[min_idx] == 0:
            x = min_idx

            insert(
                input_data,
                data_len,
                x,
                col_idx[x],
                mask_idx[x],
                input_ids,
                np.arange(data_len),
                segment_ids,
            )
            spec.append(DataSpec(idx, x, col_idx[x], data_len))

            col_idx[x] += data_len
            idx = idx + 1
            mask_idx[x] = mask_idx[x] + 1

        if not found:
            break
    for x in range(batch_size):
        input_data[INPUT_MASK][x, col_idx[x] :] = mask_idx[x]

    total_fill = 0
    for x in range(batch_size):
        total_fill += col_idx[x]

    return DataTransfer(idx, spec, input_data), total_fill / 384 / batch_size
