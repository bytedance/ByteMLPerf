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

import logging
import os
import numpy as np
from general_perf.datasets import data_loader
from tqdm import tqdm
import collections

log = logging.getLogger("CAIL2019")

maxlen = 1024


class DataLoader(data_loader.Dataset):
    def __init__(self, config):
        super(DataLoader, self).__init__(config)

        log.info("Initial...")
        self.config = config
        self.cur_bs = 2

        batch_token_ids = np.load(
            "general_perf/datasets/{}/batch_token_ids.npy".format(
                self.config['dataset_name']),
            allow_pickle=True)
        batch_segment_ids = np.load(
            "general_perf/datasets/{}/batch_segment_ids.npy".format(
                self.config['dataset_name']),
            allow_pickle=True)
        labels = np.load("general_perf/datasets/{}/label.npy".format(
            self.config['dataset_name']),
                         allow_pickle=True)
        self.feed_dict = collections.defaultdict(list)
        self.feed_dict['batch_token_ids'] = batch_token_ids.tolist()
        self.feed_dict['batch_segment_ids'] = batch_segment_ids.tolist()
        self.feed_dict['label'] = labels.tolist()

        self.items = len(self.feed_dict['label'])
        self.batch_num = int(self.items / self.cur_bs)

        for i in range(self.items):
            batch_token_id = np.pad(
                self.feed_dict['batch_token_ids'][i],
                (0, 1024 - len(self.feed_dict['batch_token_ids'][i])),
                'constant').astype(np.float32)
            batch_segment_id = np.pad(
                self.feed_dict['batch_segment_ids'][i],
                (0, 1024 - len(self.feed_dict['batch_segment_ids'][i])),
                'constant').astype(np.float32)
            self.feed_dict['batch_token_ids'][i] = batch_token_id.tolist()
            self.feed_dict['batch_segment_ids'][i] = batch_segment_id.tolist()

    def name(self):
        return self.config['dataset_name']

    def preprocess(self):
        log.info("Preprocessing...")

        self.rebatch(self.cur_bs, skip=False)

    def rebatch(self, new_bs, skip=True):
        log.info("Rebatching batch size to: {} ...".format(new_bs))

        if self.cur_bs == new_bs and skip:
            return

        self.cur_bs = new_bs
        self.batch_num = int(self.items / self.cur_bs)
        self.batched_data = []
        self.labels = []
        for i in tqdm(range(self.batch_num)):
            split_data = {
                'input_segment:0':
                self.feed_dict["batch_segment_ids"][i * self.cur_bs:(i + 1) *
                                                    self.cur_bs],
                'input_token:0':
                self.feed_dict["batch_token_ids"][i * self.cur_bs:(i + 1) *
                                                  self.cur_bs],
            }
            self.labels.append(
                self.feed_dict["label"][i * self.cur_bs:(i + 1) * self.cur_bs])
            self.batched_data.append(split_data)
