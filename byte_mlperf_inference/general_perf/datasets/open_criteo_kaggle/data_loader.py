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
from re import T
import numpy as np
from general_perf.datasets import data_loader
from tqdm import tqdm

log = logging.getLogger("CriteoKaggle")


class DataLoader(data_loader.Dataset):
    def __init__(self, config):
        super(DataLoader, self).__init__(config)

        log.info("Initial...")
        self.config = config
        self.cur_bs = 1
        if not os.path.exists("general_perf/datasets/{}/numeric.npy".format(
                self.config['dataset_name'])):
            from general_perf.datasets.open_criteo_kaggle.preprocess_dataset import csv_to_numpy
            csv_to_numpy(
                "general_perf/datasets/{}/eval.csv".format(
                    self.config['dataset_name']),
                "general_perf/datasets/{}/".format(self.config['dataset_name']))

        num = np.load("general_perf/datasets/{}/numeric.npy".format(
            self.config['dataset_name']))
        cat = np.load("general_perf/datasets/{}/categorical.npy".format(
            self.config['dataset_name']))
        label = np.load("general_perf/datasets/{}/label.npy".format(
            self.config['dataset_name']))
        self.items = len(num)
        self.batch_num = int(self.items / self.cur_bs)
        self.feed_dict = {}
        for i in tqdm(range(cat.shape[0])):
            if i == 0:
                self.feed_dict["new_categorical_placeholder:0"] = list(
                    cat[i].reshape(-1, 2))
                self.feed_dict["new_numeric_placeholder:0"] = list(
                    num[i].reshape(1, -1))
                self.feed_dict["label"] = list(label[i])
            else:
                self.feed_dict["new_categorical_placeholder:0"].extend(
                    cat[i].reshape(-1, 2))
                self.feed_dict["new_numeric_placeholder:0"].extend(
                    num[i].reshape(1, -1))
                self.feed_dict["label"].extend(label[i])
        self.feed_dict['new_categorical_placeholder:0'] = np.array(
            self.feed_dict['new_categorical_placeholder:0'], dtype=np.int64)
        self.feed_dict['new_numeric_placeholder:0'] = np.array(
            self.feed_dict['new_numeric_placeholder:0'], dtype=np.float32)
        self.feed_dict['label'] = np.array(self.feed_dict['label'],
                                           dtype=np.int64)

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
                'new_categorical_placeholder:0':
                self.feed_dict["new_categorical_placeholder:0"][i *
                                                                self.cur_bs *
                                                                26:(i + 1) *
                                                                self.cur_bs *
                                                                26, ],
                'new_numeric_placeholder:0':
                self.feed_dict["new_numeric_placeholder:0"][
                    i * self.cur_bs:(i + 1) * self.cur_bs, ],
            }
            self.labels.append(
                self.feed_dict["label"][i * self.cur_bs:(i + 1) *
                                        self.cur_bs, ])
            self.batched_data.append(split_data)
