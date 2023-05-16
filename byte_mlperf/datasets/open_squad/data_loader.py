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
# To support feature cache.
import pickle
from transformers import BertTokenizer, AutoTokenizer
from byte_mlperf.datasets.open_squad.create_squad_data import read_squad_examples, convert_examples_to_features
import collections
from byte_mlperf.datasets import data_loader
import logging
from tqdm import tqdm
import numpy as np

INPUT_TYPE = {
    "UINT8": np.uint8,
    "FLOAT32": np.float32,
    "LONG": np.long,
    "INT32": np.int32,
    "INT64": np.int64
}

max_seq_length = 384
max_query_length = 64
doc_stride = 128

log = logging.getLogger("SQUAD")


class DataLoader(data_loader.Dataset):
    def __init__(self, config):
        super(DataLoader, self).__init__(config)

        log.info("Initial...")
        self.config = config
        model = self.config["model"]
        total_count_override = None
        perf_count_override = None
        eval_features = []
        # Load features if cached, convert from examples otherwise.
        input_file = "byte_mlperf/datasets/open_squad/dev-v1.1.json"
        cache_path = 'byte_mlperf/datasets/open_squad/eval_features_' + self.config[
            'model'] + '.pickle'
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as cache_file:
                eval_features = pickle.load(cache_file)
            eval_examples = read_squad_examples(input_file=input_file,
                                                is_training=False,
                                                version_2_with_negative=False)
        else:
            log.info("Start to generate data")
            if "roberta" in self.config['model']:
                tokenizer = AutoTokenizer.from_pretrained(
                    "csarron/roberta-base-squad-v1")
            elif "albert" in self.config['model']:
                tokenizer = AutoTokenizer.from_pretrained(
                    "madlag/albert-base-v2-squad")
            else:
                tokenizer = BertTokenizer(
                    "byte_mlperf/datasets/open_squad/vocab.txt")
            eval_examples = read_squad_examples(input_file=input_file,
                                                is_training=False,
                                                version_2_with_negative=False)

            def append_feature(feature):
                eval_features.append(feature)

            convert_examples_to_features(examples=eval_examples,
                                         tokenizer=tokenizer,
                                         max_seq_length=max_seq_length,
                                         doc_stride=doc_stride,
                                         max_query_length=max_query_length,
                                         is_training=False,
                                         output_fn=append_feature,
                                         verbose_logging=False)

            with open(cache_path, 'wb') as cache_file:
                pickle.dump(eval_features, cache_file)

        self.eval_features = eval_features
        self.eval_examples = eval_examples
        self.count = total_count_override or len(self.eval_features)
        self.items = len(self.eval_features)
        self.perf_count = perf_count_override or self.count
        self.model = model
        self.cur_bs = 1
        self.batch_num = int(self.items / self.cur_bs)

    def name(self):
        return self.config['dataset_name']

    def preprocess(self):
        log.info("Preprocessing...")

        self.rebatch(self.batch_num, skip=False)

    def rebatch(self, new_bs, skip=True):
        log.info("Rebatching batch size to: {} ...".format(new_bs))

        if self.cur_bs == new_bs and skip:
            return

        self.cur_bs = new_bs
        self.batch_num = int(self.items / self.cur_bs)
        self.batched_data = []
        for i in tqdm(range(self.batch_num)):
            features = collections.defaultdict(list)
            for j in range(i * self.cur_bs, (i + 1) * self.cur_bs):
                if "roberta" in self.model:
                    features['input_ids.1'].append(
                        self.eval_features[j].input_ids)
                    features['attention_mask.1'].append(
                        self.eval_features[j].input_mask)
                    features['token_type_ids.1'].append(
                        np.zeros((384,)))
                elif "torch" in self.model:
                    features['input_ids.1'].append(
                        self.eval_features[j].input_ids)
                    features['attention_mask.1'].append(
                        self.eval_features[j].input_mask)
                    features['token_type_ids.1'].append(
                        self.eval_features[j].segment_ids)
                else:
                    features['input_ids:0'].append(
                        self.eval_features[j].input_ids)
                    features['input_mask:0'].append(
                        self.eval_features[j].input_mask)
                    features['segment_ids:0'].append(
                        self.eval_features[j].segment_ids)
            self.batched_data.append(features)

    def get_samples(self, sample_id):
        if sample_id >= len(self.batched_data) or sample_id < 0:
            raise ValueError("Your Input ID is out of range")
        return self.batched_data[sample_id], []

    def get_id(self, sample_id):
        if sample_id >= len(self.batched_data) or sample_id < 0:
            raise ValueError("Your Input ID is out of range")
        return [
            self.eval_features[i].unique_id
            for i in range(sample_id * self.cur_bs, (sample_id + 1) *
                           self.cur_bs)
        ]

    def get_fake_samples(self, batch_size, shape, input_type):
        data = {}

        avg_seq_len = 192
        max_seq_len = 384

        if input_type:
            i = 0
            for key, val in shape.items():
                val = [val[0] * batch_size] + val[1:]
                if i == 0:
                    # fake input id and mask
                    input_ids = np.zeros(val).astype(INPUT_TYPE[input_type[i]])
                    data[key] = input_ids
                elif i == 1:
                    # fake input array length
                    input_len = np.random.randint(low=2 * avg_seq_len -
                                                  max_seq_len,
                                                  high=max_seq_len + 1,
                                                  size=(batch_size),
                                                  dtype=np.int32)

                    input_mask = np.zeros(val).astype(
                        INPUT_TYPE[input_type[i]])

                    for b_idx, s_len in enumerate(input_len):
                        input_mask[b_idx][:s_len] = 1
                    data[key] = input_mask
                else:
                    data[key] = np.zeros(val).astype(INPUT_TYPE[input_type[i]])
                i += 1
            return data
        else:
            raise ValueError("Please provide input type")
