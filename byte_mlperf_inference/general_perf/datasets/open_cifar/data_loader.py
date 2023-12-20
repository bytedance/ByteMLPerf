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

import collections
import logging

import numpy as np
import os
import pickle
from tqdm import tqdm
from typing import Any
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from general_perf.datasets import data_loader

log = logging.getLogger("CIFAR100")

INPUT_TYPE = {
    "UINT8": np.uint8,
    "FLOAT32": np.float32,
    "LONG": np.long,
    "INT32": np.int32,
    "INT64": np.int64
}


class DataLoader(data_loader.Dataset):
    def __init__(self, config):
        super(DataLoader, self).__init__(config)
        log.info("Initial...")

        base_folder = "general_perf/datasets/{}/cifar-100-python".format(
            self.config['dataset_name'])
        test_list = [
            ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
        ]
        meta = {
            'filename': 'meta',
            'key': 'fine_label_names',
            'md5': '7973b15100ade9c7d40fb424638fde48',
        }

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in test_list:
            file_path = os.path.join(base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        transformer = _transform()
        path = os.path.join(base_folder, meta['filename'])
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[meta['key']]
        self.class_to_idx = {
            _class: i
            for i, _class in enumerate(self.classes)
        }
        self.test_data = []
        for i in tqdm(range(len(self.data))):
            img = self.data[i]
            img = Image.fromarray(img)
            img = transformer(img).detach().numpy()
            self.test_data.append(img)
        self.text_input = np.load(os.path.join(base_folder, 'text.npy'))
        self.config = config
        self.cur_bs = 1
        self.items = len(self.data)
        self.batch_num = int(self.items / self.cur_bs)

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
                'image': self.test_data[i * self.cur_bs:(i + 1) * self.cur_bs],
                'text': self.text_input,
            }
            self.labels.append(self.targets[i * self.cur_bs:(i + 1) *
                                            self.cur_bs])
            self.batched_data.append(split_data)

    def get_fake_samples(self, batch_size, shape, input_type):
        data = {}
        if input_type:
            i = 0
            for key, val in shape.items():
                if key == "image":
                    val = [val[0] * batch_size] + val[1:]
                    data[key] = np.random.random(size=val).astype(
                        INPUT_TYPE[input_type[i]])
                else:
                    data[key] = np.random.random(size=val).astype(
                        INPUT_TYPE[input_type[i]])
                i += 1
            return data
        else:
            raise ValueError("Please provide input type")


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform():
    return Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])
