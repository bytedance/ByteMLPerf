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
from os.path import split
import re
import time

import cv2
import numpy as np
import random
from tqdm import tqdm

from general_perf.datasets import data_loader

log = logging.getLogger("Imagenet")

INPUT_TYPE = {
    "UINT8": np.uint8,
    "FLOAT32": np.float32,
}


class DataLoader(data_loader.Dataset):
    def __init__(self, config):
        super(DataLoader, self).__init__(config)
        log.info("Initial...")

        self.config = config
        self.cur_bs = 1
        self.image_size = [224, 224, 3]

        if self.config['framework'] == 'Tensorflow':
            image_format = "NHWC"
            pre_process = pre_process_vgg
        else:
            image_format = "NCHW"
            if 'resnet50' in self.config['model']:
                pre_process = pre_process_imagenet_pytorch
            else:
                pre_process = pre_process_imagenet_vit

        cache_dir = os.getcwd() + \
            "/general_perf/datasets/{}".format(self.config['dataset_name'])
        self.input_name = self.config['inputs']
        self.image_list = []
        self.label_list = []
        self.count = None
        self.use_cache = 0
        self.cache_dir = os.path.join(cache_dir, "preprocessed",
                                      self.config['model'])
        self.data_path = "general_perf/datasets/{}/ILSVRC2012_img_val".format(
            self.config['dataset_name'])
        self.pre_process = pre_process
        self.items = 0
        # input images are in HWC
        self.need_transpose = True if image_format == "NCHW" else False
        not_found = 0
        os.makedirs(self.cache_dir, exist_ok=True)

        image_list = 'general_perf/datasets/{}/val_map.txt'.format(
            self.config['dataset_name'])

        start = time.time()
        with open(image_list, 'r') as f:
            for s in tqdm(f):
                image_name, label = re.split(r"\s+", s.strip())
                src = os.path.join(self.data_path, image_name)
                if not os.path.exists(src):
                    # if the image does not exists ignore it
                    not_found += 1
                    continue
                os.makedirs(os.path.dirname(
                    os.path.join(self.cache_dir, image_name)),
                            exist_ok=True)
                dst = os.path.join(self.cache_dir, image_name)
                if not os.path.exists(dst + ".npy"):
                    img_org = cv2.imread(src)
                    processed = self.pre_process(
                        img_org,
                        need_transpose=self.need_transpose,
                        dims=self.image_size)
                    np.save(dst, processed)

                self.image_list.append(image_name)
                self.label_list.append(int(label) + 1)
                self.items = len(self.image_list)

                # limit the dataset if requested
                if self.count and len(self.image_list) >= self.count:
                    break

        time_taken = time.time() - start
        if not self.image_list:
            log.error("no images in image list found")
            raise ValueError("no images in image list found")
        if not_found > 0:
            log.info("reduced image list, %d images not found", not_found)

        log.info("loaded {} images, cache={}, took={:.1f}sec".format(
            len(self.image_list), self.use_cache, time_taken))

        self.label_list = np.array(self.label_list)
        self.batch_num = int(self.items / self.cur_bs)
        self.shuffle_index = [i for i in range(self.items)]
        random.seed(7)
        random.shuffle(self.shuffle_index)

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
            split_data, labels = [], []
            for j in range(i * self.cur_bs, (i + 1) * self.cur_bs):
                output, label = self.get_item(self.shuffle_index[j])
                split_data.append(output)
                labels.append(label)

            self.labels.append(labels)
            self.batched_data.append({self.input_name: np.array(split_data)})

    def get_samples(self, sample_id):
        if sample_id >= len(self.batched_data) or sample_id < 0:
            raise ValueError("Your Input ID: {} is out of range: {}".format(
                sample_id, len(self.batched_data)))
        return self.batched_data[sample_id], self.labels[sample_id]

    def get_item(self, nr):
        """Get image by number in the list."""
        dst = os.path.join(self.cache_dir, self.image_list[nr])
        img = np.load(dst + ".npy")
        return img, self.label_list[nr]


#
# pre-processing
#
def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img


def resize_with_aspectratio(img,
                            out_height,
                            out_width,
                            scale=87.5,
                            inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img


def pre_process_vgg(img, dims=None, need_transpose=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    output_height, output_width, _ = dims
    cv2_interpol = cv2.INTER_AREA
    img = resize_with_aspectratio(img,
                                  output_height,
                                  output_width,
                                  inter_pol=cv2_interpol)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')

    # normalize image
    means = np.array([123.68, 116.78, 103.94], dtype=np.float32)
    img -= means

    # transpose if needed
    if need_transpose:
        img = img.transpose([2, 0, 1])
    return img


def pre_process_imagenet_pytorch(img, dims=None, need_transpose=False):
    from PIL import Image
    import torchvision.transforms.functional as F
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = F.resize(img, 256, Image.BILINEAR)
    img = F.center_crop(img, 224)
    img = F.to_tensor(img)
    img = F.normalize(img,
                      mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225],
                      inplace=False)
    if not need_transpose:
        img = img.permute(1, 2, 0)  # NHWC
    img = np.asarray(img, dtype='float32')
    return img

def pre_process_imagenet_vit(img, dims=None, need_transpose=False):
    from PIL import Image
    import torchvision.transforms.functional as F
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = F.resize(img, 256, Image.BILINEAR)
    img = F.center_crop(img, 384)
    img = F.to_tensor(img)
    img = F.normalize(img,
                      mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225],
                      inplace=False)
    if not need_transpose:
        img = img.permute(1, 2, 0)  # NHWC
    img = np.asarray(img, dtype='float32')
    return img


def maybe_resize(img, dims):
    img = np.array(img, dtype=np.float32)
    if len(img.shape) < 3 or img.shape[2] != 3:
        # some images might be grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if dims != None:
        im_height, im_width, _ = dims
        img = cv2.resize(img, (im_width, im_height),
                         interpolation=cv2.INTER_LINEAR)
    return img
