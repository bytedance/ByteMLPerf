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
import numpy as np
from general_perf.datasets import test_accuracy
from tqdm import tqdm
import torch

log = logging.getLogger("TestAccuracy")


class AccuracyChecker(test_accuracy.AccuracyChecker):
    def calculate_acc(self, data_percent):
        log.info("Start to calculate accuracy...")
        num = int((data_percent / 100) * self.dataloader.get_batch_count()
                  ) if data_percent else self.dataloader.get_batch_count()
        good, total = 0, 0
        diffs = []
        for i in tqdm(range(num)):
            test_data, labels = self.dataloader.get_samples(i)

            results = self.runtime_backend.predict(test_data)
            if "resnet50-tf-fp16" in self.configs["model"]:
                if 'classes' in results:
                    del results['classes']
            results = self._post_processing(results, self.configs['framework'])
            diffs.append(results)
            for j in range(len(results)):
                if np.argmax(results[j]) == labels[j]:
                    good += 1
                total += 1
        accuracy = round((good / total), 5)
        log.info('Batch size is {}, Accuracy: {}'.format(
            self.dataloader.cur_bs, accuracy))
        np.save(self.output_dir + "/{}.npy".format(self.dataloader.name()),
                diffs)
        return {"Top-1": accuracy}

    def _post_processing(self, inputs, framework):
        if framework == "Onnx":
            if isinstance(inputs, list):
                inputs = list(inputs[0])
            elif isinstance(inputs, dict):
                key = list(inputs.keys())[0]
                inputs = list(inputs[key])
        else:
            if isinstance(inputs, tuple):
                inputs = inputs[0].float().cpu().numpy().astype(float) if inputs[0].dtype==torch.bfloat16 else inputs[0].cpu().numpy().astype(float)
            else:
                inputs = inputs[list(inputs)[0]]
        if framework == "Pytorch" or framework == "Onnx":
            inputs = np.array(
                [np.insert(inputs[i], 0, 0) for i in range(len(inputs))])
        return inputs
