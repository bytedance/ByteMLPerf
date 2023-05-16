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
import collections
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from byte_mlperf.datasets.open_squad.bert.accuracy_squad import write_predictions
from byte_mlperf.datasets.open_squad.bert.evaluate import check_accuracy
from byte_mlperf.datasets import test_accuracy

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])

log = logging.getLogger("TestAccuracy")


class AccuracyChecker(test_accuracy.AccuracyChecker):
    def calculate_acc(self, data_percent):
        log.info("Start to calculate accuracy...")
        results, diffs = [], []
        num = int((data_percent / 100) * self.dataloader.get_batch_count()
                  ) if data_percent else self.dataloader.get_batch_count()
        for i in tqdm(range(num)):
            test_data, _ = self.dataloader.get_samples(i)
            unique_ids = self.dataloader.get_id(i)
            result = self.runtime_backend.predict(test_data)
            start_logits, end_logits = self._post_processing(
                result, self.configs['framework'])

            for i, u_id in enumerate(unique_ids):
                results.append(
                    RawResult(unique_id=u_id,
                              start_logits=start_logits[i],
                              end_logits=end_logits[i]))

            diffs.append(start_logits + end_logits)
        np.save(self.output_dir + "/{}.npy".format(self.dataloader.name()),
                diffs)
        data_file = os.path.abspath('.') + "/byte_mlperf/datasets/open_squad/dev-v1.1.json"
        predict_file = self.output_dir[:self.output_dir.
                                       rindex('/')] + "/predictions.json"
        write_predictions(self.dataloader.eval_examples,
                          self.dataloader.eval_features, results, 20, 30, True,
                          predict_file)
        result = check_accuracy(data_file, predict_file,
                                num * self.dataloader.cur_bs)
        log.info('Batch size is {}, F1: {}, Exact Match:{}'.format(
            self.dataloader.cur_bs, result['F1 Score'], result['Exact Match']))
        return result

    def _post_processing(self, inputs, framework):
        start_results, end_results = [], []

        if framework == "Tensorflow":
            if 'distill' in self.configs['model']:
                (start_logits, end_logits) = (inputs["output_0"],
                                              inputs["output_1"])
                for i in range(self.dataloader.cur_bs):
                    start_logit = [float(x) for x in start_logits[i].flat]
                    end_logit = [float(x) for x in end_logits[i].flat]
                    start_results.append(start_logit)
                    end_results.append(end_logit)
            else:
                tensor_name = list(inputs)[0]
                for i in range(len(inputs[tensor_name])):
                    logits = tf.transpose(np.array([inputs[tensor_name][i]]),
                                          [2, 0, 1])
                    unstacked_logits = tf.unstack(logits, axis=0)
                    if tf.executing_eagerly():
                        (start_logit,
                         end_logit) = (unstacked_logits[0].numpy(),
                                       unstacked_logits[1].numpy())
                    else:
                        with tf.compat.v1.Session():
                            (start_logit,
                             end_logit) = (unstacked_logits[0].eval(),
                                           unstacked_logits[1].eval())
                    start_logit = [float(x) for x in start_logit.flat]
                    end_logit = [float(x) for x in end_logit.flat]
                    start_results.append(start_logit)
                    end_results.append(end_logit)
        else:
            # if 'albert' in self.configs['model']:
            #     (start_logits, end_logits) = (inputs[0][0].cpu().detach().numpy(), inputs[1][0].cpu().detach().numpy())
            # else:
            if 'bert-torch-fp32' or 'roberta' in self.configs['model']:
                (start_logits, end_logits) = (inputs[0].cpu().detach().numpy(),
                                              inputs[1].cpu().detach().numpy())
            else:
                if isinstance(inputs, dict):
                    (start_logits, end_logits) = (inputs["start_logits"],
                                                  inputs["end_logits"])
                else:
                    (start_logits, end_logits) = (inputs[0], inputs[1])
            for i in range(self.dataloader.cur_bs):
                start_logit = [float(x) for x in start_logits[i].flat]
                end_logit = [float(x) for x in end_logits[i].flat]
                start_results.append(start_logit)
                end_results.append(end_logit)

        return start_results, end_results
