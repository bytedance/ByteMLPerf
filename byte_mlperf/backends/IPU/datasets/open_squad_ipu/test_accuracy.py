# Copyright 2023 ByteDance and/or its affiliates.
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
import collections
import json
import logging
import os
import time

import numpy as np
from tqdm import tqdm

from byte_mlperf.datasets import test_accuracy
from byte_mlperf.datasets.open_squad.bert.accuracy_squad import write_predictions
from byte_mlperf.datasets.open_squad.bert.evaluate import check_accuracy

RawResult = collections.namedtuple(
    "RawResult", ["unique_id", "start_logits", "end_logits"]
)

log = logging.getLogger("TestAccuracy")


class AccuracyChecker(test_accuracy.AccuracyChecker):
    def calculate_acc(self, data_percent):
        log.info("Start to calculate accuracy...")
        results, start_diffs, end_diffs = [], [], []
        self.dataloader.pack()

        num = (
            int((data_percent / 100) * self.dataloader.input_queue.qsize())
            if data_percent
            else self.dataloader.input_queue.qsize()
        )
        item_processes = 0

        for i in tqdm(range(num)):
            pack_data = self.dataloader.get_pack_samples(i)
            pack_unique_ids = self.dataloader.get_pack_id(i)
            if "roberta" in self.configs["model"]:
                for i in range(len(pack_data["position_ids"])):
                    pack_data["position_ids"][i] += 1
            unique_ids = pack_unique_ids
            result = self.runtime_backend.predict(pack_data)
            if "distill" in self.configs["model"]:
                result = {"start_logits": result[0], "end_logits": result[1]}
            start_logits, end_logits = self.dataloader.unpack(result)
            for i, u_id in enumerate(pack_unique_ids):
                results.append(
                    RawResult(
                        unique_id=u_id,
                        start_logits=start_logits[i],
                        end_logits=end_logits[i],
                    )
                )
                start_diffs.append(start_logits[i])
                end_diffs.append(end_logits[i])

            item_processes += len(unique_ids)

        # cpu result used for diff caculating equal to total - (total % bs)
        cpu_diffs_batch_num = (
            int((data_percent / 100) * self.dataloader.get_batch_count())
            if data_percent
            else self.dataloader.get_batch_count()
        )
        diffs = self._reshape_to_cpu_result(
            cpu_diffs_batch_num * self.dataloader.cur_bs, start_diffs, end_diffs
        )

        diffs = np.array(diffs)
        diffs = diffs.flatten()
        # post process the ipu's result
        self._modify_masked_position(diffs)

        np.save(self.output_dir + "/{}.npy".format(self.dataloader.name()), diffs)
        data_file = (
            os.path.abspath(".") + "/byte_mlperf/datasets/open_squad/dev-v1.1.json"
        )
        predict_file = (
            self.output_dir[: self.output_dir.rindex("/")] + "/predictions.json"
        )
        write_predictions(
            self.dataloader.eval_examples,
            self.dataloader.eval_features,
            results,
            20,
            30,
            True,
            predict_file,
        )
        result = check_accuracy(data_file, predict_file, item_processes)
        log.info(
            "Batch size is {}, F1: {}, Exact Match:{}".format(
                self.dataloader.cur_bs, result["F1 Score"], result["Exact Match"]
            )
        )

        self._pack_performance_cal(data_percent)

        return result

    def _pack_performance_cal(self, data_percent):
        """pack mode require different performance calculating strategy."""

        for bs in self.configs["interact_info"]["batch_sizes"]:
            log.info("Start to test performance with batch size {0}...".format(bs))
            # reload backend for performance test
            self.runtime_backend.load(bs)
            self.latency_list = []
            self.tput_list = []

            # reload dataloader
            self.dataloader.cur_bs = bs
            self.dataloader.pack()
            num = (
                int((data_percent / 100) * self.dataloader.input_queue.qsize())
                if data_percent
                else self.dataloader.input_queue.qsize()
            )
            item_processes = 0

            for i in tqdm(range(num)):
                pack_data = self.dataloader.get_pack_samples(i)
                pack_unique_ids = self.dataloader.get_pack_id(i)
                if "roberta" in self.configs["model"]:
                    for i in range(len(pack_data["position_ids"])):
                        pack_data["position_ids"][i] += 1
                unique_ids = pack_unique_ids
                start = time.time()
                result = self.runtime_backend.predict(pack_data)
                end = time.time()
                start_logits, end_logits = self.dataloader.unpack(result)

                self.latency_list.append(end - start)
                self.tput_list.append(len(pack_unique_ids) / (end - start))
                item_processes += len(unique_ids)

            latency_list = self.latency_list[1:]
            tput_list = self.tput_list[1:]
            latency = round(1000 * sum(latency_list) / len(latency_list), 2)
            thoughput = round(int(sum(tput_list) / len(tput_list)), 2)
            latency_list.sort()
            tail_latency = round(1000 * latency_list[int(len(latency_list) * 0.99)], 2)
            log.info(
                "Batch size is {}, QPS: {}, Avg Latency:{}, Tail Latency:{}".format(
                    self.dataloader.cur_bs, thoughput, latency, tail_latency
                )
            )
            result = {
                "BS": self.dataloader.cur_bs,
                "QPS": thoughput,
                "AVG Latency": latency,
                "P99 Latency": tail_latency,
            }

            with open(
                self.output_dir + "/performance_bs{0}.json".format(bs), "w"
            ) as out:
                json.dump(result, out)

    def _modify_masked_position(self, flatten_ipu_data):
        # load cpu result, modify ipu's masked positions
        cpu_data_path = os.path.abspath(
            "byte_mlperf/reports/CPU/" + self.configs["model"]
        )
        if not os.path.exists(cpu_data_path):
            log.info("Fetch CPU Data Failed")
            return {}
        cpu_data = np.load(cpu_data_path + "/{}.npy".format(self.dataloader.name()))
        flatten_cpu_data = cpu_data.flatten()
        assert flatten_cpu_data.shape == flatten_ipu_data.shape

        mask_pos = flatten_ipu_data == -10
        flatten_ipu_data[mask_pos] = flatten_cpu_data[mask_pos]
        return cpu_data

    def _reshape_to_cpu_result(self, cpu_len, start_diffs, end_diffs):
        """Modify packed ipu result to the same order as cpu's squad result."""
        start, end = [], []
        reshaped_diffs = []
        assert len(start_diffs) == len(end_diffs)
        assert cpu_len <= len(
            start_diffs
        ), "for diffs comparsion, the number of ipu result should have no less than cpu result."
        for i in range(cpu_len):
            if i and len(start) == self.dataloader.cur_bs:
                reshaped_diffs.append(start + end)
                start, end = [], []
            start.append(start_diffs[i])
            end.append(end_diffs[i])
        reshaped_diffs.append(start + end)
        return reshaped_diffs
