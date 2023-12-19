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

from tqdm import tqdm
import json
import collections
import numpy as np
from bert4keras.tokenizers import Tokenizer
import jieba
jieba.initialize()

test_data = []
with open("test.json", encoding='utf-8') as f:
    for l in f:
        l = json.loads(l)
        assert l['label'] in 'BC'
        if l['label'] == 'B':
            test_data.append((l['A'], l['B'], l['C']))
        else:
            test_data.append((l['A'], l['C'], l['B']))

tokenizer = Tokenizer("vocab.txt",
                      do_lower_case=True,
                      pre_tokenize=lambda s: jieba.cut(s, HMM=False))

feed_dict = collections.defaultdict(list)
maxlen = 1024
for i in tqdm(range(len(test_data))):
    (text1, text2, text3) = test_data[i]
    token_ids, segment_ids = tokenizer.encode(text1, text2, maxlen=maxlen)
    feed_dict["batch_token_ids"].append(token_ids)
    feed_dict["batch_segment_ids"].append(segment_ids)
    feed_dict["label"].append([1])
    token_ids, segment_ids = tokenizer.encode(text1, text3, maxlen=maxlen)
    feed_dict["batch_token_ids"].append(token_ids)
    feed_dict["batch_segment_ids"].append(segment_ids)
    feed_dict["label"].append([0])

np.save("{}.npy".format('batch_token_ids'),
        feed_dict["batch_token_ids"],
        allow_pickle=True)
np.save("{}.npy".format('batch_segment_ids'),
        feed_dict["batch_segment_ids"],
        allow_pickle=True)
np.save("{}.npy".format('label'), feed_dict["label"], allow_pickle=True)
