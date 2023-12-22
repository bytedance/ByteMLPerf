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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pandas
import argparse
import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        type=str,
                        default="eval.csv",
                        help='full path of data file e.g. eval.csv',
                        dest='evaldatafile_path',
                        required=True)

    args = parser.parse_args()
    return args


def version_is_less_than(a, b):
    a_parts = a.split('.')
    b_parts = b.split('.')

    for i in range(len(a_parts)):
        if int(a_parts[i]) < int(b_parts[i]):
            print('{} < {}, version_is_less_than() returning False'.format(
                a_parts[i], b_parts[i]))
            return True
    return False


def csv_to_numpy(eval_csv_file, output):
    print("TensorFlow version {}".format(tf.__version__))
    required_tf_version = '2.0.0'

    if version_is_less_than(tf.__version__, required_tf_version):
        tf.compat.v1.enable_eager_execution()

    # args = parse_args()
    # eval_csv_file = args.evaldatafile_path

    csv = pandas.read_csv(eval_csv_file, header=None)
    if len(csv.columns) == 39:
        dataset_type = 'test'
    else:
        dataset_type = 'eval'

    fill_na_dict = {}
    if dataset_type == 'test':
        for i in range(0, 13):
            fill_na_dict[i] = 0.0
        for i in range(13, 39):
            fill_na_dict[i] = ""
    else:
        for i in range(1, 14):
            fill_na_dict[i] = 0.0
        for i in range(14, 40):
            fill_na_dict[i] = ""

    csv = csv.fillna(value=fill_na_dict).values

    LABEL_COLUMN = ["clicked"]
    CATEGORICAL_COLUMNS1 = ["C" + str(i) + "_embedding" for i in range(1, 27)]
    NUMERIC_COLUMNS1 = ["I" + str(i) for i in range(1, 14)]
    CATEGORICAL_COLUMNS2 = ["C" + str(i) + "_embedding" for i in range(1, 27)]
    NUMERIC_COLUMNS2 = ["I" + str(i) for i in range(1, 14)]

    DATA_COLUMNS = LABEL_COLUMN + NUMERIC_COLUMNS1 + CATEGORICAL_COLUMNS1

    CATEGORICAL_COLUMNS1.sort()
    NUMERIC_COLUMNS1.sort()

    with open(eval_csv_file, 'r') as f:
        nums = [line.strip('\n\r').split(',') for line in f.readlines()]
        numpy_arr = np.array(nums)
        numpy_arr[numpy_arr == ''] = '0'
        min_list, max_list, range_list = [], [], []

        for i in range(len(DATA_COLUMNS)):
            if DATA_COLUMNS[i] in NUMERIC_COLUMNS1:
                col_min = numpy_arr[:, i].astype(np.float32).min()
                col_max = numpy_arr[:, i].astype(np.float32).max()
                min_list.append(col_min)
                max_list.append(col_max)
                range_list.append(col_max - col_min)

        print('min list', min_list)
        print('max list', max_list)
        print('range list', range_list)

    all_data = []
    no_of_rows = 0
    for row in csv:
        no_of_rows = no_of_rows + 1
        unnormalized_vals = np.array(row[1:14])
        normalized_vals = (unnormalized_vals - min_list) / range_list
        new_categorical_dict = dict(zip(CATEGORICAL_COLUMNS2, row[14:40]))

        new_categorical_list = []
        for i in CATEGORICAL_COLUMNS1:
            if pandas.isnull(new_categorical_dict[i]):
                new_categorical_list.append("")
            else:
                new_categorical_list.append(new_categorical_dict[i])

        if tf.executing_eagerly():
            hash_values = tf.strings.to_hash_bucket_fast(
                new_categorical_list, 1000).numpy()
        else:
            hash_tensor = tf.strings.to_hash_bucket_fast(
                new_categorical_list, 1000)
            with tf.compat.v1.Session() as sess:
                hash_values = hash_tensor.eval()

        new_numerical_dict = dict(zip(NUMERIC_COLUMNS2, normalized_vals))

        item_data = {
            "new_numeric_placeholder": [],
            "new_categorical_placeholder": [],
            "label": []
        }

        for i in NUMERIC_COLUMNS1:
            item_data["new_numeric_placeholder"].extend(
                [new_numerical_dict[i]])

        for i in range(0, 26):
            item_data["new_categorical_placeholder"].extend([i])
            item_data["new_categorical_placeholder"].extend([hash_values[i]])

        item_data["label"].append(row[0])

        all_data.append(item_data)

    wnd_num = []
    wnd_cate = []
    wnd_lable = []

    for data in all_data:
        wnd_num.append(data["new_numeric_placeholder"])
        wnd_cate.append(data["new_categorical_placeholder"])
        wnd_lable.append(data["label"])

    np.save(os.path.join(output, "numeric.npy"), np.array(wnd_num))
    np.save(os.path.join(output, "categorical.npy"), np.array(wnd_cate))
    np.save(os.path.join(output, "label.npy"), np.array(wnd_lable))

    print('Total number of rows ', no_of_rows)
    print(
        'Generated output file name : wnd_num.npy, wnd_cate.npy, wnd_label.npy'
    )


if __name__ == "__main__":
    csv_to_numpy()
