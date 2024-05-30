# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.


import os.path as ospath

from typing import NamedTuple, Union, List, Mapping

from dltest.log_parser import DEFAULT_NEAREST_MATCH_CHARS


class LogComparatorArgs(NamedTuple):
    threshold: Union[float, Mapping]
    patterns: List[str] = None
    pattern_names: List[str] = None
    use_re: bool = False
    nearest_distance: int = DEFAULT_NEAREST_MATCH_CHARS
    start_line_pattern_flag: str = None
    end_line_pattern_flag: str = None
    split_pattern: Union[str, List] = None
    split_sep: List = None
    split_idx: List = None
    only_last: bool = True
    allow_greater_than: bool = True

    def to_dict(self):
        return self._asdict()


class ArgsModelsTuple(NamedTuple):

    args: LogComparatorArgs
    models: List[str]


class BaseConfig:

    def __getitem__(self, item):
        return self.__class__.__dict__[item]

    def __getattr__(self, item):
        return self.__class__.__dict__[item]

    def __iter__(self):
        for attr, value in self.__class__.__dict__.items():
            if isinstance(value, ArgsModelsTuple):
                yield attr

    def iter_items(self):
        for attr, value in self.__class__.__dict__.items():
            if isinstance(value, ArgsModelsTuple):
                yield attr, value


class _TFComparatorConfig(BaseConfig):

    cnn_benchmarks = ArgsModelsTuple(
        args=LogComparatorArgs(
            threshold=0.08,
            patterns=["Accuracy @ 1 =", "Accuracy @ 5 ="],
            pattern_names=["Acc@1", "Acc@5"]
        ),
        models=["alexnet", "inceptionv3", "resnet50", "resnet101", "vgg16"]
    )

    dist_cnn_becnmarks = ArgsModelsTuple(
        args=LogComparatorArgs(
            threshold=0.08,
            split_sep=[' ', ' '],
            split_idx=[9, 10],
            split_pattern="[\s\S]*?images/sec:[\s\S]*?jitter",
            pattern_names=['Acc@1', 'Acc@5']
        ),
        models=[
            "alexnet_dist", "inceptionv3_dist", "resnet50_dist", "resnet101_dist", "vgg16_dist"
        ]
    )

    bert = ArgsModelsTuple(
        args=LogComparatorArgs(
            threshold=0.08,
            patterns=["eval_accuracy ="],
            pattern_names=["Accuracy"]
        ),
        models=["bert"]
    )

    ssd = ArgsModelsTuple(
        args=LogComparatorArgs(
            threshold=0.08,
            patterns=["acc="],
            pattern_names=["Acc@1"]
        ),
        models=["ssd"]
    )

    yolov3 = ArgsModelsTuple(
        args=LogComparatorArgs(
            threshold=0.8,
            patterns=["mAP"]
        ),
        models=["yolov3"]
    )

    vnet = ArgsModelsTuple(
        args=LogComparatorArgs(
            threshold=0.08,
            patterns=["background_dice", "anterior_dice", "posterior_dice"]
        ),
        models=["vnet"]
    )


class _TorchComparatorConfig(BaseConfig):
    classification = ArgsModelsTuple(
        args=LogComparatorArgs(
            threshold=8.0, patterns=['Acc@1', 'Acc@5'],
            start_line_pattern_flag="Start training",
        ),
        models=[
            'googlenet', 'inceptionv3', 'mobilenetv3', 'resnet', 'shufflenetv2',
            'vgg', 'resnet50_dali', 'resnext', 'densenet'
        ]
    )

    detection = ArgsModelsTuple(
        args=LogComparatorArgs(
            threshold=0.03,
            patterns=[
                "Average Precision  \(AP\) @\[ IoU=0.50:0.95 \| area=   all \| maxDets=100 \] ="
            ],
            pattern_names=["mAP"],
            start_line_pattern_flag="IoU metric: bbox",
            end_line_pattern_flag="IoU metric: segm"
        ),
        models=[
            'maskrcnn', 'retinanet', 'ssd'
        ]
    )

    bert_cola = ArgsModelsTuple(
        args=LogComparatorArgs(
            threshold=0.08,
            patterns=['mcc']
        ),
        models=['bert_cola']
    )

    bert_mrpc = ArgsModelsTuple(
        args=LogComparatorArgs(
            threshold=0.08,
            patterns=['acc']
        ),
        models=['bert_mrpc']
    )

    bert_pretrain_apex = ArgsModelsTuple(
        args=LogComparatorArgs(
            threshold=0.08,
            patterns=['eval_mlm_accaracy']
        ),
        models=['bert_pretrain_apex']
    )

    segmentation = ArgsModelsTuple(
        args=LogComparatorArgs(
            threshold=8.0,
            patterns=['mean IoU:'],
            pattern_names=['mIoU']
        ),
        models=[
            'deeplabv3', 'fcn'
        ]
    )

    t5 = ArgsModelsTuple(
        args=LogComparatorArgs(
            threshold=5.0,
            split_pattern="eval_bleu[\s\S]*?=",
            split_sep=["="],
            split_idx=[1],
            pattern_names=['EvalBleu']
        ),
        models=['t5']
    )

    yolov3 = ArgsModelsTuple(
        args=LogComparatorArgs(
            threshold=0.08,
            patterns=["mAP"]
        ),
        models=['yolov3']
    )

    yolov5 = ArgsModelsTuple(
        args=LogComparatorArgs(
            threshold=0.08,
            patterns=[
                "Average Precision  \(AP\) @\[ IoU=0.50:0.95 \| area=   all \| maxDets=100 \] ="
            ],
            pattern_names=["mAP"],
        ),
        models=['yolov5'],
    )

    yolov5s_coco128 = ArgsModelsTuple(
        args=LogComparatorArgs(
            threshold=0.08,
            split_pattern="[\s]+?all[\s\S]*?[1-9]\d*[\s]+?[1-9]\d*",
            split_sep=[" ", " "],
            split_idx=[5, 6],
            pattern_names=["AP50", "mAP"]
        ),
        models=['yolov5s_coco128']
    )
    
    centernet_resnet18 = ArgsModelsTuple(
        args=LogComparatorArgs(
            threshold=0.08,
            split_pattern="[\s]+?all[\s\S]*?[1-9]\d*[\s]+?[1-9]\d*",
            split_sep=[" ", " "],
            split_idx=[5, 6],
            pattern_names=["AP50", "mAP"]
        ),
        models=['centernet_resnet18']
    )
    
    fcos_resnet50_fpn = ArgsModelsTuple(
        args=LogComparatorArgs(
            threshold=0.08,
            split_pattern="[\s]+?all[\s\S]*?[1-9]\d*[\s]+?[1-9]\d*",
            split_sep=[" ", " "],
            split_idx=[5, 6],
            pattern_names=["AP50", "mAP"]
        ),
        models=['fcos_resnet50_fpn']
    )

    ocr_recognition = ArgsModelsTuple(
        args=LogComparatorArgs(
            threshold=0.5,  patterns=["0_word_acc"],
        ),
        models=[
            "sar", "satrn"
        ]
    )



class ComparatorConfig:

    _configs = dict(tf=_TFComparatorConfig(), torch=_TorchComparatorConfig())

    @classmethod
    def get_frameworks(cls) -> List:
        return list(cls._configs.keys())

    @classmethod
    def get(cls, tf_or_torch, name, default=None):
        for model_kind, comb in cls._configs[tf_or_torch].iter_items():
            if name in comb.models:
                return comb.args
        if default is not None:
            return default
        raise KeyError("Not found config, but got {name} for {fw}".format(name=name, fw=tf_or_torch))

    @classmethod
    def find_config(cls, script_path: str) -> LogComparatorArgs:
        tf_or_torch = script_path.split('.')[-2].split('_')[-1]

        # Find by the name of script
        script_name = ospath.basename(script_path).rsplit('.', maxsplit=1)[0]
        if script_name.startswith('train_'):
            script_name = script_name.replace("train_", "", 1)
        while script_name not in [None, "", "/", "\\"]:
            try:
                config = cls.get(tf_or_torch, script_name)
                return config
            except:
                pass
            script_name = script_name.rsplit('_', maxsplit=1)
            if len(script_name) <= 1:
                break
            script_name = script_name[0]

        # Find by the name of model's dir
        model_dir_name = ospath.basename(ospath.dirname(script_path))
        try:
            config = cls.get(tf_or_torch, model_dir_name)
            return config
        except:
            raise RuntimeError("Not found for", script_path)


def get_compare_config_with_full_path(script_path: str, to_dict=True):
    config = ComparatorConfig.find_config(script_path)
    if to_dict:
        return config.to_dict()
    return config

