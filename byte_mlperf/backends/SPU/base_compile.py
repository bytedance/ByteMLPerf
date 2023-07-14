from torch.utils.data import DataLoader as DataLoaderX
from dataset.dataset import ImageNetDataset,MZJBertDataset,DummyDataset
from nn_compiler.common.constants import OpType
from common_compile import SparsertBaseBuilder

class Resnet50Builder(SparsertBaseBuilder):
    def __init__(self, onnx_path, dump_dir, dataset_dir, dataset_cfg, dtype, batch_size, verify, **kwargs):
        super(Resnet50Builder, self).__init__(
            onnx_path, dump_dir, dataset_dir, dataset_cfg, dtype, batch_size, verify, **kwargs)

    def set_dataset_config(self):
        # calibration dataset config
        dataset = ImageNetDataset(self.dataset_dir, transform_file=self.dataset_cfg)
        self.config.dataloader = DataLoaderX(dataset, batch_size=self.batch_size)
        self.config.calib_batch = 1

        # model inputs info
        self.config.input_dict = {"actual_input": "img"}
        self.input_shape_dict = {"actual_input": (self.batch_size, 3, 224, 224)}

        # you can also set other configs here
        self.config.do_kl = True
        self.config.opt_level = 8
        self.config.total_cores = 1


class BertBaseBuilder(SparsertBaseBuilder):
    def __init__(self, onnx_path, dump_dir, dataset_dir, dataset_cfg, dtype, batch_size, verify, **kwargs):
        super(BertBaseBuilder, self).__init__(
            onnx_path, dump_dir, dataset_dir, dataset_cfg, dtype, batch_size, verify, **kwargs)

    def set_dataset_config(self):
        # model inputs info
        self.config.input_dict = {
            "input_ids": "input_ids",
            "attention_mask": "attention_mask",
            "token_type_ids": "token_type_ids"
        }
        self.input_shape_dict = {
            "input_ids": (self.batch_size, 384),
            "attention_mask": (self.batch_size, 384),
            "token_type_ids": (self.batch_size, 384)
        }

        # calibration dataset config
        dataset = MZJBertDataset(data_path=self.dataset_dir, input_info=self.config.input_dict)
        self.config.dataloader = DataLoaderX(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.config.calib_batch = 1

        # you can also set other configs here
        self.config.do_kl = False
        self.config.opt_level = 5
        self.config.safe_exp = False
        self.config.quantized_patterns = [[OpType.BatchMatmul]]


class AlbertBuilder(SparsertBaseBuilder):
    def __init__(self, onnx_path, dump_dir, dataset_dir, dataset_cfg, dtype, batch_size, verify, **kwargs):
        super(AlbertBuilder, self).__init__(
            onnx_path, dump_dir, dataset_dir, dataset_cfg, dtype, batch_size, verify, **kwargs)

    def set_dataset_config(self):
        # model inputs info
        self.config.input_dict = {
            "input_ids": "input_ids",
            "attention_mask": "attention_mask",
            "token_type_ids": "token_type_ids"
        }
        self.input_shape_dict = {
            "input_ids": (self.batch_size, 384),
            "attention_mask": (self.batch_size, 384),
            "token_type_ids": (self.batch_size, 384)
        }

        # calibration dataset config
        dataset = MZJBertDataset(data_path=self.dataset_dir, input_info=self.config.input_dict)
        self.config.dataloader = DataLoaderX(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.config.calib_batch = 1

        # you can also set other configs here
        self.config.do_kl = False
        self.config.opt_level = 5
        self.config.safe_exp = False
        self.config.quantized_patterns = [[OpType.BatchMatmul]]


class RobertaBuilder(SparsertBaseBuilder):
    def __init__(self, onnx_path, dump_dir, dataset_dir, dataset_cfg, dtype, batch_size, verify, **kwargs):
        super(RobertaBuilder, self).__init__(
            onnx_path, dump_dir, dataset_dir, dataset_cfg, dtype, batch_size, verify, **kwargs)

    def set_dataset_config(self):
        # model inputs info
        self.config.input_dict = {
            "input_ids": "input_ids",
            "attention_mask": "attention_mask",
            "position_ids": "position_ids"
        }
        self.input_shape_dict = {
            "input_ids": (self.batch_size, 384),
            "attention_mask": (self.batch_size, 384),
            "position_ids": (self.batch_size, 384)
        }

        # calibration dataset config
        dataset = MZJBertDataset(data_path=self.dataset_dir, input_info=self.config.input_dict)
        self.config.dataloader = DataLoaderX(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.config.calib_batch = 1

        # you can also set other configs here
        self.config.do_kl = False
        self.config.opt_level = 5
        self.config.safe_exp = False
        self.config.quantized_patterns = [[OpType.BatchMatmul]]

class ConformerBuilder(SparsertBaseBuilder):
    def __init__(self, onnx_path, dump_dir, dataset_dir, dataset_cfg, dtype, batch_size, verify, **kwargs):
        super(ConformerBuilder, self).__init__(
            onnx_path, dump_dir, dataset_dir, dataset_cfg, dtype, batch_size, verify, **kwargs)

    def set_dataset_config(self):
        # model inputs info
        self.config.input_dict = {
            "src": "src",
            "src_pad_mask": "src_pad_mask"
        }
        self.input_shape_dict = {
            "src": (self.batch_size,3,64,512),
            "src_pad_mask": (self.batch_size, 128)
        }

        # calibration dataset config
        dataset = DummyDataset(input_info=self.config.input_dict)
        self.config.dataloader = DataLoaderX(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.config.calib_batch = 1

        # you can also set other configs here
        self.config.do_kl = False
        self.config.opt_level = 5
        self.config.safe_exp = False
        self.config.quantized_patterns = [[OpType.BatchMatmul]]
