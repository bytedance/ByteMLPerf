import os
import numpy as np
from PIL import Image
from collections import defaultdict

import tensorflow as tf
try:
    tf = tf.compat.v1
except ImportError:
    tf = tf
tf.enable_eager_execution()

import torch
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from pycocotools.coco import COCO

from .coco_metric import *

_igie_cache_dir = os.path.expanduser("~/.igie_cache")
_bulitin_data_url = "http://10.113.3.3/data/CI_DATA/ci_data.tar.gz"
_builtin_data_path = os.path.join(_igie_cache_dir, "modelzoo_data")
_symbolic_link_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


### Tensorflow image pre-process function
def _mean_image_subtraction(image, means):
    """Subtracts the given means from each image channel."""
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)

def _central_crop(image, crop_height, crop_width):
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    amount_to_be_cropped_h = (height - crop_height)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_width)
    crop_left = amount_to_be_cropped_w // 2
    return tf.slice(image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])

def _aspect_preserving_resize(image, resize_min):
    """Resize images preserving the original aspect ratio.
    """
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    new_height, new_width = _smallest_size_at_least(height, width, resize_min)
    return _resize_image(image, new_height, new_width)

def _smallest_size_at_least(height, width, resize_min):
    resize_min = tf.cast(resize_min, tf.float32)
    # Convert to floats to make subsequent calculations go smoothly.
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
    smaller_dim = tf.minimum(height, width)
    scale_ratio = resize_min / smaller_dim
    # Convert back to ints to make heights and widths that TF ops will accept.
    new_height = tf.cast(height * scale_ratio, tf.int32)
    new_width = tf.cast(width * scale_ratio, tf.int32)
    return new_height, new_width

def _resize_image(image, height, width):
    return tf.image.resize(image, [height, width], method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False)



### Pytorch image pre-process function
def _torch_imagenet_preprocess(image_path):
    img = Image.open(image_path).convert('RGB')
    # preprocess image to nomalized tensor for pytorch
    _PYTORCH_IMAGENET_PREPROCESS = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225]),
        ]
    )
    img = _PYTORCH_IMAGENET_PREPROCESS(img)
    return img


### Tensorflow image pre-process function
def _tf_imagenet_preprocess(image_path):
    img = Image.open(image_path).convert('RGB')
    _TF_IMAGENET_PREPROCESS = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    
    img = _TF_IMAGENET_PREPROCESS(img)
    img *= 255.0
    assert len(img.shape) == 3
    img = transforms.Normalize(mean=[123.68, 116.78, 103.94], std=[1, 1, 1])(img)
    img = img.permute((1, 2, 0)) # CHW -> HWC
    
    return img


class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir_path, label_dir_path="", layout="NHWC", image_size=(224, 224)):
        super().__init__()
        self.image_dir_path = image_dir_path
        self.label_dir_path = label_dir_path
        self.layout = layout
        
        if len(image_size) == 1:
            self.image_height = self.image_width = image_size
        if len(image_size) == 2:
            self.image_height = image_size[0]
            self.image_width = image_size[1]
        assert self.layout in ["NHWC", "NCHW"], f"layout should be NHWC or NCHW, got {self.layout} "
        self.img_list = os.listdir(self.image_dir_path)
        self.label_dict = self.get_label_dict()
        
        self.images = []
        self.length = 0

        for image_dir in self.img_list:
            image_path = os.path.join(self.image_dir_path, image_dir)
            if os.path.isdir(image_path):
                for image in os.listdir(image_path):
                    self.images.append(os.path.join(image_path, image))
                    self.length += 1

    def __getitem__(self, index):
        ## NHWC pre-process for tensorflow
        if self.layout == "NHWC":
            processed_image = _tf_imagenet_preprocess(self.images[index])
            # image = cv2.imread(self.images[index])
            # image = cv2.cvtColor(image, 4)
            # resize_image = _aspect_preserving_resize(image, 256)
            # crop_image = _central_crop(resize_image, self.image_height, self.image_width)  
            # crop_image.set_shape([self.image_height, self.image_width, 3])
            # crop_image = tf.to_float(crop_image)
            # processed_image = _mean_image_subtraction(crop_image, [123.68, 116.78, 103.94]).numpy()
        
        ## NCHW pre-process for Pytorch
        elif self.layout == "NCHW":
            processed_image = _torch_imagenet_preprocess(self.images[index])
        else:
            raise ValueError("Unsupported data layout")

        image_name = self.images[index].split('/')[-1].strip()
        label = self.label_dict[image_name]

        return processed_image, label

    def __len__(self):
        return self.length

    def get_label_dict(self):
        image_label = {}
        label_path = os.path.join(self.image_dir_path, 'val.txt')
        if self.label_dir_path != "":
            label_path = self.label_dir_path
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                lines = file.readlines()
        
            for line in lines:
                image = line.split(' ')[0].strip()
                label = line.split(' ')[1].strip()
                image_label[image] = int(label)
        
        return image_label

def get_imagenet_dataloader(data_path, batch_size, num_workers, model_framework, input_layout):
    if model_framework == "tensorflow":
        val_dir = os.path.join(data_path, "val")
        dataset = ImageNetDataset(val_dir, layout="NHWC")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=num_workers, drop_last=True)

    else:
        assert input_layout == "NCHW"
        val_dir = os.path.join(data_path, 'validation')
        assert os.path.isdir(val_dir), f"{val_dir} does not exist, please specify correct data path"

        dataset = torchvision.datasets.ImageFolder(
            val_dir,
            transforms.Compose(
                [
                    transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
                    transforms.CenterCrop(224),
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ]
            )
        )

        dataloader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=num_workers, drop_last=True)

    return dataloader

class COCO2017Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 image_dir_path,
                 label_json_path,
                 image_size=640,
                 pad_color=114,
                 val_mode=True,
                 input_layout="NCHW"):

        self.image_dir_path = image_dir_path
        self.label_json_path = label_json_path
        self.image_size = image_size
        self.pad_color = pad_color
        self.val_mode = val_mode
        self.input_layout = input_layout

        self.coco = COCO(annotation_file=self.label_json_path)
        
        if self.val_mode:
            self.img_ids = list(sorted(self.coco.imgs.keys()))  # 5000
        else:  # train mode need images with labels
            self.img_ids = sorted(list(self.coco.imgToAnns.keys()))  # 4952

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_path = self._get_image_path(index)
        img, (h0, w0), (h, w) = self._load_image(index)

        img, ratio, pad = letterbox(img,
                                    self.image_size,
                                    color=(self.pad_color, self.pad_color, self.pad_color))
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        # load label
        raw_label = self._load_json_label(index)
        # normalized xywh to pixel xyxy format
        raw_label[:, 1:] = xywhn2xyxy(raw_label[:, 1:],
                                      ratio[0] * w,
                                      ratio[1] * h,
                                      padw=pad[0],
                                      padh=pad[1])

        raw_label[:, 1:] = xyxy2xywhn(raw_label[:, 1:],
                                      w=img.shape[1],
                                      h=img.shape[0],
                                      clip=True,
                                      eps=1E-3)

        nl = len(raw_label)  # number of labels
        labels_out = np.zeros((nl, 6))
        labels_out[:, 1:] = raw_label

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img) / 255.0  # 0~1 np array
        if self.input_layout == "NHWC":
            img = img.transpose((1, 2, 0))

        return img, labels_out, img_path, shapes

    def _get_image_path(self, index):
        idx = self.img_ids[index]
        path = self.coco.loadImgs(idx)[0]["file_name"]
        img_path = os.path.join(self.image_dir_path, path)
        return img_path

    def _load_image(self, index):
        img_path = self._get_image_path(index)

        im = cv2.imread(img_path)  # BGR
        h0, w0 = im.shape[:2]  # orig hw
        r = self.image_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
        return im.astype("float32"), (h0, w0), im.shape[:2]  # im, hw_original, hw_resized

    def _load_json_label(self, index):
        _, (h0, w0), _ = self._load_image(index)

        idx = self.img_ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=idx)
        targets = self.coco.loadAnns(ids=ann_ids)

        labels = []
        for target in targets:
            cat = target["category_id"]
            coco80_cat = coco91_to_coco80_dict[cat]
            cat = np.array([[coco80_cat]])

            x, y, w, h = target["bbox"]
            x1, y1, x2, y2 = x, y, int(x + w), int(y + h)
            xyxy = np.array([[x1, y1, x2, y2]])
            xywhn = xyxy2xywhn(xyxy, w0, h0)
            labels.append(np.hstack((cat, xywhn)))

        if labels:
            labels = np.vstack(labels)
        else:
            if self.val_mode:
                # for some image without label
                labels = np.zeros((1, 5))
            else:
                raise ValueError(f"set val_mode = False to use images with labels")

        return labels

    @staticmethod
    def collate_fn(batch):
        im, label, path, shapes = zip(*batch)
        for i, lb in enumerate(label):
            lb[:, 0] = i
        return np.concatenate([i[None] for i in im], axis=0), np.concatenate(label, 0), path, shapes

# Datasets just for Yolox
class COCO2017DatasetForYolox(COCO2017Dataset):
    def __getitem__(self, index):
        img_path = self._get_image_path(index)
        img = self._load_image(img_path)

        img, r = self.preproc(img, input_size=self.image_size)
        
        return img, img_path, r

    def _load_image(self, img_path):
        img = cv2.imread(img_path)
        assert img is not None, f"file {img_path} not found"

        return img
    
    def preproc(self, img, input_size, swap=(2, 0, 1)):
        if len(img.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114
        
        org_img = (img.shape[0], img.shape[1])
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, org_img

    @staticmethod
    def collate_fn(batch):
        im, img_path, r = zip(*batch)
        return np.concatenate([i[None] for i in im], axis=0), img_path, r

# Datasets just for Yolox
class COCO2017DatasetForYolov4(COCO2017DatasetForYolox):
    def preproc(self, img, input_size, swap=(2, 0, 1)):
        org_img = (img.shape[0], img.shape[1])
        img_ = cv2.resize(img, (input_size[0], input_size[1]))
        img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        img_ = img_.transpose(swap) / 255.0
        img_ = np.ascontiguousarray(img_, dtype=np.float32)
        return img_, org_img
    
def get_coco2017_dataloader(data_path, label_path, batch_size, image_size, num_workers, model_framework, input_layout, custom_option=None):
    # TODO(chen.chen)
    # we only support pytorch-like coco2017 data preprocess
    # some problems may occur when the data preprocess is different, e.g. tensorflow
    assert model_framework != "tensorflow"
    if custom_option == 'yolox':
        dataset = COCO2017DatasetForYolox(data_path, label_path, image_size=(image_size, image_size), input_layout=input_layout)
    elif custom_option == 'yolov4':
        dataset = COCO2017DatasetForYolov4(data_path, label_path, image_size=(image_size, image_size), input_layout=input_layout)
    else:
        dataset = COCO2017Dataset(data_path, label_path, image_size, input_layout=input_layout)
        
    # NOTE(chen.chen)
    # we should validate all images in the datasets to use pycocotools
    # so we do not drop last batch which maybe smaller than a normal batch
    # you should pad the batch dimension in the outside
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            drop_last=False,
                                            num_workers=num_workers,
                                            collate_fn=dataset.collate_fn)

    return dataloader


class FakeDataSet(torch.utils.data.Dataset):
    def __init__(self, input_name_list, input_shape_list, input_dtype_list):
        self.input_name_list = input_name_list
        self.input_shape_list = input_shape_list
        self.input_dtype_list = input_dtype_list

        self.max_length = 100000

    def __len__(self):
        return self.max_length
        
    def __getitem__(self, _):
        input_data = []
        for shape, dtype in zip(self.input_shape_list, self.input_dtype_list):
            if dtype.startswith("float"):
                data = np.random.randn(*shape[1:]).astype(dtype)
            elif dtype.startswith("int"):
                data = np.random.randint(0, 10, shape[1:]).astype(dtype)
            else:
                raise ValueError(f"unsupported dtype: {dtype}")
        
            input_data.append(data)
            
        return tuple(input_data)
        

    @staticmethod
    def collate_fn(batch):
        batch_input_data = []
        for i in zip(*batch):
            data = np.concatenate([j[np.newaxis,:] for j in i], axis=0)
            batch_input_data.append(data)
        return tuple(batch_input_data)        

class NumpyDataSet(torch.utils.data.Dataset):
    def __init__(self, input_name_list, input_shape_list, input_dtype_list, path):
        self.input_name_list = input_name_list
        self.input_shape_list = input_shape_list
        self.input_dtype_list = input_dtype_list
        self.path = path

        self.ext = os.path.splitext(self.path)[-1]
        assert self.ext.endswith(".npy") or self.ext.endswith(".data")

        self.dtype_size_map = {
            "fp32": np.dtype("float32"),
            "float32": np.dtype("float32"),
            "fp16": np.dtype("float16"),
            "float16": np.dtype("float16"),
            "int8": np.dtype("int8")
        }
        
        self._process_numpy_data()
   
    def _process_numpy_data(self):
        if self.ext.endswith(".npy"):
            self.total_data_number = len(self.input_name_list)
            
            self.data = np.load(self.path, allow_pickle=True)
            assert len(self.data) == self.total_data_number, f"np data length should be {self.total_data_number}, got {len(self.data)}"        
            self.length = self.data[0].shape[0]
        
        elif self.ext.endswith(".data"): 
            with open(self.path, mode='rb') as f:
                calibrate_data = f.read()
            
            total_bytes = 0
            input_size_list = []
            for shape, dtype in zip(self.input_shape_list, self.input_dtype_list):
                size = np.prod(shape) * self.dtype_size_map[dtype].itemsize
                input_size_list.append(size)
                total_bytes += size
            
            assert (len(calibrate_data) % total_bytes == 0), f"calibrate_data size({len(calibrate_data)}) don't match one batch size({total_bytes}) multiple."
            
            index = 0
            npy_data_dict = defaultdict(list)
            while index < len(calibrate_data):
                for name, shape, dtype, size in zip(self.input_name_list, self.input_shape_list, self.input_dtype_list, input_size_list):   
                    data = np.frombuffer(calibrate_data[index: index + size], dtype=dtype).reshape(shape)
                    npy_data_dict[name].append(data)
                    index += size

            self.data = []
            for v in npy_data_dict.values():
                data = np.concatenate(v, axis=0)
                self.data.append(data)
                
            self.length = self.data[0].shape[0]
        else:
            raise 

    def __len__(self):
        return self.length
        
    def __getitem__(self, index):
        input_data = []
        for i in self.data:
            input_data.append(i[index])
        return tuple(input_data)
        
    @staticmethod
    def collate_fn(batch):
        batch_input_data = []
        for i in zip(*batch):
            data = np.concatenate([j[np.newaxis,:] for j in i], axis=0)
            batch_input_data.append(data)
        return tuple(batch_input_data)        

def download_builtin_data():
    if not os.path.exists(_builtin_data_path):
        if not os.path.exists(_igie_cache_dir):
            os.makedirs(_igie_cache_dir)

        pwd = os.getcwd()
        os.chdir(_igie_cache_dir)
        
        cmd = f"wget {_bulitin_data_url}"
        os.system(cmd)

        cmd = f"tar -xzf ci_data.tar.gz"
        os.system(cmd)
                
        os.chdir(pwd)
    
    if os.path.exists(_builtin_data_path) and not os.path.exists(_symbolic_link_data_path):
        cmd = f"ln -s {_builtin_data_path} {_symbolic_link_data_path}"
        os.system(cmd)
        
    print(f"Use builtin dataset path: {_builtin_data_path}")
        

def get_dataloader_from_args(args):
    ## use built-in dataset
    if args.use_builtin_data:
        download_builtin_data()
 
        if args.use_imagenet:
            args.data_path = os.path.join(_builtin_data_path, "datasets", "imagenet")
            
            return get_imagenet_dataloader(args.data_path, args.batch_size, args.num_workers, args.model_framework, args.input_layout)
            
        elif args.use_coco2017:
            args.data_path = os.path.join(_builtin_data_path, "datasets", "coco", "images", "val2017")
            args.label_path = os.path.join(_builtin_data_path, "datasets", "coco", "annotations", "instances_val2017.json")

            input_shape = args.input_shape_list[0]            
            assert len(input_shape) == 4, f"input should be a 4d tensor, format as NCHW or NHWC, got {len(input_shape)}"
            if args.input_layout == "NCHW":
                assert input_shape[2] == input_shape[3], f"HW should be the same, got {input_shape[2]} and {input_shape[3]}"
                args.image_size = input_shape[2]
            else: #NHWC
                assert input_shape[1] == input_shape[2], f"HW should be the same, got {input_shape[1]} and {input_shape[2]}"
                args.image_size = input_shape[1]

            # use custom option do preprocessing
            if args.custom_option is not None  and 'process' in args.custom_option:
                return get_coco2017_dataloader(args.data_path, args.label_path, args.batch_size, args.image_size, args.num_workers, args.model_framework, args.input_layout, args.custom_option['process'])
            else:   
                return get_coco2017_dataloader(args.data_path, args.label_path, args.batch_size, args.image_size, args.num_workers, args.model_framework, args.input_layout)
            
    
    elif args.calibration_file_path is not None:
        ## NOTE(chen.chen)
        ## user-provided dataset, just use it as calibration data
        ## we support two format .npy and .data
        
        ## if extetion is .npy, it should be a single npy file,
        ## each input should be saved in a np.ndarray which has beed preprocessed
        ## e.g. for two inputs model
        ## the npy should be a list of two array, the shape of each array is like below
        ## ((100, 3, 224, 224), (100, 1000))
        
        ## if extension is .data, we will call np.frombuffer to load the data
        ## this is for paddle-igie compatibility and only support single input now
        
        
        calibration_file_path = args.calibration_file_path
        assert os.path.exists(calibration_file_path), f"can not find calibration file:{calibration_file_path}"
        ext = os.path.splitext(calibration_file_path)[-1]
        
        assert ext in [".npy", ".data"], f"unspported calibration file format {ext}, it should be .npy or .data"
        
        dataset = NumpyDataSet(args.input_name_list, args.input_shape_list, args.input_dtype_list, calibration_file_path)
        
        dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers, drop_last=True, collate_fn=dataset.collate_fn)
          
        return dataloader
    
    else:
        ## NOTE(chen.chen)
        ## use fake data for calibration, just used for perf test
        ## here we should know the shape/dtype info of the input to generate the fake input data
        dataset = FakeDataSet(args.input_name_list, args.input_shape_list, args.input_dtype_list)
        dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers, drop_last=True, collate_fn=dataset.collate_fn)

        return dataloader
    
