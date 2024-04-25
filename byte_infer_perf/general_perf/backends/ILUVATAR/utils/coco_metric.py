import os
import json
import cv2
import numpy as np

import torch
import torchvision
from pycocotools.coco import COCO

def get_coco_accuracy(pred_json, ann_json):
    coco = COCO(annotation_file=ann_json)
    coco_pred = coco.loadRes(pred_json)
    try:
        from .fastCoCoeval.fast_coco_eval_api import COCOeval_opt as COCOeval
        coco_evaluator = COCOeval(cocoGt=coco, cocoDt=coco_pred, iouType="bbox")
    except:
        from pycocotools.cocoeval import COCOeval
        print("Can't import fastCoCoeval, Using PyCoCcotools API ...")
        coco_evaluator = COCOeval(cocoGt=coco, cocoDt=coco_pred, iouType="bbox")
            
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    return coco_evaluator.stats

coco80_to_coco91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88,
    89, 90
]

coco80_to_coco91_dict = {idx: i for idx, i in enumerate(coco80_to_coco91)}
coco91_to_coco80_dict = {i: idx for idx, i in enumerate(coco80_to_coco91)}


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[
        1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im,
                            top,
                            bottom,
                            left,
                            right,
                            cv2.BORDER_CONSTANT,
                            value=color)  # add border
    return im, ratio, (dw, dh)


def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter + eps)


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # top left x
    y[:, 1] = h * x[:, 1] + padh  # top left y
    return y


def segment2box(segment, width=640, height=640):
    # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(),
                     y.max()]) if any(x) else np.zeros((1, 4))  # xyxy


def segments2boxes(segments):
    # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def resample_segments(segments, n=1000):
    # Up-sample an (n,2) segment
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([
            np.interp(x, xp, s[:, i]) for i in range(2)
        ]).reshape(2, -1).T  # segment xy
    return segments


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def scale_segments(img1_shape, segments, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    segments[:, 0] -= pad[0]  # x padding
    segments[:, 1] -= pad[1]  # y padding
    segments /= gain
    clip_segments(segments, img0_shape)
    return segments


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def clip_segments(boxes, shape):
    # Clip segments (xy1,xy2,...) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x
        boxes[:, 1].clamp_(0, shape[0])  # y
    else:  # np.array (faster grouped)
        boxes[:, 0] = boxes[:, 0].clip(0, shape[1])  # x
        boxes[:, 1] = boxes[:, 1].clip(0, shape[0])  # y


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=True,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    if isinstance(
            prediction, (list, tuple)
    ):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    # t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(
            x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat(
                (box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask),
                          1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(
                descending=True)[:max_nms]]  # sort by confidence
        else:
            x = x[x[:, 4].argsort(descending=True)]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:,
                                        4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n <
                      3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
    return output




#NOTE(chen.chen):
# just work for coco2017 val using pycocotools
# maybe we need some abstraction here for generic coco-like dataset
class COCO2017Evaluator:    
    def __init__(self,
                 label_path,
                 image_size=640,
                 with_nms=False,
                 conf_thres=0.001,
                 iou_thres=0.65):
        self.with_nms = with_nms
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.label_path = label_path
        self.image_size = image_size

        self.jdict = []

        # iou vector for mAP@0.5:0.95
        self.iouv = torch.linspace(0.5, 0.95, 10)  
        self.niou = self.iouv.numel()
    
    def evaluate(self, pred, all_inputs, nms_count=None):
        im = all_inputs[0]
        targets = all_inputs[1]
        paths = all_inputs[2]
        shapes = all_inputs[3]

        _, _, height, width = im.shape
        targets[:, 2:] *= np.array((width, height, width, height))
        
        if self.with_nms:
            assert nms_count is not None
            tmp_out = []
            for boxes, count in zip(pred, nms_count):
                count = count[0]
                boxes = boxes[:count, :]
                boxes_cp = boxes.copy()
                # (x1,y1,x2,y2,class_id,score)
                # To (x1,y1,x2,y2,score,class_id)
                boxes[:, 4] = boxes_cp[:, 5]
                boxes[:, 5] = boxes_cp[:, 4]
                tmp_out.append(torch.from_numpy(boxes))
            pred = tmp_out   
        else:
            pred = torch.from_numpy(pred)
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        for idx, det in enumerate(pred):
            img_path = paths[idx]

            predn = det
            shape = shapes[idx][0]
            scale_boxes(im[idx].shape[1:], predn[:, :4], shape, shapes[idx][1])  # native-space pred

            self._save_one_json(predn, self.jdict, img_path, coco80_to_coco91)  # append to COCO-JSON dictionary
        

    def _save_one_json(self, predn, jdict, path, class_map):
        # Save one JSON result in the format
        # {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
        image_id = int(os.path.splitext(os.path.basename(path))[0])
        box = xyxy2xywh(predn[:, :4])
        box[:, :2] -= box[:, 2:] / 2
        for p, b in zip(predn.tolist(), box.tolist()):
            jdict.append({
                'image_id': image_id,
                'category_id': class_map[int(p[5])],
                'bbox': [round(x, 3) for x in b],
                'score': round(p[4], 5)
            })


    def summary(self):
        if len(self.jdict):
            pred_json = os.path.join("coco2017_predictions.json")
            with open(pred_json, 'w') as f:
                json.dump(self.jdict, f)
            result = get_coco_accuracy(pred_json, self.label_path)
        else:
            raise ValueError("can not find generated json dict for pycocotools")
        return result

# coco2017 val evaluator For Yolox
class COCO2017EvaluatorForYolox(COCO2017Evaluator):
    def evaluate(self, pred, all_inputs):
        im = all_inputs[0]
        img_path = all_inputs[1]
        img_info = all_inputs[2]
        
        _, _, height, width = im.shape
        img_size = [height, width]

        pred = torch.from_numpy(self.Detect(pred, img_size=[height, width]))

        nms_outputs = self.postprocess(
                    pred, conf_thre=self.conf_thres, nms_thre=self.iou_thres
                )

        for (output, org_img, path) in zip(nms_outputs, img_info, img_path):
            if output is None:
                continue
            
            bboxes = output[:, 0:4]

            img_h, img_w = org_img

            scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))

            bboxes /= scale
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            
            bboxes = self._xyxy2xywh(bboxes)

            self._save_one_json(bboxes, cls, scores, self.jdict, path, coco80_to_coco91)

    def Detect(self, outputs, img_size):
        grids = []
        expanded_strides = []

        strides = [8, 16, 32]

        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
        
        return outputs
    
    def postprocess(self, prediction, num_classes=80, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]

        for i, image_pred in enumerate(prediction):
            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            
            if not detections.size(0):
                continue
            if class_agnostic:
                nms_out_index = torchvision.ops.nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    nms_thre,
                )
            else:
                nms_out_index = torchvision.ops.batched_nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    detections[:, 6],
                    nms_thre,
                )
            detections = detections[nms_out_index]

            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))

        return output

    def _xyxy2xywh(self, bboxes):
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
        return bboxes

    def _save_one_json(self, bboxes, class_, scores, jdict, path, class_map):
        image_id = int(os.path.splitext(os.path.basename(path))[0])
        for box, score, cls in zip(bboxes.numpy().tolist(), scores.numpy().tolist(), class_.numpy().tolist()):
            jdict.append({
                'image_id': image_id,
                'category_id': class_map[int(cls)],
                'bbox': box,
                'score': score
            })


# coco2017 val evaluator For Yolov4
class COCO2017EvaluatorForYolov4(COCO2017EvaluatorForYolox):
    def evaluate(self, pred, all_inputs):
        im = all_inputs[0]
        img_path = all_inputs[1]
        img_info = all_inputs[2]

        boxes = torch.squeeze(torch.from_numpy(pred[0]), dim=2)
        confs = torch.from_numpy(pred[1])
        detections = torch.cat((boxes, confs.float()), 2)

        nms_outputs = self.postprocess(
            detections, conf_thre=self.conf_thres, nms_thre=self.iou_thres
        )

        for (output, org_img, path) in zip(nms_outputs, img_info, img_path):
            if output is None:
                continue
            
            bboxes = output[:, 0:4]
            img_h, img_w = org_img
            bboxes[:, 0] *= img_w
            bboxes[:, 2] *= img_w
            bboxes[:, 1] *= img_h
            bboxes[:, 3] *= img_h

            cls = output[:, 5]
            scores = output[:, 4]
            
            bboxes = self._xyxy2xywh(bboxes)
            self._save_one_json(bboxes, cls, scores, self.jdict, path, coco80_to_coco91)
    
    def postprocess(self, prediction, num_classes=80, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
        output = [None for _ in range(len(prediction))]

        for i, image_pred in enumerate(prediction):
            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 4: 4 + num_classes], 1, keepdim=True)

            conf_mask = (class_conf.squeeze() >= conf_thre).squeeze()
            detections = torch.cat((image_pred[:, :4], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]

            if not detections.size(0):
                continue
            if class_agnostic:
                nms_out_index = torchvision.ops.nms(
                    detections[:, :4],
                    detections[:, 4],
                    nms_thre,
                )
            else:
                nms_out_index = torchvision.ops.batched_nms(
                    detections[:, :4],
                    detections[:, 4],
                    detections[:, 5],
                    nms_thre,
                )
            detections = detections[nms_out_index]

            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))

        return output