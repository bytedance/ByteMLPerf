import torch
import numpy as np


def get_topk_accuracy(pred, label):
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
        
    if isinstance(label, np.ndarray):
        label = torch.from_numpy(label)
    
    top1_acc = 0
    top5_acc = 0
    for idx in range(len(label)):
        label_value = label[idx]
        if label_value == torch.topk(pred[idx].float(), 1).indices.data:
            top1_acc += 1
            top5_acc += 1

        elif label_value in torch.topk(pred[idx].float(), 5).indices.data:
            top5_acc += 1
            
    return top1_acc, top5_acc