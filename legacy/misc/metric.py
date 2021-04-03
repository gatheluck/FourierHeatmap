import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import torch
import torchvision

def get_num_correct(output, target, topk=1):
    with torch.no_grad():
        _, pred = output.topk(topk, dim=1) # top-k index: size (B, k)
        pred = pred.t() # size (k, B)
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        num_correct = correct[:topk].float().sum().item()
    
        return num_correct