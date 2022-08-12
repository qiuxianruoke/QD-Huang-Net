import argparse
import logging
import math
import os
import random
import time
from pathlib import Path
from warnings import warn

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from models.models import *


def train(opt):
    model = Darknet(opt.cfg, \
                batch_size=opt.batch_size).to(opt.device)
    # test
    x = torch.rand(8,3,640,640)
    print(model(x).shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--backbone_name', type=str, default='backbone/yolor_b', help='backbone name')
    parser.add_argument('--cfg', type=str, default='cfg/MobileYOLO_backbone.cfg', help='model.yaml path')
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    train(opt)
    print('\n\n[=========finish===========]')