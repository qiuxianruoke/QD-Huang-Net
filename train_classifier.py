import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from models.models import *


def train(opt):
    model = Darknet(opt.cfg, \
                batch_size=opt.batch_size).to(opt.device)
    # print(model.parameters)
    # test
    x = torch.rand(8,3,640,640)
    result =model(x) # result未归一化
    print(result.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--backbone_name', type=str, default='backbone/yolor_b', help='backbone name')
    parser.add_argument('--cfg', type=str, default='cfg/MobileYOLO_backbone.cfg', help='model.yaml path')
    # cfg: cfg/MobileYOLO_backbone.cfg cfg/yolor_backbone.cfg cfg/yolor_lite_backbone.cfg
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    train(opt)
    print('\n\n[=========finish===========]')