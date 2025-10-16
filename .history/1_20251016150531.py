import sys
import os
import torch
from model.networks.res12_PPA import ResNet_PPA

pretrain = './saves/initialization/miniimagenet/protonet-5-shot.pth'

model = ResNet_PPA()

for key in model_dict['params'].keys():
    print(key)