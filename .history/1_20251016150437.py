import sys
import os
import torch

pretrain = './saves/initialization/miniimagenet/protonet-5-shot.pth'

model_dict = torch.load(pretrain)

for key in model_dict['params'].keys():
    print(key)