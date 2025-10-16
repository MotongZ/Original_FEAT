import sys
import os
import torch
from model.networks.res12_PPA import ResNet_PPA

pretrain = './saves/initialization/miniimagenet/protonet-5-shot.pth'

model = ResNet_PPA()
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

print("\n=== 模型状态字典键 ===")
for key in model.state_dict().keys():
    print(key)