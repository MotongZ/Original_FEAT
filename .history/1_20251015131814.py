import os
import sys

# 模拟命令行参数
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 现在导入 torch
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

# 测试权重加载
weight_path = "./saves/initialization/miniimagenet/protonet-5-shot.pth"
try:
    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
    print("✓ 权重加载成功")
except Exception as e:
    print(f"✗ 权重加载失败: {e}")