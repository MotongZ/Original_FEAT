import torch
import os

# 设置 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

# 测试加载权重
weight_path = "./saves/initialization/miniimagenet/protonet-5-shot.pth"

try:
    # 方法1：直接加载到 CPU
    checkpoint = torch.load(weight_path, map_location='cpu')
    print("✓ 成功加载到 CPU")
    
    # 方法2：加载到指定 GPU
    if torch.cuda.is_available():
        checkpoint = torch.load(weight_path, map_location='cuda:0')
        print("✓ 成功加载到 GPU")
        
except Exception as e:
    print(f"✗ 加载失败: {e}")