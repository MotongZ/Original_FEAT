import sys
import os
import torch

print("=== Python 环境诊断 ===")
print(f"Python 可执行文件路径: {sys.executable}")
print(f"Python 版本: {sys.version}")
print(f"当前工作目录: {os.getcwd()}")

print("\n=== Python 路径 ===")
for i, path in enumerate(sys.path):
    print(f"{i}: {path}")

print("\n=== PyTorch 信息 ===")
print(f"PyTorch 版本: {torch.__version__}")
print(f"PyTorch 文件位置: {torch.__file__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")

print("\n=== 环境变量 ===")
print(f"CONDA_DEFAULT_ENV: {os.environ.get('CONDA_DEFAULT_ENV', 'Not set')}")
print(f"CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', 'Not set')}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")