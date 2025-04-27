"""
统一的导入模块，用于在Jupyter Notebook或交互式环境中快速加载所有组件
"""
import os
import sys
import numpy as np
from datetime import datetime
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from config import *
from models import ResNetModel, DistilledModel


def set_seed(seed=42):
    """设置随机种子以确保实验可重复"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def show_gpu_info():
    """显示GPU信息"""
    if torch.cuda.is_available():
        print("\nGPU信息:")
        print(f"  设备名称: {torch.cuda.get_device_name(0)}")
        print(f"  总内存: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f}GB")
        print(f"  保留内存: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f}GB")
        print(f"  分配内存: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f}GB")
        print(f"  CUDA版本: {torch.version.cuda}")
    else:
        print("\n警告: 未检测到可用的CUDA设备，将使用CPU进行计算")

def init():
    """初始化环境"""
    set_seed(DEFAULT_SEED)
    show_gpu_info()
    print(f"\n初始化完成，使用设备: {DEFAULT_DEVICE}")
    return f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# 执行初始化
if __name__ == "__main__":
    exp_id = init()
    print(f"实验ID: {exp_id}")