"""
配置模块，包含全局配置参数
"""
import os
from pathlib import Path

# Project directories
ROOT_DIR = Path("/root/autodl-tmp/pycharm_project_687")  # Remove trailing slash

# Directory configurations
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
CIRCUIT_DIR = ROOT_DIR / "circuits"
CACHE_DIR = DATA_DIR  # Use DATA_DIR as cache directory for datasets
ZKP_PTAU_PATH = CIRCUIT_DIR / "powersOfTau28_hez_final_08.ptau"

# Create necessary directories
for dir_path in [DATA_DIR, MODELS_DIR, REPORTS_DIR, CIRCUIT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# CIFAR-10数据集均值和标准差
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# 默认置信水平
DEFAULT_CONFIDENCE = 0.95

# 默认随机种子
DEFAULT_SEED = 42

# 默认批次大小
# Add these constants
DEFAULT_DEVICE = "cuda"
DEFAULT_BATCH_SIZE = 128

# 默认训练轮次
DEFAULT_EPOCHS = 50

# 默认学习率
DEFAULT_LEARNING_RATE = 0.1

# 默认蒸馏参数
DEFAULT_DISTILLATION_ALPHA = 0.5
DEFAULT_DISTILLATION_TEMPERATURE = 4.0

# 默认验证参数
DEFAULT_NUM_BASELINE_MODELS = 5
# 修改默认参数以符合实验要求
DEFAULT_EPOCHS = 100  # 训练轮次
DEFAULT_BATCH_SIZE = 256  # 批次大小
DEFAULT_LEARNING_RATE = 3e-4  # 学习率
DEFAULT_NUM_SAMPLES = 1000  # KL散度计算的样本数