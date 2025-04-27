# 零知识证明神经网络验证系统

本系统用于验证神经网络是否为知识蒸馏模型，采用黑盒+白盒联合证明方法，通过零知识证明（ZKP）技术实现可验证性。

## 功能特点

- **黑盒验证**：通过KL散度检测模型输出分布特征
- **白盒抽样**：验证训练过程中是否存在蒸馏组件
- **零知识证明**：生成模型验证的密码学证明
- **统计分析**：计算置信区间和阈值T
- **可视化结果**：生成多种直观可视化图表
- **详细报告**：自动生成HTML和JSON格式验证报告

## 系统架构

- `config.py`: 系统配置文件
- `models.py`: 模型定义模块
- `train.py`: 模型训练模块
- `verify.py`: 模型验证模块
- `stats.py`: 统计分析模块
- `zkp.py`: 零知识证明生成模块
- `visualization.py`: 可视化模块
- `report.py`: 报告生成模块
- `main.py`: 主程序入口

## 依赖环境

- Python 3.7+
- PyTorch 1.7+
- NumPy
- Matplotlib
- scikit-learn
- snarkjs (零知识证明库)
- circom (电路编译器)

## 安装方法

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/zkp-neural-verification.git
cd zkp-neural-verification
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 安装零知识证明工具：
```bash
npm install -g snarkjs
npm install -g circom
```

4. 下载Powers of Tau文件：
```bash
mkdir -p ptau
# 下载10阶Powers of Tau文件
wget -P ptau https://hermez.s3-eu-west-1.amazonaws.com/powersOfTau28_hez_final_10.ptau
```

## 使用方法

### 完整验证流程

```bash
python main.py --mode full --baselines 5 --epochs 10 --batch-size 128 --temperature 1.0 --generate-zkp
```

### 仅训练模型

```bash
python main.py --mode train --baselines 5 --epochs 10 --batch-size 128 --force-retrain
```

### 仅验证模型

```bash
python main.py --mode verify --temperature 1.0
```

### 仅生成零知识证明

```bash
python main.py --mode zkp --generate-zkp
```

### 包含蒸馏模型（对照实验）

```bash
python main.py --mode full --train-distilled --distill-alpha 0.5 --distill-temp 4.0
```

## 参数说明

- `--mode`: 运行模式，选项为 train(仅训练), verify(仅验证), zkp(仅零知识证明), full(完整流程)
- `--baselines`: 基线模型数量，默认为5
- `--epochs`: 训练轮次，默认为10
- `--batch-size`: 训练批次大小，默认为128
- `--temperature`: 分布温度参数，默认为1.0
- `--device`: 计算设备，'cuda'或'cpu'
- `--gpu-mem-limit`: GPU内存限制比例(0-1)，默认为0.7
- `--train-distilled`: 是否训练蒸馏模型（用于对照）
- `--distill-alpha`: 蒸馏损失权重，默认为0.5
- `--distill-temp`: 蒸馏温度，默认为4.0
- `--generate-zkp`: 是否生成零知识证明
- `--force-retrain`: 强制重新训练模型
- `--exp-id`: 实验ID，不指定则自动生成

## 代码结构

- `MODELS_DIR`: 用于存储训练好的模型
- `REPORTS_DIR`: 用于存储验证报告
- `CACHE_DIR`: 用于存储数据集缓存
- `CIRCUIT_DIR`: 用于存储零知识证明电路
- `STATS_DIR`: 用于存储统计分析结果

## 实验报告

验证完成后，系统会生成两种格式的报告：

1. **JSON格式报告**: 包含详细的验证数据，适合程序分析
2. **HTML格式报告**: 包含可视化结果和分析摘要，适合人类阅读

报告内容包括：

- 基线模型统计信息
- 目标模型验证结果
- KL散度分析
- 零知识证明结果
- 可视化图表
- 验证结论

## 优化建议

1. 对于家用电脑（如RTX 4060），建议：
   - 减小批次大小 (--batch-size 64)
   - 减少训练轮次 (--epochs 5)
   - 设置GPU内存限制 (--gpu-mem-limit 0.7)

2. 如果遇到内存不足问题：
   - 降低基线模型数量 (--baselines 3)
   - 使用更少的训练轮次
   - 如果仍然不足，可以考虑使用较小的网络架构

## 注意事项

- 首次运行时会下载CIFAR-10数据集（约160MB）
- 零知识证明生成需要额外的计算资源
- 报告和模型会保存在本地目录，可能占用较大的磁盘空间

## 示例输出

```
===== 零知识证明神经网络验证系统 =====
运行模式: full
设备: cuda
基线模型数量: 5
训练轮次: 10
批次大小: 64