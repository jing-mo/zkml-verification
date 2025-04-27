# ZKML验证系统
基于zkp的零知识验证模块代码
代码还在完善中，目前基础功能已实现
## 项目概述
ZKML验证系统是一个用于神经网络模型验证的零知识证明框架。该系统能够验证模型是否为蒸馏模型，并生成零知识证明，确保模型的可信度和安全性。系统支持黑盒验证、白盒验证和零知识证明验证三种验证方式，可以有效地检测模型训练过程中的异常行为。

## 项目结构
```plaintext
src/
├── experiments/                # 实验相关代码
│   ├── attack_test.py         # 攻击测试
│   ├── run_experiments.py     # 运行实验
│   ├── run_verification.py    # 运行验证
│   ├── sampling_test.py       # 抽样测试
│   └── validity_test.py       # 有效性测试
├── models/                     # 模型定义
│   ├── resnet50.py            # ResNet50模型
│   └── ...
├── new_zkp_verify/            # 核心验证模块
│   ├── config.py              # 配置模块
│   ├── main.py                # 主模块
│   ├── models.py              # 模型模块
│   ├── train.py               # 训练模块
│   ├── verify.py              # 验证模块
│   ├── zkp.py                 # 零知识证明模块
│   ├── report.py              # 报告模块
│   ├── run.sh                 # 运行脚本
│   └── ...
├── utils/                      # 工具函数
│   ├── data_utils.py          # 数据处理工具
│   ├── json_utils.py          # JSON处理工具
│   └── ...
└── zkp_nn_verification.py     # 零知识证明验证
 ```
```

## 安装指南
### 环境要求
- Python 3.8+
- PyTorch 1.7.0+
- CUDA 10.2+（可选，用于GPU加速）


## 使用方法
### 基本用法
```bash
python -m src.new_zkp_verify.main
 ```

### 参数说明 基本参数
- --batch-size ：批次大小，默认为128
- --epochs ：训练轮次，默认为50
- --lr ：学习率，默认为0.1
- --seed ：随机种子，默认为42
- --num-baseline-models ：基线模型数量，默认为5 蒸馏参数
- --train-distilled ：是否训练蒸馏模型，默认为False
- --alpha ：软标签权重，默认为0.5
- --temperature ：温度参数，默认为4.0 验证参数
- --confidence-level ：置信水平，默认为0.95
- --num-samples ：抽样数量，默认为10 其他参数
- --cpu ：强制使用CPU，默认为False
- --mixed-precision ：是否使用混合精度训练，默认为False
- --num-workers ：数据加载线程数，默认为4
- --zkp-sample-rate ：训练批次ZKP抽样率，默认为0.1（即10%）
- --force-recompile-circuit ：是否强制重新编译ZKP电路，默认为False
### 使用示例
训练基线模型并验证
```bash
python -m src.new_zkp_verify.main --num-baseline-models 3 --epochs 10
 ```
```
```

## 系统架构
### 工作流程
1. 训练阶段
   
   - 训练多个基线模型
   - 选择一个基线模型作为教师模型
   - 训练蒸馏模型（可选）
2. 验证阶段
   
   - 黑盒验证：计算目标模型与基线模型的KL散度
   - 白盒验证：分析模型梯度来源
   - 零知识证明验证：生成和验证零知识证明
3. 报告阶段
   
   - 生成JSON报告
   - 生成HTML报告
   - 生成Markdown报告
   - 生成性能报告
### 模块说明
- config.py ：配置模块，包含全局配置参数
- models.py ：模型模块，包含神经网络模型定义
- train.py ：训练模块，用于训练模型
- verify.py ：验证模块，用于验证模型是否为蒸馏模型
- zkp.py ：零知识证明模块，用于生成和验证零知识证明
- report.py ：报告模块，用于生成实验报告
- main.py ：主模块，用于运行零知识神经网络验证系统
## 验证方法详解
### 黑盒验证
黑盒验证通过计算目标模型与基线模型的KL散度来判断目标模型是否为蒸馏模型。系统会计算目标模型与每个基线模型的KL散度，并与基线模型之间的KL散度进行比较。如果目标模型的KL散度显著低于基线模型之间的KL散度，则认为目标模型可能是蒸馏模型。

```python
# 黑盒验证示例代码
verification_result = verifier.verify_black_box(
    target_model=target_model,
    baseline_models=baseline_models,
    test_loader=test_loader,
    confidence_level=0.95
)
 ```
```

### 白盒验证
白盒验证通过分析模型训练过程中的梯度来源，判断模型是否受到教师模型的影响。系统会在训练过程中抽样一定比例的批次，记录梯度信息，并生成白盒验证证明。

```python
# 白盒验证示例代码
verification_result = verifier.verify_white_box(
    model=model,
    proof_path="path/to/proof.json"
)
 ```
```

### 零知识证明验证
零知识证明验证通过生成和验证零知识证明，确保模型的可信度。系统使用Circom和SnarkJS生成和验证零知识证明，支持KL散度证明和批次证明。

```python
# 零知识证明验证示例代码
verification_result = verifier.verify_with_zkp(
    target_model=target_model,
    baseline_models=baseline_models,
    test_loader=test_loader,
    num_samples=10
)
 ```
```

## 实验模块
实验模块提供了多种测试和验证方法，包括：

- attack_test.py ：测试系统对各种攻击的抵抗能力
- validity_test.py ：测试系统验证方法的有效性
- sampling_test.py ：测试不同抽样率对验证结果的影响
- run_experiments.py ：运行完整的实验流程
- run_verification.py ：运行验证流程
