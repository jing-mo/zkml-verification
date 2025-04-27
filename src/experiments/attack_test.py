import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional
import copy

from ..models.resnet50 import VerifiableResNet50, TeacherResNet50, StudentResNet50
from ..training.original_trainer import OriginalTrainer
from ..training.distill_trainer import DistillationTrainer
from ..verification.whitebox import WhiteBoxVerifier
from ..verification.blackbox import BlackBoxVerifier
from ..verification.proof_aggregator import ProofAggregator
from ..training.sampling import VerifiableSampler
from ..utils.model_loader import load_trained_models, create_model_results, evaluate_model_accuracy

# 配置常量
BATCH_SIZE = 32  # 减小批次大小
NUM_EPOCHS = 2  # 减少训练轮数
NUM_CLASSES = 10  # CIFAR-10类别数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_PATH = "./experiments/results"
USE_SMALL_DATASET = True  # 使用较小的数据集以加速训练


class GradientTamperingAttack:
    """
    梯度篡改攻击，尝试隐藏教师梯度贡献
    """

    def __init__(self, teacher_model, student_model, tampering_rate=0.5):
        """
        初始化攻击

        Args:
            teacher_model: 教师模型
            student_model: 学生模型
            tampering_rate: 篡改率，控制篡改强度
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.tampering_rate = tampering_rate

        # 保存原始方法引用，以便后续恢复
        self.original_backward = torch.autograd.backward

    def tampered_backward(self, tensors, grad_tensors=None, retain_graph=None, create_graph=False, inputs=None,
                          allow_unused=False):
        """
        篡改后的反向传播
        """
        try:
            # 如果inputs是布尔值，我们需要完全避免使用它
            if isinstance(inputs, bool):
                # 调用原始backward函数但不传入inputs参数
                self.original_backward(tensors, grad_tensors, retain_graph, create_graph, None, allow_unused)
            else:
                # 正常调用，使用提供的inputs
                self.original_backward(tensors, grad_tensors, retain_graph, create_graph, inputs, allow_unused)

            # 如果是学生模型的梯度，尝试篡改
            if isinstance(tensors, torch.Tensor) and hasattr(tensors, 'grad') and tensors.grad is not None:
                # 获取当前梯度
                current_grad = tensors.grad.clone()

                # 生成一些随机噪声来混淆教师信号
                noise = torch.randn_like(current_grad) * self.tampering_rate

                # 应用篡改
                tensors.grad = current_grad + noise
        except Exception as e:
            print(f"处理反向传播时出错: {e}")
            print("尝试使用更简单的反向传播调用...")
            # 最简单的调用方式，不使用可能导致问题的参数
            self.original_backward(tensors)

    def convert_to_serializable(obj):
        """将NumPy类型转换为可JSON序列化的Python原生类型"""
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    def activate(self):
        """激活攻击"""
        # 将反向传播函数替换为篡改版本
        torch.autograd.backward = self.tampered_backward

    def deactivate(self):
        """停用攻击"""
        # 恢复原始反向传播函数
        torch.autograd.backward = self.original_backward


class OutputDistributionAttack:
    """
    输出分布攻击，尝试调整模型输出分布以欺骗KL检测
    """

    def __init__(self, model, target_distribution='uniform', strength=0.3):
        """
        初始化攻击

        Args:
            model: 目标模型
            target_distribution: 目标分布类型，'uniform'为均匀分布
            strength: 攻击强度，0到1之间
        """
        self.model = model
        self.target_distribution = target_distribution
        self.strength = strength

        # 保存原始前向传播方法
        self.original_forward = model.forward

    def tampered_forward(self, x):
        """
        篡改后的前向传播
        """
        # 执行原始前向传播
        original_output = self.original_forward(x)

        # 创建目标分布
        if self.target_distribution == 'uniform':
            # 均匀分布
            target_output = torch.ones_like(original_output) / original_output.size(1)
        else:
            # 默认使用均匀分布
            target_output = torch.ones_like(original_output) / original_output.size(1)

        # 混合原始输出和目标分布
        tampered_output = (1 - self.strength) * original_output + self.strength * target_output

        return tampered_output

    def activate(self):
        """激活攻击"""
        # 将前向传播方法替换为篡改版本
        self.model.forward = self.tampered_forward.__get__(self.model, type(self.model))

    def deactivate(self):
        """停用攻击"""
        # 恢复原始前向传播方法
        self.model.forward = self.original_forward


class SamplingEvaderAttack:
    """
    抽样规避攻击，尝试预测抽样批次并特殊处理
    """

    def __init__(self, trainer, prediction_accuracy=0.8):
        """
        初始化攻击

        Args:
            trainer: 训练器实例
            prediction_accuracy: 预测抽样的准确率
        """
        self.trainer = trainer
        self.prediction_accuracy = prediction_accuracy

        # 保存原始抽样方法
        self.original_should_sample = trainer.sampler.should_sample

        # 创建篡改后的抽样决策字典
        self.tampered_decisions = {}

    def predict_sampling(self, batch_idx):
        """
        预测某批次是否会被抽样
        """
        # 获取原始抽样决策
        original_decision = self.original_should_sample(batch_idx)

        # 根据预测准确率决定是否返回正确结果
        if np.random.random() < self.prediction_accuracy:
            # 正确预测
            return original_decision
        else:
            # 错误预测
            return not original_decision

    def tampered_should_sample(self, batch_idx):
        """
        篡改后的抽样决策方法
        """
        # 如果已经为此批次做出决定，返回之前的结果
        if batch_idx in self.tampered_decisions:
            return self.tampered_decisions[batch_idx]

        # 预测抽样决策
        decision = self.predict_sampling(batch_idx)

        # 存储决定
        self.tampered_decisions[batch_idx] = decision

        return decision

    def activate(self):
        """激活攻击"""
        # 将抽样决策方法替换为篡改版本
        self.trainer.sampler.should_sample = self.tampered_should_sample.__get__(self.trainer.sampler,
                                                                                 type(self.trainer.sampler))

    def deactivate(self):
        """停用攻击"""
        # 恢复原始抽样决策方法
        self.trainer.sampler.should_sample = self.original_should_sample


def setup_experiment_directories():
    """创建实验结果目录"""
    # 创建基础目录
    os.makedirs(BASE_PATH, exist_ok=True)

    # 创建子目录
    dirs = [
        "models",
        "proofs",
        "results",
        "images",
        "attacks"
    ]

    for d in dirs:
        os.makedirs(os.path.join(BASE_PATH, d), exist_ok=True)

    print(f"Created experiment directories at {BASE_PATH}")


def load_dataset(use_small=USE_SMALL_DATASET):
    """
    加载CIFAR-10数据集

    Args:
        use_small: 是否使用较小的数据集

    Returns:
        训练和测试数据加载器
    """
    import torchvision
    import torchvision.transforms as transforms

    # 定义数据转换
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 加载数据集
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    # 如果需要使用较小的数据集，则使用子集
    if use_small:
        # 训练集使用10%的数据
        train_size = int(0.1 * len(trainset))
        indices = torch.randperm(len(trainset))[:train_size]
        trainset = torch.utils.data.Subset(trainset, indices)

        # 测试集使用20%的数据
        test_size = int(0.2 * len(testset))
        indices = torch.randperm(len(testset))[:test_size]
        testset = torch.utils.data.Subset(testset, indices)

    # 创建数据加载器 - 将num_workers改为0以避免多进程问题
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Loaded dataset: {len(trainset)} training samples, {len(testset)} test samples")

    return trainloader, testloader


def train_model_with_attack(model, trainloader, testloader, attack_type, attack_params):
    """
    使用特定攻击训练模型

    Args:
        model: 待训练的模型
        trainloader: 训练数据加载器
        testloader: 测试数据加载器
        attack_type: 攻击类型
        attack_params: 攻击参数

    Returns:
        训练结果
    """
    print(f"\n=== Training Model with {attack_type} Attack ===\n")

    # 初始化训练器
    trainer = OriginalTrainer(
        model=model,
        train_loader=trainloader,
        val_loader=testloader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(model.parameters(), lr=0.001),
        device=DEVICE,
        sampling_rate=0.2,  # 提高抽样率以便更好地测试攻击
        sampling_seed=42,
        enable_verification=True
    )

    # 初始化攻击
    if attack_type == "gradient_tampering":
        teacher_model = TeacherResNet50(num_classes=NUM_CLASSES, pretrained=True)
        teacher_model.to(DEVICE)
        attack = GradientTamperingAttack(
            teacher_model=teacher_model,
            student_model=model,
            tampering_rate=attack_params.get("tampering_rate", 0.5)
        )
    elif attack_type == "output_distribution":
        attack = OutputDistributionAttack(
            model=model,
            target_distribution=attack_params.get("target_distribution", "uniform"),
            strength=attack_params.get("strength", 0.3)
        )
    elif attack_type == "sampling_evader":
        attack = SamplingEvaderAttack(
            trainer=trainer,
            prediction_accuracy=attack_params.get("prediction_accuracy", 0.8)
        )
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

    # 激活攻击
    attack.activate()

    # 训练模型
    try:
        training_result = trainer.train(NUM_EPOCHS)
    finally:
        # 确保攻击被停用
        attack.deactivate()

    # 保存训练证明
    proof_path = os.path.join(BASE_PATH, "proofs", f"{attack_type}_attack_proof.json")
    trainer.save_training_proof(proof_path)

    # 保存模型
    model_path = os.path.join(BASE_PATH, "models", f"{attack_type}_attack_model.pth")
    torch.save(model.state_dict(), model_path)

    # 返回结果
    result = {
        "attack_type": attack_type,
        "attack_params": attack_params,
        "accuracy": training_result["metadata"]["final_accuracy"],
        "proof_path": proof_path,
        "model_path": model_path
    }

    return result, trainer, model


def verify_attacked_model(model, attack_result, trainloader, testloader, original_models):
    """
    验证受攻击模型

    Args:
        model: 受攻击的模型
        attack_result: 攻击训练结果
        trainloader: 训练数据加载器
        testloader: 测试数据加载器
        original_models: 原始模型列表，用于黑盒验证

    Returns:
        验证结果
    """
    attack_type = attack_result["attack_type"]
    print(f"\n=== Verifying Model Under {attack_type} Attack ===\n")

    # 白盒验证
    print("Performing WhiteBox Verification...")
    wb_verifier = WhiteBoxVerifier(attack_result["proof_path"])
    wb_result = wb_verifier.verify()
    wb_result_path = os.path.join(BASE_PATH, "results", f"{attack_type}_attack_whitebox_result.json")
    wb_verifier.save_results(wb_result_path)

    # 黑盒验证
    print("Performing BlackBox Verification...")
    bb_verifier = BlackBoxVerifier(
        test_model=model,
        baseline_models=original_models,
        test_loader=testloader,
        device=DEVICE,
        confidence_level=0.95,
        num_batches=10  # 减少批次数量以加速验证
    )
    bb_result = bb_verifier.verify()
    bb_result_path = os.path.join(BASE_PATH, "results", f"{attack_type}_attack_blackbox_result.json")
    bb_verifier.save_results(bb_result_path)

    # 组合验证
    print("Performing Combined Verification...")
    aggregator = ProofAggregator(wb_result_path, bb_result_path)
    model_hash = model.get_parameter_hash()
    combined_proof = aggregator.aggregate_proofs(model_hash)
    combined_verified = aggregator.verify_combined_proof()
    summary = aggregator.get_verification_summary()
    combined_result_path = os.path.join(BASE_PATH, "results", f"{attack_type}_attack_combined_result.json")
    aggregator.save_combined_proof(combined_result_path)

    # 收集结果
    verification_result = {
        "attack_type": attack_type,
        "whitebox": {
            "passed": wb_result["passed"],
            "result_path": wb_result_path
        },
        "blackbox": {
            "passed": bb_result["passed"],
            "result_path": bb_result_path
        },
        "combined": {
            "passed": summary["verified"],
            "result_path": combined_result_path,
            "summary": summary
        }
    }

    return verification_result


def run_attack_experiment(load_existing_models=False):
    """
    运行抗攻击实验

    Args:
        load_existing_models: 是否加载已有模型而不是重新训练

    Returns:
        实验结果
    """
    print("\n===== 开始抗攻击实验 =====\n")

    # 创建实验目录
    setup_experiment_directories()

    # 加载数据集
    trainloader, testloader = load_dataset(use_small=True)

    # 如果要加载已有模型
    if load_existing_models:
        print("\n==== 加载已训练的原始模型 ====\n")
        original_models = load_trained_models(
            model_dir=os.path.join(BASE_PATH, "models"),
            model_type="original",
            num_models=2
        )

        # 如果没有找到已训练的模型，则训练新模型
        if len(original_models) == 0:
            print("没有找到已训练的原始模型，将重新训练...")
            original_models = []
            for i in range(2):
                model = VerifiableResNet50(num_classes=NUM_CLASSES, pretrained=False)
                # 创建损失函数和优化器
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                trainer = OriginalTrainer(
                    model=model,
                    train_loader=trainloader,
                    val_loader=testloader,
                    criterion=criterion,  # 添加这个参数
                    optimizer=optimizer,  # 添加这个参数
                    device=DEVICE,
                    sampling_rate=0.1,
                    enable_verification=True
                )
                print(f"Training original model {i + 1}/2...")
                trainer.train(NUM_EPOCHS)
                original_models.append(model)

                # 保存模型
                model_path = os.path.join(BASE_PATH, "models", f"original_model_{i + 1}.pth")
                torch.save(model.state_dict(), model_path)
                print(f"Saved model to {model_path}")
    else:
        # 如果没有提供原始模型，训练一些
        original_models = []
        for i in range(2):  # 只训练2个基准模型以节省时间
            model = VerifiableResNet50(num_classes=NUM_CLASSES, pretrained=False)
            trainer = OriginalTrainer(
                model=model,
                train_loader=trainloader,
                val_loader=testloader,
                device=DEVICE,
                sampling_rate=0.1,
                enable_verification=True
            )
            print(f"Training original model {i + 1}/2...")
            trainer.train(NUM_EPOCHS)
            original_models.append(model)

            # 保存模型
            model_path = os.path.join(BASE_PATH, "models", f"original_model_{i + 1}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Saved model to {model_path}")

    # 定义攻击类型和参数
    attacks = [
        {
            "type": "gradient_tampering",
            "params": {"tampering_rate": 0.5},
            "description": "梯度篡改攻击 (篡改率=0.5)"
        },
        {
            "type": "output_distribution",
            "params": {"strength": 0.3},
            "description": "输出分布攻击 (强度=0.3)"
        },
        {
            "type": "sampling_evader",
            "params": {"prediction_accuracy": 0.8},
            "description": "抽样规避攻击 (预测准确率=0.8)"
        }
    ]

    # 存储实验结果
    attack_results = []
    verification_results = []
    attack_models = []

    # 针对每种攻击类型进行实验
    for attack in attacks:
        attack_type = attack["type"]

        # 检查是否有已训练的攻击模型
        if load_existing_models:
            print(f"\n==== 检查已训练的 {attack_type} 攻击模型 ====\n")
            attack_model_path = os.path.join(BASE_PATH, "models", f"{attack_type}_attack_model.pth")
            proof_path = os.path.join(BASE_PATH, "proofs", f"{attack_type}_attack_proof.json")

            if os.path.exists(attack_model_path) and os.path.exists(proof_path):
                print(f"找到已训练的 {attack_type} 攻击模型，直接加载...")
                model = VerifiableResNet50(num_classes=NUM_CLASSES, pretrained=False)
                model.load_state_dict(torch.load(attack_model_path, map_location=DEVICE))
                model.to(DEVICE)
                model.eval()

                attack_result = {
                    "attack_type": attack_type,
                    "attack_params": attack["params"],
                    "accuracy": None,  # 可以通过评估获取
                    "proof_path": proof_path,
                    "model_path": attack_model_path
                }

                # 评估准确率
                accuracy = evaluate_model_accuracy(model, testloader, DEVICE)
                attack_result["accuracy"] = accuracy
                print(f"{attack_type} 攻击模型准确率: {accuracy:.2f}%")

                attack_results.append(attack_result)
                attack_models.append(model)
                continue

        # 创建新模型
        model = VerifiableResNet50(num_classes=NUM_CLASSES, pretrained=False)

        # 使用攻击训练模型
        attack_result, trainer, trained_model = train_model_with_attack(
            model=model,
            trainloader=trainloader,
            testloader=testloader,
            attack_type=attack["type"],
            attack_params=attack["params"]
        )

        # 添加描述
        attack_result["description"] = attack["description"]

        # 验证模型
        verification_result = verify_attacked_model(
            model=trained_model,
            attack_result=attack_result,
            trainloader=trainloader,
            testloader=testloader,
            original_models=original_models
        )

        # 添加描述
        verification_result["description"] = attack["description"]

        # 存储结果
        attack_results.append(attack_result)
        verification_results.append(verification_result)
        attack_models.append(trained_model)

    # 生成对抗成功率分析
    success_rates = analyze_attack_success(verification_results)

    # 保存实验结果
    save_attack_experiment_results(attack_results, verification_results, success_rates)

    # 创建可视化
    create_attack_visualizations(attack_results, verification_results, success_rates)

    print("\n===== 抗攻击实验完成 =====\n")

    return {
        "attack_results": attack_results,
        "verification_results": verification_results,
        "success_rates": success_rates,
        "attack_models": attack_models
    }


def analyze_attack_success(verification_results):
    """
    分析攻击成功率

    Args:
        verification_results: 验证结果列表

    Returns:
        攻击成功率分析
    """
    success_rates = {
        "whitebox": {},
        "blackbox": {},
        "combined": {}
    }

    for verification in verification_results:
        attack_type = verification["attack_type"]

        # 白盒验证
        if verification["whitebox"]["passed"]:
            # 如果验证通过，说明攻击失败
            success_rates["whitebox"][attack_type] = 0.0
        else:
            # 如果验证失败，说明攻击成功
            success_rates["whitebox"][attack_type] = 1.0

        # 黑盒验证
        if verification["blackbox"]["passed"]:
            success_rates["blackbox"][attack_type] = 0.0
        else:
            success_rates["blackbox"][attack_type] = 1.0

        # 组合验证
        if verification["combined"]["passed"]:
            success_rates["combined"][attack_type] = 0.0
        else:
            success_rates["combined"][attack_type] = 1.0

    return success_rates


def save_attack_experiment_results(attack_results, verification_results, success_rates):
    """
    保存攻击实验结果

    Args:
        attack_results: 攻击训练结果
        verification_results: 验证结果
        success_rates: 攻击成功率
    """
    # 创建结果目录
    results_dir = os.path.join(BASE_PATH, "attacks")
    os.makedirs(results_dir, exist_ok=True)

    # 转换为可序列化对象
    serializable_attack_results = convert_to_serializable(attack_results)
    serializable_verification_results = convert_to_serializable(verification_results)
    serializable_success_rates = convert_to_serializable(success_rates)

    # 保存攻击训练结果
    with open(os.path.join(results_dir, "attack_training_results.json"), 'w') as f:
        json.dump(serializable_attack_results, f, indent=2)

    # 保存验证结果
    with open(os.path.join(results_dir, "attack_verification_results.json"), 'w') as f:
        json.dump(serializable_verification_results, f, indent=2)

    # 保存攻击成功率
    with open(os.path.join(results_dir, "attack_success_rates.json"), 'w') as f:
        json.dump(serializable_success_rates, f, indent=2)

    print(f"Attack experiment results saved to {results_dir}")

def convert_to_serializable(obj):
    """将NumPy类型转换为可JSON序列化的Python原生类型"""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def create_attack_visualizations(attack_results, verification_results, success_rates):
    """
    创建攻击实验可视化

    Args:
        attack_results: 攻击训练结果
        verification_results: 验证结果
        success_rates: 攻击成功率
    """
    # 创建图像目录
    images_dir = os.path.join(BASE_PATH, "images")
    os.makedirs(images_dir, exist_ok=True)

    # 1. 创建攻击成功率柱状图
    plt.figure(figsize=(12, 6))

    # 提取数据
    attack_types = list(success_rates["whitebox"].keys())

    # 创建默认描述，以防attack_results中缺少description
    attack_descriptions = []
    for attack_type in attack_types:
        # 尝试从attack_results中找到对应的描述
        description = None
        for attack in attack_results:
            if attack["attack_type"] == attack_type and "description" in attack:
                description = attack["description"]
                break

        # 如果找不到描述，创建一个默认描述
        if description is None:
            if attack_type == "gradient_tampering":
                description = "梯度篡改攻击 (篡改率=0.5)"
            elif attack_type == "output_distribution":
                description = "输出分布攻击 (强度=0.3)"
            elif attack_type == "sampling_evader":
                description = "抽样规避攻击 (预测准确率=0.8)"
            else:
                description = attack_type

        attack_descriptions.append(description)

    wb_rates = [success_rates["whitebox"].get(at, 0) for at in attack_types]
    bb_rates = [success_rates["blackbox"].get(at, 0) for at in attack_types]
    combined_rates = [success_rates["combined"].get(at, 0) for at in attack_types]

    # 设置X轴位置
    x = np.arange(len(attack_types))
    width = 0.25

    # 绘制柱状图
    plt.bar(x - width, wb_rates, width, label='白盒验证', color='skyblue')
    plt.bar(x, bb_rates, width, label='黑盒验证', color='lightgreen')
    plt.bar(x + width, combined_rates, width, label='组合验证', color='salmon')

    # 添加标签和图例
    plt.xlabel('攻击类型')
    plt.ylabel('攻击成功率')
    plt.title('不同验证方法对抗各类攻击的成功率')
    plt.xticks(x, attack_descriptions)
    plt.ylim(0, 1.1)
    plt.legend()

    # 为每个柱子添加标签
    for i, v in enumerate(wb_rates):
        plt.text(i - width, v + 0.05, f'{v:.1f}', ha='center')
    for i, v in enumerate(bb_rates):
        plt.text(i, v + 0.05, f'{v:.1f}', ha='center')
    for i, v in enumerate(combined_rates):
        plt.text(i + width, v + 0.05, f'{v:.1f}', ha='center')

    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "attack_success_rates.png"), dpi=300)
    plt.close()

    # 2. 创建攻击对比雷达图
    attack_descriptions = []
    for attack_type in attack_types:
        # 尝试从attack_results中找到对应的描述
        description = None
        for attack in attack_results:
            if attack["attack_type"] == attack_type and "description" in attack:
                description = attack["description"]
                break

        # 如果找不到描述，创建一个默认描述
        if description is None:
            if attack_type == "gradient_tampering":
                description = "梯度篡改攻击"
            elif attack_type == "output_distribution":
                description = "输出分布攻击"
            elif attack_type == "sampling_evader":
                description = "抽样规避攻击"
            else:
                description = attack_type

        attack_descriptions.append(description)

    # 提取各指标
    # 为了简化，我们使用攻击成功率作为主要指标
    wb_defense = [1 - success_rates["whitebox"].get(at, 0) for at in attack_types]
    bb_defense = [1 - success_rates["blackbox"].get(at, 0) for at in attack_types]
    combined_defense = [1 - success_rates["combined"].get(at, 0) for at in attack_types]

    # 计算角度
    angles = np.linspace(0, 2 * np.pi, len(attack_types), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    wb_defense += wb_defense[:1]
    bb_defense += bb_defense[:1]
    combined_defense += combined_defense[:1]
    attack_descriptions += attack_descriptions[:1]

    # 创建雷达图
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

    ax.plot(angles, wb_defense, 'o-', linewidth=2, label='白盒验证', color='skyblue')
    ax.plot(angles, bb_defense, 'o-', linewidth=2, label='黑盒验证', color='lightgreen')
    ax.plot(angles, combined_defense, 'o-', linewidth=2, label='组合验证', color='salmon')

    ax.set_thetagrids(np.degrees(angles[:-1]), attack_descriptions[:-1])
    ax.set_ylim(0, 1)
    ax.set_title('不同验证方法的防御能力对比', fontsize=15)
    ax.legend(loc='upper right')

    ax.grid(True)

    # 保存雷达图
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "attack_defense_radar.png"), dpi=300)
    plt.close()

    print(f"Attack visualizations saved to {images_dir}")


if __name__ == "__main__":
    # 运行实验
    result = run_attack_experiment()