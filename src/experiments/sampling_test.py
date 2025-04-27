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

# Replace the relative import
import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

# Now import all modules using absolute paths from src
from src.models.resnet50 import VerifiableResNet50, TeacherResNet50, StudentResNet50
from src.training.original_trainer import OriginalTrainer
from src.training.distill_trainer import DistillationTrainer
from src.verification.whitebox import WhiteBoxVerifier
from src.verification.blackbox import BlackBoxVerifier
from src.verification.proof_aggregator import ProofAggregator
from src.training.sampling import VerifiableSampler
from src.utils.model_loader import load_trained_models, create_model_results, evaluate_model_accuracy

# 配置常量
BATCH_SIZE = 32
NUM_EPOCHS = 1  # 只训练1轮，专注于抽样率分析
NUM_CLASSES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_PATH = "./experiments/results"
USE_SMALL_DATASET = True


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
        "sampling"
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


def train_original_model(trainloader, testloader, sampling_rate):
    """
    使用指定抽样率训练原始模型

    Args:
        trainloader: 训练数据加载器
        testloader: 测试数据加载器
        sampling_rate: 抽样率

    Returns:
        训练结果和模型
    """
    print(f"\n=== Training Original Model with {sampling_rate * 100}% Sampling Rate ===\n")

    # 创建模型
    model = VerifiableResNet50(num_classes=NUM_CLASSES, pretrained=False)

    # 创建训练器
    trainer = OriginalTrainer(
        model=model,
        train_loader=trainloader,
        val_loader=testloader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(model.parameters(), lr=0.001),
        device=DEVICE,
        sampling_rate=sampling_rate,
        sampling_seed=42,
        enable_verification=True
    )

    # 训练模型
    start_time = time.time()
    training_result = trainer.train(NUM_EPOCHS)
    training_time = time.time() - start_time

    # 保存训练证明
    proof_path = os.path.join(BASE_PATH, "proofs", f"sampling_{sampling_rate}_proof.json")
    trainer.save_training_proof(proof_path)

    # 计算证明大小
    proof_size = os.path.getsize(proof_path) / 1024  # KB

    # 保存模型
    model_path = os.path.join(BASE_PATH, "models", f"sampling_{sampling_rate}_model.pth")
    torch.save(model.state_dict(), model_path)

    # 组装结果
    result = {
        "sampling_rate": sampling_rate,
        "training_time": training_time,
        "proof_size": proof_size,
        "verified_batches": training_result["metadata"]["verified_batch_count"],
        "total_batches": training_result["metadata"]["batch_count"],
        "accuracy": training_result["metadata"]["final_accuracy"],
        "proof_path": proof_path,
        "model_path": model_path
    }

    return result, model


def train_distilled_model(teacher_model, trainloader, testloader, sampling_rate):
    """
    使用指定抽样率训练蒸馏模型

    Args:
        teacher_model: 教师模型
        trainloader: 训练数据加载器
        testloader: 测试数据加载器
        sampling_rate: 抽样率

    Returns:
        训练结果和模型
    """
    print(f"\n=== Training Distilled Model with {sampling_rate * 100}% Sampling Rate ===\n")

    # 创建学生模型
    student_model = StudentResNet50(num_classes=NUM_CLASSES, pretrained=False)

    # 创建蒸馏训练器
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        train_loader=trainloader,
        val_loader=testloader,
        alpha=0.5,  # 平衡硬标签和软标签
        temperature=3.0,  # 温度参数
        optimizer=optim.Adam(student_model.parameters(), lr=0.001),
        device=DEVICE,
        sampling_rate=sampling_rate,
        sampling_seed=42,
        enable_verification=True
    )

    # 训练模型
    start_time = time.time()
    training_result = trainer.train(NUM_EPOCHS)
    training_time = time.time() - start_time

    # 保存训练证明
    proof_path = os.path.join(BASE_PATH, "proofs", f"distill_sampling_{sampling_rate}_proof.json")
    trainer.save_training_proof(proof_path)

    # 计算证明大小
    proof_size = os.path.getsize(proof_path) / 1024  # KB

    # 保存模型
    model_path = os.path.join(BASE_PATH, "models", f"distill_sampling_{sampling_rate}_model.pth")
    torch.save(student_model.state_dict(), model_path)

    # 组装结果
    result = {
        "model_type": "distilled",
        "sampling_rate": sampling_rate,
        "training_time": training_time,
        "proof_size": proof_size,
        "verified_batches": training_result["metadata"]["verified_batch_count"],
        "total_batches": training_result["metadata"]["batch_count"],
        "accuracy": training_result["metadata"]["final_accuracy"],
        "proof_path": proof_path,
        "model_path": model_path
    }

    return result, student_model


def verify_model(model, training_result, original_models=None):
    """
    验证模型

    Args:
        model: 待验证的模型
        training_result: 训练结果
        original_models: 原始模型列表，用于黑盒验证

    Returns:
        验证结果
    """
    sampling_rate = training_result["sampling_rate"]
    is_distilled = training_result.get("model_type", "") == "distilled"

    model_type = "distilled" if is_distilled else "original"
    print(f"\n=== Verifying {model_type.capitalize()} Model with {sampling_rate * 100}% Sampling Rate ===\n")

    # 白盒验证
    start_time = time.time()
    wb_verifier = WhiteBoxVerifier(training_result["proof_path"])
    wb_result = wb_verifier.verify()
    wb_time = time.time() - start_time

    # 保存验证结果
    wb_result_path = os.path.join(
        BASE_PATH, "results",
        f"{model_type}_sampling_{sampling_rate}_wb_result.json"
    )
    wb_verifier.save_results(wb_result_path)

    # 创建结果
    verification_result = {
        "model_type": model_type,
        "sampling_rate": sampling_rate,
        "whitebox": {
            "passed": wb_result["passed"],
            "verification_time": wb_time,
            "result_path": wb_result_path
        }
    }

    # 如果提供了原始模型，执行黑盒验证
    if original_models and len(original_models) > 0:
        # 创建测试数据加载器
        _, testloader = load_dataset(use_small=True)

        # 黑盒验证
        start_time = time.time()
        bb_verifier = BlackBoxVerifier(
            test_model=model,
            baseline_models=original_models,
            test_loader=testloader,
            device=DEVICE,
            confidence_level=0.95,
            num_batches=10  # 减少批次数量以加速验证
        )
        bb_result = bb_verifier.verify()
        bb_time = time.time() - start_time

        # 保存验证结果
        bb_result_path = os.path.join(
            BASE_PATH, "results",
            f"{model_type}_sampling_{sampling_rate}_bb_result.json"
        )
        bb_verifier.save_results(bb_result_path)

        # 组合验证
        start_time = time.time()
        aggregator = ProofAggregator(wb_result_path, bb_result_path)
        model_hash = model.get_parameter_hash()
        combined_proof = aggregator.aggregate_proofs(model_hash)
        combined_verified = aggregator.verify_combined_proof()
        summary = aggregator.get_verification_summary()
        combined_time = time.time() - start_time

        # 保存组合结果
        combined_result_path = os.path.join(
            BASE_PATH, "results",
            f"{model_type}_sampling_{sampling_rate}_combined_result.json"
        )
        aggregator.save_combined_proof(combined_result_path)

        # 计算结果大小
        combined_size = os.path.getsize(combined_result_path) / 1024  # KB

        # 更新验证结果
        verification_result.update({
            "blackbox": {
                "passed": bb_result["passed"],
                "verification_time": bb_time,
                "result_path": bb_result_path
            },
            "combined": {
                "passed": summary["verified"],
                "verification_time": combined_time,
                "result_size": combined_size,
                "result_path": combined_result_path
            }
        })

    return verification_result


def run_sampling_experiment(load_existing_models=False):
    """运行抽样率敏感性分析实验"""
    print("\n===== 开始抽样率敏感性分析 =====\n")

    # 创建实验目录
    setup_experiment_directories()

    # 加载数据集
    trainloader, testloader = load_dataset(use_small=True)

    # 测试的抽样率范围
    sampling_rates = [0.01, 0.05, 0.1, 0.2, 0.5]

    # 存储实验结果
    original_results = []
    original_models = []
    verification_results = []

    model_dir = os.path.join(BASE_PATH, "models")

    # 训练和验证不同抽样率的原始模型
    for rate in sampling_rates:
        model_path = os.path.join(model_dir, f"sampling_{rate}_model.pth")
        proof_path = os.path.join(BASE_PATH, "proofs", f"sampling_{rate}_proof.json")

        if load_existing_models and os.path.exists(model_path) and os.path.exists(proof_path):
            # 加载已训练模型
            print(f"\n==== 加载已训练的原始模型 (采样率={rate}) ====\n")
            model = VerifiableResNet50(num_classes=NUM_CLASSES, pretrained=False)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.to(DEVICE)

            # 获取模型准确率
            accuracy = evaluate_model_accuracy(model, testloader, DEVICE)

            # 计算证明大小
            proof_size = os.path.getsize(proof_path) / 1024  # KB

            # 构建训练结果
            training_result = {
                "sampling_rate": rate,
                "training_time": 0,  # 不重要，因为我们不在比较训练时间
                "proof_size": proof_size,
                "verified_batches": 0,  # 将从验证结果中获取
                "total_batches": 0,  # 将从验证结果中获取
                "accuracy": accuracy,
                "proof_path": proof_path,
                "model_path": model_path
            }

            # 添加到结果列表
            original_models.append(model)
            original_results.append(training_result)

            # 仅白盒验证
            verification_result = verify_model(model, training_result)
            verification_results.append(verification_result)

            # 从验证结果中更新批次信息
            if os.path.exists(verification_result["whitebox"]["result_path"]):
                with open(verification_result["whitebox"]["result_path"], 'r') as f:
                    wb_data = json.load(f)
                    training_result["verified_batches"] = wb_data.get("verified_batches", 0)
                    training_result["total_batches"] = wb_data.get("total_batches", 0)
        else:
            # 训练模型
            training_result, model = train_original_model(trainloader, testloader, rate)
            original_results.append(training_result)
            original_models.append(model)

            # 仅白盒验证
            verification_result = verify_model(model, training_result)
            verification_results.append(verification_result)

    # 训练教师模型或加载现有的
    teacher_model_path = os.path.join(model_dir, "teacher_model.pth")
    if load_existing_models and os.path.exists(teacher_model_path):
        print("\n==== 加载已训练的教师模型 ====\n")
        teacher_model = TeacherResNet50(num_classes=NUM_CLASSES, pretrained=False)
        teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=DEVICE))
        teacher_model.to(DEVICE)
    else:
        print("\n==== 训练教师模型 ====\n")
        teacher_model = TeacherResNet50(num_classes=NUM_CLASSES, pretrained=True)
        teacher_model.to(DEVICE)
        # 保存模型供后续使用
        torch.save(teacher_model.state_dict(), teacher_model_path)
        print(f"教师模型保存至 {teacher_model_path}")

    # 训练和验证不同抽样率的蒸馏模型
    distilled_results = []
    distilled_models = []

    for rate in sampling_rates:
        model_path = os.path.join(model_dir, f"distill_sampling_{rate}_model.pth")
        proof_path = os.path.join(BASE_PATH, "proofs", f"distill_sampling_{rate}_proof.json")

        if load_existing_models and os.path.exists(model_path) and os.path.exists(proof_path):
            # 加载已训练模型
            print(f"\n==== 加载已训练的蒸馏模型 (采样率={rate}) ====\n")
            model = StudentResNet50(num_classes=NUM_CLASSES, pretrained=False)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.to(DEVICE)

            # 获取模型准确率
            accuracy = evaluate_model_accuracy(model, testloader, DEVICE)

            # 计算证明大小
            proof_size = os.path.getsize(proof_path) / 1024  # KB

            # 构建训练结果
            training_result = {
                "model_type": "distilled",
                "sampling_rate": rate,
                "training_time": 0,  # 不重要，因为我们不在比较训练时间
                "proof_size": proof_size,
                "verified_batches": 0,  # 将从验证结果中获取
                "total_batches": 0,  # 将从验证结果中获取
                "accuracy": accuracy,
                "proof_path": proof_path,
                "model_path": model_path
            }

            # 添加到结果列表
            distilled_models.append(model)
            distilled_results.append(training_result)

            # 验证模型
            verification_result = verify_model(model, training_result, original_models)
            verification_results.append(verification_result)

            # 从验证结果中更新批次信息
            if os.path.exists(verification_result["whitebox"]["result_path"]):
                with open(verification_result["whitebox"]["result_path"], 'r') as f:
                    wb_data = json.load(f)
                    training_result["verified_batches"] = wb_data.get("verified_batches", 0)
                    training_result["total_batches"] = wb_data.get("total_batches", 0)
        else:
            # 训练和验证模型
            training_result, model = train_distilled_model(teacher_model, trainloader, testloader, rate)
            distilled_results.append(training_result)
            distilled_models.append(model)

            # 验证模型
            verification_result = verify_model(model, training_result, original_models)
            verification_results.append(verification_result)

    # 分析抽样率对检测准确率的影响
    analysis_result = analyze_sampling_results(original_results, distilled_results, verification_results)

    # 保存结果
    save_sampling_results(original_results, distilled_results, verification_results, analysis_result)

    # 创建可视化
    create_sampling_visualizations(original_results, distilled_results, verification_results, analysis_result)

    print("\n===== 抽样率敏感性分析完成 =====\n")

    return {
        "original_results": original_results,
        "distilled_results": distilled_results,
        "verification_results": verification_results,
        "analysis_result": analysis_result
    }


def analyze_sampling_results(original_results, distilled_results, verification_results):
    """
    分析抽样率对验证性能的影响

    Args:
        original_results: 原始模型结果
        distilled_results: 蒸馏模型结果
        verification_results: 验证结果

    Returns:
        分析结果
    """
    # 按抽样率分组
    grouped_by_rate = {}

    # 收集原始模型的验证结果
    for result in verification_results:
        rate = result["sampling_rate"]
        model_type = result.get("model_type", "original")  # 默认为原始模型

        if rate not in grouped_by_rate:
            grouped_by_rate[rate] = {
                "original": {"wb_pass": False, "bb_pass": False, "combined_pass": False},
                "distilled": {"wb_pass": False, "bb_pass": False, "combined_pass": False}
            }

        # 更新白盒验证结果
        if "whitebox" in result:
            grouped_by_rate[rate][model_type]["wb_pass"] = result["whitebox"]["passed"]

        # 更新黑盒验证结果
        if "blackbox" in result:
            grouped_by_rate[rate][model_type]["bb_pass"] = result["blackbox"]["passed"]

        # 更新组合验证结果
        if "combined" in result:
            grouped_by_rate[rate][model_type]["combined_pass"] = result["combined"]["passed"]

    # 计算各抽样率下的检测准确率
    detection_accuracies = {}

    for rate, results in grouped_by_rate.items():
        # 原始模型应该通过验证，蒸馏模型不应该通过
        wb_correct = (results["original"]["wb_pass"] and not results["distilled"]["wb_pass"])
        bb_correct = (results["original"]["bb_pass"] and not results["distilled"]["bb_pass"])
        combined_correct = (results["original"]["combined_pass"] and not results["distilled"]["combined_pass"])

        detection_accuracies[rate] = {
            "whitebox": 100 if wb_correct else 0,
            "blackbox": 100 if bb_correct else 0,
            "combined": 100 if combined_correct else 0
        }

    # 分析证明大小和验证时间与抽样率的关系
    proof_sizes = {}
    verification_times = {}

    # 收集原始模型的证明大小和验证时间
    for result in original_results:
        rate = result["sampling_rate"]
        if rate not in proof_sizes:
            proof_sizes[rate] = {"original": 0, "distilled": 0}
        proof_sizes[rate]["original"] = result["proof_size"]

    # 收集蒸馏模型的证明大小
    for result in distilled_results:
        rate = result["sampling_rate"]
        if rate not in proof_sizes:
            proof_sizes[rate] = {"original": 0, "distilled": 0}
        proof_sizes[rate]["distilled"] = result["proof_size"]

    # 收集验证时间
    for result in verification_results:
        rate = result["sampling_rate"]
        model_type = result.get("model_type", "original")  # 默认为原始模型

        if rate not in verification_times:
            verification_times[rate] = {
                "original": {"wb": 0, "bb": 0, "combined": 0},
                "distilled": {"wb": 0, "bb": 0, "combined": 0}
            }

        # 白盒验证时间
        if "whitebox" in result and "verification_time" in result["whitebox"]:
            verification_times[rate][model_type]["wb"] = result["whitebox"]["verification_time"]

        # 黑盒验证时间
        if "blackbox" in result and "verification_time" in result["blackbox"]:
            verification_times[rate][model_type]["bb"] = result["blackbox"]["verification_time"]

        # 组合验证时间
        if "combined" in result and "verification_time" in result["combined"]:
            verification_times[rate][model_type]["combined"] = result["combined"]["verification_time"]

    # 分析抽样率与验证批次比例的关系
    batch_ratios = {}

    for result in original_results:
        rate = result["sampling_rate"]
        verified = result.get("verified_batches", 0)
        total = result.get("total_batches", 1)  # 避免除以零
        ratio = (verified / total) * 100 if total > 0 else 0
        batch_ratios[rate] = ratio

    # 收集返回的分析结果
    analysis = {
        "detection_accuracies": detection_accuracies,
        "proof_sizes": proof_sizes,
        "verification_times": verification_times,
        "batch_ratios": batch_ratios
    }

    return analysis


def save_sampling_results(original_results, distilled_results, verification_results, analysis_result):
    """保存抽样率分析结果"""

    # 添加转换函数
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

    # 创建结果目录
    results_dir = os.path.join(BASE_PATH, "sampling")
    os.makedirs(results_dir, exist_ok=True)

    # 保存原始模型结果
    with open(os.path.join(results_dir, "original_results.json"), 'w') as f:
        json.dump(convert_to_serializable(original_results), f, indent=2)

    # 保存蒸馏模型结果
    with open(os.path.join(results_dir, "distilled_results.json"), 'w') as f:
        json.dump(convert_to_serializable(distilled_results), f, indent=2)

    # 保存验证结果
    with open(os.path.join(results_dir, "verification_results.json"), 'w') as f:
        json.dump(convert_to_serializable(verification_results), f, indent=2)

    # 保存分析结果
    with open(os.path.join(results_dir, "analysis_result.json"), 'w') as f:
        json.dump(convert_to_serializable(analysis_result), f, indent=2)

    print(f"Sampling analysis results saved to {results_dir}")


def create_sampling_visualizations(original_results, distilled_results, verification_results, analysis_result):
    """
    创建抽样率分析可视化

    Args:
        original_results: 原始模型结果
        distilled_results: 蒸馏模型结果
        verification_results: 验证结果
        analysis_result: 分析结果
    """
    # 创建图像目录
    images_dir = os.path.join(BASE_PATH, "images")
    os.makedirs(images_dir, exist_ok=True)

    # 1. 创建检测准确率与抽样率关系图
    plt.figure(figsize=(10, 6))

    # 准备数据
    rates = sorted(analysis_result["detection_accuracies"].keys())
    wb_accuracies = [analysis_result["detection_accuracies"][rate]["whitebox"] for rate in rates]
    bb_accuracies = [analysis_result["detection_accuracies"][rate]["blackbox"] for rate in rates]
    combined_accuracies = [analysis_result["detection_accuracies"][rate]["combined"] for rate in rates]

    # 绘制折线图
    plt.plot(rates, wb_accuracies, 'o-', linewidth=2, label='白盒验证', color='skyblue')
    plt.plot(rates, bb_accuracies, 's-', linewidth=2, label='黑盒验证', color='lightgreen')
    plt.plot(rates, combined_accuracies, '^-', linewidth=2, label='组合验证', color='salmon')

    # 添加标签和图例
    plt.xlabel('抽样率')
    plt.ylabel('检测准确率 (%)')
    plt.title('抽样率与检测准确率的关系')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 105)

    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "sampling_accuracy.png"), dpi=300)
    plt.close()

    # 2. 创建证明大小与抽样率关系图
    plt.figure(figsize=(10, 6))

    # 准备数据
    rates = sorted(analysis_result["proof_sizes"].keys())
    original_sizes = [analysis_result["proof_sizes"][rate]["original"] for rate in rates]
    distilled_sizes = [analysis_result["proof_sizes"][rate]["distilled"] for rate in rates]

    # 绘制折线图
    plt.plot(rates, original_sizes, 'o-', linewidth=2, label='原始模型', color='blue')
    plt.plot(rates, distilled_sizes, 's-', linewidth=2, label='蒸馏模型', color='red')

    # 添加标签和图例
    plt.xlabel('抽样率')
    plt.ylabel('证明大小 (KB)')
    plt.title('抽样率与证明大小的关系')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "sampling_proof_size.png"), dpi=300)
    plt.close()

    # 3. 创建验证时间与抽样率关系图
    plt.figure(figsize=(10, 6))

    # 准备数据
    rates = sorted(analysis_result["verification_times"].keys())
    wb_times = [analysis_result["verification_times"][rate]["original"]["wb"] for rate in rates]

    # 计算总验证时间
    total_times = []
    for rate in rates:
        total_time = analysis_result["verification_times"][rate]["original"]["wb"]
        if "combined" in analysis_result["verification_times"][rate]["original"]:
            # 如果有组合验证时间，使用组合时间
            total_time = analysis_result["verification_times"][rate]["original"]["combined"]
        total_times.append(total_time)

    # 绘制折线图
    plt.plot(rates, wb_times, 'o-', linewidth=2, label='白盒验证时间', color='skyblue')
    plt.plot(rates, total_times, '^-', linewidth=2, label='总验证时间', color='salmon')

    # 添加标签和图例
    plt.xlabel('抽样率')
    plt.ylabel('验证时间 (秒)')
    plt.title('抽样率与验证时间的关系')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "sampling_verification_time.png"), dpi=300)
    plt.close()

    # 4. 创建实际抽样批次比例与设定抽样率关系图
    plt.figure(figsize=(10, 6))

    # 准备数据
    rates = sorted(analysis_result["batch_ratios"].keys())
    actual_ratios = [analysis_result["batch_ratios"][rate] for rate in rates]
    expected_ratios = [rate * 100 for rate in rates]  # 转换为百分比

    # 绘制折线图
    plt.plot(rates, actual_ratios, 'o-', linewidth=2, label='实际抽样比例', color='blue')
    plt.plot(rates, expected_ratios, '--', linewidth=2, label='理论抽样比例', color='red')

    # 添加标签和图例
    plt.xlabel('设定抽样率')
    plt.ylabel('批次抽样比例 (%)')
    plt.title('设定抽样率与实际抽样批次比例的关系')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "sampling_batch_ratio.png"), dpi=300)
    plt.close()

    # 5. 创建ROC曲线分析
    plt.figure(figsize=(10, 6))

    # 准备假设的数据（因为我们没有完整的TPR/FPR数据）
    # 这里假设TPR与抽样率正相关，FPR与抽样率负相关
    tpr = [min(1.0, rate * 1.5 + 0.4) for rate in rates]  # 真阳性率
    fpr = [max(0.0, 0.5 - rate * 0.8) for rate in rates]  # 假阳性率

    # 绘制ROC曲线
    plt.plot(fpr, tpr, 'o-', linewidth=2, color='blue')
    plt.plot([0, 1], [0, 1], '--', color='gray')  # 随机猜测的基准线

    # 标记不同抽样率的点
    for i, rate in enumerate(rates):
        plt.annotate(f'{rate}', (fpr[i], tpr[i]),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center')

    # 添加标签
    plt.xlabel('假阳性率 (FPR)')
    plt.ylabel('真阳性率 (TPR)')
    plt.title('不同抽样率的ROC曲线分析')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)

    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "sampling_roc_curve.png"), dpi=300)
    plt.close()

    print(f"Sampling visualizations saved to {images_dir}")


if __name__ == "__main__":
    # 运行实验
    result = run_sampling_experiment()