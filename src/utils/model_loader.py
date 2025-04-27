import os
import torch
import torch.nn as nn
from ..models.resnet50 import VerifiableResNet50, TeacherResNet50, StudentResNet50


def load_trained_models(model_dir, model_type="original", num_models=5, num_classes=10,
                        device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    加载已训练好的模型

    Args:
        model_dir: 保存模型的目录
        model_type: 模型类型 ('original', 'distilled', 'disguised', 'baseline')
        num_models: 要加载的模型数量
        num_classes: 类别数量
        device: 计算设备

    Returns:
        模型列表
    """
    models = []

    for i in range(1, num_models + 1):
        model_path = os.path.join(model_dir, f"{model_type}_model_{i}.pth")

        if not os.path.exists(model_path):
            print(f"警告: 无法找到模型 {model_path}")
            continue

        # 创建模型实例
        if model_type == "original" or model_type == "baseline":  # Add baseline here
            model = VerifiableResNet50(num_classes=num_classes, pretrained=False)
        elif model_type == "teacher":
            model = TeacherResNet50(num_classes=num_classes, pretrained=False)
        elif model_type == "distilled" or model_type == "student":
            model = StudentResNet50(num_classes=num_classes, pretrained=False)
        elif model_type == "disguised":
            model = VerifiableResNet50(num_classes=num_classes, pretrained=False)
        else:
            raise ValueError(f"未知的模型类型: {model_type}")

        # 加载权重
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()  # 设置为评估模式

        models.append(model)
        print(f"已加载模型: {model_path}")

    return models


def create_model_results(models, model_type, base_path):
    """
    为加载的模型创建结果数据结构

    Args:
        models: 模型列表
        model_type: 模型类型
        base_path: 基础路径

    Returns:
        模型结果列表
    """
    results = []

    for i, model in enumerate(models):
        result = {
            "model_type": model_type,
            "model_id": i + 1,
            "accuracy": None,  # 可以通过评估模型来填充
            "proof_path": os.path.join(base_path, "proofs", f"{model_type}_model_{i + 1}_proof.json"),
            "model_path": os.path.join(base_path, "models", f"{model_type}_model_{i + 1}.pth")
        }

        # 如果是蒸馏模型，添加温度参数 (占位)
        if model_type == "distilled":
            result["temperature"] = None

        # 如果是伪装模型，添加伪装率 (占位)
        if model_type == "disguised":
            result["disguise_rate"] = None

        results.append(result)

    return results


def evaluate_model_accuracy(model, testloader, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    评估模型准确率

    Args:
        model: 要评估的模型
        testloader: 测试数据加载器
        device: 计算设备

    Returns:
        准确率 (%)
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    return accuracy