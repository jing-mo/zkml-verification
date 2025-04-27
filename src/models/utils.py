import torch
import torch.nn as nn
import numpy as np
import hashlib
from scipy.special import kl_div
from typing import Dict, List, Tuple, Any, Optional


def calculate_model_hash(model: nn.Module) -> str:
    """
    计算模型架构和参数的哈希值

    Args:
        model: PyTorch模型

    Returns:
        模型哈希值的十六进制字符串
    """
    # 创建一个哈希对象
    hasher = hashlib.sha256()

    # 添加模型架构信息
    hasher.update(str(model).encode())

    # 添加每个参数的值
    for name, param in model.named_parameters():
        if param.requires_grad:
            hasher.update(param.data.cpu().numpy().tobytes())

    return hasher.hexdigest()


def calculate_batch_hash(batch_data: torch.Tensor, batch_labels: torch.Tensor) -> str:
    """
    计算数据批次的哈希值

    Args:
        batch_data: 输入数据批次
        batch_labels: 对应的标签

    Returns:
        批次哈希值的十六进制字符串
    """
    hasher = hashlib.sha256()

    # 添加数据
    hasher.update(batch_data.cpu().numpy().tobytes())

    # 添加标签
    hasher.update(batch_labels.cpu().numpy().tobytes())

    return hasher.hexdigest()


def calculate_kl_divergence(p: torch.Tensor, q: torch.Tensor, epsilon: float = 1e-10) -> float:
    """
    计算两个概率分布之间的KL散度

    Args:
        p: 第一个概率分布
        q: 第二个概率分布
        epsilon: 小值，防止除零错误

    Returns:
        KL散度值
    """
    # 确保输入是有效的概率分布
    p = p.clamp(min=epsilon)
    q = q.clamp(min=epsilon)

    # 归一化
    p = p / p.sum(dim=1, keepdim=True)
    q = q / q.sum(dim=1, keepdim=True)

    # 计算KL散度: sum(p * log(p/q))
    kl = (p * torch.log(p / q)).sum(dim=1).mean().item()

    return kl


def compare_distributions(model_outputs: List[torch.Tensor],
                          reference_outputs: List[torch.Tensor]) -> Dict[str, float]:
    """
    比较模型输出分布与参考分布

    Args:
        model_outputs: 模型输出分布列表
        reference_outputs: 参考输出分布列表

    Returns:
        包含KL散度统计信息的字典
    """
    kl_values = []

    for p, q in zip(model_outputs, reference_outputs):
        kl_values.append(calculate_kl_divergence(p, q))

    # 计算统计量
    mean_kl = np.mean(kl_values)
    std_kl = np.std(kl_values)

    return {
        "kl_values": kl_values,
        "mean_kl": mean_kl,
        "std_kl": std_kl,
        "min_kl": min(kl_values),
        "max_kl": max(kl_values)
    }


def check_gradient_sources(gradients: Dict[str, Tuple],
                           forbidden_sources: List[str] = ["teacher_model"]) -> bool:
    """
    检查梯度是否来自禁止的源

    Args:
        gradients: 梯度字典
        forbidden_sources: 禁止的源列表

    Returns:
        如果没有来自禁止源的梯度，则为True
    """
    for key, grad_tuple in gradients.items():
        # 检查梯度输入
        grad_inputs, _ = grad_tuple

        for grad in grad_inputs:
            if grad is not None:
                # 在实际应用中，我们需要更复杂的逻辑来检测梯度来源
                # 这里只是一个简化的示例
                for source in forbidden_sources:
                    if source in str(grad):
                        return False

    return True


def check_loss_components(loss_dict: Dict[str, torch.Tensor],
                          forbidden_components: List[str] = ["kl_div", "kl_loss"]) -> bool:
    """
    检查损失函数是否包含禁止的组件

    Args:
        loss_dict: 损失组件字典
        forbidden_components: 禁止的组件列表

    Returns:
        如果没有禁止的组件，则为True
    """
    for component_name in loss_dict.keys():
        for forbidden in forbidden_components:
            if forbidden.lower() in component_name.lower():
                return False

    return True