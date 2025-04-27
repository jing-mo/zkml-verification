import hashlib
import torch
import numpy as np
from typing import Any, Dict, List, Union


def calculate_string_hash(text: str) -> str:
    """
    计算字符串的哈希值

    Args:
        text: 输入字符串

    Returns:
        哈希值的十六进制字符串
    """
    return hashlib.sha256(text.encode()).hexdigest()


def calculate_bytes_hash(data: bytes) -> str:
    """
    计算字节数据的哈希值

    Args:
        data: 输入字节数据

    Returns:
        哈希值的十六进制字符串
    """
    return hashlib.sha256(data).hexdigest()


def calculate_tensor_hash(tensor: torch.Tensor) -> str:
    """
    计算PyTorch张量的哈希值

    Args:
        tensor: 输入张量

    Returns:
        哈希值的十六进制字符串
    """
    # 将张量转换为NumPy数组，再转换为字节
    tensor_bytes = tensor.detach().cpu().numpy().tobytes()
    return calculate_bytes_hash(tensor_bytes)


def calculate_dict_hash(data: Dict[str, Any]) -> str:
    """
    计算字典的哈希值

    Args:
        data: 输入字典

    Returns:
        哈希值的十六进制字符串
    """
    # 将字典转换为排序后的字符串表示
    dict_str = str(sorted(data.items()))
    return calculate_string_hash(dict_str)


def calculate_list_hash(data: List[Any]) -> str:
    """
    计算列表的哈希值

    Args:
        data: 输入列表

    Returns:
        哈希值的十六进制字符串
    """
    # 将列表转换为字符串表示
    list_str = str(data)
    return calculate_string_hash(list_str)


def calculate_model_params_hash(model_params: Dict[str, torch.Tensor]) -> str:
    """
    计算模型参数的哈希值

    Args:
        model_params: 模型参数字典

    Returns:
        哈希值的十六进制字符串
    """
    # 创建哈希对象
    hasher = hashlib.sha256()

    # 对所有参数名进行排序，确保结果可重现
    for name in sorted(model_params.keys()):
        param = model_params[name]
        # 更新哈希值
        hasher.update(name.encode())
        hasher.update(param.detach().cpu().numpy().tobytes())

    return hasher.hexdigest()


def create_merkle_tree(items: List[str]) -> Dict[str, Any]:
    """
    创建默克尔树

    Args:
        items: 项目列表，每个项目是一个哈希字符串

    Returns:
        包含树结构的字典
    """
    if not items:
        return {'root': None, 'levels': []}

    # 创建叶子节点级别
    current_level = items
    levels = [current_level]

    # 构建树的每一层
    while len(current_level) > 1:
        next_level = []
        # 处理当前层的每对节点
        for i in range(0, len(current_level), 2):
            # 如果还有一对节点
            if i + 1 < len(current_level):
                # 合并哈希值
                combined = current_level[i] + current_level[i + 1]
                next_hash = calculate_string_hash(combined)
            else:
                # 奇数个节点时，最后一个节点上移
                next_hash = current_level[i]

            next_level.append(next_hash)

        # 更新当前层
        current_level = next_level
        levels.append(current_level)

    # 根节点是最后一层的唯一节点
    root = levels[-1][0] if levels else None

    return {
        'root': root,
        'levels': levels
    }


def verify_merkle_path(item: str, path: List[Dict[str, str]], root: str) -> bool:
    """
    验证默克尔路径

    Args:
        item: 要验证的项目
        path: 默克尔路径
        root: 根哈希值

    Returns:
        如果路径有效，则为True
    """
    current_hash = item

    for step in path:
        position = step['position']  # 'left' 或 'right'
        sibling = step['hash']

        if position == 'left':
            # 当前哈希在左边
            combined = current_hash + sibling
        else:
            # 当前哈希在右边
            combined = sibling + current_hash

        current_hash = calculate_string_hash(combined)

    # 最终哈希应该等于根哈希
    return current_hash == root


def generate_commitment(data: Any) -> Dict[str, str]:
    """
    生成对数据的承诺

    Args:
        data: 任意数据

    Returns:
        包含承诺和随机数的字典
    """
    # 生成随机数
    import os
    randomness = os.urandom(32).hex()

    # 将数据转换为字符串
    data_str = str(data)

    # 创建承诺：数据与随机数的组合哈希
    commitment = calculate_string_hash(data_str + randomness)

    return {
        'commitment': commitment,
        'randomness': randomness
    }


def verify_commitment(data: Any, commitment: str, randomness: str) -> bool:
    """
    验证对数据的承诺

    Args:
        data: 要验证的数据
        commitment: 承诺哈希
        randomness: 生成承诺时使用的随机数

    Returns:
        如果承诺有效，则为True
    """
    # 将数据转换为字符串
    data_str = str(data)

    # 计算承诺哈希
    computed_commitment = calculate_string_hash(data_str + randomness)

    # 检查计算的承诺是否与给定的承诺匹配
    return computed_commitment == commitment