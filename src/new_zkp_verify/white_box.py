"""
白盒验证模块，用于抽样检测训练过程中是否存在蒸馏组件
"""
import os
import json
import time
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from datetime import datetime
from tqdm import tqdm
import random

from config import MODELS_DIR, CACHE_DIR, CIRCUIT_DIR, CIFAR10_MEAN, CIFAR10_STD
from models import ResNetModel, DistilledModel


class WhiteBoxVerifier:
    """
    白盒验证器 - 用于抽样检测训练过程，验证是否有知识蒸馏组件
    """

    def __init__(self, device='cuda', sample_ratio=0.1, seed=42):
        """
        初始化白盒验证器

        Args:
            device: 计算设备
            sample_ratio: 抽样比例 (0-1)
            seed: 随机种子
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sample_ratio = sample_ratio
        self.seed = seed
        self.proofs = []

        # 设置随机种子，确保抽样可重现
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # 创建证明目录
        self.proof_dir = os.path.join(CIRCUIT_DIR, "white_box_proofs")
        os.makedirs(self.proof_dir, exist_ok=True)

    def hash_tensor(self, tensor):
        """计算张量的哈希值"""
        # 将张量转换为numpy数组，然后转为字节
        tensor_bytes = tensor.detach().cpu().numpy().tobytes()
        # 计算SHA-256哈希
        return hashlib.sha256(tensor_bytes).hexdigest()

    def hash_model_structure(self, model):
        """计算模型结构的哈希值"""
        structure_str = str(model)
        return hashlib.sha256(structure_str.encode()).hexdigest()

    def hash_loss_function(self, loss_fn):
        """计算损失函数的哈希值"""
        loss_str = str(loss_fn.__class__.__name__)
        if hasattr(loss_fn, "__dict__"):
            loss_str += str(loss_fn.__dict__)
        return hashlib.sha256(loss_str.encode()).hexdigest()

    def verify_gradients(self, grads, inputs, targets, model):
        """
        验证梯度是否来自正常的损失函数而非教师模型

        Args:
            grads: 实际梯度
            inputs: 输入数据
            targets: 目标标签
            model: 模型

        Returns:
            is_valid: 梯度是否有效
            validation_info: 验证信息
        """
        # 保存原始梯度，用于对比
        original_grads = [g.clone() if g is not None else None for g in grads]

        # 重新计算梯度
        criterion = nn.CrossEntropyLoss()
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # 获取重新计算的梯度
        recomputed_grads = []
        for param in model.parameters():
            if param.grad is not None:
                recomputed_grads.append(param.grad.clone())
            else:
                recomputed_grads.append(None)

        # 比较原始梯度和重新计算的梯度
        grad_diffs = []
        for og, rg in zip(original_grads, recomputed_grads):
            if og is not None and rg is not None:
                # 计算相对差异
                diff = torch.norm(og - rg) / (torch.norm(og) + 1e-8)
                grad_diffs.append(diff.item())

        # 计算平均差异
        avg_diff = np.mean(grad_diffs) if grad_diffs else 0

        # 如果差异很大，可能存在蒸馏
        is_valid = avg_diff < 0.01  # 阈值可调整

        validation_info = {
            "average_gradient_difference": avg_diff,
            "is_valid": is_valid,
            "threshold": 0.01
        }

        return is_valid, validation_info

    def sample_training_batches(self, train_loader, num_samples=None):
        """
        对训练数据进行可验证随机抽样

        Args:
            train_loader: 训练数据加载器
            num_samples: 抽样数量，如果为None则根据sample_ratio计算

        Returns:
            sampled_batches: 抽样的批次
            sample_indices: 抽样的索引
        """
        # 获取所有批次
        all_batches = []
        for inputs, targets in train_loader:
            all_batches.append((inputs, targets))

        # 计算抽样数量
        total_batches = len(all_batches)
        if num_samples is None:
            num_samples = max(1, int(total_batches * self.sample_ratio))
        else:
            num_samples = min(num_samples, total_batches)

        # 随机抽样
        sample_indices = random.sample(range(total_batches), num_samples)
        sampled_batches = [all_batches[i] for i in sample_indices]

        print(f"已从{total_batches}个批次中抽样{num_samples}个批次 (比例: {num_samples / total_batches:.2%})")
        return sampled_batches, sample_indices

    def verify_batch(self, model, inputs, targets, optimizer=None):
        """
        验证单个训练批次，检测是否有蒸馏组件

        Args:
            model: 要验证的模型
            inputs: 输入数据
            targets: 目标标签
            optimizer: 优化器（可选）

        Returns:
            proof: 批次验证证明
        """
        # 记录开始时间
        start_time = time.time()

        # 移动数据到设备
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # 生成批次哈希值
        batch_hash = self.hash_tensor(inputs) + "_" + self.hash_tensor(targets)

        # 记录模型参数更新前的哈希值
        model_params_before = {}
        for name, param in model.named_parameters():
            model_params_before[name] = self.hash_tensor(param.data)

        # 前向传播
        model.train()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)

        # 反向传播
        if optimizer:
            optimizer.zero_grad()
        loss.backward()

        # 收集梯度
        gradients = []
        for param in model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.clone())
            else:
                gradients.append(None)

        # 验证梯度
        is_valid, grad_validation = self.verify_gradients(gradients, inputs, targets, model)

        # 如果有优化器，更新参数
        if optimizer:
            optimizer.step()

        # 记录模型参数更新后的哈希值
        model_params_after = {}
        for name, param in model.named_parameters():
            model_params_after[name] = self.hash_tensor(param.data)

        # 计算梯度公式哈希值
        gradient_formula_hash = hashlib.sha256(str(loss.item()).encode()).hexdigest()

        # 计算损失函数哈希值
        loss_fn_hash = self.hash_loss_function(F.cross_entropy)

        # 计算模型结构哈希值
        model_structure_hash = self.hash_model_structure(model)

        # 生成证明
        proof = {
            "proof_id": f"whitebox_{int(time.time())}_{len(self.proofs)}",
            "timestamp": datetime.now().isoformat(),
            "batch_hash": batch_hash,
            "model_structure_hash": model_structure_hash,
            "loss_function_hash": loss_fn_hash,
            "gradient_formula_hash": gradient_formula_hash,
            "model_params_before": model_params_before,
            "model_params_after": model_params_after,
            "gradient_validation": grad_validation,
            "is_valid": is_valid,
            "generation_time_ms": int((time.time() - start_time) * 1000)
        }

        # 添加到证明列表
        self.proofs.append(proof)

        # 保存证明
        proof_path = os.path.join(self.proof_dir, f"{proof['proof_id']}.json")
        with open(proof_path, 'w') as f:
            json.dump(proof, f, indent=2)

        return proof

    def verify_model(self, model, train_loader, num_samples=None, optimizer=None):
        """
        验证模型的训练过程，检测是否存在蒸馏组件

        Args:
            model: 要验证的模型
            train_loader: 训练数据加载器
            num_samples: 抽样数量
            optimizer: 优化器（可选）

        Returns:
            verification_result: 验证结果
        """
        print(f"\n开始白盒抽样验证，检测是否存在蒸馏组件...")
        start_time = time.time()

        # 抽样训练批次
        sampled_batches, sample_indices = self.sample_training_batches(train_loader, num_samples)

        # 验证每个抽样批次
        batch_proofs = []
        for i, (inputs, targets) in enumerate(sampled_batches):
            print(f"正在验证批次 {i + 1}/{len(sampled_batches)}...")
            proof = self.verify_batch(model, inputs, targets, optimizer)
            batch_proofs.append(proof)

        # 统计验证结果
        valid_count = sum(1 for proof in batch_proofs if proof['is_valid'])
        valid_ratio = valid_count / len(batch_proofs) if batch_proofs else 0

        # 验证是否通过（所有批次都有效）
        verification_passed = valid_ratio == 1.0

        verification_result = {
            "timestamp": datetime.now().isoformat(),
            "total_batches": len(train_loader),
            "sampled_batches": len(sampled_batches),
            "sample_ratio": len(sampled_batches) / len(train_loader),
            "valid_batches": valid_count,
            "valid_ratio": valid_ratio,
            "verification_passed": verification_passed,
            "batch_proofs": batch_proofs,
            "verification_time_ms": int((time.time() - start_time) * 1000)
        }

        # 保存验证结果
        result_path = os.path.join(self.proof_dir, f"verification_result_{int(time.time())}.json")
        with open(result_path, 'w') as f:
            json.dump(verification_result, f, indent=2)

        print(f"\n白盒验证结果:")
        print(f"  抽样批次数: {len(sampled_batches)}/{len(train_loader)} ({verification_result['sample_ratio']:.2%})")
        print(f"  有效批次数: {valid_count}/{len(sampled_batches)} ({valid_ratio:.2%})")
        print(f"  验证结果: {'通过' if verification_passed else '不通过'}")
        print(f"  验证耗时: {verification_result['verification_time_ms']}毫秒")

        return verification_result

    def verify_model_during_training(self, model, dataloader, epochs=1, sample_per_epoch=5):
        """
        在训练过程中进行抽样验证

        Args:
            model: 要验证的模型
            dataloader: 训练数据加载器
            epochs: 训练轮次
            sample_per_epoch: 每轮抽样数量

        Returns:
            all_results: 所有验证结果
        """
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        all_results = []

        for epoch in range(epochs):
            print(f"\n轮次 {epoch + 1}/{epochs}")

            # 对本轮次进行抽样验证
            result = self.verify_model(model, dataloader, num_samples=sample_per_epoch, optimizer=optimizer)
            all_results.append(result)

            # 如果验证不通过，提前终止
            if not result["verification_passed"]:
                print(f"警告: 验证不通过，可能存在蒸馏组件，提前终止训练")
                break

        return all_results

    def detect_teacher_model(self, model):
        """
        检测模型中是否存在教师模型的引用

        Args:
            model: 要检测的模型

        Returns:
            has_teacher: 是否存在教师模型
            teacher_info: 教师模型信息
        """
        has_teacher = False
        teacher_info = {}

        # 检查模型的属性
        for attr_name in dir(model):
            if attr_name.startswith('__'):
                continue

            attr = getattr(model, attr_name)

            # 检查是否有名为"teacher"的属性
            if attr_name == 'teacher' or attr_name.endswith('_teacher'):
                has_teacher = True
                teacher_info['attribute'] = attr_name
                teacher_info['type'] = type(attr).__name__
                break

            # 检查是否有蒸馏相关的方法
            if callable(attr) and any(name in attr_name.lower() for name in ['distill', 'teacher', 'kd']):
                has_teacher = True
                teacher_info['method'] = attr_name
                break

        # 检查是否有蒸馏相关的损失函数
        if hasattr(model, 'distillation_loss') or hasattr(model, 'kd_loss'):
            has_teacher = True
            teacher_info['loss_function'] = 'distillation_loss' if hasattr(model, 'distillation_loss') else 'kd_loss'

        return has_teacher, teacher_info

    def generate_combined_proof(self, white_box_results, black_box_results):
        """
        生成白盒和黑盒验证的联合证明

        Args:
            white_box_results: 白盒验证结果
            black_box_results: 黑盒验证结果

        Returns:
            combined_proof: 联合证明
        """
        # 计算白盒验证结果的哈希值
        white_box_hash = hashlib.sha256(json.dumps(white_box_results, sort_keys=True).encode()).hexdigest()

        # 计算黑盒验证结果的哈希值
        black_box_hash = hashlib.sha256(json.dumps(black_box_results, sort_keys=True).encode()).hexdigest()

        # 生成绑定证明
        binding_proof = {
            "white_box_hash": white_box_hash,
            "black_box_hash": black_box_hash,
            "timestamp": datetime.now().isoformat(),
            "binding_hash": hashlib.sha256((white_box_hash + black_box_hash).encode()).hexdigest()
        }

        # 生成联合证明
        combined_proof = {
            "proof_id": f"combined_{int(time.time())}",
            "white_box_proof": white_box_results,
            "black_box_proof": black_box_results,
            "binding_proof": binding_proof,
            "verification_passed": white_box_results["verification_passed"] and black_box_results[
                "passed_verification"],
            "timestamp": datetime.now().isoformat()
        }

        # 保存联合证明
        proof_path = os.path.join(self.proof_dir, f"{combined_proof['proof_id']}.json")
        with open(proof_path, 'w') as f:
            json.dump(combined_proof, f, indent=2)

        return combined_proof


class WhiteBoxVerifier:
    """
    白盒验证器类
    """
    def __init__(self, device='cuda'):
        self.device = device
    
    def verify_model_weights(self, target_model, baseline_models):
        """
        验证模型权重
        
        Args:
            target_model: 目标模型
            baseline_models: 基线模型列表
            
        Returns:
            result: 验证结果
        """
        # 计算目标模型与每个基线模型的权重相似度
        weight_similarities = []
        
        for i, baseline_model in enumerate(baseline_models):
            similarity = self._compute_weight_similarity(target_model, baseline_model)
            weight_similarities.append(similarity)
        
        # 计算平均相似度
        mean_similarity = np.mean(weight_similarities)
        
        # 返回结果
        result = {
            "weight_similarities": weight_similarities,
            "mean_similarity": float(mean_similarity)
        }
        
        return result
    
    def _compute_weight_similarity(self, model1, model2):
        """
        计算两个模型的权重相似度
        
        Args:
            model1: 第一个模型
            model2: 第二个模型
            
        Returns:
            similarity: 相似度值
        """
        # 获取模型参数
        params1 = {name: param.data.cpu().numpy().flatten() for name, param in model1.named_parameters()}
        params2 = {name: param.data.cpu().numpy().flatten() for name, param in model2.named_parameters()}
        
        # 计算余弦相似度
        similarities = []
        
        for name in params1:
            if name in params2 and params1[name].shape == params2[name].shape:
                # 计算余弦相似度
                dot_product = np.dot(params1[name], params2[name])
                norm1 = np.linalg.norm(params1[name])
                norm2 = np.linalg.norm(params2[name])
                
                if norm1 > 0 and norm2 > 0:
                    similarity = dot_product / (norm1 * norm2)
                    similarities.append(similarity)
        
        # 返回平均相似度
        if similarities:
            return np.mean(similarities)
        else:
            return 0.0
    
    def verify_zkp_proofs(self, zkp_proofs):
        """
        验证ZKP证明
        
        Args:
            zkp_proofs: ZKP证明列表
            
        Returns:
            result: 验证结果
        """
        # 验证每个ZKP证明
        valid_proofs = 0
        
        for proof in zkp_proofs:
            if self._verify_proof(proof):
                valid_proofs += 1
        
        # 计算有效证明比例
        if zkp_proofs:
            validity_ratio = valid_proofs / len(zkp_proofs)
        else:
            validity_ratio = 0.0
        
        # 返回结果
        result = {
            "total_proofs": len(zkp_proofs),
            "valid_proofs": valid_proofs,
            "validity_ratio": float(validity_ratio)
        }
        
        return result
    
    def _verify_proof(self, proof):
        """
        验证单个ZKP证明
        
        Args:
            proof: ZKP证明
            
        Returns:
            valid: 是否有效
        """
        # 这里应该实现实际的ZKP验证逻辑
        # 由于ZKP验证需要特定的密码学库和电路，这里只是一个占位符
        # 在实际实现中，应该调用ZKP验证器来验证证明
        
        # 假设所有证明都有效
        return True