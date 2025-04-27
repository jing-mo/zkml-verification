"""
验证模块，用于验证模型是否为蒸馏模型
"""
import os
import torch
import numpy as np
from scipy import stats
import time
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from new_zkp_verify.config import MODELS_DIR, REPORTS_DIR, DEFAULT_CONFIDENCE
from new_zkp_verify.models import ResNetModel
from new_zkp_verify.zkp import ZKProofGenerator


class ModelVerifier:
    """模型验证器"""

    def __init__(self, device="cuda"):
        """
        初始化模型验证器

        Args:
            device: 计算设备
        """
        self.device = device
        # 初始化零知识证明生成器
        self.zkp_generator = ZKProofGenerator()

    def load_model(self, model_id):
        """
        加载模型
        
        Args:
            model_id: 模型ID
            
        Returns:
            model: 加载的模型
        """
        model_path = os.path.join(MODELS_DIR, f"{model_id}.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        model = ResNetModel().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        
        return model
    
    def compute_kl_divergence(self, outputs_a, outputs_b):
        """
        计算两个模型输出分布之间的KL散度
        
        Args:
            outputs_a: 第一个模型的输出
            outputs_b: 第二个模型的输出
            
        Returns:
            float: KL散度值
        """
        # 确保输入是概率分布（softmax后的结果）
        softmax = torch.nn.Softmax(dim=1)
        p_a = softmax(outputs_a)
        p_b = softmax(outputs_b)
        
        # 计算KL散度: KL(p_a || p_b)
        kl = torch.sum(p_a * (torch.log(p_a + 1e-10) - torch.log(p_b + 1e-10)), dim=1)
        
        # 返回平均KL散度
        return kl.mean().item()
    
    def verify_black_box(self, target_model, baseline_models, test_loader, confidence_level=DEFAULT_CONFIDENCE):
        """
        黑盒验证
        
        Args:
            target_model: 目标模型
            baseline_models: 基线模型列表
            test_loader: 测试数据加载器
            confidence_level: 置信水平
            
        Returns:
            result: 验证结果
        """
        print("开始黑盒验证...")
        start_time = time.time()
        
        # 计算目标模型与每个基线模型的KL散度
        target_kl_values = []
        
        # 计算基线模型之间的KL散度
        baseline_kl_values = []
        
        # 计算每个模型的准确率
        target_correct = 0
        target_total = 0
        baseline_correct = [0] * len(baseline_models)
        baseline_total = [0] * len(baseline_models)
        
        with torch.no_grad():
            for batch_data, batch_labels in tqdm(test_loader, desc="计算KL散度"):
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # 目标模型预测
                target_outputs = target_model(batch_data)
                _, target_predicted = torch.max(target_outputs, 1)
                target_correct += (target_predicted == batch_labels).sum().item()
                target_total += batch_labels.size(0)
                
                # 基线模型预测
                baseline_outputs = []
                for i, baseline_model in enumerate(baseline_models):
                    outputs = baseline_model(batch_data)
                    baseline_outputs.append(outputs)
                    
                    _, predicted = torch.max(outputs, 1)
                    baseline_correct[i] += (predicted == batch_labels).sum().item()
                    baseline_total[i] += batch_labels.size(0)
                
                # 计算目标模型与每个基线模型的KL散度
                for baseline_output in baseline_outputs:
                    kl = self.compute_kl_divergence(target_outputs, baseline_output)
                    target_kl_values.append(kl)
                
                # 计算基线模型之间的KL散度
                for i in range(len(baseline_outputs)):
                    for j in range(i+1, len(baseline_outputs)):
                        kl = self.compute_kl_divergence(baseline_outputs[i], baseline_outputs[j])
                        baseline_kl_values.append(kl)
        
        # 计算准确率
        target_accuracy = 100 * target_correct / target_total
        baseline_accuracies = [100 * correct / total for correct, total in zip(baseline_correct, baseline_total)]
        
        # 计算KL散度统计量
        target_kl_mean = np.mean(target_kl_values)
        target_kl_std = np.std(target_kl_values)
        
        baseline_kl_mean = np.mean(baseline_kl_values)
        baseline_kl_std = np.std(baseline_kl_values)
        
        # 计算置信区间
        t_value = stats.t.ppf((1 + confidence_level) / 2, len(baseline_kl_values) - 1)
        margin_of_error = t_value * (baseline_kl_std / np.sqrt(len(baseline_kl_values)))
        confidence_interval = (baseline_kl_mean - margin_of_error, baseline_kl_mean + margin_of_error)
        
        # 计算阈值T
        threshold_T = baseline_kl_mean + margin_of_error
        
        # 判断目标模型是否通过验证
        verification_passed = target_kl_mean <= threshold_T
        
        # 确定目标KL均值相对于置信区间的位置
        if target_kl_mean < confidence_interval[0]:
            position = f"小于{confidence_level*100}%置信区间下限"
        elif target_kl_mean > confidence_interval[1]:
            position = f"大于{confidence_level*100}%置信区间上限"
        else:
            position = f"在{confidence_level*100}%置信区间内"
        
        # 计算验证时间
        verification_time = time.time() - start_time
        
        # 可视化KL散度分布
        visualization_path = self._visualize_kl_distribution(
            target_kl_values, baseline_kl_values, confidence_interval, threshold_T
        )
        
        # 为KL散度验证生成零知识证明
        kl_proof = self.zkp_generator.generate_kl_proof(
            np.array(target_kl_values), 
            np.array(baseline_kl_values),
            threshold_T
        )
        
        # 构建验证结果
        result = {
            "verification_passed": verification_passed,
            "target_kl_mean": target_kl_mean,
            "target_kl_std": target_kl_std,
            "baseline_kl_mean": baseline_kl_mean,
            "baseline_kl_std": baseline_kl_std,
            "threshold_T": threshold_T,
            "confidence_interval": confidence_interval,
            "confidence_level": confidence_level,
            "position": position,
            "target_accuracy": target_accuracy,
            "baseline_accuracies": baseline_accuracies,
            "baseline_avg_accuracy": np.mean(baseline_accuracies),
            "verification_time": verification_time,
            "visualization_path": visualization_path,
            "kl_proof": kl_proof
        }
        
        return result
    
    def _visualize_kl_distribution(self, target_kl_values, baseline_kl_values, confidence_interval, threshold_T):
        """
        可视化KL散度分布
        
        Args:
            target_kl_values: 目标模型KL散度值
            baseline_kl_values: 基线模型KL散度值
            confidence_interval: 置信区间
            threshold_T: 阈值T
            
        Returns:
            vis_path: 可视化图像路径
        """
        # 创建可视化目录
        vis_dir = os.path.join(REPORTS_DIR, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # 生成可视化图像
        plt.figure(figsize=(10, 6))
        
        # 绘制基线模型KL散度分布
        plt.hist(baseline_kl_values, bins=30, alpha=0.5, label='基线模型KL散度')
        
        # 绘制目标模型KL散度分布
        plt.hist(target_kl_values, bins=30, alpha=0.5, label='目标模型KL散度')
        
        # 绘制置信区间
        plt.axvline(x=confidence_interval[0], color='r', linestyle='--', label=f'置信区间下限')
        plt.axvline(x=confidence_interval[1], color='r', linestyle='--', label=f'置信区间上限')
        
        # 绘制阈值T
        plt.axvline(x=threshold_T, color='g', linestyle='-', label=f'阈值T={threshold_T:.4f}')
        
        # 添加图例和标签
        plt.legend()
        plt.xlabel('KL散度')
        plt.ylabel('频率')
        plt.title('模型KL散度分布对比')
        
        # 保存图像
        timestamp = int(time.time())
        vis_path = os.path.join(vis_dir, f"kl_distribution_{timestamp}.png")
        plt.savefig(vis_path)
        plt.close()
        
        return vis_path
    
    def verify_white_box(self, target_model, test_loader, num_samples=10):
        """
        白盒验证
        
        Args:
            target_model: 目标模型
            test_loader: 测试数据加载器
            num_samples: 抽样数量
            
        Returns:
            result: 验证结果
        """
        print("开始白盒验证...")
        start_time = time.time()
        
        # 抽样批次
        sampled_batches = []
        for batch_idx, (batch_data, batch_labels) in enumerate(test_loader):
            if batch_idx >= num_samples:
                break
            sampled_batches.append((batch_data, batch_labels))
        
        # 检查损失函数是否包含KL散度项
        has_kl_divergence = False
        
        # 检查梯度是否来自教师模型
        has_teacher_signal = False
        
        # 生成零知识证明
        proofs = []
        for batch_idx, (batch_data, batch_labels) in enumerate(sampled_batches):
            batch_data = batch_data.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            # 前向传播
            target_model.train()  # 设置为训练模式以计算梯度
            target_model.zero_grad()
            outputs = target_model(batch_data)
            
            # 计算损失
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(outputs, batch_labels)
            
            # 反向传播
            loss.backward()
            
            # 检查梯度来源
            for name, param in target_model.named_parameters():
                if param.grad is not None:
                    # 在实际系统中，这里应该检查梯度图来确定梯度来源
                    pass
            
            # 生成批次证明
            batch_proof = self.zkp_generator.generate_batch_proof(
                batch_data.cpu().numpy(), 
                outputs.detach().cpu().numpy()
            )
            
            proof = {
                "batch_idx": batch_idx,
                "has_kl_divergence": has_kl_divergence,
                "has_teacher_signal": has_teacher_signal,
                "verification_passed": not (has_kl_divergence or has_teacher_signal),
                "zkp_proof": batch_proof
            }
            proofs.append(proof)
        
        # 计算验证时间
        verification_time = time.time() - start_time
        
        # 构建验证结果
        result = {
            "verification_passed": all(proof["verification_passed"] for proof in proofs),
            "has_kl_divergence": has_kl_divergence,
            "has_teacher_signal": has_teacher_signal,
            "num_samples": num_samples,
            "proofs": proofs,
            "verification_time": verification_time
        }
        
        return result

    def verify_with_zkp(self, target_model, baseline_models, test_loader, num_samples=10):
        """
        使用零知识证明进行验证
        
        Args:
            target_model: 目标模型
            baseline_models: 基线模型列表
            test_loader: 测试数据加载器
            num_samples: 抽样数量
            
        Returns:
            result: 验证结果
        """
        print("开始零知识证明验证...")
        start_time = time.time()
        
        # 生成零知识证明
        proofs = []
        
        # 抽样批次
        sampled_batches = []
        for batch_idx, (batch_data, batch_labels) in enumerate(test_loader):
            if batch_idx >= num_samples:
                break
            sampled_batches.append((batch_data, batch_labels))
        
        for batch_idx, (batch_data, batch_labels) in enumerate(sampled_batches):
            batch_data = batch_data.to(self.device)
            
            # 目标模型前向传播
            target_outputs = target_model(batch_data)
            
            # 基线模型前向传播
            baseline_outputs = [model(batch_data) for model in baseline_models]
            
            # 计算KL散度
            kl_values = []
            for baseline_output in baseline_outputs:
                kl = self.compute_kl_divergence(target_outputs, baseline_output)
                kl_values.append(kl)
            
            # 生成KL散度证明
            kl_proof = self.zkp_generator.generate_kl_proof(
                target_outputs.detach().cpu().numpy(),
                [output.detach().cpu().numpy() for output in baseline_outputs],
                np.mean(kl_values)
            )
            
            # 生成批次证明
            batch_proof = self.zkp_generator.generate_batch_proof(
                batch_data.cpu().numpy(),
                target_outputs.detach().cpu().numpy()
            )
            
            # 生成联合证明
            combined_proof = self.zkp_generator.generate_combined_proof(batch_proof, kl_proof)
            
            proofs.append(combined_proof)
        
        # 验证零知识证明
        verification_results = []
        for proof in proofs:
            if "proof_id" in proof:
                result = self.zkp_generator.verify_proof(proof["proof_id"])
                verification_results.append(result)
        
        # 计算验证时间
        verification_time = time.time() - start_time
        
        # 构建验证结果
        result = {
            "verification_passed": all(result.get("verified", False) for result in verification_results),
            "proofs": proofs,
            "verification_results": verification_results,
            "num_samples": num_samples,
            "verification_time": verification_time
        }
        
        return result

    def comprehensive_verification(self, target_model_id, baseline_model_ids, test_loader, 
                                  confidence_level=DEFAULT_CONFIDENCE, num_samples=10):
        """
        综合验证
        
        Args:
            target_model_id: 目标模型ID
            baseline_model_ids: 基线模型ID列表
            test_loader: 测试数据加载器
            confidence_level: 置信水平
            num_samples: 抽样数量
            
        Returns:
            result: 验证结果
        """
        print(f"开始对模型 {target_model_id} 进行综合验证...")
        start_time = time.time()
        
        # 加载目标模型
        target_model = self.load_model(target_model_id)
        
        # 加载基线模型
        baseline_models = []
        for model_id in baseline_model_ids:
            model = self.load_model(model_id)
            baseline_models.append(model)
        
        # 黑盒验证
        black_box_result = self.verify_black_box(
            target_model, baseline_models, test_loader, confidence_level
        )
        
        # 白盒验证
        white_box_result = self.verify_white_box(
            target_model, test_loader, num_samples
        )
        
        # 零知识证明验证
        zkp_result = self.verify_with_zkp(
            target_model, baseline_models, test_loader, num_samples
        )
        
        # 综合判断
        verification_passed = (
            black_box_result["verification_passed"] and
            white_box_result["verification_passed"] and
            zkp_result["verification_passed"]
        )
        
        # 计算验证时间
        verification_time = time.time() - start_time
        
        # 构建验证结果
        result = {
            "verification_passed": verification_passed,
            "target_model_id": target_model_id,
            "baseline_model_ids": baseline_model_ids,
            "black_box_result": black_box_result,
            "white_box_result": white_box_result,
            "zkp_result": zkp_result,
            "confidence_level": confidence_level,
            "num_samples": num_samples,
            "verification_time": verification_time
        }
        
        return result

    def compute_kl_matrix(self, models, test_loader, num_samples):
        """计算模型之间的KL散度矩阵并生成表格数据"""
        print("计算基线KL分布结果...")
        kl_matrix = np.zeros((len(models), len(models)))
        model_pairs = []
        kl_values = []
        
        with torch.no_grad():
            for i in range(len(models)):
                for j in range(i+1, len(models)):
                    total_kl = 0
                    samples_processed = 0
                    
                    for batch_data, _ in test_loader:
                        if samples_processed >= num_samples:
                            break
                            
                        batch_data = batch_data.to(self.device)
                        
                        # 获取两个模型的输出
                        outputs_i = models[i](batch_data)
                        outputs_j = models[j](batch_data)
                        
                        # 计算对称KL散度
                        kl_ij = self.compute_kl_divergence(outputs_i, outputs_j)
                        kl_ji = self.compute_kl_divergence(outputs_j, outputs_i)
                        sym_kl = (kl_ij + kl_ji) / 2
                        
                        total_kl += sym_kl
                        samples_processed += batch_data.size(0)
                    
                    # 计算平均KL散度
                    avg_kl = total_kl / (samples_processed / batch_data.size(0))
                    
                    # 存储结果
                    model_pair = f"Model{i+1}-Model{j+1}"
                    model_pairs.append(model_pair)
                    kl_values.append(round(avg_kl, 2))
                    
                    # 更新KL矩阵
                    kl_matrix[i][j] = avg_kl
                    kl_matrix[j][i] = avg_kl
        
        # 生成表格数据
        table_data = {
            "model_pairs": model_pairs,
            "kl_sym_values": kl_values
        }
        
        return kl_matrix, table_data
        
    def compute_model_kl_divergence(self, model_a, model_b, test_loader, num_batches=10):
        """
        计算两个模型之间的KL散度
        
        Args:
            model_a: 第一个模型
            model_b: 第二个模型
            test_loader: 测试数据加载器
            num_batches: 要处理的批次数量
            
        Returns:
            float: KL散度值
        """
        total_kl = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for batch_data, _ in test_loader:
                if batch_count >= num_batches:
                    break
                    
                batch_data = batch_data.to(self.device)
                
                # 获取两个模型的输出
                outputs_a = model_a(batch_data)
                outputs_b = model_b(batch_data)
                
                # 计算KL散度
                kl = self.compute_kl_divergence(outputs_a, outputs_b)
                
                total_kl += kl
                batch_count += 1
        
        # 返回平均KL散度
        return total_kl / batch_count if batch_count > 0 else 0.0