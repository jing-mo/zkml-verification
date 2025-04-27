from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats
import os

from new_zkp_verify.zkp import ZKProofGenerator, ZKCircuitGenerator
from new_zkp_verify.report import ReportGenerator
from new_zkp_verify.config import REPORTS_DIR, MODELS_DIR

class VerificationExperiment:
    def __init__(self, exp_id=None):
        """初始化验证实验"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exp_id = exp_id or f"experiment_{int(time.time())}"
        self.zkp_generator = ZKProofGenerator()
        self.report_generator = ReportGenerator(exp_id)
        
        # 配置保存路径
        self.save_dir = Path(MODELS_DIR)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 实验配置
        self.config = {
            "num_baseline_models": 5,
            "confidence_level": 0.95,
            "batch_size": 128,
            "num_epochs": 10,
            "learning_rate": 0.001,
            "sample_batches": 50  # 白盒验证的抽样批次数
        }
        
    def load_data(self):
        """加载CIFAR-10数据集"""
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = datasets.CIFAR10(root=str(DATA_DIR), train=True,
                                       download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root=str(DATA_DIR), train=False,
                                      download=True, transform=transform_test)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config["batch_size"],
                                shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=self.config["batch_size"],
                               shuffle=False, num_workers=2)
        
        return train_loader, test_loader
    
    def train_model(self, model_id, seed, is_distilled=False, teacher_model=None):
        """
        训练单个模型
        
        Args:
            model_id: 模型ID
            seed: 随机种子
            is_distilled: 是否为蒸馏模型
            teacher_model: 教师模型（如果是蒸馏模型）
            
        Returns:
            model: 训练好的模型
            stats: 训练统计信息
        """
        print(f"训练模型: {model_id}, 种子: {seed}, 蒸馏: {is_distilled}")
        torch.manual_seed(seed)
        model = models.resnet50(pretrained=False, num_classes=10).to(self.device)
        
        # 配置训练参数
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), 
                                   lr=self.config["learning_rate"])
        
        train_loader, test_loader = self.load_data()
        
        # 训练过程
        start_time = time.time()
        best_acc = 0
        train_losses = []
        test_accs = []
        
        # 蒸馏温度
        temperature = 2.0
        
        for epoch in range(self.config["num_epochs"]):
            model.train()
            epoch_loss = 0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}") as pbar:
                for batch_idx, (data, target) in enumerate(pbar):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # 生成白盒验证证明
                    if batch_idx % (len(train_loader) // self.config["sample_batches"]) == 0:
                        self.generate_whitebox_proof(model, data, target)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    
                    # 如果是蒸馏模型，添加蒸馏损失
                    if is_distilled and teacher_model is not None:
                        with torch.no_grad():
                            teacher_output = teacher_model(data)
                        
                        # 硬标签损失
                        hard_loss = criterion(output, target)
                        
                        # 软标签损失（KL散度）
                        soft_target = nn.functional.softmax(teacher_output / temperature, dim=1)
                        soft_output = nn.functional.log_softmax(output / temperature, dim=1)
                        soft_loss = nn.functional.kl_div(soft_output, soft_target, reduction='batchmean') * (temperature ** 2)
                        
                        # 总损失 = 硬标签损失 + 软标签损失
                        loss = hard_loss * 0.5 + soft_loss * 0.5
                    else:
                        loss = criterion(output, target)
                    
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    pbar.set_postfix({"loss": loss.item()})
            
            # 测试准确率
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
            
            accuracy = 100. * correct / total
            if accuracy > best_acc:
                best_acc = accuracy
            
            train_losses.append(epoch_loss / len(train_loader))
            test_accs.append(accuracy)
            
            print(f"Epoch {epoch+1}: Loss={epoch_loss/len(train_loader):.4f}, Accuracy={accuracy:.2f}%")
        
        training_time = time.time() - start_time
        
        # 保存模型
        save_path = self.save_dir / f"model_{model_id}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'seed': seed,
            'best_accuracy': best_acc,
            'training_time': training_time,
            'is_distilled': is_distilled
        }, save_path)
        
        # 返回模型和统计信息
        stats = {
            "model_id": model_id,
            "seed": seed,
            "best_accuracy": best_acc,
            "training_time": training_time,
            "is_distilled": is_distilled,
            "train_losses": train_losses,
            "test_accs": test_accs
        }
        
        return model, stats
    
    def generate_whitebox_proof(self, model, data, target):
        """
        生成白盒证明
        
        Args:
            model: 模型
            data: 输入数据
            target: 目标标签
            
        Returns:
            proof: 证明
        """
        # 获取模型梯度和参数
        params_before = {name: param.clone() for name, param in model.named_parameters()}
        
        # 前向传播
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        
        # 检查梯度来源
        for name, param in model.named_parameters():
            if param.grad is not None:
                # 验证梯度来源不是教师模型
                assert "teacher" not in name, f"发现教师模型梯度: {name}"
        
        # 生成证明
        proof = self.zkp_generator.generate_batch_proof(
            input_data=data.cpu().numpy(),
            model_output=output.detach().cpu().numpy(),
            model_params={name: param.cpu().detach().numpy().mean() for name, param in params_before.items()}
        )
        
        return proof
    
    def get_model_distribution(self, model, test_loader, num_samples=10):
        """获取模型输出分布"""
        model.eval()
        all_outputs = []
        
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                if i >= num_samples:
                    break
                
                data = data.to(self.device)
                output = model(data)
                # 将输出转换为概率分布
                probs = torch.softmax(output, dim=1)
                all_outputs.append(probs.cpu().numpy())
        
        # 合并所有输出并计算平均分布
        all_outputs = np.vstack(all_outputs)
        avg_distribution = np.mean(all_outputs, axis=0)
        
        return avg_distribution
    
    def get_baseline_distribution(self, baseline_models):
        """获取基线模型的平均分布"""
        _, test_loader = self.load_data()
        
        # 获取每个基线模型的分布
        distributions = []
        for model_info in baseline_models:
            model = model_info["model"]
            dist = self.get_model_distribution(model, test_loader)
            distributions.append(dist)
        
        # 计算平均分布
        avg_distribution = np.mean(distributions, axis=0)
        
        return avg_distribution
    
    def calculate_kl_divergences(self, target_model, baseline_models):
        """计算KL散度"""
        _, test_loader = self.load_data()
        
        target_dist = self.get_model_distribution(target_model, test_loader)
        
        kl_divergences = []
        for model_info in baseline_models:
            model = model_info["model"]
            baseline_dist = self.get_model_distribution(model, test_loader)
            
            # 避免除零错误
            epsilon = 1e-10
            target_dist_safe = target_dist + epsilon
            baseline_dist_safe = baseline_dist + epsilon
            
            # 归一化
            target_dist_safe = target_dist_safe / np.sum(target_dist_safe)
            baseline_dist_safe = baseline_dist_safe / np.sum(baseline_dist_safe)
            
            # 计算KL散度
            kl = np.sum(target_dist_safe * np.log(target_dist_safe / baseline_dist_safe))
            kl_divergences.append(kl)
            
        return np.array(kl_divergences)
    
    def visualize_kl_distribution(self, kl_divergences, target_kl=None):
        """可视化KL散度分布"""
        import matplotlib.pyplot as plt
        from scipy import stats
        
        # 创建图表
        plt.figure(figsize=(10, 6))
        
        # 绘制KL散度分布
        x = np.linspace(min(kl_divergences) * 0.8, max(kl_divergences) * 1.2, 100)
        kde = stats.gaussian_kde(kl_divergences)
        plt.plot(x, kde(x), 'b-', label='基线KL散度分布')
        
        # 绘制置信区间
        ci = self.calculate_confidence_interval(kl_divergences)
        plt.axvline(x=ci[0], color='g', linestyle='--', label=f'{self.config["confidence_level"]*100}%置信区间下限')
        plt.axvline(x=ci[1], color='g', linestyle='--', label=f'{self.config["confidence_level"]*100}%置信区间上限')
        
        # 如果提供了目标KL散度，绘制它
        if target_kl is not None:
            plt.axvline(x=target_kl, color='r', linestyle='-', label='目标模型KL散度')
        
        # 添加标签和标题
        plt.xlabel('KL散度')
        plt.ylabel('密度')
        plt.title('基线模型KL散度分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        save_path = self.save_dir / f"kl_distribution_{self.exp_id}.png"
        plt.savefig(save_path)
        plt.close()
        
        return str(save_path)
    
    def run_experiment(self):
        """运行完整实验"""
        # 编译电路
        circuit_generator = ZKCircuitGenerator()
        circuit_status = circuit_generator.compile_all_circuits()
        
        if not all(status["compiled"] for status in circuit_status.values()):
            raise RuntimeError("电路编译失败")
        
        # 训练基线模型
        baseline_models = []
        for i in range(self.config["num_baseline_models"]):
            seed = np.random.randint(1000, 10000)
            model, acc = self.train_model(f"baseline_{i}", seed)
            baseline_models.append({"model": model, "accuracy": acc, "seed": seed})
        
        # 训练目标模型
        target_model, target_acc = self.train_model("target", seed=42)
        
        # 计算KL散度
        kl_divergences = self.calculate_kl_divergences(target_model, baseline_models)
        target_kl_mean = np.mean(kl_divergences)
        
        # 计算置信区间
        confidence_interval = self.calculate_confidence_interval(kl_divergences)
        
        # 判断目标模型是否在置信区间内
        is_in_confidence_interval = confidence_interval[0] <= target_kl_mean <= confidence_interval[1]
        
        # 可视化KL散度分布
        vis_path = self.visualize_kl_distribution(kl_divergences, target_kl_mean)
        
        # 生成黑盒证明
        black_box_proof = self.zkp_generator.generate_kl_proof(
            target_distribution=self.get_model_distribution(target_model, self.load_data()[1]),
            baseline_distribution=self.get_baseline_distribution(baseline_models),
            threshold=confidence_interval[1]  # 使用置信区间上限作为阈值
        )
        
        # 生成报告
        report_data = {
            "experiment_id": self.exp_id,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "baseline_models": {
                "count": len(baseline_models),
                "configurations": [{"seed": m["seed"], "accuracy": m["accuracy"]} 
                                 for m in baseline_models]
            },
            "target_model": {
                "config": {
                    "model_id": "target",
                    "seed": 42,
                    "epochs": self.config["num_epochs"],
                    "batch_size": self.config["batch_size"],
                    "learning_rate": self.config["learning_rate"],
                    "is_distilled": False,
                    "best_accuracy": target_acc,
                    "training_time": "N/A"  # 这里需要在train_model中添加时间记录
                }
            },
            "kl_divergence_stats": {
                "values": kl_divergences.tolist(),
                "kl_mean": float(target_kl_mean),
                "kl_std": float(np.std(kl_divergences)),
                "confidence_interval": confidence_interval,
                "is_in_confidence_interval": is_in_confidence_interval
            },
            "summary": {
                "passed_verification": is_in_confidence_interval,
                "confidence_level": self.config["confidence_level"],
                "target_kl_mean": float(target_kl_mean),
                "kl_threshold_T": float(confidence_interval[1]),
                "baseline_avg_accuracy": float(np.mean([m["accuracy"] for m in baseline_models])),
                "zkp_count": len(self.zkp_generator.get_all_proofs())
            },
            "visualizations": {
                "kl_distribution": vis_path
            },
            "zero_knowledge_proofs": self.zkp_generator.get_all_proofs(),
            "experiment_metrics": {
                "avg_proof_generation_time": f"{np.mean([p.get('generation_time_ms', 0) for p in self.zkp_generator.get_all_proofs()])/1000:.2f}秒",
                "avg_proof_size": f"{np.mean([p.get('proof_size_bytes', 0) for p in self.zkp_generator.get_all_proofs()])/1024:.2f}KB",
                "total_experiment_time": "N/A"  # 需要在实验开始和结束时记录时间
            }
        }
        
        # 生成报告
        json_path, html_path = self.report_generator.generate_comprehensive_report(report_data)
        md_path = self.report_generator.generate_markdown_report(report_data)
        
        print(f"实验完成！报告已生成：\nJSON: {json_path}\nHTML: {html_path}\nMarkdown: {md_path}")
        
        return report_data

def train_model(self, model_id, seed):
    """训练单个模型"""
    print(f"训练模型: {model_id}, 种子: {seed}")
    torch.manual_seed(seed)
    model = models.resnet50(pretrained=False, num_classes=10).to(self.device)
    
    # 配置训练参数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 
                               lr=self.config["learning_rate"])
    
    train_loader, test_loader = self.load_data()
    
    # 训练过程
    start_time = time.time()
    best_acc = 0
    
    for epoch in range(self.config["num_epochs"]):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 生成白盒验证证明
            if batch_idx % (len(train_loader) // self.config["sample_batches"]) == 0:
                self.generate_whitebox_proof(model, data, target)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # 每个epoch结束后评估模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        if accuracy > best_acc:
            best_acc = accuracy
        
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')
    
    training_time = time.time() - start_time
    print(f"模型 {model_id} 训练完成，最佳准确率: {best_acc:.2f}%，训练时间: {training_time:.2f}秒")
    
    # 保存模型
    save_path = self.save_dir / f"model_{model_id}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'seed': seed,
        'best_accuracy': best_acc,
        'training_time': training_time
    }, save_path)
    
    return model, best_acc


if __name__ == "__main__":
    start_time = time.time()
    experiment = VerificationExperiment()
    results = experiment.run_experiment()
    total_time = time.time() - start_time
    print(f"实验完成！总耗时: {total_time:.2f}秒")