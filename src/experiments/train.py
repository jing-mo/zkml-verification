"""
训练模块，用于训练模型
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from new_zkp_verify.config import MODELS_DIR, CIFAR10_MEAN, CIFAR10_STD, CACHE_DIR
from new_zkp_verify.models import ResNetModel
import numpy as np  # 添加这行


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, device="cuda", batch_size=128, epochs=50, lr=0.1, model_id=None, seed=None, scaler=None):
        """
        初始化模型训练器
        
        Args:
            device: 计算设备
            batch_size: 批次大小
            epochs: 训练轮次
            lr: 学习率
            model_id: 模型ID
            seed: 随机种子
            scaler: 混合精度训练的GradScaler
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.model_id = model_id or f"model_{int(time.time())}"
        self.seed = seed
        self.scaler = scaler  # Add this line
        
        # 设置随机种子
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

        # Initialize model
        self.model = ResNetModel().to(self.device)
        
        # 创建优化器和学习率调度器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=5e-4
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
    
    def _get_data_loaders(self):
        """
        获取数据加载器
        
        Returns:
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
        """
        # 数据变换
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
        
        # 加载训练集和测试集
        train_dataset = datasets.CIFAR10(
            root=CACHE_DIR, train=True, download=True, transform=train_transform
        )
        
        test_dataset = datasets.CIFAR10(
            root=CACHE_DIR, train=False, download=True, transform=test_transform
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, 
            num_workers=4, pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=100, shuffle=False, 
            num_workers=4, pin_memory=True
        )
        
        return train_loader, test_loader
    
    def train(self, force_retrain=False):
        """
        训练模型
        
        Args:
            force_retrain: 是否强制重新训练
            
        Returns:
            model: 训练好的模型
            stats: 训练统计信息
        """
        model_path = os.path.join(MODELS_DIR, f"{self.model_id}.pth")
        
        # 如果模型已存在且不强制重新训练，则直接加载
        if os.path.exists(model_path) and not force_retrain:
            print(f"加载现有模型: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            
            # 评估模型
            _, test_loader = self._get_data_loaders()
            test_acc = self._evaluate(test_loader)
            
            return self.model, {
                "best_acc": test_acc,
                "training_time": 0,
                "loaded_from_file": True
            }
        
        # 获取数据加载器
        train_loader, test_loader = self._get_data_loaders()
        
        # 训练模型
        print(f"开始训练模型: {self.model_id}")
        start_time = time.time()
        
        best_acc = 0
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []
        
        for epoch in range(self.epochs):
            # 训练一个轮次
            train_loss, train_acc = self._train_epoch(train_loader, epoch)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # 评估模型
            test_loss, test_acc = self._evaluate(test_loader, compute_loss=True)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            
            # 更新学习率
            self.scheduler.step()
            
            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(self.model.state_dict(), model_path)
                print(f"保存最佳模型，准确率: {best_acc:.2f}%")
        
        training_time = time.time() - start_time
        print(f"模型训练完成，耗时: {training_time:.2f}秒，最佳准确率: {best_acc:.2f}%")
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # 返回训练好的模型和统计信息
        return self.model, {
            "best_acc": best_acc,
            "training_time": training_time,
            "train_losses": train_losses,
            "train_accs": train_accs,
            "test_losses": test_losses,
            "test_accs": test_accs,
            "epochs": self.epochs,
            "loaded_from_file": False
        }
    
    def _train_epoch(self, train_loader, epoch):
        """
        训练一个轮次
        
        Args:
            train_loader: 训练数据加载器
            epoch: 当前轮次
            
        Returns:
            avg_loss: 平均损失
            accuracy: 准确率
        """
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            avg_loss = train_loss / (batch_idx + 1)
            accuracy = 100. * correct / total
            pbar.set_postfix({
                'loss': f"{avg_loss:.4f}",
                'acc': f"{accuracy:.2f}%"
            })
        
        return avg_loss, accuracy
    
    def _evaluate(self, test_loader, compute_loss=False):
        """
        评估模型
        
        Args:
            test_loader: 测试数据加载器
            compute_loss: 是否计算损失
            
        Returns:
            avg_loss: 平均损失（如果compute_loss为True）
            accuracy: 准确率
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 计算损失
                if compute_loss:
                    loss = self.criterion(outputs, targets)
                    test_loss += loss.item()
                
                # 统计
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        
        if compute_loss:
            avg_loss = test_loss / len(test_loader)
            return avg_loss, accuracy
        else:
            return accuracy
    
    def train_distilled(self, teacher_model, alpha=0.5, temperature=4.0, force_retrain=False):
        """
        使用知识蒸馏训练模型
        
        Args:
            teacher_model: 教师模型
            alpha: 软标签权重
            temperature: 温度参数
            force_retrain: 是否强制重新训练
            
        Returns:
            model: 训练好的模型
            stats: 训练统计信息
        """
        model_path = os.path.join(MODELS_DIR, f"{self.model_id}_distilled.pth")
        
        # 如果模型已存在且不强制重新训练，则直接加载
        if os.path.exists(model_path) and not force_retrain:
            print(f"加载现有蒸馏模型: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            
            # 评估模型
            _, test_loader = self._get_data_loaders()
            test_acc = self._evaluate(test_loader)
            
            return self.model, {
                "best_acc": test_acc,
                "training_time": 0,
                "loaded_from_file": True,
                "is_distilled": True
            }
        
        # 获取数据加载器
        train_loader, test_loader = self._get_data_loaders()
        
        # 训练模型
        print(f"开始蒸馏训练模型: {self.model_id}")
        start_time = time.time()
        
        best_acc = 0
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []
        
        # 设置教师模型为评估模式
        teacher_model.eval()
        
        for epoch in range(self.epochs):
            # 训练一个轮次
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                # 教师模型输出
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)
                
                # 计算蒸馏损失
                # 硬标签损失
                hard_loss = self.criterion(outputs, targets)
                
                # 软标签损失
                soft_targets = torch.nn.functional.softmax(teacher_outputs / temperature, dim=1)
                soft_outputs = torch.nn.functional.log_softmax(outputs / temperature, dim=1)
                soft_loss = torch.nn.functional.kl_div(soft_outputs, soft_targets, reduction='batchmean') * (temperature ** 2)
                
                # 总损失
                loss = alpha * soft_loss + (1 - alpha) * hard_loss
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                # 统计
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 更新进度条
                avg_loss = train_loss / (batch_idx + 1)
                accuracy = 100. * correct / total
                pbar.set_postfix({
                    'loss': f"{avg_loss:.4f}",
                    'acc': f"{accuracy:.2f}%"
                })
            
            train_losses.append(avg_loss)
            train_accs.append(accuracy)
            
            # 评估模型
            test_loss, test_acc = self._evaluate(test_loader, compute_loss=True)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            
            # 更新学习率
            self.scheduler.step()
            
            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(self.model.state_dict(), model_path)
                print(f"保存最佳蒸馏模型，准确率: {best_acc:.2f}%")
        
        training_time = time.time() - start_time
        print(f"蒸馏模型训练完成，耗时: {training_time:.2f}秒，最佳准确率: {best_acc:.2f}%")
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # 返回训练好的模型和统计信息
        return self.model, {
            "best_acc": best_acc,
            "training_time": training_time,
            "train_losses": train_losses,
            "train_accs": train_accs,
            "test_losses": test_losses,
            "test_accs": test_accs,
            "epochs": self.epochs,
            "loaded_from_file": False,
            "is_distilled": True,
            "distillation_params": {
                "alpha": alpha,
                "temperature": temperature
            }
        }
    
    def train_with_zkp_sampling(self, train_loader, test_loader, zkp_generator, sample_rate=0.1):
        """
        训练模型并进行ZKP抽样验证
        
        Args:
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            zkp_generator: ZKP证明生成器
            sample_rate: 抽样率
            
        Returns:
            model: 训练好的模型
            stats: 训练统计信息
            zkp_proofs: ZKP证明列表
        """
        # 训练模型
        model, stats = self.train()
        
        # 收集模型输出分布
        model_outputs = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                outputs = torch.softmax(model(inputs), dim=1)
                model_outputs.extend(outputs.cpu().numpy().tolist())
        
        model_outputs = np.array(model_outputs)
        
        # 创建一个基线分布用于比较
        # 这里我们创建一个均匀分布作为基线
        num_classes = 10  # CIFAR-10 有 10 个类别
        baseline_distribution = np.ones((len(model_outputs), num_classes)) / num_classes
        
        # 计算KL散度并生成证明
        kl_proofs = []
        target_kl = 1.27  # 基线KL散度均值
        kl_threshold = 0.21  # 允许的偏差范围
        
        # 生成ZKP证明
        proof = zkp_generator.generate_kl_proof(
            target_distribution=model_outputs,
            baseline_distribution=baseline_distribution,  # 使用创建的基线分布
            threshold=target_kl + kl_threshold
        )
        
        # 验证KL散度是否在允许范围内
        if proof["kl_divergence"] <= target_kl + kl_threshold and \
           proof["kl_divergence"] >= target_kl - kl_threshold:
            proof["verification_result"] = "通过"
            print(f"KL散度验证通过: {proof['kl_divergence']:.4f}")
        else:
            proof["verification_result"] = "失败"
            print(f"KL散度验证失败: {proof['kl_divergence']:.4f}")
        
        kl_proofs.append(proof)
        
        # 更新统计信息
        stats.update({
            "kl_divergence": proof["kl_divergence"],
            "kl_threshold": target_kl + kl_threshold,
            "verification_result": proof["verification_result"]
        })
        
        return model, stats, kl_proofs

    def train_step(self, inputs, targets, optimizer):
        """单步训练"""
        # Zero gradients
        optimizer.zero_grad()
        
        if self.scaler is not None:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            # Scale loss and compute gradients
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Regular training
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        return loss.item(), outputs

    def save_model(self, model, model_path):
        """
        保存模型
        
        Args:
            model: 要保存的模型
            model_path: 保存路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # 保存模型
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存至: {model_path}")