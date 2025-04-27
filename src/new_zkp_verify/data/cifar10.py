import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_cifar10_loaders(batch_size=128, num_workers=4, test_only=False, test_samples=1000):
    """
    获取CIFAR-10数据加载器
    
    Args:
        batch_size: 批次大小
        num_workers: 数据加载线程数
        test_only: 是否只返回测试加载器
        test_samples: 测试样本数量
        
    Returns:
        如果test_only为False，返回(train_loader, test_loader)
        否则只返回test_loader
    """
    # 定义数据变换
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
    
    # 加载测试集
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # 如果指定了测试样本数量，则创建子集
    if test_samples < len(testset):
        # 固定随机种子以确保一致性
        np.random.seed(42)
        indices = np.random.choice(len(testset), test_samples, replace=False)
        testset = Subset(testset, indices)
    
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    if test_only:
        return test_loader
    
    # 加载训练集
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    return train_loader, test_loader