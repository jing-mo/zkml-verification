import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from new_zkp_verify.config import CACHE_DIR, CIFAR10_MEAN, CIFAR10_STD

def get_data_loaders(batch_size, num_workers=4):
    """
    获取CIFAR-10数据集的训练和测试数据加载器
    
    Args:
        batch_size: 批次大小
        num_workers: 数据加载线程数
        
    Returns:
        train_loader, test_loader: 训练和测试数据加载器
    """
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
    
    train_dataset = datasets.CIFAR10(
        root=CACHE_DIR, train=True, download=True, transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=CACHE_DIR, train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader