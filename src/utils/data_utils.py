import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_cifar10_loaders(batch_size=32, num_workers=0, use_small=False):
    """
    加载 CIFAR-10 数据集并返回训练和测试数据加载器。

    Args:
        batch_size (int): 数据加载器的批次大小。
        num_workers (int): 数据加载器的工作线程数。
        use_small (bool): 是否使用较小的数据集。

    Returns:
        Tuple[DataLoader, DataLoader]: 训练和测试数据加载器。
    """
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
        train_size = int(0.1 * len(trainset))
        indices = torch.randperm(len(trainset))[:train_size]
        trainset = torch.utils.data.Subset(trainset, indices)

        test_size = int(0.2 * len(testset))
        indices = torch.randperm(len(testset))[:test_size]
        testset = torch.utils.data.Subset(testset, indices)

    # 创建数据加载器
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader