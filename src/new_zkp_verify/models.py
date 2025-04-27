"""
模型模块，包含神经网络模型定义
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BasicBlock(nn.Module):
    """ResNet基本块"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """ResNet瓶颈块"""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet模型"""
    
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    """ResNet18模型"""
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    """ResNet34模型"""
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    """ResNet50模型"""
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    """ResNet101模型"""
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    """ResNet152模型"""
    return ResNet(Bottleneck, [3, 8, 36, 3])


class ResNetModel(nn.Module):
    """ResNet模型封装"""
    
    def __init__(self, model_type='resnet18', num_classes=10):
        """
        初始化ResNet模型
        
        Args:
            model_type: 模型类型，可选值为'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
            num_classes: 类别数
        """
        super(ResNetModel, self).__init__()
        
        if model_type == 'resnet18':
            self.model = ResNet18()
        elif model_type == 'resnet34':
            self.model = ResNet34()
        elif model_type == 'resnet50':
            self.model = ResNet50()
        elif model_type == 'resnet101':
            self.model = ResNet101()
        elif model_type == 'resnet152':
            self.model = ResNet152()
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 修改最后一层以适应类别数
        if num_classes != 10:
            in_features = self.model.linear.in_features
            self.model.linear = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            output: 输出张量
        """
        return self.model(x)


def get_resnet18(num_classes=10, pretrained=False):
    """
    获取ResNet-18模型
    
    Args:
        num_classes: 类别数量
        pretrained: 是否使用预训练权重
        
    Returns:
        model: ResNet-18模型
    """
    model = models.resnet18(pretrained=pretrained)
    
    # 修改第一个卷积层以适应CIFAR-10
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # 移除最大池化层
    
    # 修改全连接层
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


def get_resnet50(num_classes=10, pretrained=False):
    """
    获取ResNet-50模型
    
    Args:
        num_classes: 类别数量
        pretrained: 是否使用预训练权重
        
    Returns:
        model: ResNet-50模型
    """
    model = models.resnet50(pretrained=pretrained)
    
    # 修改第一个卷积层以适应CIFAR-10
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # 移除最大池化层
    
    # 修改全连接层
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


class ModelArchitecture(nn.Module):
    """
    模型架构类，用于创建ResNet模型
    """
    def __init__(self, model_type='resnet50', num_classes=10, pretrained=False):
        super(ModelArchitecture, self).__init__()
        
        if model_type == 'resnet18':
            self.model = get_resnet18(num_classes, pretrained)
        elif model_type == 'resnet50':
            self.model = get_resnet50(num_classes, pretrained)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def forward(self, x):
        return self.model(x)