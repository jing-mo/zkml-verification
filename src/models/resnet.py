import torch
import torch.nn as nn
import torchvision.models as models

class ResNetModel(nn.Module):
    """ResNet50模型，适用于CIFAR-10"""
    
    def __init__(self, num_classes=10, pretrained=False):
        super(ResNetModel, self).__init__()
        
        # 加载预训练的ResNet50模型
        self.model = models.resnet50(pretrained=pretrained)
        
        # 修改第一个卷积层以适应CIFAR-10的3x32x32输入
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 移除最大池化层，因为CIFAR-10图像较小
        self.model.maxpool = nn.Identity()
        
        # 修改最后的全连接层以适应CIFAR-10的10个类别
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)