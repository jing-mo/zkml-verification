import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


def get_parameter_hash(self):
    """
    计算模型参数的哈希值，用于验证模型身份

    Returns:
        参数哈希值
    """
    params = []
    for name, param in self.named_parameters():
        params.append((name, param.data.cpu().numpy().sum()))

    # 使用简单的字符串哈希
    return str(hash(str(params)))
class VerifiableResNet50(nn.Module):
    """ResNet50模型，添加了必要的内部状态追踪，以支持零知识证明验证"""

    def __init__(self, num_classes=1000, pretrained=False):
        super(VerifiableResNet50, self).__init__()

        # 是否使用预训练权重
        if pretrained:
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.model = resnet50(weights=None)

        # 修改最后的全连接层以匹配目标类别数
        if num_classes != 1000:
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # 用于追踪内部状态的字典
        self.activation_trace = {}
        self.gradient_trace = {}
        self.hooks = []

        # 为所有层注册前向和后向钩子
        self._register_hooks()

    def _register_hooks(self):
        """注册钩子以捕获激活和梯度"""

        def get_activation(name):
            def hook(module, input, output):
                self.activation_trace[name] = output.detach()

            return hook

        def get_gradient(name):
            def hook(module, grad_input, grad_output):
                self.gradient_trace[name] = (
                    tuple(g.detach() if g is not None else None for g in grad_input),
                    tuple(g.detach() if g is not None else None for g in grad_output)
                )

            return hook

        # 注册主要层的钩子
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                self.hooks.append(module.register_forward_hook(get_activation(f"{name}_fwd")))
                self.hooks.append(module.register_backward_hook(get_gradient(f"{name}_bwd")))

    def forward(self, x):
        """前向传播"""
        return self.model(x)

    def get_parameter_hash(self):
        """计算模型参数的哈希值"""
        import hashlib

        # 将所有参数连接成一个大张量
        all_params = torch.cat([p.view(-1) for p in self.parameters()])

        # 计算SHA-256哈希
        param_bytes = all_params.cpu().detach().numpy().tobytes()
        return hashlib.sha256(param_bytes).hexdigest()

    def clear_traces(self):
        """清除激活和梯度的追踪记录"""
        self.activation_trace.clear()
        self.gradient_trace.clear()

    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __del__(self):
        """析构函数，确保钩子被移除"""
        self.remove_hooks()


class TeacherResNet50(VerifiableResNet50):
    """教师模型，用于知识蒸馏"""

    def __init__(self, num_classes=1000, pretrained=True):
        super(TeacherResNet50, self).__init__(num_classes=num_classes, pretrained=pretrained)

    def get_soft_targets(self, x, temperature=1.0):
        """生成软标签，用于知识蒸馏"""
        logits = self.forward(x)
        return F.softmax(logits / temperature, dim=1)


class StudentResNet50(VerifiableResNet50):
    """学生模型，用于知识蒸馏"""

    def __init__(self, num_classes=1000, pretrained=False):
        super(StudentResNet50, self).__init__(num_classes=num_classes, pretrained=pretrained)
        # Use the original ResNet50 architecture
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.model = resnet50(weights=weights)  # Use weights instead of pretrained
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)