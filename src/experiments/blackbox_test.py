import os
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Optional

from src.verification.blackbox import BlackBoxVerifier
from src.models.resnet import ResNetModel
from src.training.trainer import ModelTrainer

# 定义常量
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

def run_blackbox_experiment(num_baselines: int = 5, confidence: float = 0.95, 
                           load_models: bool = False, model_dir: Optional[str] = None) -> Dict:
    """
    运行黑盒KL散度验证实验
    
    Args:
        num_baselines: 基线模型数量
        confidence: 置信度水平
        load_models: 是否加载已有模型
        model_dir: 模型目录路径
        
    Returns:
        实验结果
    """
    print(f"开始黑盒KL散度验证实验 (基线模型数量: {num_baselines}, 置信度: {confidence*100}%)")
    
    # 确保目录存在
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 使用指定的模型目录
    if model_dir:
        models_path = model_dir
    else:
        models_path = MODELS_DIR
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 训练或加载模型
    baseline_paths = []
    target_path = os.path.join(models_path, "target.pth")
    
    if load_models and all(os.path.exists(os.path.join(models_path, f"baseline_{i}.pth")) 
                          for i in range(num_baselines)) and os.path.exists(target_path):
        print("加载已有模型...")
        for i in range(num_baselines):
            baseline_path = os.path.join(models_path, f"baseline_{i}.pth")
            baseline_paths.append(baseline_path)
            print(f"已加载基线模型 {i+1}/{num_baselines}: {baseline_path}")
    else:
        print("训练新模型...")
        # 训练目标模型
        print("训练目标模型...")
        target_trainer = ModelTrainer(device=device.type, batch_size=128, epochs=20)
        target_model = target_trainer.train()
        torch.save(target_model.state_dict(), target_path)
        print(f"目标模型已保存至: {target_path}")
        
        # 训练基线模型
        for i in range(num_baselines):
            print(f"训练基线模型 {i+1}/{num_baselines}...")
            # 使用不同的随机种子
            seed = 42 + i
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            
            trainer = ModelTrainer(device=device.type, batch_size=128, epochs=20)
            model = trainer.train()
            
            baseline_path = os.path.join(models_path, f"baseline_{i}.pth")
            torch.save(model.state_dict(), baseline_path)
            baseline_paths.append(baseline_path)
            print(f"基线模型 {i+1} 已保存至: {baseline_path}")
    
    # 执行黑盒验证
    print("\n开始黑盒验证...")
    verifier = BlackBoxVerifier()
    result = verifier.verify(
        test_model_path=target_path,
        baseline_models_paths=baseline_paths,
        test_data_path=DATA_DIR
    )
    
    # 生成报告
    report = {
        "experiment_type": "blackbox_kl_verification",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "num_baselines": num_baselines,
            "confidence_level": confidence,
            "architecture": "ResNet50",
            "dataset": "CIFAR-10"
        },
        "results": result,
        "conclusion": "通过" if result["passed"] else "未通过"
    }
    
    # 保存报告
    report_path = os.path.join(RESULTS_DIR, "blackbox_verification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    # 生成可读性报告
    markdown_report = generate_markdown_report(report)
    markdown_path = os.path.join(RESULTS_DIR, "blackbox_verification_report.md")
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)
    
    print(f"\n黑盒验证完成！")
    print(f"结论: {'通过' if result['passed'] else '未通过'}")
    print(f"KL散度均值: {result['test_kl']['mean']:.6f}")
    print(f"95%置信区间: [{result['confidence_interval']['lower']:.6f}, {result['confidence_interval']['upper']:.6f}]")
    print(f"详细报告已保存至: {markdown_path}")
    
    return report

def generate_markdown_report(report: Dict) -> str:
    """生成Markdown格式的报告"""
    result = report["results"]
    
    markdown = f"""# 黑盒KL散度验证报告

## 实验信息
- **实验类型**: 黑盒KL散度验证
- **时间**: {report["timestamp"]}
- **结论**: {report["conclusion"]}

## 实验参数
- **架构**: {report["parameters"]["architecture"]}
- **数据集**: {report["parameters"]["dataset"]}
- **基线模型数量**: {report["parameters"]["num_baselines"]}
- **置信度水平**: {report["parameters"]["confidence_level"] * 100}%

## 验证结果
- **KL散度均值**: {result["test_kl"]["mean"]:.6f}
- **KL散度标准差**: {result["test_kl"]["std"]:.6f}
- **置信区间**: [{result["confidence_interval"]["lower"]:.6f}, {result["confidence_interval"]["upper"]:.6f}]
- **验证通过**: {"是" if result["passed"] else "否"}

## 性能指标
- **验证时间**: {result["verification_time"]:.2f} 秒
- **证明生成时间**: {result["zkp_proof"]["metrics"].get("prove_time_ms", 0) / 1000:.2f} 秒
- **证明验证时间**: {result["zkp_proof"]["metrics"].get("verify_time_ms", 0) / 1000:.2f} 秒
- **证明大小**: {result["zkp_proof"]["metrics"].get("proof_size_bytes", 0) / 1024:.2f} KB

## 详细KL散度值
"""
    
    # 添加KL散度值表格
    markdown += "| 基线模型 | KL散度值 |\n"
    markdown += "| --- | --- |\n"
    
    for i, kl_value in enumerate(result["test_kl"]["values"]):
        markdown += f"| 基线 {i+1} | {kl_value:.6f} |\n"
    
    return markdown