import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List

class ReportGenerator:
    """实验报告生成器"""
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir

    
    def generate_verification_report(self, 
                                   wb_metrics: Dict,
                                   bb_metrics: Dict,
                                   kl_stats: Dict) -> str:
        """生成验证报告"""
        
        # 计算综合指标
        total_time = (wb_metrics.get("prove_time_ms", 0) + 
                     bb_metrics.get("prove_time_ms", 0))
        total_size = (wb_metrics.get("proof_size_bytes", 0) + 
                     bb_metrics.get("proof_size_bytes", 0))
        
        # 生成报告内容
        report = [
            "**独立性验证报告**\n",
            
            "1. **基线模型配置**:",
            "   - 架构: ResNet-18",
            "   - 训练数据: CIFAR-10 (50,000样本)",
            "   - 超参数: SGD(lr=0.1, momentum=0.9), 200 epochs",
            "   - 随机种子: [42, 123, 456, 789, 1024]\n",
            
            "2. **基线KL分布**:",
            f"   - 样本量: {kl_stats['sample_size']}对",
            f"   - 均值±标准差: {kl_stats['mean']:.2f} ± {kl_stats['std']:.2f}",
            f"   - 95%置信区间: [{kl_stats['ci_lower']:.2f}, {kl_stats['ci_upper']:.2f}]\n",
            
            "3. **目标模型验证**:",
            f"   - KL均值: {kl_stats['test_mean']:.2f}",
            f"   - t检验结果: t={kl_stats['t_stat']:.2f}, p={kl_stats['p_value']:.2f}",
            f"   - 结论: {'符合独立训练特征' if kl_stats['p_value'] > 0.05 else '不符合独立训练特征'} "
            f"(p {'>' if kl_stats['p_value'] > 0.05 else '<='} 0.05)。\n",
            
            "4. **性能指标**:",
            f"   - 编译电路时间: {wb_metrics['compile_time_ms']} ms",
            f"   - 创建电路时间: {wb_metrics['setup_time_ms']} ms", 
            f"   - 生成证明时间: {total_time} ms",
            f"   - 验证时间: {wb_metrics['verify_time_ms']} ms",
            f"   - 证明内存占用: {total_size} 字节",
            f"   - PK长度: {wb_metrics['pk_size_bytes']} Bytes",
            f"   - VK长度: {wb_metrics['vk_size_bytes']} Bytes\n"
        ]
        
        # 保存报告
        report_path = os.path.join(self.results_dir, 
                                 f"verification_report_{datetime.now():%Y%m%d_%H%M%S}.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report))
            
        return report_path