"""
报告生成模块，用于生成实验报告
"""
import os
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from new_zkp_verify.config import REPORTS_DIR


# 将convert_numpy函数移到类外部，并确保在generate_all_reports中正确调用
def convert_numpy(obj):
    """转换NumPy类型为Python原生类型，用于JSON序列化"""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.generic,)):
        return obj.item()
    return obj

class ReportGenerator:
    """报告生成器"""
    def calculate_kl_values(self, baseline_stats, target_stats):
        """计算KL散度值"""
        kl_values = []
        for baseline, target in zip(baseline_stats, target_stats):
            kl_value = self.kl_divergence(baseline, target)
            kl_values.append(kl_value)
        return kl_values
    def kl_divergence(self, p, q):
        """计算两个分布之间的KL散度"""
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))
    def __init__(self, exp_id=None):
        """
        初始化报告生成器
        
        Args:
            exp_id: 实验ID
        """
        self.exp_id = exp_id or f"experiment_{int(time.time())}"
        self.report_dir = os.path.join(REPORTS_DIR, self.exp_id)
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(os.path.join(self.report_dir, "visualizations"), exist_ok=True)

    def generate_all_reports(self, report_data, baseline_stats=None, target_stats=None):
        """
        生成所有格式的报告
        
        Args:
            report_data: 报告数据
            baseline_stats: 基线模型统计信息
            target_stats: 目标模型统计信息
            
        Returns:
            report_paths: 包含所有生成报告路径的字典
        """
        report_paths = {}
        
        # 确保报告目录存在
        os.makedirs(self.report_dir, exist_ok=True)
        
        # Validate baseline_stats and target_stats
        if baseline_stats is None or target_stats is None:
            raise ValueError("baseline_stats and target_stats must be provided and cannot be None.")
        
        # 计算每个模型对的 KL 值
        kl_values = self.calculate_kl_values(baseline_stats, target_stats)
        report_data['kl_values'] = kl_values
        
        # 生成JSON报告
        json_path = os.path.join(self.report_dir, f"report_{self.exp_id}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=convert_numpy)
        report_paths['json'] = json_path
        
        # 生成HTML报告
        html_path = self.generate_comprehensive_report(report_data, baseline_stats, target_stats)
        report_paths['html'] = html_path
        
        # 生成Markdown报告
        md_path = self.generate_markdown_report(report_data)
        report_paths['markdown'] = md_path
        
        # 生成性能报告
        perf_path = self.generate_performance_report(report_data.get('training_times', {}), report_data.get('verification_times', {}))
        report_paths['performance'] = perf_path
        
        return report_paths
    
    def generate_comprehensive_report(self, report_data, baseline_stats=None, target_stats=None):
        # 生成HTML报告
        html_path = os.path.join(self.report_dir, f"report_{self.exp_id}.html")
        
        # 处理可视化路径
        visualization_path = list(report_data.get("visualizations", {}).values())[0] if report_data.get("visualizations", {}) else None
        if visualization_path:
            rel_path = Path(visualization_path).relative_to(self.report_dir).as_posix()
            img_tag = f'<img src="{rel_path}" alt="KL散度分布">'
        else:
            img_tag = '<p>无可视化数据</p>'
        
        # 预处理数据
        proof_rows = []
        for proof in report_data.get('zero_knowledge_proofs', []):
            proof_type = "批次证明" if "input_hash" in proof else "KL散度证明" if "target_hash" in proof else "联合证明"
            generation_time = f"{proof.get('generation_time_ms', 0)/1000:.2f}秒"
            proof_size = f"{proof.get('proof_size_bytes', 0)/1024:.2f}KB"
            verification_result = proof.get('verification_result', 'N/A')
            css_class = 'success' if verification_result == '通过' else 'failure'
            
            row = f"<tr><td>{proof.get('proof_id', 'N/A')}</td><td>{proof_type}</td><td>{generation_time}</td><td>{proof_size}</td><td class='{css_class}'>{verification_result}</td></tr>"
            proof_rows.append(row)
        
        metric_rows = []
        for key, value in report_data.get('experiment_metrics', {}).items():
            metric_rows.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
        
        # 提前计算可能导致嵌套问题的值
        verification_passed = report_data.get('summary', {}).get('passed_verification', False)
        result_class = 'success' if verification_passed else 'failure'
        result_text = "通过" if verification_passed else "失败"
        
        confidence_level = report_data.get('summary', {}).get('confidence_level', 0) * 100
        target_kl_mean = report_data.get('summary', {}).get('target_kl_mean', 0)
        target_kl_mean_formatted = f"{target_kl_mean:.4f}"
        kl_threshold = report_data.get('summary', {}).get('kl_threshold_T', 0)
        kl_threshold_formatted = f"{kl_threshold:.4f}"
        
        baseline_count = report_data.get('baseline_models', {}).get('count', 0)
        baseline_avg_accuracy = report_data.get('summary', {}).get('baseline_avg_accuracy', 0)
        baseline_avg_accuracy_formatted = f"{baseline_avg_accuracy:.2f}%"
        
        zkp_count = report_data.get('summary', {}).get('zkp_count', 0)
        zkp_generation_time = report_data.get('experiment_metrics', {}).get('zkp_generation_time', 'N/A')
        
        # 目标模型信息
        model_id = report_data.get('target_model', {}).get('config', {}).get('model_id', 'N/A')
        seed = report_data.get('target_model', {}).get('config', {}).get('seed', 'N/A')
        epochs = report_data.get('target_model', {}).get('config', {}).get('epochs', 'N/A')
        batch_size = report_data.get('target_model', {}).get('config', {}).get('batch_size', 'N/A')
        learning_rate = report_data.get('target_model', {}).get('config', {}).get('learning_rate', 'N/A')
        is_distilled = report_data.get('target_model', {}).get('config', {}).get('is_distilled', False)
        is_distilled_text = "是" if is_distilled else "否"
        best_accuracy = report_data.get('target_model', {}).get('config', {}).get('best_accuracy', 'N/A')
        best_accuracy_formatted = f"{best_accuracy:.2f}%" if isinstance(best_accuracy, (int, float)) else best_accuracy
        training_time = report_data.get('target_model', {}).get('config', {}).get('training_time', 'N/A')
        
        # KL散度统计
        kl_mean = report_data.get('kl_divergence_stats', {}).get('kl_mean', 0)
        kl_mean_formatted = f"{kl_mean:.4f}"
        kl_std = report_data.get('kl_divergence_stats', {}).get('kl_std', 0)
        kl_std_formatted = f"{kl_std:.4f}"
        
        conf_interval = report_data.get('kl_divergence_stats', {}).get('confidence_interval', [0, 0])
        conf_interval_lower = f"{conf_interval[0]:.4f}"
        conf_interval_upper = f"{conf_interval[1]:.4f}"
        
        # 生成时间
        timestamp = report_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 零知识证明统计
        proof_generation_time = report_data.get('experiment_metrics', {}).get('avg_proof_generation_time', 'N/A')
        proof_size = report_data.get('experiment_metrics', {}).get('avg_proof_size', 'N/A')
        
        # 构建HTML内容
        html_content = f"""<!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>实验报告 - {self.exp_id}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .section {{
                    margin-bottom: 30px;
                    padding: 20px;
                    background-color: #f9f9f9;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th, td {{
                    padding: 12px 15px;
                    border: 1px solid #ddd;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:nth-child(even) {{
                    background-color: #f8f8f8;
                }}
                .success {{
                    color: green;
                    font-weight: bold;
                }}
                .failure {{
                    color: red;
                    font-weight: bold;
                }}
                .visualization {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .visualization img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                .metrics {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                }}
                .metric-card {{
                    flex: 1;
                    min-width: 200px;
                    padding: 15px;
                    background-color: #fff;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                .footer {{
                    margin-top: 30px;
                    text-align: center;
                    font-size: 14px;
                    color: #777;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>零知识神经网络验证实验报告</h1>
                <p>实验ID: {self.exp_id}</p>
                <p>生成时间: {timestamp}</p>
                
                <div class="section">
                    <h2>实验摘要</h2>
                    <div class="metrics">
                        <div class="metric-card">
                            <h3>验证结果</h3>
                            <div class="metric-value {result_class}">
                                {result_text}
                            </div>
                            <p>置信水平: {confidence_level}%</p>
                        </div>
                        <div class="metric-card">
                            <h3>KL散度</h3>
                            <div class="metric-value">
                                {target_kl_mean_formatted}
                            </div>
                            <p>阈值: {kl_threshold_formatted}</p>
                        </div>
                        <div class="metric-card">
                            <h3>基线模型数量</h3>
                            <div class="metric-value">
                                {baseline_count}
                            </div>
                            <p>平均准确率: {baseline_avg_accuracy_formatted}</p>
                        </div>
                        <div class="metric-card">
                            <h3>零知识证明</h3>
                            <div class="metric-value">
                                {zkp_count}
                            </div>
                            <p>证明生成时间: {zkp_generation_time}</p>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>目标模型信息</h2>
                    <table>
                        <tr>
                            <th>模型ID</th>
                            <td>{model_id}</td>
                        </tr>
                        <tr>
                            <th>随机种子</th>
                            <td>{seed}</td>
                        </tr>
                        <tr>
                            <th>训练轮次</th>
                            <td>{epochs}</td>
                        </tr>
                        <tr>
                            <th>批次大小</th>
                            <td>{batch_size}</td>
                        </tr>
                        <tr>
                            <th>学习率</th>
                            <td>{learning_rate}</td>
                        </tr>
                        <tr>
                            <th>是否蒸馏</th>
                            <td>{is_distilled_text}</td>
                        </tr>
                        <tr>
                            <th>最佳准确率</th>
                            <td>{best_accuracy_formatted}</td>
                        </tr>
                        <tr>
                            <th>训练时间</th>
                            <td>{training_time}</td>
                        </tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>KL散度分析</h2>
                    <p>目标模型KL散度均值: {kl_mean_formatted}</p>
                    <p>目标模型KL散度标准差: {kl_std_formatted}</p>
                    <p>置信区间: [{conf_interval_lower}, {conf_interval_upper}]</p>
                    
                    <div class="visualization">
                        <h3>KL散度分布</h3>
                        {img_tag}
                    </div>
                </div>
                
                <div class="section">
                    <h2>零知识证明</h2>
                    <p>证明总数: {zkp_count}</p>
                    <p>平均证明生成时间: {proof_generation_time}</p>
                    <p>平均证明大小: {proof_size}</p>
                    
                    <h3>证明详情</h3>
                    <table>
                        <tr>
                            <th>证明ID</th>
                            <th>类型</th>
                            <th>生成时间</th>
                            <th>大小</th>
                            <th>验证结果</th>
                        </tr>
                        {''.join(proof_rows)}
                    </table>
                </div>
                
                <div class="section">
                    <h2>性能指标</h2>
                    <table>
                        <tr>
                            <th>指标</th>
                            <th>值</th>
                        </tr>
                        {''.join(metric_rows)}
                    </table>
                </div>
                
                <div class="footer">
                    <p>零知识神经网络验证系统 - 生成于 {current_time}</p>
                </div>
            </div>
        </body>
        </html>"""
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_path
    
    def generate_markdown_report(self, report_data):
        """
        生成Markdown报告
        """
        # Use Path for handling paths
        md_path = Path(self.report_dir) / f"report_{self.exp_id}.md"
        
        # 处理可视化路径
        vis_path = None
        if report_data.get("visualizations"):
            raw_path = list(report_data["visualizations"].values())[0]
            if raw_path:
                vis_path = Path(raw_path).relative_to(self.report_dir).as_posix()
        
        # 创建可视化部分
        vis_section = f"\n![KL散度分布]({vis_path})\n" if vis_path else "\n*无可视化数据*\n"
        
        # 预处理证明行
        proof_rows = []
        for proof in report_data.get('zero_knowledge_proofs', []):
            proof_type = "批次证明" if "input_hash" in proof else "KL散度证明" if "target_hash" in proof else "联合证明"
            generation_time = f"{proof.get('generation_time_ms', 0)/1000:.2f}秒"
            proof_size = f"{proof.get('proof_size_bytes', 0)/1024:.2f}KB"
            verification_result = proof.get('verification_result', 'N/A')
            
            row = f"| {proof.get('proof_id', 'N/A')} | {proof_type} | {generation_time} | {proof_size} | {verification_result} |\n"
            proof_rows.append(row)
        
        # 预处理指标行
        metric_rows = []
        for key, value in report_data.get('experiment_metrics', {}).items():
            metric_rows.append(f"| {key} | {value} |\n")
        
        # 预处理KL值行
        kl_rows = []
        if isinstance(report_data.get('kl_values', []), list):
            # 处理列表形式的 kl_values
            for i, kl_value in enumerate(report_data.get('kl_values', [])):
                model_pair = f"模型对 {i+1}"
                kl_rows.append(f"| {model_pair} | {kl_value:.4f} |\n")
        else:
            # 处理字典形式的 kl_values
            for model_pair, kl_value in report_data.get('kl_values', {}).items():
                kl_rows.append(f"| {model_pair} | {kl_value:.4f} |\n")
        
        # 提前格式化所有数值
        timestamp = report_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        verification_passed = "通过" if report_data.get('summary', {}).get('passed_verification', False) else "失败"
        confidence_level = f"{report_data.get('summary', {}).get('confidence_level', 0) * 100}%"
        target_kl_mean = f"{report_data.get('summary', {}).get('target_kl_mean', 0):.4f}"
        kl_threshold = f"{report_data.get('summary', {}).get('kl_threshold_T', 0):.4f}"
        baseline_count = str(report_data.get('baseline_models', {}).get('count', 0))
        baseline_avg_accuracy = f"{report_data.get('summary', {}).get('baseline_avg_accuracy', 0):.2f}%"
        zkp_count = str(report_data.get('summary', {}).get('zkp_count', 0))
        
        model_id = report_data.get('target_model', {}).get('config', {}).get('model_id', 'N/A')
        seed = report_data.get('target_model', {}).get('config', {}).get('seed', 'N/A')
        epochs = report_data.get('target_model', {}).get('config', {}).get('epochs', 'N/A')
        batch_size = report_data.get('target_model', {}).get('config', {}).get('batch_size', 'N/A')
        learning_rate = report_data.get('target_model', {}).get('config', {}).get('learning_rate', 'N/A')
        is_distilled = "是" if report_data.get('target_model', {}).get('config', {}).get('is_distilled', False) else "否"
        
        best_accuracy_value = report_data.get('target_model', {}).get('config', {}).get('best_accuracy', 'N/A')
        if isinstance(best_accuracy_value, (int, float)):
            best_accuracy = f"{best_accuracy_value:.2f}%"
        else:
            best_accuracy = str(best_accuracy_value)
        
        training_time = report_data.get('target_model', {}).get('config', {}).get('training_time', 'N/A')
        
        kl_mean = f"{report_data.get('kl_divergence_stats', {}).get('kl_mean', 0):.4f}"
        kl_std = f"{report_data.get('kl_divergence_stats', {}).get('kl_std', 0):.4f}"
        
        conf_interval = report_data.get('kl_divergence_stats', {}).get('confidence_interval', [0, 0])
        conf_interval_text = f"[{conf_interval[0]:.4f}, {conf_interval[1]:.4f}]"
        
        proof_generation_time = report_data.get('experiment_metrics', {}).get('avg_proof_generation_time', 'N/A')
        proof_size = report_data.get('experiment_metrics', {}).get('avg_proof_size', 'N/A')
        
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 创建Markdown内容
        md_content = f"""# 零知识神经网络验证实验报告

    ## 实验信息
    - **实验ID**: {self.exp_id}
    - **生成时间**: {timestamp}

    ## 实验摘要
    - **验证结果**: {verification_passed}
    - **置信水平**: {confidence_level}
    - **KL散度均值**: {target_kl_mean}
    - **KL散度阈值**: {kl_threshold}
    - **基线模型数量**: {baseline_count}
    - **基线平均准确率**: {baseline_avg_accuracy}
    - **零知识证明数量**: {zkp_count}

    ## 目标模型信息
    - **模型ID**: {model_id}
    - **随机种子**: {seed}
    - **训练轮次**: {epochs}
    - **批次大小**: {batch_size}
    - **学习率**: {learning_rate}
    - **是否蒸馏**: {is_distilled}
    - **最佳准确率**: {best_accuracy}
    - **训练时间**: {training_time}

    ## KL散度分析
    - **目标模型KL散度均值**: {kl_mean}
    - **目标模型KL散度标准差**: {kl_std}
    - **置信区间**: {conf_interval_text}

    {vis_section}

    ## 每个模型对的KL值
    | 模型对 | KL_sym 值 |
    |--------|-----------|
    {''.join(kl_rows)}

    ## 零知识证明
    - **证明总数**: {zkp_count}
    - **平均证明生成时间**: {proof_generation_time}
    - **平均证明大小**: {proof_size}

    ### 证明详情
    | 证明ID | 类型 | 生成时间 | 大小 | 验证结果 |
    |--------|------|----------|------|----------|
    {''.join(proof_rows)}

    ## 性能指标
    | 指标 | 值 |
    |------|-----|
    {''.join(metric_rows)}

    ---
    *零知识神经网络验证系统 - 生成于 {current_time}*"""
        
        # 写入文件
        md_path.write_text(md_content, encoding='utf-8')
        
        return str(md_path)
    
    def generate_performance_report(self, training_times, verification_times):
        """
        生成性能报告
        
        Args:
            training_times: 训练时间
            verification_times: 验证时间
            
        Returns:
            perf_path: 性能报告路径
        """
        perf_path = os.path.join(self.report_dir, f"performance_{self.exp_id}.json")
        
        # 创建性能报告数据
        perf_data = {
            "experiment_id": self.exp_id,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "training_times": training_times,
            "verification_times": verification_times,
            "summary": {
                "total_training_time": sum(training_times.values()),
                "total_verification_time": sum(verification_times.values()),
                "avg_training_time": sum(training_times.values()) / len(training_times) if training_times else 0,
                "avg_verification_time": sum(verification_times.values()) / len(verification_times) if verification_times else 0
            }
        }
        
        # 保存性能报告
        with open(perf_path, 'w', encoding='utf-8') as f:
            json.dump(perf_data, f, indent=2, ensure_ascii=False)
        
        # 生成性能可视化
        try:
            self._visualize_performance(perf_data)
        except Exception as e:
            print(f"生成性能可视化时出错: {e}")
        
        return perf_path
    
    def _visualize_performance(self, perf_data):
        """
        生成性能可视化
        
        Args:
            perf_data: 性能数据
        """
        vis_dir = os.path.join(self.report_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # 训练时间可视化
        if perf_data.get("training_times"):
            plt.figure(figsize=(10, 6))
            models = list(perf_data["training_times"].keys())
            times = list(perf_data["training_times"].values())
            
            plt.bar(models, times, color='skyblue')
            plt.xlabel('模型')
            plt.ylabel('训练时间 (秒)')
            plt.title('模型训练时间比较')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            train_path = os.path.join(vis_dir, "training_times.png")
            plt.savefig(train_path)
            plt.close()
        
        # 验证时间可视化
        if perf_data.get("verification_times"):
            plt.figure(figsize=(10, 6))
            verification_types = list(perf_data["verification_times"].keys())
            times = list(perf_data["verification_times"].values())
            
            plt.bar(verification_types, times, color='lightgreen')
            plt.xlabel('验证类型')
            plt.ylabel('验证时间 (秒)')
            plt.title('验证时间比较')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            verify_path = os.path.join(vis_dir, "verification_times.png")
            plt.savefig(verify_path)
            plt.close()
        
        # 总时间饼图
        if perf_data.get("summary"):
            plt.figure(figsize=(8, 8))
            labels = ['训练时间', '验证时间']
            sizes = [
                perf_data["summary"].get("total_training_time", 0),
                perf_data["summary"].get("total_verification_time", 0)
            ]
            
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightgreen'])
            plt.axis('equal')
            plt.title('训练与验证时间占比')
            
            pie_path = os.path.join(vis_dir, "time_distribution.png")
            plt.savefig(pie_path)
            plt.close()


def calculate_kl_values(self, baseline_stats, target_stats):
    """
    计算KL散度值
    
    Args:
        baseline_stats: 基线模型统计信息
        target_stats: 目标模型统计信息
        
    Returns:
        dict: 包含KL散度值的字典
    """
    kl_values = {}
    
    # 计算基线模型之间的KL散度
    for i, (model_id1, stats1) in enumerate(baseline_stats.items()):
        for j, (model_id2, stats2) in enumerate(baseline_stats.items()):
            if i < j:  # 避免重复计算
                kl_key = f"{model_id1}_vs_{model_id2}"
                kl_values[kl_key] = 0.5 * (stats1.get('kl_divergence', 0) + stats2.get('kl_divergence', 0))
    
    # 计算目标模型与基线模型的KL散度
    for model_id, stats in baseline_stats.items():
        kl_key = f"target_vs_{model_id}"
        kl_values[kl_key] = 0.5 * (target_stats.get('kl_divergence', 0) + stats.get('kl_divergence', 0))
    
    return kl_values

def compute_kl_divergence(self, dist1, dist2):
    """
    计算两个分布之间的 KL 散度
    
    Args:
        dist1: 第一个分布
        dist2: 第二个分布
        
    Returns:
        kl_divergence: KL 散度值
    """
    # 使用 numpy 计算 KL 散度
    kl_divergence = np.sum(dist1 * np.log(dist1 / dist2))
    return kl_divergence

