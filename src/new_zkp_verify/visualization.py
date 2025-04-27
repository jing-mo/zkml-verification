"""
可视化模块，用于生成实验结果的图表
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime

from new_zkp_verify.config import REPORTS_DIR

class Visualizer:
    """可视化工具类"""

    def __init__(self, exp_id=None):
        """
        初始化可视化工具

        Args:
            exp_id: 实验ID，如果不提供则使用时间戳
        """
        if exp_id is None:
            exp_id = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.exp_id = exp_id
        self.vis_dir = os.path.join(REPORTS_DIR, exp_id, "visualizations")
        os.makedirs(self.vis_dir, exist_ok=True)

    def visualize_distributions(self, target_dist, baseline_dists, distilled_dist=None):
        """
        可视化模型输出分布

        Args:
            target_dist: 目标模型的输出分布
            baseline_dists: 基线模型的输出分布列表
            distilled_dist: 蒸馏模型的输出分布（可选）

        Returns:
            dist_path: 保存的分布图路径
        """
        plt.figure(figsize=(12, 8))
        x = np.arange(len(target_dist))

        # 计算柱状图宽度
        num_models = len(baseline_dists) + (2 if distilled_dist is not None else 1)
        width = 1.0 / (num_models + 1)

        # 绘制目标模型分布
        plt.bar(x, target_dist, width=width, label='目标模型', color='blue')

        # 绘制基线模型分布
        for i, dist in enumerate(baseline_dists):
            plt.bar(x + width * (i + 1), dist, width=width,
                    label=f'基线 {i + 1}', alpha=0.7,
                    color=plt.cm.tab10(i % 10))

        # 如果有蒸馏模型，绘制其分布
        if distilled_dist is not None:
            plt.bar(x + width * (len(baseline_dists) + 1), distilled_dist,
                    width=width, label='蒸馏模型', color='red')

        # 设置图表
        plt.xlabel('类别', fontsize=12)
        plt.ylabel('概率', fontsize=12)
        plt.title('模型输出分布对比', fontsize=14)
        plt.xticks(x + width * (num_models / 2),
                   [f"类别 {i}" for i in range(len(target_dist))],
                   fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # 保存图表
        dist_path = os.path.join(self.vis_dir, "distributions.png")
        plt.savefig(dist_path, dpi=300)
        plt.close()

        print(f"分布可视化已保存至: {dist_path}")
        return dist_path

    def visualize_kl_divergences(self, target_kl, baseline_kl_matrix,
                               threshold=None, conf_interval=None,
                               distilled_kl=None):
        """
        可视化KL散度

        Args:
            target_kl: 目标模型与每个基线模型的KL散度
            baseline_kl_matrix: 基线模型之间的KL散度矩阵
            threshold: 阈值T（可选）
            conf_interval: 置信区间（可选）
            distilled_kl: 蒸馏模型与每个基线模型的KL散度（可选）

        Returns:
            kl_bar_path: 保存的KL散度条形图路径
            kl_heatmap_path: 保存的KL散度热图路径
        """
        # 1. 绘制KL散度条形图
        plt.figure(figsize=(12, 6))

        # 提取基线之间的KL散度值（不包括对角线）
        n_baselines = len(target_kl)
        baseline_pairs = []
        for i in range(n_baselines):
            for j in range(n_baselines):
                if i != j:
                    baseline_pairs.append(baseline_kl_matrix[i, j])

        # 设置柱状图位置
        bar_width = 0.3
        baseline_x = np.arange(len(baseline_pairs))
        target_x = np.arange(len(target_kl))

        # 绘制基线模型之间的KL散度
        plt.bar(baseline_x, baseline_pairs, width=bar_width,
                label='基线模型之间', alpha=0.7, color='green')

        # 绘制目标模型与基线模型的KL散度
        plt.bar(np.max(baseline_x) + 1 + target_x, target_kl, width=bar_width,
                label='目标模型与基线模型', alpha=0.7, color='blue')

        # 如果有蒸馏模型，绘制其KL散度
        if distilled_kl is not None:
            plt.bar(np.max(baseline_x) + 1 + np.max(target_x) + 1 + np.arange(len(distilled_kl)),
                    distilled_kl, width=bar_width,
                    label='蒸馏模型与基线模型', alpha=0.7, color='red')

        # 如果有阈值和置信区间，绘制这些线
        if threshold is not None:
            plt.axhline(y=threshold, linestyle='--', color='red',
                        label=f'阈值 T')

        if conf_interval is not None:
            plt.axhline(y=conf_interval[0], linestyle=':', color='orange',
                        label=f'置信区间下限')
            plt.axhline(y=conf_interval[1], linestyle=':', color='orange',
                        label=f'置信区间上限')

        # 设置图表
        plt.xlabel('模型对比', fontsize=12)
        plt.ylabel('KL散度', fontsize=12)
        plt.title('KL散度对比', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # 保存图表
        kl_bar_path = os.path.join(self.vis_dir, "kl_divergence_bars.png")
        plt.savefig(kl_bar_path, dpi=300)
        plt.close()

        # 2. 绘制KL散度热图
        plt.figure(figsize=(10, 8))

        # 创建包含所有模型的大矩阵
        n_all = n_baselines + (2 if distilled_kl is not None else 1)
        all_labels = [f"基线 {i+1}" for i in range(n_baselines)] + ["目标模型"]
        if distilled_kl is not None:
            all_labels.append("蒸馏模型")

        full_matrix = np.zeros((n_all, n_all))

        # 填充基线模型部分
        full_matrix[:n_baselines, :n_baselines] = baseline_kl_matrix

        # 填充目标模型行和列
        for i in range(n_baselines):
            full_matrix[n_baselines, i] = target_kl[i]
            # KL散度不是对称的，但为了可视化，假设对称
            full_matrix[i, n_baselines] = target_kl[i]

        # 如果有蒸馏模型，填充其行和列
        if distilled_kl is not None:
            for i in range(n_baselines):
                full_matrix[n_baselines+1, i] = distilled_kl[i]
                full_matrix[i, n_baselines+1] = distilled_kl[i]

        # 创建自定义的热图颜色映射
        cmap = LinearSegmentedColormap.from_list("custom_cmap", ['#f7fcf5', '#74c476', '#00441b'])

        # 绘制热图
        sns.heatmap(full_matrix, annot=True, cmap=cmap, square=True,
                    xticklabels=all_labels, yticklabels=all_labels,
                    cbar_kws={'label': 'KL散度'})

        plt.title('所有模型间的KL散度矩阵', fontsize=14)
        plt.tight_layout()

        # 保存热图
        kl_heatmap_path = os.path.join(self.vis_dir, "kl_divergence_heatmap.png")
        plt.savefig(kl_heatmap_path, dpi=300)
        plt.close()

        print(f"KL散度可视化已保存至:\n  {kl_bar_path}\n  {kl_heatmap_path}")
        return kl_bar_path, kl_heatmap_path

    def visualize_accuracy(self, baseline_stats, target_stats, distilled_stats=None):
        """
        可视化模型准确率

        Args:
            baseline_stats: 基线模型的性能统计信息列表
            target_stats: 目标模型的性能统计信息
            distilled_stats: 蒸馏模型的性能统计信息（可选）

        Returns:
            acc_path: 保存的准确率图路径
        """
        plt.figure(figsize=(10, 6))

        # 基线模型准确率
        baseline_acc = [stats.get('accuracy', stats.get('final_val_accuracy', 0))
                        for stats in baseline_stats]
        baseline_x = np.arange(len(baseline_acc))
        plt.bar(baseline_x, baseline_acc, width=0.6,
                label='基线模型', alpha=0.7, color='green')

        # 目标模型准确率
        target_acc = target_stats.get('accuracy', target_stats.get('final_val_accuracy', 0))
        plt.bar(np.max(baseline_x) + 1, target_acc, width=0.6,
                label='目标模型', alpha=0.7, color='blue')

        # 如果有蒸馏模型，添加其准确率
        if distilled_stats is not None:
            distilled_acc = distilled_stats.get('accuracy',
                                          distilled_stats.get('final_val_accuracy', 0))
            plt.bar(np.max(baseline_x) + 2, distilled_acc, width=0.6,
                    label='蒸馏模型', alpha=0.7, color='red')

        # 设置图表
        plt.xlabel('模型', fontsize=12)
        plt.ylabel('准确率 (%)', fontsize=12)
        plt.title('模型准确率对比', fontsize=14)

        # 设置x轴标签
        labels = [f"基线 {i+1}" for i in range(len(baseline_acc))]
        labels.append("目标模型")
        if distilled_stats is not None:
            labels.append("蒸馏模型")
        plt.xticks(range(len(labels)), labels, rotation=45)

        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # 保存图表
        acc_path = os.path.join(self.vis_dir, "accuracy.png")
        plt.savefig(acc_path, dpi=300)
        plt.close()

        print(f"准确率可视化已保存至: {acc_path}")
        return acc_path

    def visualize_training_metrics(self, baseline_stats, target_stats, distilled_stats=None):
        """
        可视化训练过程中的指标

        Args:
            baseline_stats: 基线模型的统计信息
            target_stats: 目标模型的统计信息
            distilled_stats: 蒸馏模型的统计信息（可选）

        Returns:
            train_loss_path: 训练损失图路径
            val_acc_path: 验证准确率图路径
        """
        # 1. 绘制训练损失图
        plt.figure(figsize=(12, 6))

        # 绘制基线模型的训练损失
        for i, stats in enumerate(baseline_stats):
            if 'train_metrics' in stats:
                epochs = [m['epoch'] for m in stats['train_metrics']]
                losses = [m['loss'] for m in stats['train_metrics']]
                plt.plot(epochs, losses, alpha=0.5, label=f'基线 {i+1}')

        # 绘制目标模型的训练损失
        if 'train_metrics' in target_stats:
            epochs = [m['epoch'] for m in target_stats['train_metrics']]
            losses = [m['loss'] for m in target_stats['train_metrics']]
            plt.plot(epochs, losses, 'b-', linewidth=2, label='目标模型')

        # 如果有蒸馏模型，绘制其训练损失
        if distilled_stats is not None and 'train_metrics' in distilled_stats:
            epochs = [m['epoch'] for m in distilled_stats['train_metrics']]
            losses = [m['loss'] for m in distilled_stats['train_metrics']]
            plt.plot(epochs, losses, 'r-', linewidth=2, label='蒸馏模型')

        # 设置图表
        plt.xlabel('轮次', fontsize=12)
        plt.ylabel('损失', fontsize=12)
        plt.title('训练损失对比', fontsize=14)
        plt.legend()
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()

        # 保存图表
        train_loss_path = os.path.join(self.vis_dir, "training_loss.png")
        plt.savefig(train_loss_path, dpi=300)
        plt.close()

        # 2. 绘制验证准确率图
        plt.figure(figsize=(12, 6))

        # 绘制基线模型的验证准确率
        for i, stats in enumerate(baseline_stats):
            if 'val_metrics' in stats:
                epochs = [m['epoch'] for m in stats['val_metrics']]
                accuracies = [m['accuracy'] for m in stats['val_metrics']]
                plt.plot(epochs, accuracies, alpha=0.5, label=f'基线 {i+1}')

        # 绘制目标模型的验证准确率
        if 'val_metrics' in target_stats:
            epochs = [m['epoch'] for m in target_stats['val_metrics']]
            accuracies = [m['accuracy'] for m in target_stats['val_metrics']]
            plt.plot(epochs, accuracies, 'b-', linewidth=2, label='目标模型')

        # 如果有蒸馏模型，绘制其验证准确率
        if distilled_stats is not None and 'val_metrics' in distilled_stats:
            epochs = [m['epoch'] for m in distilled_stats['val_metrics']]
            accuracies = [m['accuracy'] for m in distilled_stats['val_metrics']]
            plt.plot(epochs, accuracies, 'r-', linewidth=2, label='蒸馏模型')

        # 设置图表
        plt.xlabel('轮次', fontsize=12)
        plt.ylabel('准确率 (%)', fontsize=12)
        plt.title('验证准确率对比', fontsize=14)
        plt.legend()
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()

        # 保存图表
        val_acc_path = os.path.join(self.vis_dir, "validation_accuracy.png")
        plt.savefig(val_acc_path, dpi=300)
        plt.close()

        print(f"训练指标可视化已保存至:\n  {train_loss_path}\n  {val_acc_path}")
        return train_loss_path, val_acc_path

    def visualize_confidence_interval(self, baseline_kl_values, target_kl_mean,
                                    threshold=None, conf_interval=None):
        """
        可视化置信区间和目标模型KL散度

        Args:
            baseline_kl_values: 基线KL散度值列表
            target_kl_mean: 目标模型KL散度均值
            threshold: 阈值T（可选）
            conf_interval: 置信区间（可选）

        Returns:
            conf_path: 保存的置信区间图路径
        """
        plt.figure(figsize=(10, 6))

        # 绘制基线KL散度直方图
        plt.hist(baseline_kl_values, bins=10, alpha=0.5, color='green', label='基线模型KL散度分布')

        # 添加密度曲线
        density = sns.kdeplot(baseline_kl_values, color='green', linewidth=2)

        # 绘制目标模型KL散度
        plt.axvline(x=target_kl_mean, color='blue', linewidth=2, label=f'目标模型KL均值: {target_kl_mean:.6f}')

        # 如果有阈值和置信区间，绘制这些线
        if threshold is not None:
            plt.axvline(x=threshold, linestyle='--', color='red',
                        label=f'阈值T: {threshold:.6f}')

        if conf_interval is not None:
            plt.axvspan(conf_interval[0], conf_interval[1], alpha=0.2, color='green',
                    label=f'95%置信区间: [{conf_interval[0]:.6f}, {conf_interval[1]:.6f}]')

        # 计算并显示均值和标准差
        mean = np.mean(baseline_kl_values)
        std = np.std(baseline_kl_values, ddof=1)
        plt.text(0.95, 0.95, f'均值: {mean:.6f}\n标准差: {std:.6f}',
                transform=plt.gca().transAxes, verticalalignment='top',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 设置图表
        plt.xlabel('KL散度', fontsize=12)
        plt.ylabel('频数', fontsize=12)
        plt.title('KL散度分布与置信区间分析', fontsize=14)
        plt.legend()
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()

        # 保存图表
        conf_path = os.path.join(self.vis_dir, "confidence_interval.png")
        plt.savefig(conf_path, dpi=300)
        plt.close()

        print(f"置信区间可视化已保存至: {conf_path}")
        return conf_path

    def create_report_summary(self, baseline_stats, target_stats, kl_stats, verification_result,
                            distilled_stats=None, distilled_verification=None):
        """
        创建实验结果摘要图

        Args:
            baseline_stats: 基线模型统计信息
            target_stats: 目标模型统计信息
            kl_stats: KL散度统计信息
            verification_result: 验证结果
            distilled_stats: 蒸馏模型统计信息（可选）
            distilled_verification: 蒸馏模型验证结果（可选）

        Returns:
            summary_path: 摘要图路径
        """
        plt.figure(figsize=(14, 10))

        # 创建网格布局
        gs = plt.GridSpec(3, 2, figure=plt.gcf())

        # 1. 准确率对比图
        ax1 = plt.subplot(gs[0, 0])
        baseline_acc = [stats.get('accuracy', stats.get('final_val_accuracy', 0))
                        for stats in baseline_stats]
        target_acc = target_stats.get('accuracy', target_stats.get('final_val_accuracy', 0))

        acc_data = baseline_acc + [target_acc]
        labels = [f"基线 {i+1}" for i in range(len(baseline_acc))] + ["目标模型"]
        colors = ['green'] * len(baseline_acc) + ['blue']

        if distilled_stats is not None:
            distilled_acc = distilled_stats.get('accuracy', distilled_stats.get('final_val_accuracy', 0))
            acc_data.append(distilled_acc)
            labels.append("蒸馏模型")
            colors.append('red')

        ax1.bar(range(len(acc_data)), acc_data, color=colors, alpha=0.7)
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.set_ylabel('准确率 (%)')
        ax1.set_title('模型准确率对比')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # 2. KL散度对比图
        ax2 = plt.subplot(gs[0, 1])
        baseline_kl_mean = kl_stats["kl_mean"]
        target_kl_mean = verification_result["target_kl_mean"]

        kl_data = [baseline_kl_mean, target_kl_mean]
        kl_labels = ["基线平均", "目标模型"]
        kl_colors = ['green', 'blue']

        if distilled_verification is not None:
            distilled_kl_mean = distilled_verification["distilled_kl_mean"]
            kl_data.append(distilled_kl_mean)
            kl_labels.append("蒸馏模型")
            kl_colors.append('red')

        ax2.bar(range(len(kl_data)), kl_data, color=kl_colors, alpha=0.7)
        ax2.set_xticks(range(len(kl_labels)))
        ax2.set_xticklabels(kl_labels)
        ax2.set_ylabel('KL散度')
        ax2.set_title('KL散度对比')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)

        # 3. 置信区间图
        ax3 = plt.subplot(gs[1, :])
        sns.kdeplot(kl_stats["kl_values"], color='green', linewidth=2, ax=ax3, label='基线KL散度分布')
        ax3.axvline(x=verification_result["target_kl_mean"], color='blue', linewidth=2,
                label=f'目标模型: {verification_result["target_kl_mean"]:.6f}')

        if distilled_verification is not None:
            ax3.axvline(x=distilled_verification["distilled_kl_mean"], color='red', linewidth=2,
                    label=f'蒸馏模型: {distilled_verification["distilled_kl_mean"]:.6f}')

        ax3.axvspan(kl_stats["confidence_interval"][0], kl_stats["confidence_interval"][1],
                alpha=0.2, color='green',
                label=f'95%置信区间: [{kl_stats["confidence_interval"][0]:.6f}, {kl_stats["confidence_interval"][1]:.6f}]')

        ax3.axvline(x=kl_stats["threshold_T"], linestyle='--', color='red',
                label=f'阈值T: {kl_stats["threshold_T"]:.6f}')

        ax3.set_xlabel('KL散度')
        ax3.set_ylabel('密度')
        ax3.set_title('KL散度分布与置信区间')
        ax3.legend()
        ax3.grid(linestyle='--', alpha=0.7)

        # 4. 验证结果摘要
        ax4 = plt.subplot(gs[2, :])
        ax4.axis('off')  # 关闭坐标轴

        result_text = f"""
        验证结果摘要:
        
        基于{kl_stats['confidence_level']*100}%置信度的统计分析，目标模型的KL散度均值({verification_result['target_kl_mean']:.6f})
        {verification_result['position']}
        ({kl_stats['confidence_interval'][0]:.6f}, {kl_stats['confidence_interval'][1]:.6f})
        
        验证结论: {'通过' if verification_result['passed_verification'] else '不通过'}
        
        • 基线模型KL散度均值: {kl_stats['kl_mean']:.6f}
        • 基线模型KL散度标准差: {kl_stats['kl_std']:.6f}
        • 目标模型KL散度均值: {verification_result['target_kl_mean']:.6f}
        • 置信区间: [{kl_stats['confidence_interval'][0]:.6f}, {kl_stats['confidence_interval'][1]:.6f}]
        • 阈值T: {kl_stats['threshold_T']:.6f}
        """

        if distilled_verification is not None:
            result_text += f"""
        • 蒸馏模型KL散度均值: {distilled_verification['distilled_kl_mean']:.6f}
        • 蒸馏模型验证结果: {'通过' if distilled_verification['passed_verification'] else '不通过'}
        """

        ax4.text(0.5, 0.5, result_text, fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='#eaf2f8', alpha=0.8))

        # 保存图表
        plt.tight_layout()
        summary_path = os.path.join(self.vis_dir, "summary.png")
        plt.savefig(summary_path, dpi=300)
        plt.close()

        print(f"实验摘要可视化已保存至: {summary_path}")
        return summary_path