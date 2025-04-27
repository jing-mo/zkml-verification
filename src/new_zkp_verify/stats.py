"""
统计分析模块，用于计算KL散度、置信区间和阈值T
"""
import os
import json
import numpy as np
import scipy.stats as stats
from datetime import datetime

from new_zkp_verify.config import STATS_DIR, THRESHOLD_FILE, CONFIDENCE_LEVEL


class StatsAnalyzer:
    """统计分析类，用于分析模型分布和计算统计量"""

    def __init__(self):
        """初始化统计分析器"""
        os.makedirs(STATS_DIR, exist_ok=True)

    @staticmethod
    def kl_divergence(p, q):
        """计算KL散度，处理零概率问题"""
        # 添加小常数避免除零错误
        epsilon = 1e-10
        p = np.array(p) + epsilon
        q = np.array(q) + epsilon

        # 归一化
        p = p / np.sum(p)
        q = q / np.sum(q)

        # 计算KL散度
        return np.sum(p * np.log(p / q))

    @staticmethod
    def js_divergence(p, q):
        """计算JS散度（对称版KL散度）"""
        # 添加小常数避免除零错误
        epsilon = 1e-10
        p = np.array(p) + epsilon
        q = np.array(q) + epsilon

        # 归一化
        p = p / np.sum(p)
        q = q / np.sum(q)

        # 计算中间分布
        m = 0.5 * (p + q)

        # 计算JS散度
        return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))

    def calculate_baseline_kl_matrix(self, baseline_dists):
        """计算基线模型之间的KL散度矩阵"""
        n_baselines = len(baseline_dists)
        kl_matrix = np.zeros((n_baselines, n_baselines))

        for i in range(n_baselines):
            for j in range(n_baselines):
                if i != j:
                    kl_matrix[i, j] = self.kl_divergence(baseline_dists[i], baseline_dists[j])

        # 返回上三角矩阵中的所有KL散度（排除对角线上的零）
        kl_values = []
        for i in range(n_baselines):
            for j in range(i + 1, n_baselines):
                kl_values.append(kl_matrix[i, j])
                kl_values.append(kl_matrix[j, i])  # 添加对称值，因为KL散度是不对称的

        return kl_matrix, kl_values

    def calculate_baseline_js_matrix(self, baseline_dists):
        """计算基线模型之间的JS散度矩阵"""
        n_baselines = len(baseline_dists)
        js_matrix = np.zeros((n_baselines, n_baselines))

        for i in range(n_baselines):
            for j in range(n_baselines):
                if i != j:
                    js_matrix[i, j] = self.js_divergence(baseline_dists[i], baseline_dists[j])

        # 返回上三角矩阵中的所有JS散度（排除对角线上的零）
        js_values = []
        for i in range(n_baselines):
            for j in range(i + 1, n_baselines):
                js_values.append(js_matrix[i, j])

        return js_matrix, js_values

    def calculate_target_baseline_kl(self, target_dist, baseline_dists):
        """计算目标模型与每个基线模型之间的KL散度"""
        kl_values = []
        for baseline_dist in baseline_dists:
            kl_values.append(self.kl_divergence(target_dist, baseline_dist))
        return kl_values

    def calculate_threshold(self, kl_values, confidence_level=CONFIDENCE_LEVEL):
        """
        统计分析工具模块
        """
        import numpy as np
        import scipy.stats as stats
        from typing import List, Dict, Tuple, Optional, Any
        
        
        def calculate_mean_std(data: List[float]) -> Tuple[float, float]:
            """
            计算均值和标准差
            
            Args:
                data: 数据列表
                
            Returns:
                (均值, 标准差)的元组
            """
            return float(np.mean(data)), float(np.std(data, ddof=1))
        
        
        def calculate_confidence_interval(
                data: List[float],
                confidence: float = 0.95
        ) -> Tuple[float, float]:
            """
            计算置信区间
            
            Args:
                data: 数据列表
                confidence: 置信水平，默认为0.95（95%置信区间）
                
            Returns:
                (下界, 上界)的元组
            """
            mean, std = calculate_mean_std(data)
            n = len(data)
            
            # 计算t分布的临界值
            t_critical = stats.t.ppf((1 + confidence) / 2, df=n - 1)
            
            # 计算误差幅度
            margin_of_error = t_critical * (std / np.sqrt(n))
            
            # 计算置信区间
            lower_bound = mean - margin_of_error
            upper_bound = mean + margin_of_error
            
            return lower_bound, upper_bound
        
        
        def perform_t_test(
                sample1: List[float],
                sample2: List[float],
                alternative: str = 'two-sided'
        ) -> Dict[str, float]:
            """
            执行独立样本t检验
            
            Args:
                sample1: 第一个样本
                sample2: 第二个样本
                alternative: 备择假设类型，可以是'two-sided'、'less'或'greater'
                
            Returns:
                包含t统计量和p值的字典
            """
            t_stat, p_value = stats.ttest_ind(sample1, sample2, alternative=alternative)
            
            return {
                't_statistic': float(t_stat),
                'p_value': float(p_value)
            }
        
        
        def calculate_kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
            """
            计算KL散度: KL(p||q)
            
            Args:
                p: 分布p
                q: 分布q
                epsilon: 小值，防止除零错误
                
            Returns:
                KL散度值
            """
            # 确保p和q是概率分布（和为1）
            p = np.asarray(p, dtype=np.float64)
            q = np.asarray(q, dtype=np.float64)
            
            p = p / np.sum(p)
            q = q / np.sum(q)
            
            # 添加小值防止log(0)
            p = np.maximum(p, epsilon)
            q = np.maximum(q, epsilon)
            
            # 计算KL散度
            return float(np.sum(p * np.log(p / q)))
        
        
        def calculate_kl_divergence_matrix(distributions: List[np.ndarray]) -> np.ndarray:
            """
            计算多个分布之间的KL散度矩阵
            
            Args:
                distributions: 分布列表
                
            Returns:
                KL散度矩阵
            """
            n = len(distributions)
            kl_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    if i != j:  # 对角线上的KL散度为0
                        kl_matrix[i, j] = calculate_kl_divergence(distributions[i], distributions[j])
            
            return kl_matrix
        
        
        def calculate_kl_threshold(
                kl_values: List[float],
                confidence: float = 0.95
        ) -> Dict[str, Any]:
            """
            计算KL散度阈值和统计信息
            
            Args:
                kl_values: KL散度值列表
                confidence: 置信水平
                
            Returns:
                包含阈值和统计信息的字典
            """
            mean, std = calculate_mean_std(kl_values)
            lower, upper = calculate_confidence_interval(kl_values, confidence)
            
            # 计算阈值T（使用上界）
            threshold_T = upper
            
            return {
                "mean": mean,
                "std": std,
                "confidence": confidence,
                "lower_bound": lower,
                "upper_bound": upper,
                "threshold_T": threshold_T,
                "sample_size": len(kl_values)
            }
        
        
        def is_normal_distribution(
                data: List[float],
                significance_level: float = 0.05
        ) -> Dict[str, Any]:
            """
            检验数据是否服从正态分布
            
            Args:
                data: 数据列表
                significance_level: 显著性水平
                
            Returns:
                包含检验结果的字典
            """
            # 执行Shapiro-Wilk检验
            sw_stat, sw_p = stats.shapiro(data)
            
            # 执行D'Agostino-Pearson检验
            dp_stat, dp_p = stats.normaltest(data)
            
            # 判断是否为正态分布
            is_normal = (sw_p > significance_level) and (dp_p > significance_level)
            
            return {
                'is_normal': is_normal,
                'shapiro_wilk': {
                    'statistic': float(sw_stat),
                    'p_value': float(sw_p),
                    'test_passed': sw_p > significance_level
                },
                'dagostino_pearson': {
                    'statistic': float(dp_stat),
                    'p_value': float(dp_p),
                    'test_passed': dp_p > significance_level
                }
            }

        # 计算基线KL散度的均值和标准差
        mean = np.mean(kl_values)
        std = np.std(kl_values, ddof=1)  # 使用无偏估计

        # 计算置信区间
        n = len(kl_values)
        t_critical = stats.t.ppf((1 + confidence_level) / 2, df=n - 1)
        margin_error = t_critical * (std / np.sqrt(n))

        conf_interval = (mean - margin_error, mean + margin_error)

        # 计算阈值T（上置信限）
        threshold = conf_interval[1]

        # 保存阈值信息到文件
        threshold_info = {
            "timestamp": datetime.now().isoformat(),
            "kl_values": kl_values,
            "mean": float(mean),
            "std": float(std),
            "sample_size": n,
            "confidence_level": confidence_level,
            "t_critical": float(t_critical),
            "margin_error": float(margin_error),
            "confidence_interval": [float(conf_interval[0]), float(conf_interval[1])],
            "threshold_T": float(threshold)
        }

        with open(THRESHOLD_FILE, 'w') as f:
            json.dump(threshold_info, f, indent=2)

        return threshold, mean, std, conf_interval

    def perform_t_test(self, target_kl, baseline_kl_mean, baseline_kl_std, baseline_kl_n):
        """
        执行t检验，检验目标模型的KL散度是否显著不同于基线分布

        Args:
            target_kl: 目标模型的平均KL散度
            baseline_kl_mean: 基线模型KL散度的均值
            baseline_kl_std: 基线模型KL散度的标准差
            baseline_kl_n: 基线模型KL散度的样本量

        Returns:
            t_stat: t统计量
            p_value: p值
            result: 检验结果描述
        """
        # 计算t统计量
        t_stat = (target_kl - baseline_kl_mean) / (baseline_kl_std / np.sqrt(baseline_kl_n))

        # 计算双侧p值
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=baseline_kl_n - 1))

        # 检验结果
        if p_value < 0.05:
            result = "拒绝原假设，目标模型KL散度与基线分布有显著差异"
        else:
            result = "接受原假设，目标模型KL散度与基线分布无显著差异"

        return t_stat, p_value, result

    def check_within_confidence_interval(self, target_kl, conf_interval):
        """
        检查目标模型的KL散度是否在置信区间内

        Args:
            target_kl: 目标模型的平均KL散度
            conf_interval: 置信区间(下限, 上限)

        Returns:
            is_within: 是否在置信区间内
            position: 位置描述
        """
        if target_kl >= conf_interval[0] and target_kl <= conf_interval[1]:
            return True, "在置信区间内"
        elif target_kl < conf_interval[0]:
            return False, "低于置信区间下限"
        else:
            return False, "高于置信区间上限"