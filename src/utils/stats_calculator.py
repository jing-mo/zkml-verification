import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings

class StatsCalculator:
    """统计计算器"""
    
    @staticmethod
    def calculate_kl_stats(baseline_kls: List[float], 
                          test_kls: List[float],
                          confidence_level: float = 0.95) -> Dict:
        """计算KL散度统计量"""
        
        # 基本统计量
        baseline_mean = np.mean(baseline_kls)
        baseline_std = np.std(baseline_kls, ddof=1)
        test_mean = np.mean(test_kls)
        
        # 计算置信区间
        df = len(baseline_kls) - 1
        t_crit = stats.t.ppf((1 + confidence_level) / 2, df)
        margin = t_crit * baseline_std / np.sqrt(len(baseline_kls))
        
        # 进行t检验
        t_stat, p_value = stats.ttest_ind(baseline_kls, test_kls)
        
        return {
            "sample_size": len(baseline_kls),
            "mean": baseline_mean,
            "std": baseline_std,
            "ci_lower": baseline_mean - margin,
            "ci_upper": baseline_mean + margin,
            "test_mean": test_mean,
            "t_stat": t_stat,
            "p_value": p_value
        }

    @staticmethod
    def calculate_sample_size(effect_size: float,
                            power: float = 0.8,
                            alpha: float = 0.05) -> int:
        """
        计算所需的样本量
        
        Args:
            effect_size: 预期效应量
            power: 统计检验力
            alpha: 显著性水平
        """
        return int(np.ceil(stats.norm.ppf(1-alpha/2) + stats.norm.ppf(power) / effect_size)**2)

    @staticmethod
    def analyze_sampling_efficiency(sampling_rates: List[float],
                                  detection_rates: List[float]) -> Dict:
        """
        分析采样效率
        
        Args:
            sampling_rates: 采样率列表
            detection_rates: 对应的检测率列表
        """
        # 计算相关系数
        correlation = np.corrcoef(sampling_rates, detection_rates)[0,1]
        
        # 找到最佳采样率（检测率最高的点）
        optimal_idx = np.argmax(detection_rates)
        optimal_rate = sampling_rates[optimal_idx]
        
        # 计算效率指标（检测率/采样率）
        efficiency = [d/r for d, r in zip(detection_rates, sampling_rates)]
        
        return {
            "correlation": correlation,
            "optimal_sampling_rate": optimal_rate,
            "efficiency_scores": efficiency,
            "mean_efficiency": np.mean(efficiency)
        }

    @staticmethod
    def compute_confidence_bounds(data: List[float],
                                confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        计算置信区间边界
        
        Args:
            data: 数据列表
            confidence_level: 置信水平
        """
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        # 计算置信区间
        n = len(data)
        se = std / np.sqrt(n)
        t_value = stats.t.ppf((1 + confidence_level) / 2, n-1)
        margin = t_value * se
        
        return mean - margin, mean + margin

    @staticmethod
    def test_distribution_normality(data: List[float]) -> Dict:
        """
        检验数据分布的正态性
        
        Args:
            data: 待检验的数据
        """
        # Shapiro-Wilk检验
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w_stat, p_value = stats.shapiro(data)
        
        return {
            "is_normal": p_value > 0.05,
            "w_statistic": w_stat,
            "p_value": p_value
        }