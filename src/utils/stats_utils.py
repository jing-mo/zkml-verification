import numpy as np
import scipy.stats as stats
from typing import List, Dict, Tuple, Optional, Union, Any


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


def perform_z_test(
        sample: List[float],
        population_mean: float,
        population_std: Optional[float] = None,
        alternative: str = 'two-sided'
) -> Dict[str, float]:
    """
    执行单样本z检验

    Args:
        sample: 样本数据
        population_mean: 总体均值
        population_std: 总体标准差，如果为None，则使用样本标准差
        alternative: 备择假设类型，可以是'two-sided'、'less'或'greater'

    Returns:
        包含z统计量和p值的字典
    """
    sample_mean = np.mean(sample)
    sample_size = len(sample)

    # 如果未提供总体标准差，使用样本标准差
    if population_std is None:
        population_std = np.std(sample, ddof=1)

    # 计算z统计量
    z_stat = (sample_mean - population_mean) / (population_std / np.sqrt(sample_size))

    # 计算p值
    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    elif alternative == 'less':
        p_value = stats.norm.cdf(z_stat)
    elif alternative == 'greater':
        p_value = 1 - stats.norm.cdf(z_stat)
    else:
        raise ValueError(f"Invalid alternative: {alternative}")

    return {
        'z_statistic': float(z_stat),
        'p_value': float(p_value)
    }


def perform_kolmogorov_smirnov_test(
        sample: List[float],
        reference_distribution: str = 'norm',
        reference_params: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    执行Kolmogorov-Smirnov（KS）检验

    Args:
        sample: 样本数据
        reference_distribution: 参考分布类型，例如'norm'表示正态分布
        reference_params: 参考分布的参数

    Returns:
        包含KS统计量和p值的字典
    """
    # 如果未提供参考分布参数，则估计
    if reference_params is None:
        if reference_distribution == 'norm':
            reference_params = {'loc': np.mean(sample), 'scale': np.std(sample, ddof=1)}

    # 获取参考分布
    if reference_distribution == 'norm':
        distribution = stats.norm
    elif reference_distribution == 'uniform':
        distribution = stats.uniform
    elif reference_distribution == 'expon':
        distribution = stats.expon
    else:
        raise ValueError(f"Unsupported reference distribution: {reference_distribution}")

    # 执行KS检验
    ks_stat, p_value = stats.kstest(sample, distribution.name, args=tuple(reference_params.values()))

    return {
        'ks_statistic': float(ks_stat),
        'p_value': float(p_value)
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


def outlier_detection(
        data: List[float],
        method: str = 'z_score',
        threshold: float = 3.0
) -> Dict[str, Any]:
    """
    检测异常值

    Args:
        data: 数据列表
        method: 检测方法，可以是'z_score'、'iqr'或'grubbs'
        threshold: 阈值

    Returns:
        包含异常值信息的字典
    """
    data_array = np.array(data)

    if method == 'z_score':
        # Z分数法
        mean = np.mean(data_array)
        std = np.std(data_array, ddof=1)
        z_scores = np.abs((data_array - mean) / std)
        outliers_idx = np.where(z_scores > threshold)[0]

    elif method == 'iqr':
        # 四分位距法
        q1 = np.percentile(data_array, 25)
        q3 = np.percentile(data_array, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outliers_idx = np.where((data_array < lower_bound) | (data_array > upper_bound))[0]

    elif method == 'grubbs':
        # Grubbs检验
        from scipy.stats import t
        mean = np.mean(data_array)
        std = np.std(data_array, ddof=1)
        n = len(data_array)

        # 计算Grubbs统计量的临界值
        t_critical = t.ppf(1 - 0.05 / (2 * n), n - 2)
        G_critical = ((n - 1) * t_critical) / np.sqrt(n * (n - 2 + t_critical ** 2))

        # 计算每个点的Grubbs统计量
        G = np.abs(data_array - mean) / std
        outliers_idx = np.where(G > G_critical)[0]

    else:
        raise ValueError(f"Unsupported method: {method}")

    # 获取异常值
    outliers = data_array[outliers_idx].tolist()
    outliers_index = outliers_idx.tolist()

    return {
        'outliers': outliers,
        'outliers_index': outliers_index,
        'n_outliers': len(outliers),
        'outliers_ratio': len(outliers) / len(data)
    }


def calculate_effect_size(
        sample1: List[float],
        sample2: List[float],
        type: str = 'cohen_d'
) -> float:
    """
    计算效应量

    Args:
        sample1: 第一个样本
        sample2: 第二个样本
        type: 效应量类型，可以是'cohen_d'或'hedges_g'

    Returns:
        效应量
    """
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    n1 = len(sample1)
    n2 = len(sample2)

    # 计算合并标准差
    s1 = np.var(sample1, ddof=1)
    s2 = np.var(sample2, ddof=1)

    if type == 'cohen_d':
        # Cohen's d
        # 使用合并方差
        s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        effect_size = (mean1 - mean2) / s_pooled

    elif type == 'hedges_g':
        # Hedges' g（小样本修正版的Cohen's d）
        s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        cohen_d = (mean1 - mean2) / s_pooled

        # 应用小样本修正
        correction = 1 - 3 / (4 * (n1 + n2 - 2) - 1)
        effect_size = cohen_d * correction

    else:
        raise ValueError(f"Unsupported effect size type: {type}")

    return float(effect_size)


def calculate_sample_size(
        effect_size: float,
        power: float = 0.8,
        significance_level: float = 0.05,
        test_type: str = 'two_sample'
) -> int:
    """
    计算样本大小

    Args:
        effect_size: 期望检测到的效应量
        power: 检验效能，默认为0.8
        significance_level: 显著性水平，默认为0.05
        test_type: 检验类型，可以是'one_sample'、'two_sample'或'paired'

    Returns:
        所需的样本大小
    """
    if test_type == 'one_sample':
        # 单样本t检验
        sample_size = stats.t.ppf(1 - significance_level / 2, 100) ** 2 * (1 / effect_size ** 2)

    elif test_type == 'two_sample':
        # 独立样本t检验
        # 计算非中心参数
        from statsmodels.stats.power import TTestIndPower
        power_analysis = TTestIndPower()
        sample_size = power_analysis.solve_power(
            effect_size=effect_size,
            power=power,
            alpha=significance_level,
            ratio=1.0
        )

    elif test_type == 'paired':
        # 配对样本t检验
        from statsmodels.stats.power import TTestPower
        power_analysis = TTestPower()
        sample_size = power_analysis.solve_power(
            effect_size=effect_size,
            power=power,
            alpha=significance_level
        )

    else:
        raise ValueError(f"Unsupported test type: {test_type}")

    # 向上取整获得整数样本大小
    return int(np.ceil(sample_size))