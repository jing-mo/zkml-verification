def verify_black_box(self, target_model, baseline_models, test_loader, confidence_level=0.95):
    """黑盒验证方法"""
    # 计算基线模型之间的KL散度
    kl_matrix, baseline_kl_values = self.stats_analyzer.compute_baseline_kl_matrix(baseline_models, test_loader)
    
    # 计算基线KL散度均值和标准差
    baseline_kl_mean = np.mean(baseline_kl_values)
    baseline_kl_std = np.std(baseline_kl_values)
    
    print(f"基线模型KL散度均值 μ_base: {baseline_kl_mean:.4f}")
    print(f"基线模型KL散度标准差 σ_base: {baseline_kl_std:.4f}")
    
    # 计算置信区间
    t_value = stats.t.ppf((1 + confidence_level) / 2, len(baseline_kl_values) - 1)
    margin_of_error = t_value * (baseline_kl_std / np.sqrt(len(baseline_kl_values)))
    confidence_interval = (baseline_kl_mean - margin_of_error, baseline_kl_mean + margin_of_error)
    
    print(f"{confidence_level*100}%置信区间: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")
    
    # 计算目标模型与每个基线模型的KL散度
    target_kl_values = []
    for i, baseline_model in enumerate(baseline_models):
        target_outputs = self.stats_analyzer.get_model_outputs(target_model, test_loader)
        baseline_outputs = self.stats_analyzer.get_model_outputs(baseline_model, test_loader)
        
        # 计算对称KL散度
        kl = self.stats_analyzer.compute_kl_divergence(target_outputs, baseline_outputs)
        kl_reverse = self.stats_analyzer.compute_kl_divergence(baseline_outputs, target_outputs)
        sym_kl = (kl + kl_reverse) / 2
        target_kl_values.append(sym_kl)
        
        print(f"目标模型与基线模型 {i} 的对称KL散度: {sym_kl:.4f}")
    
    # 计算目标KL均值
    target_kl_mean = np.mean(target_kl_values)
    print(f"目标KL均值 μ_target: {target_kl_mean:.4f}")
    
    # 使用t检验比较目标KL均值与基线分布
    t_stat, p_value = stats.ttest_ind_from_stats(
        target_kl_mean, np.std(target_kl_values), len(target_kl_values),
        baseline_kl_mean, baseline_kl_std, len(baseline_kl_values),
        equal_var=False
    )
    
    print(f"t检验结果: t = {t_stat:.2f}, 自由度 = {len(baseline_kl_values) + len(target_kl_values) - 2}, p = {p_value:.4f}")
    
    # 判断目标模型是否在置信区间内
    verification_passed = target_kl_mean <= confidence_interval[1]
    
    # 输出独立性验证报告
    print("\n===== 独立性验证报告 =====")
    print("1. 基线模型配置:")
    print("   架构: ResNet-50")
    print("   训练数据: CIFAR-10")
    print("   超参数:")
    print(f"   - 批次大小: {DEFAULT_BATCH_SIZE}")
    print(f"   - 学习率: {DEFAULT_LEARNING_RATE}")
    print(f"   - 训练轮次: {DEFAULT_EPOCHS}")
    print("   - 随机种子: 多个")
    
    print("\n2. 基线KL分布:")
    print(f"   样本量: {len(baseline_kl_values)}")
    print(f"   均值±标准差: {baseline_kl_mean:.4f} ± {baseline_kl_std:.4f}")
    print(f"   {confidence_level*100}%置信区间: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")
    
    print("\n3. 目标模型验证:")
    print(f"   KL均值: {target_kl_mean:.4f}")
    print(f"   t检验结果: t = {t_stat:.2f}, p = {p_value:.4f}")
    print(f"   结论: {'通过验证' if verification_passed else '未通过验证'}")
    
    return {
        "baseline_kl_matrix": kl_matrix.tolist(),
        "baseline_kl_values": baseline_kl_values,
        "baseline_kl_mean": baseline_kl_mean,
        "baseline_kl_std": baseline_kl_std,
        "confidence_interval": confidence_interval,
        "target_kl_values": target_kl_values,
        "target_kl_mean": target_kl_mean,
        "t_stat": t_stat,
        "p_value": p_value,
        "passed_verification": verification_passed
    }