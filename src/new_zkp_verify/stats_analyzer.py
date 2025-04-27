def compute_baseline_kl_matrix(self, baseline_models, test_loader):
    """计算所有基线模型之间的对称KL散度矩阵"""
    n_models = len(baseline_models)
    kl_matrix = np.zeros((n_models, n_models))
    baseline_outputs = []
    
    # 首先获取所有基线模型的输出
    print("获取基线模型输出...")
    for model in baseline_models:
        outputs = self.get_model_outputs(model, test_loader)
        baseline_outputs.append(outputs)
    
    # 计算KL散度矩阵
    print("计算基线模型之间的KL散度...")
    baseline_kl_values = []
    for i in range(n_models):
        for j in range(i+1, n_models):  # 只计算上三角矩阵，避免重复
            outputs_i = baseline_outputs[i]
            outputs_j = baseline_outputs[j]
            
            # 计算对称KL散度
            kl_ij = self.compute_kl_divergence(outputs_i, outputs_j)
            kl_ji = self.compute_kl_divergence(outputs_j, outputs_i)
            sym_kl = (kl_ij + kl_ji) / 2
            
            # 存储到矩阵
            kl_matrix[i, j] = sym_kl
            kl_matrix[j, i] = sym_kl  # 对称矩阵
            
            # 添加到列表中，用于后续统计分析
            baseline_kl_values.append(sym_kl)
            
            print(f"基线模型 {i} 和 {j} 之间的对称KL散度: {sym_kl:.4f}")
    
    return kl_matrix, baseline_kl_values