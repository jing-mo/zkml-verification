"""
主模块，用于运行零知识神经网络验证系统
"""
import os
import argparse
import time
import json
import torch
import numpy as np
import scipy.stats as stats
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pathlib import Path
from new_zkp_verify.report import ReportGenerator  # 修改为从report.py导入
from new_zkp_verify.config import (
    CACHE_DIR, CIFAR10_MEAN, CIFAR10_STD, DEFAULT_CONFIDENCE, 
    DEFAULT_SEED, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE,
    DEFAULT_DISTILLATION_ALPHA, DEFAULT_DISTILLATION_TEMPERATURE,
    DEFAULT_NUM_BASELINE_MODELS, DEFAULT_NUM_SAMPLES, MODELS_DIR, REPORTS_DIR
)
from new_zkp_verify.train import ModelTrainer
from new_zkp_verify.verify import ModelVerifier
# 移除以下导入
# from new_zkp_verify.report import ReportGenerator
from new_zkp_verify.utils import get_data_loaders  # 保留此导入
from new_zkp_verify.zkp import ZKCircuitGenerator, ZKProofGenerator
from experiments.run_verification import VerificationExperiment
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
def parse_args():
    parser = argparse.ArgumentParser(description='运行零知识证明验证实验')
    
    # 实验基本参数
    parser.add_argument('--num-baseline-models', '--num_baseline_models', type=int, default=DEFAULT_NUM_BASELINE_MODELS,
                      help='基线模型数量（默认：5）')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                      help='训练轮次（默认：50）')
    parser.add_argument('--batch-size', '--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                      help='批次大小（默认：128）')
    parser.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE,
                      help='学习率（默认：0.1）')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                      help='随机种子（默认：42）')
    
    # 验证参数
    parser.add_argument('--confidence', type=float, default=DEFAULT_CONFIDENCE,
                      help='置信水平（默认：0.95）')
    parser.add_argument('--sample-batches', type=int, default=10,
                      help='每轮训练抽样验证的批次数（默认：10）')
    parser.add_argument('--num-samples', type=int, default=DEFAULT_NUM_SAMPLES,
                      help='KL散度计算的样本数（默认：100）')
    
    # 蒸馏参数
    parser.add_argument('--alpha', type=float, default=DEFAULT_DISTILLATION_ALPHA,
                      help='蒸馏软标签权重（默认：0.5）')
    parser.add_argument('--temperature', type=float, default=DEFAULT_DISTILLATION_TEMPERATURE,
                      help='蒸馏温度参数（默认：2.0）')
    
    # 实验模式
    parser.add_argument('--train-distilled', action='store_true',
                      help='是否训练蒸馏模型（默认：False）')
    parser.add_argument('--debug', action='store_true',
                      help='是否启用调试模式（默认：False）')
    parser.add_argument('--cpu', action='store_true',
                      help='是否强制使用CPU（默认：False）')
    
    # 模型和数据管理
    parser.add_argument('--model-dir', type=str, default=MODELS_DIR,
                      help=f'模型保存目录（默认：{MODELS_DIR}）')
    parser.add_argument('--load-models', action='store_true',
                      help='是否加载已有模型（默认：False）')
    parser.add_argument('--force-retrain', action='store_true',
                      help='是否强制重新训练模型（默认：False）')
    
    # 性能优化
    parser.add_argument('--mixed-precision', action='store_true',
                      help='是否使用混合精度训练（默认：False）')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='数据加载线程数（默认：4）')
    
    # ZKP相关参数
    parser.add_argument('--zkp-sample-rate', type=float, default=0.1,
                      help='训练批次ZKP抽样率（默认：0.1，即10%）')
    parser.add_argument('--force-recompile-circuit', action='store_true',
                      help='是否强制重新编译ZKP电路（默认：False）')
    
    # Add the device argument
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='计算设备（默认：cuda，如果可用）')
    return parser.parse_args()

def setup_experiment_dirs(args):
    """设置实验目录"""
    # 确保模型目录存在
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 确保报告目录存在
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # 创建实验ID
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_id = f"exp_{timestamp}"
    
    # 创建实验目录
    exp_dir = os.path.join(REPORTS_DIR, exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    
    # 创建模型目录
    models_dir = os.path.join(args.model_dir, exp_id)
    os.makedirs(models_dir, exist_ok=True)
    
    return {
        "exp_id": exp_id,
        "exp_dir": exp_dir,
        "models_dir": models_dir
    }

def save_experiment_config(args, dirs):
    """保存实验配置"""
    # Convert args to a dictionary and ensure all values are JSON serializable
    args_dict = vars(args)
    
    # Convert any Path objects to strings
    for key, value in args_dict.items():
        if isinstance(value, Path):
            args_dict[key] = str(value)
    
    config = {
        "exp_id": dirs["exp_id"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": args_dict,  # Use the processed dictionary
        "hardware": {
            "device": "cuda" if torch.cuda.is_available() and not args.cpu else "cpu",
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and not args.cpu else "N/A",
            "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory/1024/1024/1024:.1f}GB" if torch.cuda.is_available() and not args.cpu else "N/A",
        }
    }
    
    # Convert any remaining Path objects in the config
    for key, value in config.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, Path):
                    config[key][sub_key] = str(sub_value)
        elif isinstance(value, Path):
            config[key] = str(value)
    
    config_path = os.path.join(dirs["exp_dir"], "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    return config_path

def main():
    """主函数"""
    args = parse_args()
    
    # 设置实验目录
    dirs = setup_experiment_dirs(args)
    
    # 保存实验配置
    config_path = save_experiment_config(args, dirs)
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    
    # 初始化验证器
    verifier = ModelVerifier(device=device)
    
    # 性能优化设置
    if args.mixed_precision and device == "cuda":
        torch.backends.cudnn.benchmark = True
        scaler = torch.cuda.amp.GradScaler()
        print("启用混合精度训练")
    else:
        scaler = None
    
    # 优化GPU内存使用
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    
    # 获取数据加载器
    train_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print("\n=== 实验配置 ===")
    print(f"实验ID: {dirs['exp_id']}")
    print("硬件环境:")
    print(f"- 设备: {device}")
    if device == "cuda":
        print(f"- GPU: {torch.cuda.get_device_name(0)}")
        print(f"- 显存: {torch.cuda.get_device_properties(0).total_memory/1024/1024/1024:.1f}GB")
    
    print("\n模型配置:")
    print(f"- 架构: ResNet-50")
    print(f"- 数据集: CIFAR-10")
    print(f"- 基线模型数量: {args.num_baseline_models}")
    print(f"- 训练轮次: {args.epochs}")
    print(f"- 批次大小: {args.batch_size}")
    print(f"- 学习率: {args.lr}")
    print(f"- 随机种子: {args.seed}")
    
    print("\n验证配置:")
    print(f"- 置信水平: {args.confidence}")
    print(f"- 抽样批次数: {args.sample_batches}")
    print(f"- ZKP抽样率: {args.zkp_sample_rate * 100}%")
    
    # 创建ZKP电路生成器
    zkp_circuit_generator = ZKCircuitGenerator()
    
    # 创建KL散度电路
    kl_circuit_path = zkp_circuit_generator.create_circom_file(force_recreate=args.force_recompile_circuit)
    print(f"KL散度电路路径: {kl_circuit_path}")
    
    # 创建联合证明电路
    combined_circuit_path = zkp_circuit_generator.create_combined_circom_file(force_recreate=args.force_recompile_circuit)
    print(f"联合证明电路路径: {combined_circuit_path}")
    
    # 编译电路
    success, zkey_path = zkp_circuit_generator.compile_circuit(
        combined_circuit_path, 
        force_recompile=args.force_recompile_circuit
    )
    
    if not success:
        print("电路编译失败，退出实验")
        return
    
    print(f"电路编译成功，zkey路径: {zkey_path}")
    
    # 创建ZKP证明生成器
    zkp_generator = ZKProofGenerator()
    
    # 初始化ZKP指标
    all_zkp_proofs = []
    zkp_metrics = {
        "proof_generation_time": 0,
        "proof_size": 0,
        "verification_time": 0,
        "num_proofs": 0
    }
    
    # 训练基线模型
    baseline_models = []
    baseline_stats = {}
    baseline_paths = []
    training_times = {}
    
    # 使用指定的随机种子集合
    baseline_seeds = [42, 123, 456, 789, 1024]  # 根据实验要求指定的种子
    
    print("\n=== 训练基线模型 ===")
    for i in range(args.num_baseline_models):
        model_id = f"baseline_{i+1}"
        model_seed = baseline_seeds[i] if i < len(baseline_seeds) else args.seed + i
        model_path = os.path.join(dirs["models_dir"], f"{model_id}.pth")
        
        # 检查是否已存在模型
        if os.path.exists(model_path) and args.load_models and not args.force_retrain:
            print(f"加载已有基线模型 {i+1}/{args.num_baseline_models}，模型ID: {model_id}")
            
            # 创建模型训练器
            trainer = ModelTrainer(
                device=device,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                model_id=model_id,
                seed=model_seed
            )
            
            # 加载模型
            model = trainer.load_model(model_path)
            
            # 评估模型
            accuracy = trainer._evaluate(model, test_loader)
            
            # 创建统计信息
            model_stats = {
                "model_id": model_id,
                "seed": model_seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "is_distilled": False,
                "best_acc": accuracy,
                "training_time": 0,  # 加载模型不计入训练时间
                "model_path": model_path
            }
        else:
            print(f"训练基线模型 {i+1}/{args.num_baseline_models}，模型ID: {model_id}，随机种子: {model_seed}")
            
            # 创建模型训练器
            trainer = ModelTrainer(
                device=device,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                model_id=model_id,
                seed=model_seed,
                scaler=scaler
            )
            
            # 训练模型，同时进行ZKP抽样验证
            # 从 train_with_zkp_sampling 方法调用中移除 sample_rate 参数
            start_time = time.time()
            model, model_stats, zkp_proofs = trainer.train_with_zkp_sampling(
                train_loader=train_loader,
                test_loader=test_loader,
                zkp_generator=zkp_generator
                # 不传入 sample_rate 参数
            )
            training_time = time.time() - start_time
            
            # 处理 zkp_proofs
            print(f"基线模型 {model_id} 的 zkp_proofs 类型: {type(zkp_proofs)}")
            if zkp_proofs:
                if isinstance(zkp_proofs, list):
                    if all(isinstance(p, dict) for p in zkp_proofs):
                        all_zkp_proofs.extend(zkp_proofs)
                    else:
                        print(f"警告：{model_id} 的 zkp_proofs 包含非字典元素")
                elif isinstance(zkp_proofs, str):
                    print(f"警告：{model_id} 的 zkp_proofs 是字符串: '{zkp_proofs[:100]}'")
                else:
                    print(f"警告：{model_id} 的 zkp_proofs 类型未知: {type(zkp_proofs)}")
            
            # 保存模型
            trainer.save_model(model, model_path)
            
            # 更新统计信息
            model_stats.update({
                "model_id": model_id,
                "seed": model_seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "is_distilled": False,
                "training_time": training_time,
                "model_path": model_path
            })
        
        # 保存模型和统计信息
        baseline_models.append(model)
        baseline_stats[model_id] = model_stats
        baseline_paths.append(model_path)
        training_times[model_id] = model_stats["training_time"]
    
    # 训练或加载目标模型
    print("\n=== 训练目标模型 ===")
    if args.train_distilled:
        # 选择一个基线模型作为教师模型
        teacher_model = baseline_models[0]
        
        target_model_id = f"distilled_{int(time.time())}"
        target_model_path = os.path.join(dirs["models_dir"], f"{target_model_id}.pth")
        
        # 创建模型训练器
        trainer = ModelTrainer(
                device=device,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                model_id=target_model_id,  # 修复：使用 target_model_id 而不是 model_id
                seed=1234
        )
        
        # 训练模型，同时进行ZKP抽样验证
        start_time = time.time()
        try:
            # 尝试使用没有 sample_rate 参数的调用方式
            target_model, target_stats, zkp_proofs = trainer.train_with_zkp_sampling(
                train_loader=train_loader,
                test_loader=test_loader,
                zkp_generator=zkp_generator
            )
        except TypeError:
            # 如果有错误，尝试传入 sample_rate 参数
            print("警告：第一次调用 train_with_zkp_sampling 失败，尝试使用 sample_rate 参数")
            target_model, target_stats, zkp_proofs = trainer.train_with_zkp_sampling(
                train_loader=train_loader,
                test_loader=test_loader,
                zkp_generator=zkp_generator,
                sample_rate=args.zkp_sample_rate
            )
        
        training_time = time.time() - start_time
        
        # 处理 zkp_proofs
        print(f"目标模型的 zkp_proofs 类型: {type(zkp_proofs)}")
        if zkp_proofs:
            if isinstance(zkp_proofs, list):
                if all(isinstance(p, dict) for p in zkp_proofs):
                    all_zkp_proofs.extend(zkp_proofs)
                else:
                    print(f"警告：目标模型的 zkp_proofs 包含非字典元素")
            elif isinstance(zkp_proofs, str):
                print(f"警告：目标模型的 zkp_proofs 是字符串: '{zkp_proofs[:100]}'")
            else:
                print(f"警告：目标模型的 zkp_proofs 类型未知: {type(zkp_proofs)}")
        
        # 保存模型
        trainer.save_model(target_model, target_model_path)
        
        # 更新目标模型统计信息
        target_stats.update({
            "model_id": target_model_id,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "is_distilled": True,
            "alpha": args.alpha,
            "temperature": args.temperature,
            "training_time": training_time,
            "model_path": target_model_path
        })
    else:
        # 使用一个独立训练的模型作为目标模型
        target_model_id = f"target_{int(time.time())}"
        target_model_path = os.path.join(dirs["models_dir"], f"{target_model_id}.pth")
        
        # 创建新的训练器实例
        trainer = ModelTrainer(
            device=device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            model_id="target_model",
            seed=1234,
            scaler=scaler
        )
        
        # 训练模型，同时进行ZKP抽样验证
        start_time = time.time()
        try:
            # 尝试使用没有 sample_rate 参数的调用方式
            target_model, target_stats, zkp_proofs = trainer.train_with_zkp_sampling(
                train_loader=train_loader,
                test_loader=test_loader,
                zkp_generator=zkp_generator
            )
        except TypeError:
            # 如果有错误，尝试传入 sample_rate 参数
            print("警告：第一次调用 train_with_zkp_sampling 失败，尝试使用 sample_rate 参数")
            target_model, target_stats, zkp_proofs = trainer.train_with_zkp_sampling(
                train_loader=train_loader,
                test_loader=test_loader,
                zkp_generator=zkp_generator,
                sample_rate=args.zkp_sample_rate
            )
        
        training_time = time.time() - start_time
        
        # 处理 zkp_proofs
        print(f"目标模型的 zkp_proofs 类型: {type(zkp_proofs)}")
        if zkp_proofs:
            if isinstance(zkp_proofs, list):
                if all(isinstance(p, dict) for p in zkp_proofs):
                    all_zkp_proofs.extend(zkp_proofs)
                else:
                    print(f"警告：目标模型的 zkp_proofs 包含非字典元素")
            elif isinstance(zkp_proofs, str):
                print(f"警告：目标模型的 zkp_proofs 是字符串: '{zkp_proofs[:100]}'")
            else:
                print(f"警告：目标模型的 zkp_proofs 类型未知: {type(zkp_proofs)}")
        
        # 保存模型
        trainer.save_model(target_model, target_model_path)
        
        # 更新统计信息
        target_stats.update({
            "model_id": target_model_id,
            "seed": 1234,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "is_distilled": False,
            "training_time": training_time,
            "model_path": target_model_path
        })
    
    # 计算 ZKP 指标
    if all_zkp_proofs:
        try:
            # 检查字典中可能的字段名
            sample_proof = all_zkp_proofs[0]
            print(f"ZKP 证明字段: {list(sample_proof.keys())}")
            
            # 寻找生成时间字段
            gen_time_field = None
            for field in ["generation_time", "generation_time_ms"]:
                if field in sample_proof:
                    gen_time_field = field
                    break
            
            # 寻找证明大小字段
            size_field = None
            for field in ["proof_size", "proof_size_bytes"]:
                if field in sample_proof:
                    size_field = field
                    break
            
            # 寻找验证时间字段
            verify_time_field = None
            for field in ["verification_time", "verification_time_ms"]:
                if field in sample_proof:
                    verify_time_field = field
                    break
            
            # 计算指标
            if gen_time_field:
                time_values = [p.get(gen_time_field, 0) for p in all_zkp_proofs]
                if gen_time_field.endswith("_ms"):
                    zkp_metrics["proof_generation_time"] = sum(time_values) / 1000.0  # 毫秒转为秒
                else:
                    zkp_metrics["proof_generation_time"] = sum(time_values)
            
            if size_field:
                zkp_metrics["proof_size"] = sum([p.get(size_field, 0) for p in all_zkp_proofs])
            
            if verify_time_field:
                time_values = [p.get(verify_time_field, 0) for p in all_zkp_proofs]
                if verify_time_field.endswith("_ms"):
                    zkp_metrics["verification_time"] = sum(time_values) / 1000.0  # 毫秒转为秒
                else:
                    zkp_metrics["verification_time"] = sum(time_values)
            
            zkp_metrics["num_proofs"] = len(all_zkp_proofs)
        except Exception as e:
            print(f"计算 ZKP 指标时发生错误: {str(e)}")
    
    # 验证模型
    print("\n=== 验证模型 ===")
    
    # 添加白盒验证
    print("\n=== 白盒验证 ===")
    start_time = time.time()
    white_box_result = verifier.verify_white_box(
        target_model=target_model,
        test_loader=test_loader
    )
    
    # 黑盒验证
    start_time = time.time()
    verification_result = verifier.verify_black_box(
        target_model=target_model,
        baseline_models=baseline_models,
        test_loader=test_loader,
        confidence_level=args.confidence
    )
    
    verification_time = time.time() - start_time
    verification_result["verification_time"] = verification_time
    
    white_box_time = time.time() - start_time
    white_box_result["verification_time"] = white_box_time
    
    # 合并验证结果
    verification_result.update({
        "white_box_result": white_box_result
    })
    
    # 计算 KL 散度矩阵
    kl_matrix, table_data = verifier.compute_kl_matrix(baseline_models, test_loader, num_samples=args.num_samples)
    
    # 计算每对基线模型之间的对称KL散度
    baseline_kl_values = []
    for i in range(len(baseline_models)):
        for j in range(i + 1, len(baseline_models)):
            kl_ij = kl_matrix[i][j]
            kl_ji = kl_matrix[j][i]
            symmetric_kl = 0.5 * (kl_ij + kl_ji)
            baseline_kl_values.append(symmetric_kl)
    
    # 计算基线KL散度的统计量
    baseline_kl_mean = np.mean(baseline_kl_values)
    baseline_kl_std = np.std(baseline_kl_values)
    
    # 计算置信区间
    t_critical = stats.t.ppf((1 + args.confidence) / 2, len(baseline_kl_values) - 1)
    margin_of_error = t_critical * (baseline_kl_std / np.sqrt(len(baseline_kl_values)))
    ci_lower = baseline_kl_mean - margin_of_error
    ci_upper = baseline_kl_mean + margin_of_error
    
    print(f"\n基线统计量:")
    print(f"- 均值 μ_base: {baseline_kl_mean:.4f}")
    print(f"- 标准差 σ_base: {baseline_kl_std:.4f}")
    print(f"- 95% 置信区间: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # 计算目标模型与基线模型的KL散度
    target_kl_values = []
    for i in range(len(baseline_models)):
        kl_target_baseline = verifier.compute_model_kl_divergence(target_model, baseline_models[i], test_loader)
        kl_baseline_target = verifier.compute_model_kl_divergence(baseline_models[i], target_model, test_loader)
        symmetric_kl = 0.5 * (kl_target_baseline + kl_baseline_target)
        target_kl_values.append(symmetric_kl)
    
    target_kl_mean = np.mean(target_kl_values)
    target_kl_std = np.std(target_kl_values)
    
    print(f"\n目标模型验证:")
    print(f"- 目标KL均值 μ_target: {target_kl_mean:.4f}")
    
    # 进行t检验
    n_base = len(baseline_kl_values)
    n_target = len(target_kl_values)
    
    t_stat = (target_kl_mean - baseline_kl_mean) / np.sqrt((baseline_kl_std**2/n_base) + (target_kl_std**2/n_target))
    
    # 计算自由度
    df = ((baseline_kl_std**2/n_base + target_kl_std**2/n_target)**2) / \
         ((baseline_kl_std**4/(n_base**2 * (n_base-1))) + (target_kl_std**4/(n_target**2 * (n_target-1))))
    df = int(df)
    
    # 计算p值
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    print(f"- t检验结果: t = {t_stat:.2f}, 自由度 = {df}, p = {p_value:.2f}")
    
    # 结果判断
    if p_value >= 0.05:
        conclusion = "支持独立训练 (无法拒绝原假设)"
    else:
        conclusion = "目标模型可能经过蒸馏或存在异常"
    
    print(f"- 结论: {conclusion}")
    
    # 保存验证结果
    verification_result.update({
        "baseline_kl_values": baseline_kl_values,
        "baseline_kl_mean": baseline_kl_mean,
        "baseline_kl_std": baseline_kl_std,
        "baseline_kl_ci": [ci_lower, ci_upper],
        "target_kl_values": target_kl_values,
        "target_kl_mean": target_kl_mean,
        "target_kl_std": target_kl_std,
        "t_stat": t_stat,
        "p_value": p_value,
        "conclusion": conclusion,
        "baseline_models_info": baseline_stats,
        "target_model_info": target_stats,
        "experiment_config": {
            "architecture": "ResNet-50",
            "dataset": "CIFAR-10",
            "num_baseline_models": args.num_baseline_models,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "seed": args.seed,
            "confidence_level": args.confidence,
            "zkp_sample_rate": args.zkp_sample_rate
        },
        "zkp_metrics": zkp_metrics  # 使用预处理的 zkp_metrics
    })
    
    # 从模型输出中提取分布信息
    baseline_distributions = []
    for model in baseline_models:
        with torch.no_grad():
            outputs = []
            for data, _ in test_loader:
                data = data.to(device)
                output = model(data)
                outputs.append(torch.softmax(output, dim=1).cpu().numpy())
            baseline_distributions.append(np.concatenate(outputs))
    
    # 从目标模型中提取分布信息
    target_distributions = []
    with torch.no_grad():
        outputs = []
        for data, _ in test_loader:
            data = data.to(device)
            output = target_model(data)
            outputs.append(torch.softmax(output, dim=1).cpu().numpy())
        target_distributions.append(np.concatenate(outputs))
    
    # 生成报告
    print("\n=== 生成实验报告 ===")
    report_generator = ReportGenerator(dirs["exp_id"])
    
    report_data = {
        "verification_result": verification_result,
        "baseline_stats": baseline_distributions,  # 直接使用 baseline_distributions
        "target_stats": target_distributions,      # 直接使用 target_distributions
        "experiment_config": {
            "architecture": "ResNet-50",
            "dataset": "CIFAR-10",
            "num_baseline_models": args.num_baseline_models,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "seed": args.seed,
            "confidence_level": args.confidence
        }
    }
    
    # 调用generate_all_reports并传递必要的参数
    report_path = report_generator.generate_all_reports(
        report_data, 
        baseline_stats=baseline_distributions, 
        target_stats=target_distributions
    )
    
    print(f"\n实验报告已生成并保存至: {report_path}")
    
    # 将报告路径添加到验证结果中
    verification_result["report_path"] = report_path
    
    # 保存验证结果到JSON文件
    results_path = os.path.join(dirs["exp_dir"], "verification_results.json")
    with open(results_path, "w") as f:
        json.dump(verification_result, f, indent=4, default=convert_numpy)
    
    print(f"\n验证结果已保存至: {results_path}")
    
    return verification_result

if __name__ == "__main__":
    main()

def get_test_loader(num_samples=None):
    """
    获取测试数据加载器
    
    Args:
        num_samples: 样本数量，如果为None则使用全部测试集
        
    Returns:
        test_loader: 测试数据加载器
    """
    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    
    # 加载CIFAR-10测试集
    test_dataset = datasets.CIFAR10(
        root=CACHE_DIR,
        train=False,
        download=True,
        transform=transform
    )
    
    # 如果指定了样本数量，则创建子集
    if num_samples and num_samples < len(test_dataset):
        indices = torch.randperm(len(test_dataset))[:num_samples]
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2
    )
    
    return test_loader
def print_verification_summary(verification_result):
    """打印验证结果摘要"""
    print(f"KL值: {verification_result['target_kl_mean']:.4f}")
    print(f"证明生成时间: {verification_result.get('zkp_metrics', {}).get('proof_generation_time', 0):.2f}秒")
    print(f"证明验证时间: {verification_result.get('zkp_metrics', {}).get('verification_time', 0):.2f}秒")
    print(f"证明大小: {verification_result.get('zkp_metrics', {}).get('proof_size', 0)/1024:.2f}KB")
    print(f"可行性: {'高' if verification_result.get('passed_verification', False) else '低'}")
    ci = verification_result.get('confidence_interval', [0, 0])
    print(f"置信区间: [{ci[0]:.4f}, {ci[1]:.4f}]")
if __name__ == "__main__":
    result = main()
    print_verification_summary(result)






