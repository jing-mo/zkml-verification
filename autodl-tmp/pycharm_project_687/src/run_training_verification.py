def main():
    # 修改为Linux路径
    base_dir = Path("/autodl-tmp/pycharm_project_687")
    data_dir = base_dir / "data"
    results_dir = base_dir / "results"
    os.makedirs(results_dir / "student", exist_ok=True)
    
    # ... 其余代码保持不变 ...