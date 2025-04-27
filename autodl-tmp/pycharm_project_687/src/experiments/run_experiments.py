# 全局配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "/autodl-tmp/pycharm_project_687/results"
PROOFS_DIR = "/autodl-tmp/pycharm_project_687/proofs"
MODELS_DIR = "/autodl-tmp/pycharm_project_687/models"

# 确保目录存在
for directory in [RESULTS_DIR, PROOFS_DIR, MODELS_DIR, "/autodl-tmp/pycharm_project_687/logs"]:
    os.makedirs(directory, exist_ok=True)