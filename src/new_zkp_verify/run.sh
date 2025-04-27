#!/bin/bash

# 运行脚本，分步执行零知识证明神经网络验证系统
# 参数设置为适合家用电脑（RTX 4060）的配置

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查是否安装了必要的工具
echo -e "${BLUE}正在检查环境...${NC}"

# 检查Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}错误: 未找到Python，请安装Python 3.7+${NC}"
    exit 1
fi

# 检查pip
if ! command -v pip &> /dev/null; then
    echo -e "${RED}错误: 未找到pip，请安装pip${NC}"
    exit 1
fi

# 检查CUDA
python -c "import torch; print('CUDA可用' if torch.cuda.is_available() else '警告: CUDA不可用，将使用CPU')"

# 创建虚拟环境（可选）
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}创建Python虚拟环境...${NC}"
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# 确保目录存在
mkdir -p trained_models experiment_reports data_cache zkcircuits statistics

# 解析命令行参数
BASELINES=5
EPOCHS=10
BATCH_SIZE=64
GPU_MEM_LIMIT=0.7
FORCE_RETRAIN=false
TRAIN_DISTILLED=false
GENERATE_ZKP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --baselines)
            BASELINES="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gpu-mem-limit)
            GPU_MEM_LIMIT="$2"
            shift 2
            ;;
        --force-retrain)
            FORCE_RETRAIN=true
            shift
            ;;
        --train-distilled)
            TRAIN_DISTILLED=true
            shift
            ;;
        --generate-zkp)
            GENERATE_ZKP=true
            shift
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            exit 1
            ;;
    esac
done

# 显示配置
echo -e "${BLUE}====== 运行配置 ======${NC}"
echo -e "基线模型数量: ${BASELINES}"
echo -e "训练轮次: ${EPOCHS}"
echo -e "批次大小: ${BATCH_SIZE}"
echo -e "GPU内存限制: ${GPU_MEM_LIMIT}"
echo -e "强制重新训练: ${FORCE_RETRAIN}"
echo -e "训练蒸馏模型: ${TRAIN_DISTILLED}"
echo -e "生成零知识证明: ${GENERATE_ZKP}"
echo -e "${BLUE}======================${NC}"

# 步骤1: 训练模型
echo -e "\n${GREEN}步骤1: 训练模型${NC}"
TRAIN_ARGS=""
if [ "$FORCE_RETRAIN" = true ]; then
    TRAIN_ARGS="$TRAIN_ARGS --force-retrain"
fi
if [ "$TRAIN_DISTILLED" = true ]; then
    TRAIN_ARGS="$TRAIN_ARGS --train-distilled"
fi

python main.py --mode train --baselines $BASELINES --epochs $EPOCHS --batch-size $BATCH_SIZE --gpu-mem-limit $GPU_MEM_LIMIT $TRAIN_ARGS

# 检查训练是否成功
if [ $? -ne 0 ]; then
    echo -e "${RED}模型训练失败，请检查错误信息${NC}"
    exit 1
fi

# 步骤2: 验证模型
echo -e "\n${GREEN}步骤2: 验证模型${NC}"
VERIFY_ARGS=""
if [ "$TRAIN_DISTILLED" = true ]; then
    VERIFY_ARGS="$VERIFY_ARGS --train-distilled"
fi

python main.py --mode verify --gpu-mem-limit $GPU_MEM_LIMIT $VERIFY_ARGS

# 检查验证是否成功
if [ $? -ne 0 ]; then
    echo -e "${RED}模型验证失败，请检查错误信息${NC}"
    exit 1
fi

# 步骤3: 生成零知识证明（可选）
if [ "$GENERATE_ZKP" = true ]; then
    echo -e "\n${GREEN}步骤3: 生成零知识证明${NC}"
    ZKP_ARGS=""
    if [ "$TRAIN_DISTILLED" = true ]; then
        ZKP_ARGS="$ZKP_ARGS --train-distilled"
    fi

    python main.py --mode zkp --generate-zkp --gpu-mem-limit $GPU_MEM_LIMIT $ZKP_ARGS

    # 检查ZKP生成是否成功
    if [ $? -ne 0 ]; then
        echo -e "${RED}零知识证明生成失败，请检查错误信息${NC}"
        exit 1
    fi
else
    echo -e "\n${YELLOW}步骤3: 跳过零知识证明生成${NC}"
fi

# 查找最新的实验报告目录
LATEST_REPORT=$(ls -td experiment_reports/experiment_* | head -1)

echo -e "\n${GREEN}实验完成!${NC}"
echo -e "报告保存在: ${LATEST_REPORT}"
echo -e "HTML报告: ${LATEST_REPORT}/comprehensive_report.html"

# 如果在Linux系统上，尝试使用浏览器打开HTML报告
if [ -f "/usr/bin/xdg-open" ]; then
    echo -e "${BLUE}正在打开HTML报告...${NC}"
    xdg-open "${LATEST_REPORT}/comprehensive_report.html" &
elif [ -f "/usr/bin/open" ]; then
    echo -e "${BLUE}正在打开HTML报告...${NC}"
    open "${LATEST_REPORT}/comprehensive_report.html" &
fi

echo -e "${GREEN}感谢使用零知识证明神经网络验证系统！${NC}"