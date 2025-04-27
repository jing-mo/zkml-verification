#!/bin/bash

# 零知识证明神经网络验证系统安装脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}===== 零知识证明神经网络验证系统安装程序 =====${NC}"

# 检测操作系统
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="MacOS"
else
    OS="Other"
fi

echo -e "检测到操作系统: ${YELLOW}$OS${NC}"

# 检查Python
echo -e "\n${BLUE}正在检查Python...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON="python3"
elif command -v python &> /dev/null; then
    PYTHON="python"
else
    echo -e "${RED}错误: 未找到Python，请安装Python 3.7+${NC}"
    exit 1
fi

# 检查Python版本
PY_VERSION=$($PYTHON -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "Python版本: ${YELLOW}$PY_VERSION${NC}"

if [[ $(echo "$PY_VERSION < 3.7" | bc) -eq 1 ]]; then
    echo -e "${RED}错误: Python版本必须 >= 3.7${NC}"
    exit 1
fi

# 检查pip
echo -e "\n${BLUE}正在检查pip...${NC}"
if command -v pip3 &> /dev/null; then
    PIP="pip3"
elif command -v pip &> /dev/null; then
    PIP="pip"
else
    echo -e "${RED}错误: 未找到pip，请安装pip${NC}"
    exit 1
fi

# 创建虚拟环境
echo -e "\n${BLUE}正在创建Python虚拟环境...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}虚拟环境已存在，跳过创建${NC}"
else
    $PYTHON -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}创建虚拟环境失败${NC}"
        exit 1
    fi
    echo -e "${GREEN}虚拟环境创建成功${NC}"
fi

# 激活虚拟环境
echo -e "\n${BLUE}正在激活虚拟环境...${NC}"
if [[ "$OS" == "Windows"* ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# 安装Python依赖
echo -e "\n${BLUE}正在安装Python依赖...${NC}"
$PIP install --upgrade pip
$PIP install -r requirements.txt

if [ $? -ne 0 ]; then
    echo -e "${RED}安装Python依赖失败${NC}"
    exit 1
fi
echo -e "${GREEN}Python依赖安装成功${NC}"

# 检查CUDA支持
echo -e "\n${BLUE}正在检查CUDA支持...${NC}"
$PYTHON -c "import torch; print('CUDA可用' if torch.cuda.is_available() else '警告: CUDA不可用，将使用CPU')"

# 创建必要的目录
echo -e "\n${BLUE}正在创建项目目录...${NC}"
mkdir -p trained_models experiment_reports data_cache zkcircuits statistics ptau

# 检查Node.js (用于snarkjs和circom)
echo -e "\n${BLUE}正在检查Node.js...${NC}"
if command -v node &> /dev/null; then
    NODE_VERSION=$(node -v)
    echo -e "Node.js版本: ${YELLOW}$NODE_VERSION${NC}"
else
    echo -e "${YELLOW}警告: 未找到Node.js，需要安装Node.js以使用零知识证明功能${NC}"
    echo -e "请访问 https://nodejs.org/ 安装Node.js"
fi

# 检查npm
echo -e "\n${BLUE}正在检查npm...${NC}"
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm -v)
    echo -e "npm版本: ${YELLOW}$NPM_VERSION${NC}"

    # 安装snarkjs和circom
    echo -e "\n${BLUE}正在安装snarkjs和circom...${NC}"
    npm install -g snarkjs circom

    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}警告: 安装snarkjs和circom失败${NC}"
        echo -e "您可能需要手动安装: npm install -g snarkjs circom"
    else
        echo -e "${GREEN}snarkjs和circom安装成功${NC}"
    fi
else
    echo -e "${YELLOW}警告: 未找到npm，无法安装snarkjs和circom${NC}"
    echo -e "请安装npm后再手动安装: npm install -g snarkjs circom"
fi

# 下载Powers of Tau文件
echo -e "\n${BLUE}正在检查Powers of Tau文件...${NC}"
PTAU_FILE="ptau/powersOfTau28_hez_final_10.ptau"
if [ -f "$PTAU_FILE" ]; then
    echo -e "${GREEN}Powers of Tau文件已存在${NC}"
else
    echo -e "${YELLOW}Powers of Tau文件不存在，正在尝试下载...${NC}"

    if command -v wget &> /dev/null; then
        wget -P ptau https://hermez.s3-eu-west-1.amazonaws.com/powersOfTau28_hez_final_10.ptau
    elif command -v curl &> /dev/null; then
        curl -o "$PTAU_FILE" https://hermez.s3-eu-west-1.amazonaws.com/powersOfTau28_hez_final_10.ptau
    else
        echo -e "${YELLOW}警告: 未找到wget或curl，无法自动下载Powers of Tau文件${NC}"
        echo -e "请手动下载 https://hermez.s3-eu-west-1.amazonaws.com/powersOfTau28_hez_final_10.ptau 并放置到 $PTAU_FILE"
    fi

    # 检查下载结果
    if [ -f "$PTAU_FILE" ]; then
        echo -e "${GREEN}Powers of Tau文件下载成功${NC}"
    else
        echo -e "${YELLOW}警告: Powers of Tau文件下载失败，请手动下载${NC}"
    fi
fi

# 授予运行权限
echo -e "\n${BLUE}正在设置脚本执行权限...${NC}"
chmod +x run.sh

echo -e "\n${GREEN}===== 安装完成 =====${NC}"
echo -e "您可以使用以下命令运行系统:"
echo -e "${YELLOW}./run.sh${NC} - 使用默认参数运行"
echo -e "或"
echo -e "${YELLOW}./run.sh --baselines 5 --epochs 10 --batch-size 64 --generate-zkp${NC} - 自定义参数运行"
echo -e "\n详细使用说明请查看 README.md"