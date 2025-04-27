#!/bin/bash

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8.0"

if [[ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]]; then
    echo "错误: 需要Python 3.8.0或更高版本"
    exit 1
fi

# 创建虚拟环境
echo "创建Python虚拟环境..."
python3 -m venv venv

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 升级pip
echo "升级pip..."
pip install --upgrade pip

# 安装依赖
echo "安装依赖..."
pip install -e .

# 创建必要的目录
echo "创建实验目录..."
mkdir -p experiments/results/models
mkdir -p experiments/results/proofs
mkdir -p experiments/results/results
mkdir -p experiments/results/images
mkdir -p experiments/results/attacks
mkdir -p experiments/results/efficiency
mkdir -p experiments/results/sampling

echo "安装完成！"
echo "要激活环境，请运行: source venv/bin/activate"