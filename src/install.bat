@echo off
SETLOCAL

:: 检查Python版本
python --version > NUL 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo 错误: 未找到Python，请安装Python 3.8或更高版本
    exit /b 1
)

FOR /F "tokens=2" %%I IN ('python --version 2^>^&1') DO SET python_version=%%I
echo 检测到Python版本: %python_version%

:: 创建虚拟环境
echo 创建Python虚拟环境...
python -m venv venv

:: 激活虚拟环境
echo 激活虚拟环境...
call venv\Scripts\activate.bat

:: 升级pip
echo 升级pip...
python -m pip install --upgrade pip

:: 安装依赖
echo 安装依赖...
pip install -e .

:: 创建必要的目录
echo 创建实验目录...
mkdir experiments\results\models
mkdir experiments\results\proofs
mkdir experiments\results\results
mkdir experiments\results\images
mkdir experiments\results\attacks
mkdir experiments\results\efficiency
mkdir experiments\results\sampling

echo 安装完成！
echo 要激活环境，请运行: venv\Scripts\activate.bat

ENDLOCAL