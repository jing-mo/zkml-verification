"""
零知识神经网络验证系统
"""
from new_zkp_verify.config import *
from new_zkp_verify.models import ResNetModel
from new_zkp_verify.train import ModelTrainer
from new_zkp_verify.verify import ModelVerifier
from new_zkp_verify.report import ReportGenerator
from new_zkp_verify.zkp import ZKPGenerator

__version__ = "0.1.0"