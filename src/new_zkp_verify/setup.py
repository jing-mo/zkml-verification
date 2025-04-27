from setuptools import setup, find_packages

setup(
    name="zk-model-verification",
    version="0.1.0",
    description="Zero-Knowledge Proof Framework for AI Model Verification",
    author="ZKP Verification Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.50.0",
        "cryptography>=36.0.0",
        "python-docx>=0.8.10",
        "openpyxl>=3.0.0",
        "psutil>=5.8.0",
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)