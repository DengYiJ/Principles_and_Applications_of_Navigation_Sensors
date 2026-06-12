"""
实验(3) 惯性导航实验 — 安装脚本
"""
from setuptools import setup, find_packages

setup(
    name='exp3_navigation_lab',
    version='1.0.0',
    description='实验(3) 惯性导航实验 - 捷联惯导解算系统',
    author='哈尔滨工业大学 空间控制与惯性技术研究中心',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'matplotlib>=3.5.0',
        'PyYAML>=5.4.0',
    ],
    python_requires='>=3.8',
)