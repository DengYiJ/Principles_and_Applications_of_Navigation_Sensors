"""
图表生成器
==========
生成实验报告所需的全部图表
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # 无头模式
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional


class Plotter:
    """实验报告图表生成器"""
    
    def __init__(self, save_dir: str = "./results", dpi: int = 300, fmt: str = 'png'):
        """
        初始化绘图器
        输入: save_dir — 保存目录
              dpi — 分辨率
              fmt — 图片格式
        """
        self.save_dir = save_dir
        self.dpi = dpi
        self.fmt = fmt
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def _save(self, filename: str):
        """保存图片"""
        filepath = os.path.join(self.save_dir, f"{filename}.{self.fmt}")
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"[Plotter] 已保存: {filepath}")
    
    def plot_imu_raw_data(self, imu_data: np.ndarray, 
                           title: str = "IMU原始数据曲线",
                           filename: str = "imu_raw_data"):
        """
        图1/2：陀螺仪与加速度计原始数据曲线（6轴）
        """
        time = imu_data[:, 6]
        gyro = imu_data[:, 0:3]
        acc = imu_data[:, 3:6]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 陀螺仪
        ax1.plot(time, np.rad2deg(gyro[:, 0]), label='X轴', linewidth=0.5)
        ax1.plot(time, np.rad2deg(gyro[:, 1]), label='Y轴', linewidth=0.5)
        ax1.plot(time, np.rad2deg(gyro[:, 2]), label='Z轴', linewidth=0.5)
        ax1.set_ylabel('角速率 (°/s)')
        ax1.set_title(f'{title} - 陀螺仪')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 加速度计
        ax2.plot(time, acc[:, 0], label='X轴', linewidth=0.5)
        ax2.plot(time, acc[:, 1], label='Y轴', linewidth=0.5)
        ax2.plot(time, acc[:, 2], label='Z轴', linewidth=0.5)
        ax2.set_xlabel('时间 (s)')
        ax2.set_ylabel('比力 (m/s^2)')
        ax2.set_title(f'{title} - 加速度计')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save(filename)
    
    def plot_coarse_align(self, att_history: np.ndarray,
                           ref_att: np.ndarray = None,
                           time: np.ndarray = None,
                           filename: str = "coarse_align_attitude"):
        """
        图3：粗对准曲线（姿态角收敛过程）
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 9))
        labels = ['横滚角 (°)', '俯仰角 (°)', '航向角 (°)']
        
        if time is None:
            time = np.arange(len(att_history))
        
        for i, ax in enumerate(axes):
            ax.plot(time, att_history[:, i], 'b-', label='粗对准结果', linewidth=1)
            if ref_att is not None:
                ref_interp = np.interp(time, np.arange(len(ref_att)), ref_att[:, i])
                ax.plot(time, ref_interp, 'r--', label='参考姿态', linewidth=1, alpha=0.7)
            ax.set_ylabel(labels[i])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('时间 (s)')
        fig.suptitle('双矢量法粗对准姿态角收敛曲线', fontsize=14)
        plt.tight_layout()
        self._save(filename)
    
    def plot_fine_align(self, att_history: np.ndarray,
                         ref_att: np.ndarray = None,
                         time: np.ndarray = None,
                         filename: str = "fine_align_attitude"):
        """
        图5左：精对准姿态更新曲线
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 9))
        labels = ['横滚角 (°)', '俯仰角 (°)', '航向角 (°)']
        
        if time is None:
            time = np.arange(len(att_history))
        
        for i, ax in enumerate(axes):
            ax.plot(time, att_history[:, i], 'g-', label='精对准结果', linewidth=1)
            if ref_att is not None:
                ref_interp = np.interp(time, np.arange(len(ref_att)), ref_att[:, i])
                ax.plot(time, ref_interp, 'r--', label='参考姿态', linewidth=1, alpha=0.7)
            ax.set_ylabel(labels[i])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('时间 (s)')
        fig.suptitle('卡尔曼滤波精对准姿态角收敛曲线', fontsize=14)
        plt.tight_layout()
        self._save(filename)
    
    def plot_kf_error_params(self, X_history: np.ndarray,
                              time: np.ndarray = None,
                              filename: str = "kf_error_params"):
        """
        图5右：卡尔曼滤波误差参数曲线（12维状态）
        分为4个子图：失准角、速度误差、陀螺零偏、加速度计零偏
        """
        if time is None:
            T = X_history.shape[1]
            time = np.arange(T)
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 12))
        
        # 失准角 φ
        ax = axes[0]
        ax.plot(time, np.rad2deg(X_history[0]), label='φ_E')
        ax.plot(time, np.rad2deg(X_history[1]), label='φ_N')
        ax.plot(time, np.rad2deg(X_history[2]), label='φ_U')
        ax.set_ylabel('失准角 (°)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('失准角估计')
        
        # 速度误差 δv
        ax = axes[1]
        ax.plot(time, X_history[3], label='δv_E')
        ax.plot(time, X_history[4], label='δv_N')
        ax.plot(time, X_history[5], label='δv_U')
        ax.set_ylabel('速度误差 (m/s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('速度误差估计')
        
        # 陀螺零偏 ε
        ax = axes[2]
        ax.plot(time, np.rad2deg(X_history[6]), label='ε_x')
        ax.plot(time, np.rad2deg(X_history[7]), label='ε_y')
        ax.plot(time, np.rad2deg(X_history[8]), label='ε_z')
        ax.set_ylabel('陀螺零偏 (°/s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('陀螺零偏估计')
        
        # 加速度计零偏 ∇
        ax = axes[3]
        ax.plot(time, X_history[9], label='∇_x')
        ax.plot(time, X_history[10], label='∇_y')
        ax.plot(time, X_history[11], label='∇_z')
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('加计零偏 (m/s^2)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('加速度计零偏估计')
        
        fig.suptitle('卡尔曼滤波12维误差状态估计', fontsize=14)
        plt.tight_layout()
        self._save(filename)
    
    def plot_attitude_update_comparison(self, att_history: np.ndarray,
                                         ref_aligned: np.ndarray,
                                         time: np.ndarray = None,
                                         filename: str = "attitude_update"):
        """
        图7：三轴姿态更新曲线 vs 惯导系统输出参考姿态
        """
        if time is None:
            time = np.arange(len(att_history))
        
        min_len = min(len(att_history), len(ref_aligned))
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 9))
        labels = ['横滚角 (°)', '俯仰角 (°)', '航向角 (°)']
        colors = ['#2196F3', '#4CAF50', '#FF9800']
        
        for i, ax in enumerate(axes):
            ax.plot(time[:min_len], att_history[:min_len, i], 
                    color=colors[i], label='自主解算', linewidth=1)
            ax.plot(time[:min_len], ref_aligned[:min_len, i], 
                    'r--', label='惯导参考', linewidth=1, alpha=0.7)
            ax.set_ylabel(labels[i])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('时间 (s)')
        fig.suptitle('姿态更新曲线对比（自主解算 vs 惯导参考）', fontsize=14)
        plt.tight_layout()
        self._save(filename)
    
    def plot_all_results(self, imu_data: np.ndarray,
                          coarse_ref: Dict,
                          fine_ref: Dict,
                          update_ref: Dict):
        """
        一键生成所有图表（实验报告所需6类图）
        
        输入:
            imu_data — [N×7] 预处理后数据
            coarse_ref — 含粗对准相关数据
            fine_ref — 含精对准相关数据
            update_ref — 含姿态更新相关数据
        """
        print("[Plotter] 开始生成所有图表...")
        
        # 图1: 粗对准原始数据曲线
        self.plot_imu_raw_data(imu_data, '粗对准', 'coarse_align_raw_data')
        
        # 图2: 精对准原始数据曲线（相同数据不同标题）
        self.plot_imu_raw_data(imu_data, '精对准', 'fine_align_raw_data')
        
        # 图3: 粗对准曲线
        if 'att_history' in coarse_ref:
            self.plot_coarse_align(
                coarse_ref['att_history'],
                coarse_ref.get('ref_att'),
                filename='coarse_align_attitude'
            )
        
        # 图5: 精对准曲线 + KF参数
        if 'att_history' in fine_ref:
            self.plot_fine_align(
                fine_ref['att_history'],
                fine_ref.get('ref_att'),
                filename='fine_align_attitude'
            )
        if 'X_history' in fine_ref:
            self.plot_kf_error_params(
                fine_ref['X_history'],
                filename='kf_error_params'
            )
        
        # 图7: 姿态更新对比
        if 'att_history' in update_ref and 'ref_aligned' in update_ref:
            self.plot_attitude_update_comparison(
                update_ref['att_history'],
                update_ref['ref_aligned'],
                filename='attitude_update'
            )
        
        print("[Plotter] 所有图表生成完毕")