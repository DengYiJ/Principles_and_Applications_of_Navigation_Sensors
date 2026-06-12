"""
数据保存器
==========
保存处理结果到CSV文件
"""

import numpy as np
import os
import csv
from typing import Dict, Optional


class DataSaver:
    """实验结果保存工具"""
    
    @staticmethod
    def save_csv(filepath: str, headers: list, data: np.ndarray):
        """保存数据到CSV文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for row in data:
                writer.writerow(row)
        print(f"[DataSaver] 已保存: {filepath}")
    
    @staticmethod
    def save_alignment_result(filepath: str, 
                               method: str,
                               roll: float, pitch: float, yaw: float,
                               ref_roll: float, ref_pitch: float, ref_yaw: float):
        """保存对准结果对比表"""
        headers = ['项目', '横滚角(°)', '俯仰角(°)', '航向角(°)']
        data = [
            [f'{method}结果', f'{roll:.4f}', f'{pitch:.4f}', f'{yaw:.4f}'],
            ['参考姿态(惯导输出)', f'{ref_roll:.4f}', f'{ref_pitch:.4f}', f'{ref_yaw:.4f}'],
            ['误差', f'{roll-ref_roll:.4f}', f'{pitch-ref_pitch:.4f}', f'{yaw-ref_yaw:.4f}']
        ]
        DataSaver.save_csv(filepath, headers, np.array(data))
    
    @staticmethod
    def save_error_statistics(filepath: str, stats: Dict):
        """保存误差统计"""
        headers = ['误差指标', '横滚(°)', '俯仰(°)', '航向(°)']
        data = []
        for key, values in stats.items():
            data.append([key, values[0], values[1], values[2]])
        DataSaver.save_csv(filepath, headers, np.array(data))
    
    @staticmethod
    def save_kf_estimates(filepath: str, 
                          X_history: np.ndarray, 
                          P_diag_history: np.ndarray,
                          time_vec: np.ndarray):
        """保存KF状态估计历史"""
        headers = ['time', 'phi_E', 'phi_N', 'phi_U', 
                   'dv_E', 'dv_N', 'dv_U',
                   'eps_x', 'eps_y', 'eps_z',
                   'nabla_x', 'nabla_y', 'nabla_z']
        data = np.column_stack([time_vec, X_history.T])
        # 只保存降采样数据（每100个点）
        if data.shape[0] > 10000:
            data = data[::100, :]
        DataSaver.save_csv(filepath, headers, data)