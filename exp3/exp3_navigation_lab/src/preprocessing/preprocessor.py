"""
数据预处理器
============
完成IMU数据单位转换、野值检测、格式化为N×7矩阵、参考数据对齐
"""

import numpy as np
from typing import Dict, Optional, Tuple
from .outlier_detector import OutlierDetector


class Preprocessor:
    """IMU数据预处理器"""
    
    @staticmethod
    def process(imu_raw: Dict[str, np.ndarray],
                outlier_threshold: float = 5.0,
                interpolation: str = 'linear') -> np.ndarray:
        """
        预处理IMU数据
        
        输入:
            imu_raw — DataLoader输出的原始数据Dict
            outlier_threshold — 野值检测阈值（标准差倍数）
            interpolation — 插值方法
        
        返回:
            imu_processed — [N×7] = [gyroX,gyroY,gyroZ,accX,accY,accZ,time]
        """
        gyro = imu_raw['gyro']
        acc = imu_raw['acc']
        time = imu_raw['time']
        
        # 断言数据非空
        assert len(gyro) > 0, "陀螺仪数据为空"
        assert len(acc) > 0, "加速度计数据为空"
        assert len(time) > 0, "时间戳数据为空"
        
        # 检查时间戳是否递增（有无时间跳跃）
        dt = np.diff(time)
        if np.any(dt <= 0):
            n_reversed = np.sum(dt <= 0)
            print(f"[Preprocessor] 警告: 发现 {n_reversed} 个时间戳不递增，将强制排序")
            # 按时间排序
            sort_idx = np.argsort(time)
            gyro = gyro[sort_idx]
            acc = acc[sort_idx]
            time = time[sort_idx]
        
        # 检查并处理NaN/Inf
        for name, data in [('gyro', gyro), ('acc', acc), ('time', time)]:
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                print(f"[Preprocessor] {name} 数据存在NaN/Inf，将替换为0")
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 拼接为N×7矩阵
        imu_data = np.column_stack([gyro, acc, time])
        
        # 野值检测（对陀螺和加速度计数据）
        sensor_data = imu_data[:, :6]  # 6轴数据
        outlier_mask = OutlierDetector.detect_outliers_std(
            sensor_data, threshold=outlier_threshold)
        
        n_outliers = np.sum(outlier_mask)
        if n_outliers > 0:
            print(f"[Preprocessor] 检测到 {n_outliers} 个野值点 ({n_outliers/len(imu_data)*100:.2f}%)")
            # 插值修复
            sensor_repaired = OutlierDetector.interpolate_outliers(
                sensor_data, outlier_mask, method=interpolation)
            imu_data[:, :6] = sensor_repaired
        
        print(f"[Preprocessor] 预处理完成: {imu_data.shape[0]} 个采样点")
        return imu_data
    
    @staticmethod
    def align_reference(ref_raw: Dict[str, np.ndarray],
                         imu_time: np.ndarray) -> np.ndarray:
        """
        将参考姿态时间对齐到IMU时间序列
        
        输入:
            ref_raw — DataLoader输出的$GPFPD数据Dict
            imu_time — IMU时间戳 [N]
        
        返回:
            ref_aligned — [M×3] = [heading, pitch, roll] 对齐到IMU时间点
        """
        ref_time = ref_raw['time']
        
        # $GPFPD中的顺序：Heading, Pitch, Roll
        ref_values = np.column_stack([
            ref_raw['heading'],
            ref_raw['pitch'],
            ref_raw['roll']
        ])
        
        # 插值对齐
        ref_aligned = np.zeros((len(imu_time), 3))
        for i in range(3):
            ref_aligned[:, i] = np.interp(imu_time, ref_time, ref_values[:, i])
        
        n_aligned = len(ref_aligned)
        print(f"[Preprocessor] 参考数据对齐完成: {n_aligned} 个点")
        return ref_aligned
    
    @staticmethod
    def extract_reference_for_compare(ref_raw: Dict[str, np.ndarray]) -> np.ndarray:
        """
        提取参考姿态的完整序列（用于姿态更新对比）
        返回: [M×3] = [heading, pitch, roll]
        """
        return np.column_stack([
            ref_raw['heading'],
            ref_raw['pitch'],
            ref_raw['roll']
        ])