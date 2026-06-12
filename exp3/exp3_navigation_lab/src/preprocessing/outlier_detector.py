"""
野值检测器
==========
检测和修复IMU数据中的野值
"""

import numpy as np
from scipy import interpolate
from typing import Optional


class OutlierDetector:
    """野值检测与修复"""
    
    @staticmethod
    def detect_outliers_std(data: np.ndarray, 
                            threshold: float = 5.0,
                            axis: int = 0) -> np.ndarray:
        """
        基于标准差的野值检测
        输入: data — [N×C] 多通道数据
              threshold — 标准差倍数阈值
              axis — 检测轴
        输出: mask — [N] bool数组，True=野值
        """
        mean = np.mean(data, axis=axis)
        std = np.std(data, axis=axis)
        
        if data.ndim == 1:
            mask = np.abs(data - mean) > threshold * std
        else:
            # 多通道：任一通道超限即标记
            centered = np.abs(data - mean[None, :])
            mask = np.any(centered > threshold * std[None, :], axis=1)
        
        return mask
    
    @staticmethod
    def detect_outliers_mad(data: np.ndarray, 
                            threshold: float = 3.0) -> np.ndarray:
        """
        基于MAD（中位数绝对偏差）的鲁棒野值检测
        输入: data — [N×C] 多通道数据
              threshold — MAD倍数阈值
        输出: mask — [N] bool数组
        """
        median = np.median(data, axis=0)
        mad = np.median(np.abs(data - median[None, :]), axis=0)
        # 避免MAD=0
        mad = np.maximum(mad, 1e-12)
        
        if data.ndim == 1:
            mask = np.abs(data - median) > threshold * mad / 0.6745
        else:
            centered = np.abs(data - median[None, :])
            mask = np.any(centered > threshold * mad[None, :] / 0.6745, axis=1)
        
        return mask
    
    @staticmethod
    def interpolate_outliers(data: np.ndarray, 
                             mask: np.ndarray,
                             method: str = 'linear') -> np.ndarray:
        """
        插值修复野值
        输入: data — [N×C] 原始数据
              mask — [N] 野值标记
              method — 插值方法: linear/cubic/spline
        输出: repaired — [N×C] 修复后数据
        """
        N = data.shape[0]
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        repaired = data.copy()
        good_idx = np.where(~mask)[0]
        
        if len(good_idx) < 2:
            print("[OutlierDetector] 有效数据点不足，无法插值")
            return repaired
        
        for ch in range(data.shape[1]):
            bad_idx = np.where(mask)[0]
            if len(bad_idx) == 0:
                continue
            
            if method == 'linear':
                repaired[bad_idx, ch] = np.interp(
                    bad_idx, good_idx, data[good_idx, ch])
            elif method in ['cubic', 'spline']:
                if len(good_idx) >= 4:
                    f = interpolate.interp1d(good_idx, data[good_idx, ch],
                                              kind='cubic', fill_value='extrapolate')
                    repaired[bad_idx, ch] = f(bad_idx)
                else:
                    repaired[bad_idx, ch] = np.interp(
                        bad_idx, good_idx, data[good_idx, ch])
            else:
                raise ValueError(f"不支持的插值方法: {method}")
        
        return repaired.reshape(N, -1)
    
    @staticmethod
    def sliding_average(data: np.ndarray, window_size: int = 5) -> np.ndarray:
        """滑动平均滤波"""
        from scipy.ndimage import uniform_filter1d
        return uniform_filter1d(data, size=window_size, axis=0, mode='reflect')