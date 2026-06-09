"""
异常值剔除滤波器
3σ准则剔除异常样本
"""
import numpy as np


class OutlierFilter:
    """基于3σ准则的异常值滤波器"""

    @staticmethod
    def filter_3sigma(data: np.ndarray, sigma_threshold: float = 3.0,
                      axis: int = 0, verbose: bool = True) -> np.ndarray:
        """
        3σ异常值剔除

        参数:
            data: shape=(N, D) 时间序列数据
            sigma_threshold: σ倍数阈值，默认3.0
            axis: 沿哪个轴计算统计量(0=逐通道)
        返回:
            剔除异常值后的数据
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        # 避免零标准差
        std = np.where(std < 1e-12, 1e-12, std)

        # 标记所有通道都在阈值内的样本
        z_scores = np.abs((data - mean) / std)
        valid_mask = np.all(z_scores < sigma_threshold, axis=1)

        n_total = len(data)
        n_removed = n_total - np.sum(valid_mask)
        if verbose and n_removed > 0:
            print(f"[OutlierFilter] removed {n_removed}/{n_total} samples "
                  f"({100*n_removed/n_total:.2f}%)")

        return data[valid_mask]

    @staticmethod
    def filter_by_channel(data: np.ndarray, sigma_threshold: float = 3.0,
                          verbose: bool = True) -> np.ndarray:
        """
        逐通道3σ异常值剔除（某通道异常则整行剔除）
        """
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std = np.where(std < 1e-12, 1e-12, std)

        z_scores = np.abs((data - mean) / std)
        valid_mask = np.all(z_scores < sigma_threshold, axis=1)

        n_removed = len(data) - np.sum(valid_mask)
        if verbose and n_removed > 0:
            print(f"[OutlierFilter] by_channel: removed {n_removed}/{len(data)} samples")

        return data[valid_mask]