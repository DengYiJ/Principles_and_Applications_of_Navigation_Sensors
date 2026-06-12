"""
积分器
======
提供不同数值积分策略：梯形法、辛普森法等
"""

import numpy as np


class Integrator:
    """数值积分器"""

    @staticmethod
    def trapz(y: np.ndarray, dt: float, axis: int = 0) -> np.ndarray:
        """
        梯形法数值积分
        输入: y — [N] 或 [N×C] 被积函数值
              dt — 采样间隔
              axis — 积分轴
        输出: 积分结果 [C] 或标量
        """
        if y.ndim == 1:
            return np.trapz(y, dx=dt)
        return np.trapz(y, dx=dt, axis=axis)
    
    @staticmethod
    def cumtrapz(y: np.ndarray, dt: float, initial: float = 0.0) -> np.ndarray:
        """
        累积梯形积分
        输入: y — [N] 被积函数
              dt — 采样间隔
              initial — 初始积分值
        输出: [N] 累积积分值
        """
        from scipy.integrate import cumulative_trapezoid
        result = cumulative_trapezoid(y, dx=dt, initial=initial)
        return result
    
    @staticmethod
    def simpson(y: np.ndarray, dt: float, axis: int = 0) -> np.ndarray:
        """
        辛普森法数值积分（要求N为奇数）
        输入: y — [N] 或 [N×C]
              dt — 采样间隔
        输出: 积分结果
        """
        from scipy.integrate import simpson
        return simpson(y, dx=dt, axis=axis)