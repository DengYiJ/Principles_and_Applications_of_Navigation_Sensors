"""
角度增量积分器
支持梯形法和辛普森法数值积分
"""
import numpy as np


class Integrator:
    """角度增量积分器：将角速率积分成角度"""

    @staticmethod
    def trapezoidal(gyro_data: np.ndarray, dt: float = 0.005) -> np.ndarray:
        """
        梯形法数值积分
        J = ∫ ω(t) dt ≈ Σ (ω_{i} + ω_{i+1})/2 * dt

        参数:
            gyro_data: shape=(N, 3), 角速率(°/s)
            dt: 采样间隔(s), 默认0.005s(200Hz)
        返回:
            J: shape=(3,), 积分角度增量(°)
        """
        if len(gyro_data) < 2:
            return np.zeros(3, dtype=np.float64)

        # 梯形法: (y0+y1)/2 + (y1+y2)/2 + ... = (y0/2 + y1 + ... + y_{n-1}/2)
        integral = 0.5 * (gyro_data[0] + gyro_data[-1]) + np.sum(gyro_data[1:-1], axis=0)
        return integral * dt

    @staticmethod
    def simpson(gyro_data: np.ndarray, dt: float = 0.005) -> np.ndarray:
        """
        辛普森法数值积分（要求N为奇数，否则退化为梯形法）

        参数:
            gyro_data: shape=(N, 3), 角速率(°/s)
            dt: 采样间隔(s)
        返回:
            J: shape=(3,), 积分角度增量(°)
        """
        n = len(gyro_data)
        if n < 3:
            return Integrator.trapezoidal(gyro_data, dt)

        # 若N为偶数，最后一点用梯形法
        if n % 2 == 0:
            integral = (gyro_data[0] + gyro_data[-1] + 4 * np.sum(gyro_data[1:n-1:2], axis=0)
                        + 2 * np.sum(gyro_data[2:n-2:2], axis=0)) * dt / 3.0
        else:
            integral = (gyro_data[0] + gyro_data[-1] + 4 * np.sum(gyro_data[1:-1:2], axis=0)
                        + 2 * np.sum(gyro_data[2:-2:2], axis=0)) * dt / 3.0
        return integral

    @staticmethod
    def integrate_one_revolution(gyro_data: np.ndarray, dt: float = 0.005,
                                  method: str = "trapezoidal") -> np.ndarray:
        """
        旋转一周的角度增量积分

        参数:
            gyro_data: 旋转一周内的陀螺三轴输出 shape=(N,3)
            dt: 采样间隔
            method: "trapezoidal" 或 "simpson"
        返回:
            J: shape=(3,) 角度增量(°)
        """
        if method == "simpson":
            return Integrator.simpson(gyro_data, dt)
        else:
            return Integrator.trapezoidal(gyro_data, dt)

    @staticmethod
    def compute_angle_increments(gyro_data: np.ndarray, dt: float = 0.005) -> np.ndarray:
        """
        逐秒角度增量（用于分段时间对齐）

        参数:
            gyro_data: shape=(N,3)
            dt: 采样间隔
        返回:
            angle_increments: shape=(N,) 每步的角度增量(°)
        """
        return np.sqrt(np.sum((gyro_data * dt) ** 2, axis=1))