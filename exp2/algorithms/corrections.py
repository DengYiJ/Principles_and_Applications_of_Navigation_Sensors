"""
误差修正层 — Error Correction
===============================
Module C1: ClockCorrection
Module C2: EarthRotationCorrection

对伪距和卫星坐标进行误差修正。

设计依据: 03_algorithm_design.md Stage 2 Module C1/C2
"""

from typing import Tuple
import math
import numpy as np
import configs.constants as const


def apply_clock_correction(
    eph_row: np.ndarray,
    t_obs: float,
    Ek: float,
    A: float,
    e: float
) -> float:
    """卫星钟差修正 (Module C1)

    计算卫星钟差修正量 (包含多项式项 + 相对论效应)。
    修正公式:
        Δt_sv = a_f0 + a_f1·(t-t_oc) + a_f2·(t-t_oc)² + Δt_r
        其中 Δt_r = -2√(GM) · e · √A · sin(Ek) / C²

    Args:
        eph_row: 星历参数向量 (23 列)
        t_obs: 观测时刻 [s]
        Ek: 偏近点角 [rad] (来自卫星位置解算)
        A: 轨道半长轴 [m] (来自卫星位置解算)
        e: 偏心率 (来自卫星位置解算)

    Returns:
        dt_s: 卫星钟差修正量 [s] (正值表示卫星钟偏快)

    Reference:
        指导书 3.3 节, ICD-GPS-200 20.3.3.3.3.1
    """
    toc = eph_row[18]  # 星钟参考时刻 [s]
    af0 = eph_row[20]  # a_f0 [s]
    af1 = eph_row[21]  # a_f1 [s/s]
    af2 = eph_row[22]  # a_f2 [s/s²]

    # 时间差
    dt = t_obs - toc

    # 多项式项
    dt_poly = af0 + af1 * dt + af2 * dt * dt

    # 相对论修正项
    # Δt_r = -2 * sqrt(GM) * e * sqrt(A) * sin(Ek) / C²
    dt_relativistic = (-2.0 * math.sqrt(const.GM) * e
                       * math.sqrt(A) * math.sin(Ek)
                       / (const.C * const.C))

    return dt_poly + dt_relativistic


def apply_earth_rotation_correction(
    X: float,
    Y: float,
    Z: float,
    pseudorange: float
) -> Tuple[float, float, float]:
    """地球自转 Sagnac 效应修正 (Module C2)

    卫星信号传播期间地球自转引起的坐标修正 (Sagnac 效应)。
    修正公式:
        X' = X + ω_e · τ · Y
        Y' = Y - ω_e · τ · X
        Z' = Z
        其中 τ = ρ / C (信号传播延时)

    Args:
        X, Y, Z: 修正前的卫星 ECEF 坐标 [m]
        pseudorange: 卫星伪距 [m]

    Returns:
        (X', Y', Z'): 修正后的卫星 ECEF 坐标 [m]

    Reference:
        指导书 3.3 节, ICD-GPS-200 20.3.3.4.3.3
    """
    # 信号传播延时 [s]
    tau = pseudorange / const.C

    X_corr = X + const.OMEGA_E * tau * Y
    Y_corr = Y - const.OMEGA_E * tau * X
    Z_corr = Z

    return X_corr, Y_corr, Z_corr