"""
坐标变换层 — Coordinate Transform
===================================
Module E1: EcefToGeodetic

ECEF (地心地固直角坐标) ↔ WGS-84 大地坐标转换。

设计依据: 03_algorithm_design.md Stage 2 Module E1
"""

from typing import Tuple
import math
import configs.constants as const


def ecef_to_geodetic(X: float, Y: float, Z: float) -> Tuple[float, float, float]:
    """ECEF → WGS-84 大地坐标转换 (Module E1)

    使用迭代法将地心地固直角坐标转换为大地坐标 (经度, 纬度, 高度)。
    初始纬度估计使用 Hirvonen 方法, 迭代直到收敛。

    Args:
        X, Y, Z: ECEF 坐标 [m]

    Returns:
        (lon_deg, lat_deg, h_m): 经度 [°], 纬度 [°], 椭球高度 [m]

    Reference:
        指导书 3.4 节第 6 步, 03_algorithm_design.md Stage 4 Step 9
    """
    # 经度可以直接计算
    lon = math.atan2(Y, X)
    p = math.sqrt(X * X + Y * Y)  # 水平距离

    # 极地退化情况
    if p < 1e-6:
        lat = math.copysign(const.PI / 2.0, Z)
        h = abs(Z) - const.A_WGS84 * math.sqrt(1.0 - const.E2_WGS84)
        return lon * const.R2D, lat * const.R2D, h

    # 初始纬度估计 (Hirvonen 方法)
    phi = math.atan(Z / p / (1.0 - const.E2_WGS84))

    # 迭代求解纬度和高度
    for _ in range(const.GEO_MAX_ITER):
        sin_phi = math.sin(phi)
        N = const.A_WGS84 / math.sqrt(1.0 - const.E2_WGS84 * sin_phi * sin_phi)
        h = p / math.cos(phi) - N

        phi_new = math.atan(Z / p / (1.0 - const.E2_WGS84 * N / (N + h)))

        if abs(phi_new - phi) < const.GEO_CONV_TOL:
            phi = phi_new
            break
        phi = phi_new

    # 最终高度计算
    sin_phi = math.sin(phi)
    N = const.A_WGS84 / math.sqrt(1.0 - const.E2_WGS84 * sin_phi * sin_phi)
    h = p / math.cos(phi) - N

    return lon * const.R2D, phi * const.R2D, h


def geodetic_to_ecef(
    lon_deg: float, lat_deg: float, h_m: float
) -> Tuple[float, float, float]:
    """WGS-84 大地坐标 → ECEF 转换 (逆变换)

    Args:
        lon_deg: 经度 [°]
        lat_deg: 纬度 [°]
        h_m: 椭球高度 [m]

    Returns:
        (X, Y, Z): ECEF 坐标 [m]
    """
    lon = lon_deg * const.D2R
    lat = lat_deg * const.D2R

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)

    N = const.A_WGS84 / math.sqrt(1.0 - const.E2_WGS84 * sin_lat * sin_lat)

    X = (N + h_m) * cos_lat * math.cos(lon)
    Y = (N + h_m) * cos_lat * math.sin(lon)
    Z = (N * (1.0 - const.E2_WGS84) + h_m) * sin_lat

    return X, Y, Z