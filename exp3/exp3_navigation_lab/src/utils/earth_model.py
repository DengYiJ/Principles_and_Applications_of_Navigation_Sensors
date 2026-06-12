"""
地球模型库
==========
定义地球物理常数、重力模型、地球自转投影、坐标系变换
"""

import numpy as np

# ==================== 地球常数 ====================

WGS84_A = 6378137.0          # 地球长半轴 (m)
WGS84_F = 1.0 / 298.257223563  # 扁率
WGS84_E2 = 2 * WGS84_F - WGS84_F**2  # 第一偏心率平方
WGS84_OMEGA = 7.292115e-5    # 地球自转角速度 (rad/s)
WGS84_GM = 3.986004418e14    # 地球引力常数 (m³/s^2)

# 标准重力加速度（纬度相关，此处使用哈工大纬度近似值）
G_0 = 9.7803267714           # 赤道海平面重力 (m/s^2)
G_POLAR = 9.8321863685       # 极地重力 (m/s^2)


def gravity_n(L_rad: float, h: float = 0.0) -> np.ndarray:
    """
    重力矢量在导航系(n系)下的投影
    输入: L_rad — 纬度 (rad)
          h — 高度 (m)，默认=0
    输出: g_n — shape (3,) = [0, 0, -g]，东/北/天方向
    """
    # Somigliana 公式
    sinL2 = np.sin(L_rad)**2
    g = G_0 * (1 + 0.0053024 * sinL2 - 0.0000058 * np.sin(2*L_rad)**2)
    # 高度修正
    g_h = g - (3.0877e-6 - 4.4e-9 * sinL2) * h + 0.72e-12 * h**2
    return np.array([0.0, 0.0, -g_h])


def gravity_i(t: float, L_rad: float, wie: float = WGS84_OMEGA, 
              g: float = G_0) -> np.ndarray:
    """
    重力矢量在惯性系(i系)下的投影（随时间变化，由地球自转引起）
    输入: t — 时间 (s)
          L_rad — 纬度 (rad)
          wie — 地球自转角速度 (rad/s)
          g — 当地重力加速度 (m/s^2)
    输出: g_i — shape (3,) 重力在i系下的表达式
    公式: gⁱ(t) = [g·cosL·cos(ω_ie·t), g·cosL·sin(ω_ie·t), g·sinL]
    """
    cosL = np.cos(L_rad)
    sinL = np.sin(L_rad)
    wie_t = wie * t
    return np.array([
        g * cosL * np.cos(wie_t),
        g * cosL * np.sin(wie_t),
        g * sinL
    ])


def earth_rate_n(L_rad: float, wie: float = WGS84_OMEGA) -> np.ndarray:
    """
    地球自转角速度在导航系(n系)下的投影
    输入: L_rad — 纬度 (rad)
          wie — 地球自转角速度 (rad/s)
    输出: wie_n — shape (3,) = [0, wie·cosL, wie·sinL] (东北天)
    """
    return np.array([0.0, wie * np.cos(L_rad), wie * np.sin(L_rad)])


def transport_rate_n(v_n: np.ndarray, L_rad: float, h: float = 0.0) -> np.ndarray:
    """
    导航系相对地球的旋转角速度（运输项）在n系下的投影
    输入: v_n — shape (3,) 速度向量 [vE, vN, vU] (m/s)
          L_rad — 纬度 (rad)
          h — 高度 (m)
    输出: en_n — shape (3,) 运输角速度项
    """
    Rn = WGS84_A * (1 - WGS84_E2) / (1 - WGS84_E2 * np.sin(L_rad)**2)**1.5
    Re = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(L_rad)**2)
    
    vE, vN, vU = v_n
    omega_en_n = np.array([
        vN / (Rn + h),
        vE / (Re + h),
        vE * np.tan(L_rad) / (Re + h)
    ])
    return omega_en_n


def C_n2i(t: float, L_rad: float, wie: float = WGS84_OMEGA) -> np.ndarray:
    """
    导航系(n系) → 惯性系(i系) 的变换矩阵（初始时刻之后随时间变化）
    输入: t — 时间 (s)
          L_rad — 纬度 (rad)
          wie — 地球自转角速度 (rad/s)
    输出: C — shape (3,3) C_n^i
    公式参见指导书粗对准章节
    """
    cosL = np.cos(L_rad)
    sinL = np.sin(L_rad)
    wie_t = wie * t
    
    # n系(i系在时刻t的表示): 东北天 → 惯性系
    C = np.array([
        [-np.sin(wie_t),         np.cos(wie_t),         0.0],
        [-sinL * np.cos(wie_t), -sinL * np.sin(wie_t),  cosL],
        [ cosL * np.cos(wie_t),  cosL * np.sin(wie_t),  sinL]
    ])
    return C


def C_i2n(t: float, L_rad: float, wie: float = WGS84_OMEGA) -> np.ndarray:
    """
    惯性系(i系) → 导航系(n系) 的变换矩阵
    输入: t — 时间 (s)
    输出: C — shape (3,3) C_i^n = (C_n^i)^T
    """
    return C_n2i(t, L_rad, wie).T