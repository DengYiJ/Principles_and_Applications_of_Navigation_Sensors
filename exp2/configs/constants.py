"""
物理常数与算法超参数
=========================
集中管理系统中所有可配置的常数和超参数。
遵循 ICD-GPS-200 / IS-GPS-200 标准。

设计依据: 03_algorithm_design.md Stage 7
"""

import math

# =============================================================================
# 物理常数 (Physical Constants)
# =============================================================================

# 光速 [m/s]
C: float = 299792458.0

# 地球引力常数 [m^3/s^2] (WGS-84)
GM: float = 3.986005e14

# 地球自转角速度 [rad/s] (WGS-84)
OMEGA_E: float = 7.2921151467e-5

# WGS-84 椭球参数
A_WGS84: float = 6378137.0       # 长半轴 [m]
F_WGS84: float = 1.0 / 298.257223563  # 扁率
E2_WGS84: float = 2 * F_WGS84 - F_WGS84 ** 2  # 第一偏心率的平方

# 数学常数
PI: float = math.pi
D2R: float = PI / 180.0          # 度→弧度
R2D: float = 180.0 / PI          # 弧度→度

# =============================================================================
# 算法超参数 (Algorithm Hyperparameters)
# =============================================================================

# 开普勒方程迭代收敛阈值 [rad]
# 默认 1e-12 ≈ 对应卫星位置误差 < 0.1 mm
KEPLER_CONV_TOL: float = 1e-12

# 开普勒方程最大迭代次数
KEPLER_MAX_ITER: int = 100

# 最小二乘迭代收敛阈值 [m] (位置分量 ‖ΔX[:3]‖)
LSQ_CONV_TOL: float = 1e-4

# 最小二乘最大迭代次数
LSQ_MAX_ITER: int = 20

# 星历最大允许时差 [s] (|TOE - t_obs|)
EPHEMERIS_MAX_TIME_DIFF: float = 7200.0

# 有效定位最小卫星数
MIN_SATELLITES: int = 4

# GPS 卫星 PRN 有效范围
GPS_PRN_MIN: int = 1
GPS_PRN_MAX: int = 37

# ECEF→大地坐标迭代收敛阈值 [rad]
GEO_CONV_TOL: float = 1e-10

# ECEF→大地坐标最大迭代次数
GEO_MAX_ITER: int = 100

# =============================================================================
# 数据解析参数 (Data Parsing Parameters)
# =============================================================================

# 有效 L1 C/A 码信号掩码 (低 4 位)
L1_SIGNAL_MASKS: set = {0x04, 0x08, 0x0C}

# RANGEA 观测块字段数
RANGEA_BLOCK_SIZE: int = 10

# 日志文件路径
DATA_FILE_PATH: str = "exp2/data.txt"