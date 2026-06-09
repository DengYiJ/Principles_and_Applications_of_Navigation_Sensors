"""
姿态表构建函数（理论输入矩阵）
对应实验指导书 表1(加速度计六位置) 和 表2(陀螺仪八位置)
"""
import numpy as np
from .constants import G_MAGNITUDE, EARTH_ROTATION_RATE_DEGPS, DEFAULT_LATITUDE_DEG, DEG2RAD


def build_accel_pose_table_6() -> np.ndarray:
    """
    加速度计六位置理论重力分量表 (6×3)
    
    位置1-6: X/Y/Z轴分别朝上和朝下，感受 ±g 分量
    返回: shape=(6,3), 每行 [ax, ay, az] 理论比力 (m/s²)
    
    位置1: X朝上 → [g, 0, 0]
    位置2: X朝下 → [-g, 0, 0]
    位置3: Y朝上 → [0, g, 0]
    位置4: Y朝下 → [0, -g, 0]
    位置5: Z朝上 → [0, 0, g]
    位置6: Z朝下 → [0, 0, -g]
    """
    g = G_MAGNITUDE
    table = np.array([
        [ g,  0,  0],   # 位置1: X轴朝上
        [-g,  0,  0],   # 位置2: X轴朝下
        [ 0,  g,  0],   # 位置3: Y轴朝上
        [ 0, -g,  0],   # 位置4: Y轴朝下
        [ 0,  0,  g],   # 位置5: Z轴朝上
        [ 0,  0, -g],   # 位置6: Z轴朝下
    ], dtype=np.float64)
    return table


def build_gyro_bias_pose_table_8(latitude_deg: float = DEFAULT_LATITUDE_DEG) -> np.ndarray:
    """
    陀螺仪八位置地球自转理论投影表 (8×3)
    
    8个位置的地球自转角速率在载体系的理论投影分量 (°/s)
    基于实验指导书表2的8个姿态角度
    
    返回: shape=(8,3), 每行 [ωx, ωy, ωz] (°/s)
    """
    lat = latitude_deg * DEG2RAD
    # 地球自转在导航系(n系)的分量: ω_en = [ω*cos(lat), 0, -ω*sin(lat)]
    # 但本实验采用简化定义，直接给出各姿态的理论投影
    ω = EARTH_ROTATION_RATE_DEGPS  # °/s
    
    # 表2: 八个位置对应姿态描述:
    # 位置1: X东, Y北, Z天 → ω投影 = [0, ω*cos(lat), ω*sin(lat)]
    # 位置2: X西, Y北, Z地 → ω投影
    # ... (简化: 使用8个正交方向的近似投影)
    
    ω_e = ω
    ω_n = ω * np.cos(lat)
    ω_u = ω * np.sin(lat)
    
    # 8个姿态的理论地球自转投影（简化正交配置）
    # 实际值需根据表2精确角度计算，此处给出标准正交近似
    table = np.array([
        [  0,  ω_n,  ω_u],   # 位置1
        [  0, -ω_n, -ω_u],   # 位置2
        [ ω_n,  0,  ω_u],    # 位置3
        [-ω_n,  0, -ω_u],    # 位置4
        [ ω_n, -ω_u,  0],    # 位置5
        [-ω_n,  ω_u,  0],    # 位置6
        [ ω_e,  0,   0],     # 位置7
        [-ω_e,  0,   0],     # 位置8
    ], dtype=np.float64)
    return table


def build_gyro_bias_pose_table_8_standard() -> np.ndarray:
    """
    标准地球自转投影（纬度无关简化版）
    当纬度未知时使用此表做近似标定
    
    返回: shape=(8,3), 单位 °/s
    """
    ω = EARTH_ROTATION_RATE_DEGPS
    # 在45°N近似: ω_n ≈ ω_v ≈ ω/√2 ≈ 0.0030 °/s
    ω_45 = ω * np.cos(45 * DEG2RAD)  # ≈ ω * 0.7071
    
    table = np.array([
        [  0,      ω_45,   ω_45],   # 位置1
        [  0,     -ω_45,  -ω_45],   # 位置2
        [ ω_45,    0,      ω_45],   # 位置3
        [-ω_45,    0,     -ω_45],   # 位置4
        [ ω_45,   -ω_45,   0   ],   # 位置5
        [-ω_45,    ω_45,   0   ],   # 位置6
        [ ω,       0,      0   ],   # 位置7
        [-ω,       0,      0   ],   # 位置8
    ], dtype=np.float64)
    return table