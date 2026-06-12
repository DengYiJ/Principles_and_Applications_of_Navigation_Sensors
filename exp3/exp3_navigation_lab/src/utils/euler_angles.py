"""
欧拉角运算库
============
定义欧拉角与DCM/四元数的互转，支持3-1-2转序（本实验使用）和3-2-1转序
3-1-2转序: ψ(航向) → γ(横滚) → θ(俯仰)，对应 Z → X → Y 轴旋转
"""

import numpy as np
from .quaternion import quat_normalize


# ==================== 3-1-2 转序（本实验使用）====================

def euler312_to_dcm(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    3-1-2转序欧拉角 → 方向余弦矩阵
    输入: roll — 横滚角 (rad)
          pitch — 俯仰角 (rad)
          yaw — 航向角 (rad)
    输出: C — shape (3,3) 方向余弦矩阵 C_n^b
    旋转顺序: ψ(Z) → γ(X) → θ(Y)
    C = Ry(θ) * Rx(γ) * Rz(ψ)
    """
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    
    C = np.array([
        [cy*cr - sy*sp*sr,  sy*cp,  cy*sr + sy*sp*cr],
        [-sy*cr - cy*sp*sr, cy*cp, -sy*sr + cy*sp*cr],
        [-cp*sr,           -sp,     cp*cr            ]
    ])
    return C


def dcm_to_euler312(C: np.ndarray) -> tuple:
    """
    方向余弦矩阵 → 3-1-2转序欧拉角
    输入: C — shape (3,3) 方向余弦矩阵 C_n^b
    输出: (roll, pitch, yaw) — 横滚/俯仰/航向 (rad)
    """
    pitch = np.arcsin(-C[2,1])
    
    if np.abs(np.abs(C[2,1]) - 1.0) < 1e-12:
        # 万向锁（俯仰=±90°），航向设为0
        roll = 0.0
        yaw = np.arctan2(C[0,2], C[0,0])
    else:
        roll = np.arctan2(-C[2,0], C[2,2])
        yaw = np.arctan2(C[0,1], C[1,1])
    
    return (roll, pitch, yaw)


def euler312_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    3-1-2转序欧拉角 → 四元数
    输入: roll, pitch, yaw — (rad)
    输出: q — shape (4,)
    """
    C = euler312_to_dcm(roll, pitch, yaw)
    from .dcm import dcm_to_quat
    return dcm_to_quat(C)


def quat_to_euler312(q: np.ndarray) -> tuple:
    """
    四元数 → 3-1-2转序欧拉角
    输入: q — shape (4,)
    输出: (roll, pitch, yaw) — (rad)
    """
    from .dcm import dcm_from_quat
    C = dcm_from_quat(q)
    return dcm_to_euler312(C)


# ==================== 3-2-1 转序（备用）====================

def euler321_to_dcm(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    3-2-1转序欧拉角 → DCM (标准航空航天转序)
    旋转顺序: ψ(Z) → θ(Y) → γ(X)
    C = Rx(γ) * Ry(θ) * Rz(ψ)
    """
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    
    # C = Rx(roll) @ Ry(pitch) @ Rz(yaw)
    C = np.array([
        [cp*cy,             -cp*sy,              sp],
        [sr*sp*cy + cr*sy,  -sr*sp*sy + cr*cy,  -sr*cp],
        [-cr*sp*cy + sr*sy,  cr*sp*sy + sr*cy,   cr*cp]
    ])
    return C


def dcm_to_euler321(C: np.ndarray) -> tuple:
    """
    DCM -> 3-2-1 Euler angles
    C = Rx(roll) @ Ry(pitch) @ Rz(yaw)
    Returns: (roll, pitch, yaw) in rad
    """
    pitch = np.arcsin(C[0, 2])

    if np.abs(np.abs(C[0, 2]) - 1.0) < 1e-12:
        roll = 0.0
        yaw = np.arctan2(-C[1, 0], C[1, 1])
    else:
        roll = np.arctan2(-C[1, 2], C[2, 2])
        yaw = np.arctan2(-C[0, 1], C[0, 0])

    return (roll, pitch, yaw)


# ==================== 辅助函数 ====================

def rad2deg(rad: float) -> float:
    """弧度转角度"""
    return rad * 180.0 / np.pi


def deg2rad(deg: float) -> float:
    """角度转弧度"""
    return deg * np.pi / 180.0