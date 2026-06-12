"""
四元数运算库
============
定义四元数基本运算：乘法、共轭、归一化、DCM互转、增量更新
四元数约定：标量在前 [qw, qx, qy, qz]
"""

import numpy as np


def quat_multiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    四元数乘法 r = p ⊗ q
    输入: p, q — shape (4,) 四元数 [qw, qx, qy, qz]
    输出: r — shape (4,) p ⊗ q
    """
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    r = np.array([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw
    ])
    return r


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """
    四元数共轭
    输入: q — shape (4,) [qw, qx, qy, qz]
    输出: q_inv — shape (4,) [qw, -qx, -qy, -qz]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_norm(q: np.ndarray) -> float:
    """四元数模长"""
    return np.linalg.norm(q)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """
    四元数归一化
    输入: q — shape (4,)
    输出: q_norm — shape (4,), ||q_norm|| = 1
    """
    n = quat_norm(q)
    if n < 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """
    四元数逆
    输入: q — shape (4,)
    输出: q_inv — shape (4,), q ⊗ q_inv = [1,0,0,0]
    """
    return quat_normalize(quat_conjugate(q))


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    从旋转轴和角度构造四元数
    输入: axis — shape (3,) 单位旋转轴
          angle — 旋转角度 (rad)
    输出: q — shape (4,) [cos(θ/2), sin(θ/2)*axis]
    """
    half = angle * 0.5
    s = np.sin(half)
    return np.array([np.cos(half), s*axis[0], s*axis[1], s*axis[2]])


def quat_to_axis_angle(q: np.ndarray) -> tuple:
    """
    四元数 → 旋转轴和角度
    输入: q — shape (4,)
    输出: (axis, angle) — axis shape (3,), angle in rad
    """
    q = quat_normalize(q)
    angle = 2.0 * np.arccos(np.clip(q[0], -1.0, 1.0))
    if np.sin(angle/2) < 1e-12:
        return (np.array([1.0, 0.0, 0.0]), 0.0)
    axis = q[1:4] / np.sin(angle/2)
    return (axis, angle)


def quat_update_picard_2nd(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """
    四元数更新 — 毕卡二阶法 (Picard 2nd order)
    解四元数微分方程: dq/dt = 0.5 * q ⊗ [0, ωx, ωy, ωz]
    
    输入: q — shape (4,) 当前四元数
          omega — shape (3,) 角速度 (rad/s)
          dt — 时间间隔 (s)
    输出: q_new — shape (4,) 更新后四元数
    """
    omega_norm = np.linalg.norm(omega)
    if omega_norm < 1e-15:
        return q  # 无旋转
    
    half_theta = 0.5 * omega_norm * dt
    sin_half = np.sin(half_theta) / omega_norm
    
    # 构造增量四元数
    dq = np.array([
        np.cos(half_theta),
        sin_half * omega[0],
        sin_half * omega[1],
        sin_half * omega[2]
    ])
    
    # q_new = q ⊗ dq
    q_new = quat_multiply(q, dq)
    q_new = quat_normalize(q_new)
    return q_new


def quat_update_rk4(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """
    四元数更新 — 四阶龙格库塔法 (RK4)
    
    输入: q — shape (4,) 当前四元数
          omega — shape (3,) 角速度 (rad/s)
          dt — 时间间隔 (s)
    输出: q_new — shape (4,) 更新后四元数
    """
    def dqdt(q, w):
        """四元数微分方程右端项"""
        w4 = np.array([0.0, w[0], w[1], w[2]])
        return 0.5 * quat_multiply(q, w4)
    
    k1 = dqdt(q, omega)
    k2 = dqdt(q + 0.5*dt*k1, omega)
    k3 = dqdt(q + 0.5*dt*k2, omega)
    k4 = dqdt(q + dt*k3, omega)
    
    q_new = q + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    q_new = quat_normalize(q_new)
    return q_new


def quat_update(q: np.ndarray, omega: np.ndarray, dt: float, method: str = "picard_2nd") -> np.ndarray:
    """
    四元数更新统一接口
    输入: q — shape (4,) 当前四元数
          omega — shape (3,) 角速度 (rad/s)
          dt — 时间间隔 (s)
          method — "picard_2nd" 或 "rk4"
    输出: q_new — shape (4,) 更新后四元数
    """
    if method == "picard_2nd":
        return quat_update_picard_2nd(q, omega, dt)
    elif method == "rk4":
        return quat_update_rk4(q, omega, dt)
    else:
        raise ValueError(f"不支持的积分方法: {method}，请使用 'picard_2nd' 或 'rk4'")