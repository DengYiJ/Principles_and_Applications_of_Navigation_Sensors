"""
物理边界断言工具库
==================
定义数据维度、物理量范围、姿态矩阵正交性等断言检查
"""

import numpy as np


def assert_imu_shape(imu_data: np.ndarray):
    """
    断言IMU数据格式为 N×7
    列: [gyroX, gyroY, gyroZ, accX, accY, accZ, time]
    """
    assert imu_data is not None, "IMU数据为None"
    assert imu_data.ndim == 2, f"IMU数据应为2维，实际为{imu_data.ndim}维"
    assert imu_data.shape[1] == 7, f"IMU数据列数应为7，实际为{imu_data.shape[1]}"
    assert imu_data.shape[0] > 10, f"IMU数据行数不足: {imu_data.shape[0]}"


def assert_dcm_valid(C: np.ndarray, tol: float = 1e-6, name: str = "C"):
    """
    断言DCM矩阵有效（正交且行列式为+1）
    """
    assert C is not None, f"{name} 为None"
    assert C.shape == (3, 3), f"{name} shape应为(3,3)，实际为{C.shape}"
    
    detC = np.linalg.det(C)
    assert abs(abs(detC) - 1.0) < tol, \
        f"{name} 行列式越界: det={detC:.2e}，应为±1"
    
    orth_err = np.max(np.abs(C.T @ C - np.eye(3)))
    assert orth_err < tol, \
        f"{name} 正交性误差过大: max|C^T C - I| = {orth_err:.2e}"


def assert_attitude_range(roll: float, pitch: float, yaw: float, 
                          tol: float = 1.0):
    """
    断言姿态角在合理物理范围内
    输入: roll, pitch, yaw — 单位: 度
          tol — 额外容差（度）
    """
    assert -180 - tol <= roll <= 180 + tol, \
        f"横滚角越界: {roll}° (应在[-180, 180]°范围内)"
    assert -90 - tol <= pitch <= 90 + tol, \
        f"俯仰角越界: {pitch}° (应在[-90, 90]°范围内)"
    # 航向角范围 [-180, 360]
    assert -180 - tol <= yaw <= 360 + tol, \
        f"航向角越界: {yaw}° (应在[-180, 360]°范围内)"


def assert_gravity_norm(g_vec: np.ndarray, tol: float = 0.1):
    """
    断言重力矢量模接近9.78 m/s^2
    """
    g_norm = np.linalg.norm(g_vec)
    assert abs(g_norm - 9.78) < tol, \
        f"重力矢量模异常: {g_norm:.4f} m/s^2 (应在9.78±{tol}范围内)"


def assert_latitude(L_deg: float):
    """
    断言纬度在合法范围内
    """
    assert -90 <= L_deg <= 90, \
        f"纬度越界: {L_deg}° (应在[-90, 90]°范围内)"


def assert_wie_order(wie: float):
    """
    断言地球自转角速度在10^-5量级
    """
    assert 1e-6 < wie < 1e-3, \
        f"地球自转角速度量级异常: {wie:.2e} (应为7.29e-5 rad/s)"


def assert_kf_state(X: np.ndarray, dim: int = 12):
    """
    断言KF状态向量维度正确
    """
    assert X is not None, "KF状态为None"
    assert X.shape[0] == dim, \
        f"KF状态维度应为{dim}，实际为{X.shape[0]}"


def assert_non_empty(data: np.ndarray, name: str = "data"):
    """断言数据非空"""
    assert data is not None, f"{name} 为None"
    assert len(data) > 0, f"{name} 为空"


def assert_quaternion(q: np.ndarray):
    """断言四元数格式正确且归一化"""
    assert q is not None, "四元数为None"
    assert q.shape == (4,), f"四元数shape应为(4,)，实际为{q.shape}"
    q_norm = np.linalg.norm(q)
    assert abs(q_norm - 1.0) < 1e-6, \
        f"四元数未归一化: ||q||={q_norm:.4f} (应为1.0)"


def check_singularity(mat: np.ndarray, name: str = "matrix") -> float:
    """
    检查矩阵是否奇异，返回条件数
    条件数 > 1e10 视为奇异
    """
    cond = np.linalg.cond(mat)
    if cond > 1e10:
        import warnings
        warnings.warn(f"{name} 条件数过大: cond={cond:.2e}，矩阵可能奇异")
    return cond