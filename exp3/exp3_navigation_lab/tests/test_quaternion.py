"""
四元数运算库单元测试
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.utils.quaternion import (
    quat_multiply, quat_conjugate, quat_normalize, quat_norm,
    quat_from_axis_angle, quat_to_axis_angle,
    quat_update_picard_2nd, quat_update_rk4, quat_update
)


def test_quat_multiply():
    """测试四元数乘法"""
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    q2 = np.array([1.0, 0.0, 0.0, 0.0])
    r = quat_multiply(q1, q2)
    assert np.allclose(r, [1, 0, 0, 0]), f"单位四元数乘法失败: {r}"
    
    # 旋转复合（X旋转90°再Y旋转90°）
    q_x = quat_from_axis_angle(np.array([1,0,0]), np.pi/2)
    q_y = quat_from_axis_angle(np.array([0,1,0]), np.pi/2)
    q_combined = quat_multiply(q_y, q_x)
    assert abs(quat_norm(q_combined) - 1.0) < 1e-10
    print("[PASS] test_quat_multiply")


def test_quat_conjugate():
    """测试四元数共轭"""
    q = np.array([1.0/np.sqrt(2), 1.0/np.sqrt(2), 0.0, 0.0])
    q_conj = quat_conjugate(q)
    r = quat_multiply(q, q_conj)
    assert np.allclose(r, [1.0, 0.0, 0.0, 0.0], atol=1e-10), \
        f"共轭测试失败: q⊗q* = {r}"
    print("[PASS] test_quat_conjugate")


def test_quat_normalize():
    """测试四元数归一化"""
    q = np.array([2.0, 0.0, 0.0, 0.0])
    q_norm = quat_normalize(q)
    assert abs(quat_norm(q_norm) - 1.0) < 1e-10
    assert np.allclose(q_norm, [1, 0, 0, 0])
    
    q_zero = np.array([0.0, 0.0, 0.0, 0.0])
    qz_norm = quat_normalize(q_zero)
    assert np.allclose(qz_norm, [1, 0, 0, 0])
    print("[PASS] test_quat_normalize")


def test_quat_axis_angle():
    """测试四元数与轴角互转"""
    axis = np.array([0.0, 0.0, 1.0])
    angle = np.pi/3
    q = quat_from_axis_angle(axis, angle)
    assert abs(quat_norm(q) - 1.0) < 1e-10
    assert np.allclose(q, [np.cos(angle/2), 0, 0, np.sin(angle/2)])
    
    axis_back, angle_back = quat_to_axis_angle(q)
    assert abs(angle_back - angle) < 1e-10
    print("[PASS] test_quat_axis_angle")


def test_quat_update():
    """测试四元数更新（零角速度应保持不变）"""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.zeros(3)
    dt = 0.01
    
    q_new = quat_update_picard_2nd(q, omega, dt)
    assert np.allclose(q_new, q), f"零角速度更新失败: {q_new}"
    
    q_new_rk4 = quat_update_rk4(q, omega, dt)
    assert np.allclose(q_new_rk4, q), f"零角速度RK4更新失败: {q_new_rk4}"
    print("[PASS] test_quat_update")


def test_quat_update_small():
    """测试小角度旋转（累积100步转90°）"""
    q_p = np.array([1.0, 0.0, 0.0, 0.0])
    q_r = np.array([1.0, 0.0, 0.0, 0.0])
    total_angle = np.pi/2  # 90°
    steps = 100
    omega = np.array([0.0, 0.0, total_angle])  # 总角位移
    dt = 1.0 / steps
    
    for _ in range(steps):
        q_p = quat_update_picard_2nd(q_p, omega, dt)
        q_r = quat_update_rk4(q_r, omega, dt)
    
    expected = np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])
    assert np.allclose(q_p, expected, atol=0.01), f"Picard旋转失败: {q_p}"
    assert np.allclose(q_r, expected, atol=0.01), f"RK4旋转失败: {q_r}"
    print("[PASS] test_quat_update_small")


def test_quat_update_unified():
    """测试统一接口"""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.array([0.1, 0.2, 0.3])
    dt = 0.01
    
    q1 = quat_update(q, omega, dt, method='picard_2nd')
    q2 = quat_update(q, omega, dt, method='rk4')
    assert abs(quat_norm(q1) - 1.0) < 1e-10
    assert abs(quat_norm(q2) - 1.0) < 1e-10
    print("[PASS] test_quat_update_unified")


if __name__ == '__main__':
    test_quat_multiply()
    test_quat_conjugate()
    test_quat_normalize()
    test_quat_axis_angle()
    test_quat_update()
    test_quat_update_small()
    test_quat_update_unified()
    print("\n所有四元数测试通过!")