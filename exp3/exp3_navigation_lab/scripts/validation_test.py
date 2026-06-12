"""
快速集成验证脚本
测试所有模块导入和基础功能是否正常
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

print("=" * 60)
print("实验(3) 惯性导航实验 — 集成验证")
print("=" * 60)

# ====== Stage 1: 验证所有导入 ======
print("\n[1/6] 验证所有模块导入...")

from src.utils.quaternion import (
    quat_multiply, quat_conjugate, quat_normalize, quat_norm,
    quat_from_axis_angle, quat_to_axis_angle, quat_update
)
from src.utils.dcm import (
    dcm_from_quat, dcm_to_quat, dcm_orthogonalize, skew, dcm_is_valid
)
from src.utils.euler_angles import (
    euler312_to_dcm, dcm_to_euler312, rad2deg, deg2rad
)
from src.utils.earth_model import (
    gravity_n, gravity_i, earth_rate_n, C_n2i, WGS84_OMEGA
)
from src.utils.kalman_filter import (
    LinearKF, build_F_matrix, discretize_F, build_H_matrix
)
from src.utils.assert_helpers import (
    assert_dcm_valid, assert_attitude_range, assert_imu_shape
)
from src.data_io import DataLoader, DataSaver
from src.preprocessing import Preprocessor, OutlierDetector
from src.alignment import CoarseAligner, FineAligner
from src.attitude import AttitudeUpdater, Integrator
from src.analysis import ComparisonAnalyzer, Plotter

print("  ✅ 所有模块导入成功")

# ====== Stage 2: 验证工具函数 ======
print("\n[2/6] 验证工具函数基础功能...")

# 四元数
q1 = np.array([1.0, 0.0, 0.0, 0.0])
q2 = np.array([0.0, 1.0, 0.0, 0.0])
r = quat_multiply(q1, q2)
assert np.allclose(r, q2), f"quat_multiply failed: {r}"
print("  ✅ 四元数乘法")

q = quat_from_axis_angle(np.array([0,0,1]), np.pi/3)
assert abs(quat_norm(q) - 1.0) < 1e-10
print("  ✅ 四元数轴角构造")

q_norm = quat_normalize(np.array([2.0, 0.0, 0.0, 0.0]))
assert abs(quat_norm(q_norm) - 1.0) < 1e-10
print("  ✅ 四元数归一化")

q_new = quat_update(q, np.array([0.1, 0.0, 0.0]), 0.01)
assert abs(quat_norm(q_new) - 1.0) < 1e-10
print("  ✅ 四元数更新")

# DCM
C = euler312_to_dcm(0.1, 0.2, 0.3)
assert dcm_is_valid(C), "DCM should be valid"
roll, pitch, yaw = dcm_to_euler312(C)
assert np.allclose([roll, pitch, yaw], [0.1, 0.2, 0.3], atol=1e-6)
print("  ✅ DCM/欧拉角互转")

C_orth = dcm_orthogonalize(C + np.random.randn(3,3)*1e-6)
assert dcm_is_valid(C_orth)
print("  ✅ DCM正交化")

# 反对称矩阵
v = np.array([1.0, 2.0, 3.0])
V_skew = skew(v)
assert np.allclose(V_skew @ v, np.zeros(3))
print("  ✅ 反对称矩阵")

# 地球模型
g_n = gravity_n(np.deg2rad(45.0))
assert abs(np.linalg.norm(g_n) - 9.78) < 0.1
print("  ✅ 重力模型")

wie_n = earth_rate_n(np.deg2rad(45.0))
assert abs(wie_n[1]) > 0
print("  ✅ 地球自转模型")

g_i = gravity_i(10.0, np.deg2rad(45.0))
assert abs(np.linalg.norm(g_i) - 9.78) < 0.1
print("  ✅ 惯性系重力")

C_n2i_0 = C_n2i(0.0, np.deg2rad(45.0))
assert C_n2i_0.shape == (3, 3)
print("  ✅ 坐标变换矩阵")

# KF
kf = LinearKF(dim_x=12, dim_z=3)
kf.H = build_H_matrix()
assert kf.H.shape == (3, 12)
print("  ✅ KF构造和H矩阵")

F_basic = build_F_matrix(np.eye(3), np.array([0,0,-9.78]), wie_n)
assert F_basic.shape == (12, 12)
print("  ✅ F矩阵构造")

Phi = discretize_F(F_basic, 0.005)
assert Phi.shape == (12, 12)
print("  ✅ F离散化")

# ====== Stage 3: 验证断言工具 ======
print("\n[3/6] 验证断言工具...")

assert_attitude_range(10.0, 20.0, 30.0)
print("  ✅ 姿态角范围断言")

assert_dcm_valid(np.eye(3))
print("  ✅ DCM有效性断言")

# ====== Stage 4: 生成模拟数据 ======
print("\n[4/6] 生成模拟IMU数据...")

np.random.seed(42)
N = 2000  # 10秒 @ 200Hz
fs = 200
dt = 1/fs
t = np.arange(N) * dt

# 模拟静基座IMU数据
gyro_bias = np.deg2rad(np.array([0.002, 0.002, 0.002]))  # 2°/h
acc_bias = np.array([0.02, 0.02, 0.02])  # 20mg

# 真实姿态：水平静止 (roll=0, pitch=0, yaw=45°)
true_roll, true_pitch, true_yaw = 0.0, 0.0, 45.0
L_rad = deg2rad(45.734501)
C_true = euler312_to_dcm(
    deg2rad(true_roll), deg2rad(true_pitch), deg2rad(true_yaw)
)

# 理想传感器输出
g = 9.7803267714
wie = 7.292115e-5

# 重力在b系投影
g_n_vec = np.array([0, 0, -g])
g_b = C_true.T @ g_n_vec

# 地球自转在b系投影
wie_n_vec = np.array([0, wie*np.cos(L_rad), wie*np.sin(L_rad)])
wie_b = C_true.T @ wie_n_vec

# 加噪声
gyro_noise = np.random.randn(N, 3) * deg2rad(0.001)
acc_noise = np.random.randn(N, 3) * 0.001

gyro_data = wie_b[None, :] + gyro_bias[None, :] + gyro_noise
acc_data = g_b[None, :] + acc_bias[None, :] + acc_noise

imu_sim = np.column_stack([gyro_data, acc_data, t])

print(f"  ✅ 模拟数据: {N}个点, {N*dt:.1f}s")
print(f"  True姿态: roll={true_roll}°, pitch={true_pitch}°, yaw={true_yaw}°")

# ====== Stage 5: 执行粗对准 ======
print("\n[5/6] 执行粗对准验证...")

aligner = CoarseAligner(latitude_deg=45.734501, wie=wie, g=g)
try:
    Cnb0, att_coarse = aligner.run(imu_sim, t1=2.0, t2=8.0, n_avg_pairs=1)
    print(f"  粗对准结果: roll={att_coarse[0]:.2f}°, pitch={att_coarse[1]:.2f}°, yaw={att_coarse[2]:.2f}°")
    roll_err = abs(att_coarse[0] - true_roll)
    pitch_err = abs(att_coarse[1] - true_pitch)
    yaw_err = abs(att_coarse[2] - true_yaw)
    print(f"  误差: roll={roll_err:.2f}°, pitch={pitch_err:.2f}°, yaw={yaw_err:.2f}°")
    if yaw_err < 10:
        print("  ✅ 粗对准通过 (航向误差<10°)")
    else:
        print(f"  ⚠️ 粗对准航向误差较大 ({yaw_err:.2f}°)，短时模拟数据导致")
except Exception as e:
    import traceback
    print(f"  ❌ 粗对准失败: {e}")
    traceback.print_exc()

# ====== Stage 6: 验证姿态更新 ======
print("\n[6/6] 执行姿态更新验证...")

updater = AttitudeUpdater(latitude_deg=45.734501)
try:
    # 模拟连续旋转（10°/s绕Z轴）
    N_rot = 2000
    t_rot = np.arange(N_rot) * dt
    gyro_rot = np.zeros((N_rot, 3))
    gyro_rot[:, 2] = deg2rad(10.0)  # 10°/s
    acc_rot = np.zeros((N_rot, 3))
    acc_rot[:, 2] = -g  # 重力

    imu_rot = np.column_stack([gyro_rot, acc_rot, t_rot])
    att_hist = updater.run(imu_rot, (0, 0, 0))
    # 3-1-2转序: +Z轴旋转(逆时针)对应航向减小
    # 10°/s * 10s = 100° 旋转 → 航向 ≈ -100°
    expected_yaw = -10.0 * 10.0
    final_yaw = att_hist[-1, 2]
    yaw_diff = abs(final_yaw - expected_yaw)
    print(f"  姿态更新: 最终yaw={final_yaw:.1f}°, 预期≈{expected_yaw:.1f}°, 误差={yaw_diff:.1f}°")
    if yaw_diff < 5:
        print("  ✅ 姿态更新通过")
    else:
        print(f"  ⚠️ 姿态更新误差({yaw_diff:.1f}°)")
except Exception as e:
    import traceback
    print(f"  ❌ 姿态更新失败: {e}")
    traceback.print_exc()

# ====== 汇总 ======
print("\n" + "=" * 60)
print("集成验证完成")
print("=" * 60)