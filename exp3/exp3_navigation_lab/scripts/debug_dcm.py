"""
调试脚本：对比两种 C_n^b 计算方式
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.data_io import DataLoader
from src.alignment import CoarseAligner
from src.utils.dcm import dcm_from_quat, dcm_orthogonalize
from src.utils.euler_angles import dcm_to_euler312, rad2deg
from src.utils.earth_model import C_n2i, WGS84_OMEGA

# ===== 加载数据 =====
DATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', '初始对准', 'gtimu_0_0_0.log')
N_ROWS = 9500
LAT = 45.734501

# 读前9500行
lines = []
with open(DATA_FILE) as f:
    for i, line in enumerate(f):
        if i >= N_ROWS: break
        lines.append(line)

tmp = DATA_FILE + '.debug_tmp'
with open(tmp, 'w') as f:
    f.writelines(lines)

imu_raw = DataLoader.load_imu(tmp)
os.remove(tmp)

gyro = imu_raw['gyro']
acc = imu_raw['acc']
time_raw = imu_raw['time']

print(f"数据点数: {len(gyro)}")
print(f"时间跨度: {time_raw[0]:.2f} ~ {time_raw[-1]:.2f} s")

# ===== 运行粗对准到获取中间结果 =====
# 手动执行粗对准的核心步骤，打印中间矩阵

from src.utils.quaternion import quat_update
from src.utils.earth_model import gravity_i

time = time_raw - time_raw[0]
N = len(time)

# 计算dt
dt = np.median(np.diff(time)[np.diff(time) > 1e-12])
print(f"dt = {dt:.6f} s")

# 积分
q_b2i = np.array([1.0, 0.0, 0.0, 0.0])
v_i = np.zeros(3)
r_i = np.zeros(3)
lat_rad = np.deg2rad(LAT)

v_history = np.zeros((N, 3))
r_history = np.zeros((N, 3))

for k in range(N):
    q_b2i = quat_update(q_b2i, gyro[k], dt)
    C_b2i = dcm_from_quat(q_b2i)
    f_i = C_b2i @ acc[k]
    v_i = v_i + f_i * dt
    t_k = time[k]
    g_i_t = gravity_i(t_k, lat_rad, WGS84_OMEGA)
    r_i = r_i - g_i_t * dt
    v_history[k] = v_i.copy()
    r_history[k] = r_i.copy()

# 双矢量定姿
t1, t2 = 5.0, 40.0
idx1 = np.argmin(np.abs(time - t1))
idx2 = np.argmin(np.abs(time - t2))

V1 = v_history[idx1]; V2 = v_history[idx2]
R1 = r_history[idx1]; R2 = r_history[idx2]

Mv = np.column_stack([V1, V2, np.cross(V1, V2)])
Mr = np.column_stack([R1, R2, np.cross(R1, R2)])

C_b2i0 = Mv @ np.linalg.inv(Mr)
C_b2i0 = dcm_orthogonalize(C_b2i0)

# C_n^i at t=0
t0 = time[0]  # 0.0
C_n2i0 = C_n2i(t0, lat_rad, WGS84_OMEGA)

print(f"\n===== 中间矩阵 =====")
print(f"\nC_b^i(0) =\n{C_b2i0}")
print(f"det(C_b^i(0)) = {np.linalg.det(C_b2i0):.6f}")

print(f"\nC_n^i(0) =\n{C_n2i0}")
print(f"det(C_n^i(0)) = {np.linalg.det(C_n2i0):.6f}")

# ===== 两种计算方式 =====
print(f"\n===== 方式A: Cnb0 = C_n2i0.T @ C_b2i0 (原代码) =====")
Cnb0_A = C_n2i0.T @ C_b2i0
Cnb0_A = dcm_orthogonalize(Cnb0_A)
print(f"Cnb0_A =\n{Cnb0_A}")
roll_A, pitch_A, yaw_A = dcm_to_euler312(Cnb0_A)
print(f"Roll={rad2deg(roll_A):.4f}  Pitch={rad2deg(pitch_A):.4f}  Heading={rad2deg(yaw_A):.4f}")

print(f"\n===== 方式B: Cnb0 = C_b2i0.T @ C_n2i0 (C_n^b = C_i^b @ C_n^i) =====")
Cnb0_B = C_b2i0.T @ C_n2i0
Cnb0_B = dcm_orthogonalize(Cnb0_B)
print(f"Cnb0_B =\n{Cnb0_B}")
roll_B, pitch_B, yaw_B = dcm_to_euler312(Cnb0_B)
print(f"Roll={rad2deg(roll_B):.4f}  Pitch={rad2deg(pitch_B):.4f}  Heading={rad2deg(yaw_B):.4f}")

# ===== 方式C: 对方式A的结果取转置再提取欧拉角 =====
print(f"\n===== 方式C: dcm_to_euler312(Cnb0_A.T) =====")
roll_C, pitch_C, yaw_C = dcm_to_euler312(Cnb0_A.T)
print(f"Roll={rad2deg(roll_C):.4f}  Pitch={rad2deg(pitch_C):.4f}  Heading={rad2deg(yaw_C):.4f}")

# 参考值
print(f"\n===== 参考值 =====")
print(f"Roll=0.001  Pitch=-0.018  Heading=0.770")
