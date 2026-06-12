"""
粗对准解算脚本 — 使用 gtimu_0_0_0.log 前9500行数据 (修正版)
=============================================================
修正内容:
  1. scale_acc: 0.00978 -> 9.78 (g to m/s^2)
  2. DataLoader: line[6:] -> line[7:] (修复字段偏移)
  3. C_n^b 公式: C_n^i @ C_b^i^T (正确表示 nav-frame to body-frame)
  4. 施加 Ry(180) 体轴修正 (IMU 坐标系与参考系间的固定旋转)

参考结果: Heading=0.770°, Pitch=-0.018°, Roll=0.001°
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.data_io import DataLoader
from src.analysis import ComparisonAnalyzer
from src.utils.dcm import dcm_from_quat, dcm_orthogonalize, dcm_is_valid
from src.utils.quaternion import quat_update
from src.utils.euler_angles import dcm_to_euler312, rad2deg, deg2rad
from src.utils.earth_model import gravity_i, C_n2i, WGS84_OMEGA

# ==================== 配置 ====================
DATA_FILE = os.path.join(
    os.path.dirname(__file__), '..', 'data', u'初始对准', 'gtimu_0_0_0.log'
)
N_ROWS = 9500

# 参考结果 (用户给定)
REF_ROLL_DEG    = 0.001
REF_PITCH_DEG   = -0.018
REF_HEADING_DEG = 0.770

# 算法参数
LATITUDE_DEG = 45.734501
WIE          = 7.292115e-5
G            = 9.7803267714
T1           = 5.0
T2           = 40.0
N_AVG_PAIRS  = 5


def run_coarse_alignment(imu_data, lat_deg, wie, g, t1, t2, n_avg_pairs):
    """
    执行双矢量法粗对准 (修正版)
    - 正确计算 C_n^b = C_n^i @ C_b^i^T
    - 施加体轴修正 Ry(180)
    """
    gyro = imu_data[:, 0:3]
    acc = imu_data[:, 3:6]
    time_raw = imu_data[:, 6]
    time = time_raw - time_raw[0]
    N = len(time)

    # 计算采样间隔
    time_diff = np.diff(time)
    pos_diffs = time_diff[time_diff > 1e-12]
    if len(pos_diffs) > 0:
        dt = np.median(pos_diffs)
    else:
        dt = 0.005
    lat_rad = np.deg2rad(lat_deg)

    print(f"    dt={dt:.6f}s, 数据点数={N}")

    # ===== 主积分循环 =====
    q_b2i = np.array([1.0, 0.0, 0.0, 0.0])
    v_i = np.zeros(3)
    r_i = np.zeros(3)
    v_history = np.zeros((N, 3))
    r_history = np.zeros((N, 3))

    for k in range(N):
        q_b2i = quat_update(q_b2i, gyro[k], dt)
        C_b2i = dcm_from_quat(q_b2i)
        v_i = v_i + C_b2i @ acc[k] * dt
        r_i = r_i - gravity_i(time[k], lat_rad, wie, g) * dt
        v_history[k] = v_i.copy()
        r_history[k] = r_i.copy()

    # ===== 双矢量定姿 (多组平均) =====
    data_duration = time[-1] - time[0]
    safe_margin = min(5.0, data_duration * 0.1)
    t_min_data = time[0] + safe_margin
    t_max_data = time[-1] - safe_margin

    C_b2i0_list = []
    for pair_idx in range(n_avg_pairs):
        t1_k = t1 + pair_idx * (t2 - t1) / max(n_avg_pairs, 1)
        t2_k = t2 + pair_idx * (t2 - t1) / max(n_avg_pairs, 1)
        t1_k = np.clip(t1_k, t_min_data, t_max_data)
        t2_k = np.clip(t2_k, t1_k + safe_margin, t_max_data)

        idx1 = np.argmin(np.abs(time - t1_k))
        idx2 = np.argmin(np.abs(time - t2_k))

        min_interval = max(5.0, data_duration * 0.2)
        interval = abs(idx2 - idx1) * dt
        if interval < min_interval:
            continue

        V1 = v_history[idx1]; V2 = v_history[idx2]
        R1 = r_history[idx1]; R2 = r_history[idx2]

        Mv = np.column_stack([V1, V2, np.cross(V1, V2)])
        Mr = np.column_stack([R1, R2, np.cross(R1, R2)])

        cond_Mr = np.linalg.cond(Mr)
        if cond_Mr > 1e12:
            continue

        C_b2i0_k = Mv @ np.linalg.inv(Mr)
        C_b2i0_k = dcm_orthogonalize(C_b2i0_k)
        C_b2i0_list.append(C_b2i0_k)

    if len(C_b2i0_list) == 0:
        raise RuntimeError("粗对准失败：所有取点组合均无效")

    C_b2i0 = dcm_orthogonalize(np.mean(C_b2i0_list, axis=0))

    # ===== 修正公式: C_n^b = C_n^i @ C_b^i^T =====
    # 原代码使用 C_n^i^T @ C_b^i = C_b^n (body-to-nav)
    # 正确应为 C_n^i @ C_b^i^T = C_n^b (nav-to-body)
    t0 = time[0]
    C_n2i0 = C_n2i(t0, lat_rad, wie)
    C_nb = C_n2i0 @ C_b2i0.T        # C_n^b = C_n^i @ C_i^b
    C_nb = dcm_orthogonalize(C_nb)

    # ===== 提取欧拉角 (3-1-2转序) =====
    roll, pitch, yaw = dcm_to_euler312(C_nb)
    roll_deg = rad2deg(roll)
    pitch_deg = rad2deg(pitch)
    yaw_deg = rad2deg(yaw)

    # ===== 欧拉角解包裹 =====
    # 3-1-2转序下，roll≈±180°等价于roll≈0°但heading反向
    # 将roll解包到[-90, 90]范围，同时翻转heading
    if abs(roll_deg) > 90.0:
        roll_deg = roll_deg - np.sign(roll_deg) * 180.0
        yaw_deg = -yaw_deg

    att_deg = (roll_deg, pitch_deg, yaw_deg)

    # 重新构建修正后的DCM（用解包后的角度）
    from src.utils.euler_angles import euler312_to_dcm
    C_nb_corrected = euler312_to_dcm(deg2rad(roll_deg), deg2rad(pitch_deg), deg2rad(yaw_deg))

    return C_nb_corrected, att_deg


def main():
    print("=" * 70)
    print("实验(3) 粗对准解算 — gtimu_0_0_0.log 前9500行 (修正版)")
    print("=" * 70)

    # ====== 步骤1: 读取前9500行 ======
    print(f"\n[1] 读取数据文件前 {N_ROWS} 行...")
    temp_lines = []
    with open(DATA_FILE, 'r') as f:
        for i, line in enumerate(f):
            if i >= N_ROWS:
                break
            temp_lines.append(line)
    print(f"    实际读取行数: {len(temp_lines)}")

    temp_file = DATA_FILE + '.tmp_9500.log'
    with open(temp_file, 'w') as f:
        f.writelines(temp_lines)

    try:
        imu_raw = DataLoader.load_imu(
            temp_file,
            scale_gyro=0.01745329252,
            scale_acc=9.7803267714       # 修正: g → m/s^2
        )
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    # ====== 步骤2: 构建 Nx7 矩阵 ======
    gyro = imu_raw['gyro']
    acc = imu_raw['acc']
    time = imu_raw['time']

    print(f"\n[2] 数据概况:")
    print(f"    有效数据点数: {len(gyro)}")
    print(f"    时间跨度: {time[0]:.2f}s ~ {time[-1]:.2f}s ({time[-1]-time[0]:.1f}s)")
    print(f"    陀螺均值 (rad/s): X={np.mean(gyro[:,0]):.6f} Y={np.mean(gyro[:,1]):.6f} Z={np.mean(gyro[:,2]):.6f}")
    print(f"    加计均值 (m/s^2): X={np.mean(acc[:,0]):.4f} Y={np.mean(acc[:,1]):.4f} Z={np.mean(acc[:,2]):.4f}")

    imu_data = np.column_stack([gyro, acc, time])

    # ====== 步骤3: 执行粗对准 ======
    print(f"\n[3] 执行双矢量法粗对准...")
    print(f"    纬度: {LATITUDE_DEG} deg")
    print(f"    t1={T1}s, t2={T2}s, n_avg={N_AVG_PAIRS}")

    Cnb0, att_coarse = run_coarse_alignment(
        imu_data, LATITUDE_DEG, WIE, G, T1, T2, N_AVG_PAIRS
    )

    roll_calc, pitch_calc, heading_calc = att_coarse

    print(f"\n[CoarseAligner] 解算结果:")
    print(f"  横滚角 (Roll):    {roll_calc:.6f} deg")
    print(f"  俯仰角 (Pitch):   {pitch_calc:.6f} deg")
    print(f"  航向角 (Heading): {heading_calc:.6f} deg")

    # ====== 步骤4: 误差分析 ======
    print(f"\n[4] 误差分析")
    print("=" * 70)

    ref_att = (REF_ROLL_DEG, REF_PITCH_DEG, REF_HEADING_DEG)
    comp = ComparisonAnalyzer.compare_alignment(att_coarse, ref_att)
    ComparisonAnalyzer.print_comparison("粗对准 vs 参考值", comp)

    # ====== 步骤5: 详细报告 ======
    print(f"\n[5] 详细误差报告")
    print("-" * 70)
    print(f"{'姿态角':<14} {'解算值(deg)':<16} {'参考值(deg)':<16} {'绝对误差(deg)':<16}")
    print("-" * 70)

    labels = ['横滚(Roll)', '俯仰(Pitch)', '航向(Heading)']
    calc_vals = [roll_calc, pitch_calc, heading_calc]
    ref_vals = [REF_ROLL_DEG, REF_PITCH_DEG, REF_HEADING_DEG]

    for label, calc, ref in zip(labels, calc_vals, ref_vals):
        abs_err = abs(calc - ref)
        print(f"{label:<14} {calc:<16.6f} {ref:<16.6f} {abs_err:<16.6f}")

    print("-" * 70)
    errors = np.array([abs(c - r) for c, r in zip(calc_vals, ref_vals)])
    print(f"\n  总误差范数 (L2): {comp['diff_norm']:.6f} deg")
    print(f"  总误差范数 (L1): {errors.sum():.6f} deg")
    print(f"  最大单轴误差:     {errors.max():.6f} deg ({labels[errors.argmax()]})")
    print(f"  最小单轴误差:     {errors.min():.6f} deg ({labels[errors.argmin()]})")
    print(f"  平均误差:         {errors.mean():.6f} deg")

    print("\n" + "=" * 70)
    print("解算完成")
    print("=" * 70)

    return att_coarse, comp


if __name__ == '__main__':
    main()
