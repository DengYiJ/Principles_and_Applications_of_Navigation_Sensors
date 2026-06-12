"""
姿态更新 — 3组动态数据
=======================
利用 0_0_0 精对准的陀螺零偏，对3组姿态更新数据做纯惯性递推
与 $GPFPD 惯导参考输出对比
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from src.data_io import DataLoader
from src.utils.dcm import dcm_from_quat, dcm_to_quat, dcm_orthogonalize
from src.utils.quaternion import quat_update, quat_normalize
from src.utils.euler_angles import euler312_to_dcm, euler321_to_dcm, dcm_to_euler312, rad2deg, deg2rad
from src.utils.earth_model import earth_rate_n

LAT = 45.734501
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

# 陀螺零偏 — 来自 0_0_0 精对准估计 (rad/s)
GYRO_BIAS = np.array([-3.5e-5, -1.0e-5, 0.0])

UPDATES = {
    '0_20_0':      {'gtimu': 'gtimu_0_20_0.log',      'gpfpd': 'gpfpd_0_20_0.log'},
    '0_0_90':      {'gtimu': 'gtimu_0_0_90.log',      'gpfpd': 'gpfpd_0_0_90.log'},
    '-30_-20_180': {'gtimu': 'gtimu_-30_-20_180.log', 'gpfpd': 'gpfpd_-30_-20_180.log'},
}


def run_attitude_update_dcm(imu_data, C_nb_init, gyro_bias):
    """
    纯惯性姿态递推
    返回: att_history [N×3] (Euler deg), q_history [N×4] (quaternions)
    """
    gyro = imu_data[:, 0:3]
    time = imu_data[:, 6]
    N = len(time)
    dt = float(np.median(np.diff(time)))
    lat_rad = np.deg2rad(LAT)

    C_nb = C_nb_init.copy()
    q_nb = dcm_to_quat(C_nb)
    bias = np.array(gyro_bias)
    att_history = np.zeros((N, 3))
    q_history = np.zeros((N, 4))

    for k in range(N):
        q_history[k] = q_nb.copy()
        wie_n = earth_rate_n(lat_rad)
        omega_nb_b = gyro[k] - bias - C_nb.T @ wie_n
        q_nb = quat_update(q_nb, omega_nb_b, dt)
        q_nb = quat_normalize(q_nb)
        C_nb = dcm_from_quat(q_nb)
        r, p, y = dcm_to_euler312(C_nb)
        rd, pd, yd = rad2deg(r), rad2deg(p), rad2deg(y)
        if abs(rd) > 90: rd -= np.sign(rd)*180; yd = -yd
        att_history[k] = [rd, pd, yd % 360]

    return att_history, q_history


def euler_to_quat(att_deg):
    """欧拉角(deg) → 四元数 [qw,qx,qy,qz]"""
    C = euler312_to_dcm(np.deg2rad(att_deg[0]), np.deg2rad(att_deg[1]), np.deg2rad(att_deg[2]))
    return dcm_to_quat(C)

def quat_angular_error(q1, q2):
    """两个四元数之间的角度误差 (deg)"""
    # q1, q2: [qw, qx, qy, qz]
    # 角度 = 2*arccos(|q1·q2|), 使用内积
    dot = np.abs(np.sum(q1 * q2, axis=1))
    dot = np.clip(dot, -1.0, 1.0)
    return 2 * np.arccos(dot) * 180.0 / np.pi

def compute_error_metrics(est_euler, ref_euler):
    """计算误差: 四元数空间角度误差 + 欧拉角通道误差(参考导向解包裹)"""
    N = len(est_euler)

    # 四元数角度误差 (避免万向锁)
    q_err = np.zeros(N)
    for k in range(N):
        q_est = euler_to_quat(est_euler[k])
        q_ref = euler_to_quat(ref_euler[k])
        # 处理四元数符号歧义
        if np.sum(q_est * q_ref) < 0:
            q_est = -q_est
        dot = np.clip(np.sum(q_est * q_ref), -1.0, 1.0)
        q_err[k] = 2 * np.arccos(dot) * 180.0 / np.pi

    # 欧拉角通道误差 (以参考值为基准解包裹)
    diff_roll = est_euler[:, 0] - ref_euler[:, 0]
    diff_roll = np.arctan2(np.sin(np.deg2rad(diff_roll)), np.cos(np.deg2rad(diff_roll)))
    diff_roll = np.rad2deg(diff_roll)

    diff_pitch = est_euler[:, 1] - ref_euler[:, 1]

    diff_head = est_euler[:, 2] - ref_euler[:, 2]
    diff_head = np.arctan2(np.sin(np.deg2rad(diff_head)), np.cos(np.deg2rad(diff_head)))
    diff_head = np.rad2deg(diff_head)

    diff = np.column_stack([diff_roll, diff_pitch, diff_head])

    rmse = np.sqrt(np.mean(diff**2, axis=0))
    mae = np.mean(np.abs(diff), axis=0)
    maxe = np.max(np.abs(diff), axis=0)

    return {
        'RMSE': rmse, 'MAE': mae, 'MaxError': maxe, 'diff': diff,
        'RMSE_quat': np.sqrt(np.mean(q_err**2)),
        'MAE_quat': np.mean(np.abs(q_err)),
    }


def process_update(name, cfg):
    print(f"\n{'='*60}")
    print(f" {name}")
    print(f"{'='*60}")

    imu_path = os.path.join(DATA_DIR, '姿态更新', cfg['gtimu'])
    ref_path = os.path.join(DATA_DIR, '姿态更新', cfg['gpfpd'])

    # Load
    imu_raw = DataLoader.load_imu(imu_path, scale_acc=9.7803267714)
    ref_raw = DataLoader.load_gpfpd(ref_path)

    imu_data = np.column_stack([imu_raw['gyro'], imu_raw['acc'], imu_raw['time']])
    t_imu = imu_data[:, 6] - imu_data[0, 6]

    # 参考姿态 — GPFPD 用 3-2-1 转序 (Heading→Pitch→Roll)
    # 转为 DCM 再用 3-1-2 提取, 与我们的姿态更新统一
    Nref = len(ref_raw['heading'])
    ref_att_312 = np.zeros((Nref, 3))
    for k in range(Nref):
        C_ref = euler321_to_dcm(deg2rad(ref_raw['roll'][k]),
                                deg2rad(ref_raw['pitch'][k]),
                                deg2rad(ref_raw['heading'][k]))
        r312, p312, y312 = dcm_to_euler312(C_ref)
        rd, yd = rad2deg(r312), rad2deg(y312)
        if abs(rd) > 90: rd -= np.sign(rd)*180; yd = -yd
        ref_att_312[k] = [rd, rad2deg(p312), yd % 360]

    # 初始姿态: 直接从 GPFPD DCM 构建 (绕过 Euler 万向锁)
    C_ref_init = euler321_to_dcm(deg2rad(ref_raw['roll'][0]),
                                 deg2rad(ref_raw['pitch'][0]),
                                 deg2rad(ref_raw['heading'][0]))
    q_init = dcm_to_quat(C_ref_init)
    r312, p312, y312 = dcm_to_euler312(C_ref_init)
    rd, yd = rad2deg(r312), rad2deg(y312)
    if abs(rd) > 90: rd -= np.sign(rd)*180; yd = -yd
    att_init = (rd, rad2deg(p312), yd % 360)
    print(f"  Init attitude (3-1-2): R={att_init[0]:.3f} P={att_init[1]:.3f} H={att_init[2]:.3f}")

    # 姿态递推: 从四元数初始化 (避免Euler→DCM→Quat精度损失)
    C_nb = C_ref_init.copy()
    q_nb = q_init.copy()
    print(f"  Gyro bias: {np.rad2deg(GYRO_BIAS)} deg/s")
    print(f"  Data: {len(imu_data)} pts, {t_imu[-1]:.1f}s")

    # 姿态更新 (从 GPFPD DCM 直接初始化)
    att_est, q_est = run_attitude_update_dcm(imu_data, C_ref_init, GYRO_BIAS)

    # 对齐参考 (插值到IMU时间, 使用3-1-2转换后的值)
    ref_aligned = np.zeros((len(t_imu), 3))
    ref_t = ref_raw['time'] - ref_raw['time'][0]
    for i in range(3):
        ref_aligned[:, i] = np.interp(t_imu, ref_t, ref_att_312[:, i])

    # 误差 (欧拉角 + 四元数直接比较)
    metrics = compute_error_metrics(att_est, ref_aligned)

    # 四元数直接误差 (绕过 Euler, 避免万向锁影响)
    q_ref_raw = np.zeros((len(ref_raw["heading"]), 4))
    for k in range(len(ref_raw["heading"])):
        C_ref_k = euler321_to_dcm(deg2rad(ref_raw["roll"][k]),
                                  deg2rad(ref_raw["pitch"][k]),
                                  deg2rad(ref_raw["heading"][k]))
        q_ref_raw[k] = dcm_to_quat(C_ref_k)
    q_ref_aligned = np.zeros((len(t_imu), 4))
    for j in range(4):
        q_ref_aligned[:, j] = np.interp(t_imu, ref_t, q_ref_raw[:, j])
    # 处理四元数符号歧义并计算误差
    for k in range(len(t_imu)):
        if np.sum(q_est[k] * q_ref_aligned[k]) < 0:
            q_est[k] = -q_est[k]
    dot = np.clip(np.sum(q_est * q_ref_aligned, axis=1), -1.0, 1.0)
    q_err_direct = 2 * np.arccos(dot) * 180.0 / np.pi
    metrics['RMSE_quat_direct'] = np.sqrt(np.mean(q_err_direct**2))
    metrics['MAE_quat_direct'] = np.mean(np.abs(q_err_direct))
    metrics['q_err_direct'] = q_err_direct

    print(f"  RMSE (euler):     R={metrics['RMSE'][0]:.4f} P={metrics['RMSE'][1]:.4f} H={metrics['RMSE'][2]:.4f}")
    print(f"  RMSE (quat direct): {metrics['RMSE_quat_direct']:.4f} deg  (bypasses Euler singularities)")

    return att_est, ref_aligned, t_imu, metrics


def plot_all(all_results):
    """生成3组姿态更新对比图"""
    for name, (att_est, ref_aligned, t, metrics) in all_results.items():
        save_dir = os.path.join(RESULTS_DIR, f'attitude_update_{name}')
        os.makedirs(save_dir, exist_ok=True)

        diff = metrics['diff']

        # 图1: 三轴姿态对比
        fig, axes = plt.subplots(3, 1, figsize=(14, 9))
        colors = ['#E63946', '#2A9D8F', '#264653']
        for i, (ax, lb) in enumerate(zip(axes, ['Roll', 'Pitch', 'Heading'])):
            ax.plot(t, att_est[:, i], color=colors[i], linewidth=0.8, label='INS Update')
            ax.plot(t, ref_aligned[:, i], '--', color='#E76F51', linewidth=0.8, alpha=0.8, label='GPFPD Ref')
            ax.set_ylabel(f'{lb} (deg)', fontsize=11)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3, linestyle='--')
        axes[-1].set_xlabel('Time (s)', fontsize=12)
        fig.suptitle(f'Attitude Update vs GPFPD Reference — {name}', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '01_attitude_comparison.png'), dpi=200, bbox_inches='tight')
        plt.close()

        # 图2: 三轴误差 + 四元数总误差
        fig, axes = plt.subplots(4, 1, figsize=(14, 11))
        for i, (ax, lb) in enumerate(zip(axes[:3], ['Roll Error', 'Pitch Error', 'Heading Error'])):
            ax.plot(t, diff[:, i], color=colors[i], linewidth=0.5, alpha=0.85)
            ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
            ax.set_ylabel(f'{lb} (deg)', fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--')
        # 四元数直接误差 (从 q_est vs q_ref 直接算, 绕过Euler)
        axes[3].plot(t, metrics['q_err_direct'], color='#6A0572', linewidth=0.5, alpha=0.85)
        axes[3].set_ylabel('Total Error (deg)', fontsize=11)
        q_rmse = metrics["RMSE_quat_direct"]
        axes[3].set_xlabel('Time (s)', fontsize=12)
        axes[3].grid(True, alpha=0.3, linestyle='--')
        axes[3].text(0.98, 0.92, f"Quat RMSE={q_rmse:.4f} deg", transform=axes[3].transAxes,
                    fontsize=10, ha="right", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
        fig.suptitle(f'Attitude Error — {name}', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '02_error.png'), dpi=200, bbox_inches='tight')
        plt.close()

        # CSV
        with open(os.path.join(save_dir, 'error_metrics.csv'), 'w') as f:
            f.write('Metric,Roll(deg),Pitch(deg),Heading(deg)\n')
            for mname in ['RMSE', 'MAE', 'MaxError']:
                vals = metrics[mname]
                f.write(f'{mname},{vals[0]:.6f},{vals[1]:.6f},{vals[2]:.6f}\n')

        print(f"  Plots saved: {save_dir}")


def main():
    print("=" * 70)
    print("Attitude Update — 3 dynamic datasets")
    print(f"Gyro bias from 0_0_0 fine alignment: {np.rad2deg(GYRO_BIAS)} deg/s")
    print("=" * 70)

    all_results = {}
    for name, cfg in UPDATES.items():
        all_results[name] = process_update(name, cfg)

    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':<16} {'Roll RMSE':<12} {'Pitch RMSE':<12} {'Heading RMSE':<12} {'Quat RMSE':<12}")
    print("-" * 64)
    for name, (_, _, _, m) in all_results.items():
        print(f"{name:<16} {m['RMSE'][0]:<12.4f} {m['RMSE'][1]:<12.4f} {m['RMSE'][2]:<12.4f} {m['RMSE_quat_direct']:<12.4f}")

    plot_all(all_results)
    print(f"\nDone. Results saved to: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
