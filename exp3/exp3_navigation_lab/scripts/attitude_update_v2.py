"""
姿态更新 v2 — 统一用 3-2-1 转序 (GPFPD原生格式)
===================================================
IMU解算的DCM → 提取3-2-1欧拉角 → 与GPFPD直接对比
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from src.data_io import DataLoader
from src.utils.dcm import dcm_from_quat, dcm_to_quat
from src.utils.quaternion import quat_update, quat_normalize
from src.utils.euler_angles import euler321_to_dcm, dcm_to_euler321, rad2deg, deg2rad
from src.utils.earth_model import earth_rate_n

LAT = 45.734501
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
GYRO_BIAS = np.array([-3.5e-5, -1.0e-5, 0.0])

UPDATES = {
    '0_20_0':      {'gtimu': 'gtimu_0_20_0.log',      'gpfpd': 'gpfpd_0_20_0.log'},
    '0_0_90':      {'gtimu': 'gtimu_0_0_90.log',      'gpfpd': 'gpfpd_0_0_90.log'},
    '-30_-20_180': {'gtimu': 'gtimu_-30_-20_180.log', 'gpfpd': 'gpfpd_-30_-20_180.log'},
}


def run_ins_update(imu_data, C_nb_init, gyro_bias):
    """纯惯性递推 → 返回每步的 C_n^b (即 GPFPD 对应的 DCM)"""
    gyro = imu_data[:, 0:3]
    time = imu_data[:, 6]
    N = len(time)
    dt = float(np.median(np.diff(time)))
    lat_rad = np.deg2rad(LAT)

    C_nb = C_nb_init.copy()
    q_nb = dcm_to_quat(C_nb)
    bias = np.array(gyro_bias)

    # 输出3-2-1欧拉角 (与 GPFPD 同格式: Heading, Pitch, Roll)
    att_321 = np.zeros((N, 3))

    for k in range(N):
        wie_n = earth_rate_n(lat_rad)
        omega_nb_b = gyro[k] - bias - C_nb.T @ wie_n
        q_nb = quat_update(q_nb, omega_nb_b, dt)
        q_nb = quat_normalize(q_nb)
        C_nb = dcm_from_quat(q_nb)
        # 提取3-2-1欧拉角
        roll, pitch, yaw = dcm_to_euler321(C_nb)
        att_321[k] = [rad2deg(yaw) % 360, rad2deg(pitch), rad2deg(roll)]

    return att_321


def process(name, cfg):
    print(f"\n{'='*60}\n {name}\n{'='*60}")

    imu_path = os.path.join(DATA_DIR, '姿态更新', cfg['gtimu'])
    ref_path = os.path.join(DATA_DIR, '姿态更新', cfg['gpfpd'])

    imu_raw = DataLoader.load_imu(imu_path, scale_acc=9.7803267714)
    ref_raw = DataLoader.load_gpfpd(ref_path)
    imu_data = np.column_stack([imu_raw['gyro'], imu_raw['acc'], imu_raw['time']])
    t_imu = imu_data[:, 6] - imu_data[0, 6]

    # GPFPD 参考 (原生3-2-1: Heading, Pitch, Roll)
    ref_att = np.column_stack([ref_raw['heading'], ref_raw['pitch'], ref_raw['roll']])
    print(f"  GPFPD ref range: H=[{ref_att[:,0].min():.1f}~{ref_att[:,0].max():.1f}], "
          f"P=[{ref_att[:,1].min():.1f}~{ref_att[:,1].max():.1f}], "
          f"R=[{ref_att[:,2].min():.1f}~{ref_att[:,2].max():.1f}]")

    # 初始姿态: 从 GPFPD 首帧 3-2-1 角构建 DCM
    C_init = euler321_to_dcm(deg2rad(ref_att[0, 2]),   # Roll
                             deg2rad(ref_att[0, 1]),   # Pitch
                             deg2rad(ref_att[0, 0]))   # Heading

    # 姿态更新
    att_ins = run_ins_update(imu_data, C_init, GYRO_BIAS)

    # 对齐参考到 IMU 时间
    ref_t = ref_raw['time'] - ref_raw['time'][0]
    ref_aligned = np.zeros((len(t_imu), 3))
    for i in range(3):
        ref_aligned[:, i] = np.interp(t_imu, ref_t, ref_att[:, i])

    # 误差 (3-2-1欧拉角通道, 航向做wrap)
    diff = att_ins - ref_aligned
    diff[:, 0] = np.arctan2(np.sin(np.deg2rad(diff[:, 0])),
                            np.cos(np.deg2rad(diff[:, 0])))  # heading wrap
    diff[:, 0] = np.rad2deg(diff[:, 0])
    diff[:, 2] = np.arctan2(np.sin(np.deg2rad(diff[:, 2])),
                            np.cos(np.deg2rad(diff[:, 2])))  # roll wrap
    diff[:, 2] = np.rad2deg(diff[:, 2])

    rmse = np.sqrt(np.mean(diff**2, axis=0))
    print(f"  RMSE: Heading={rmse[0]:.4f}  Pitch={rmse[1]:.4f}  Roll={rmse[2]:.4f}")

    return att_ins, ref_aligned, t_imu, diff, rmse


def plot_results(all_results):
    for name in UPDATES:
        att_ins, ref_aligned, t, diff, rmse = all_results[name]
        save_dir = os.path.join(RESULTS_DIR, f'att_update_{name}')
        os.makedirs(save_dir, exist_ok=True)

        # 图1: INS vs GPFPD 对比 (统一3-2-1)
        fig, axes = plt.subplots(3, 1, figsize=(14, 9))
        labels = ['Heading (deg)', 'Pitch (deg)', 'Roll (deg)']
        colors = ['#264653', '#2A9D8F', '#E63946']
        for i, (ax, lb, c) in enumerate(zip(axes, labels, colors)):
            ax.plot(t, att_ins[:, i], color=c, linewidth=0.8, label='INS Update')
            ax.plot(t, ref_aligned[:, i], '--', color='#E76F51', linewidth=0.8, alpha=0.8, label='GPFPD Ref')
            ax.set_ylabel(lb, fontsize=11)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3, linestyle='--')
        axes[-1].set_xlabel('Time (s)', fontsize=12)
        fig.suptitle(f'Attitude Update vs GPFPD (3-2-1) — {name}', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '01_comparison.png'), dpi=200, bbox_inches='tight')
        plt.close()

        # 图2: 误差
        fig, axes = plt.subplots(3, 1, figsize=(14, 9))
        for i, (ax, lb, c) in enumerate(zip(axes, labels, colors)):
            ax.plot(t, diff[:, i], color=c, linewidth=0.5, alpha=0.85)
            ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
            ax.set_ylabel(f'{lb.split()[0]} Error', fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.text(0.98, 0.92, f'RMSE={rmse[i]:.4f}°', transform=ax.transAxes, fontsize=10,
                    ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        axes[-1].set_xlabel('Time (s)', fontsize=12)
        fig.suptitle(f'Attitude Error (3-2-1) — {name}', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '02_error.png'), dpi=200, bbox_inches='tight')
        plt.close()

        # CSV
        with open(os.path.join(save_dir, 'error_metrics.csv'), 'w') as f:
            f.write('Metric,Heading(deg),Pitch(deg),Roll(deg)\n')
            f.write(f'RMSE,{rmse[0]:.6f},{rmse[1]:.6f},{rmse[2]:.6f}\n')
        print(f"  Saved: {save_dir}")


def main():
    print("=" * 70)
    print("Attitude Update — 3-2-1 convention (GPFPD native)")
    print(f"Gyro bias: {np.rad2deg(GYRO_BIAS)} deg/s")
    print("=" * 70)

    all_results = {}
    for name, cfg in UPDATES.items():
        all_results[name] = process(name, cfg)

    print(f"\n{'='*70}\nFINAL SUMMARY (3-2-1)\n{'='*70}")
    print(f"{'Dataset':<16} {'Head RMSE':<12} {'Pitch RMSE':<12} {'Roll RMSE':<12}")
    print("-" * 52)
    for name in UPDATES:
        _, _, _, _, rmse = all_results[name]
        print(f"{name:<16} {rmse[0]:<12.4f} {rmse[1]:<12.4f} {rmse[2]:<12.4f}")

    plot_results(all_results)
    print(f"\nDone.")
    print("=" * 70)


if __name__ == '__main__':
    main()
