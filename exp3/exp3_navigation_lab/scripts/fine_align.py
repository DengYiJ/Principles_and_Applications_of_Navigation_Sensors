"""
卡尔曼滤波精对准 — 完整流水线
==============================
对两个姿态(0_0_0, 30_0_0)执行: 粗对准 → KF精对准 → 误差分析 → 作图

用法:
    cd exp3_navigation_lab
    python scripts/fine_align.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data_io import DataLoader
from src.alignment import FineAligner
from src.analysis import ComparisonAnalyzer
from src.utils.dcm import dcm_from_quat, dcm_orthogonalize
from src.utils.quaternion import quat_update
from src.utils.euler_angles import dcm_to_euler312, rad2deg, deg2rad
from src.utils.earth_model import gravity_i, C_n2i, WGS84_OMEGA

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

LAT = 45.734501; WIE = 7.292115e-5; G = 9.7803267714
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', u'初始对准')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

DATASETS = {
    '0_0_0': {
        'gtimu': 'gtimu_0_0_0.log', 'gpfpd': 'gpfpd_0_0_0.log',
        'n_rows': 9500, 't1': 5.0, 't2': 40.0,
        'ref': (0.001, -0.018, 0.770),  # Roll, Pitch, Heading (deg)
    },
    '30_0_0': {
        'gtimu': 'gtimu_30_0_0.log', 'gpfpd': 'gpfpd_30_0_0.log',
        'n_rows': 14901, 't1': 5.0, 't2': 70.0,
        'ref': (0.003, -0.017, 330.642),
    }
}


def load_imu(filepath, n_rows):
    lines = []
    with open(filepath, 'r') as f:
        for i, l in enumerate(f):
            if i >= n_rows: break
            lines.append(l)
    tmp = filepath + '.tmp_fine'
    with open(tmp, 'w') as f: f.writelines(lines)
    imu_raw = DataLoader.load_imu(tmp, scale_acc=9.7803267714)
    os.remove(tmp)
    gyro = imu_raw['gyro']; acc = imu_raw['acc']; t = imu_raw['time']
    return np.column_stack([gyro, acc, t])


def coarse_align(imu_data, t1, t2, n_avg=5):
    """双矢量法粗对准 (修正版, 返回 C_n^b)"""
    gyro = imu_data[:, 0:3]; acc = imu_data[:, 3:6]
    time_raw = imu_data[:, 6]; time = time_raw - time_raw[0]
    N = len(time); lat_rad = np.deg2rad(LAT)
    dt = np.median(np.diff(time)[np.diff(time) > 1e-12])
    if np.isnan(dt): dt = 0.005

    q_b2i = np.array([1., 0., 0., 0.]); v_i = np.zeros(3); r_i = np.zeros(3)
    vh = np.zeros((N, 3)); rh = np.zeros((N, 3))
    for k in range(N):
        q_b2i = quat_update(q_b2i, gyro[k], dt)
        C_b2i = dcm_from_quat(q_b2i)
        v_i += C_b2i @ acc[k] * dt
        r_i -= gravity_i(time[k], lat_rad, WIE, G) * dt
        vh[k] = v_i.copy(); rh[k] = r_i.copy()

    dur = time[-1]; margin = min(5.0, dur * 0.1)
    C_list = []
    for p in range(n_avg):
        t1k = np.clip(t1 + p*(t2-t1)/max(n_avg,1), margin, dur-margin)
        t2k = np.clip(t2 + p*(t2-t1)/max(n_avg,1), t1k+margin, dur-margin)
        i1 = np.argmin(np.abs(time-t1k)); i2 = np.argmin(np.abs(time-t2k))
        if abs(i2-i1)*dt < max(5.0, dur*0.2): continue
        V1 = vh[i1]; V2 = vh[i2]; R1 = rh[i1]; R2 = rh[i2]
        Mv = np.column_stack([V1, V2, np.cross(V1, V2)])
        Mr = np.column_stack([R1, R2, np.cross(R1, R2)])
        if np.linalg.cond(Mr) > 1e12: continue
        C_list.append(dcm_orthogonalize(Mv @ np.linalg.inv(Mr)))

    if not C_list: raise RuntimeError("Coarse align failed")
    C_bi0 = dcm_orthogonalize(np.mean(C_list, axis=0))
    C_ni0 = C_n2i(time[0], lat_rad, WIE)
    C_nb = dcm_orthogonalize(C_ni0 @ C_bi0.T)  # C_n^b
    r, p, y = dcm_to_euler312(C_nb)
    rd, pd, yd = rad2deg(r), rad2deg(p), rad2deg(y)
    if abs(rd) > 90: rd -= np.sign(rd)*180; yd = -yd
    yd = yd % 360
    return C_nb, (rd, pd, yd)


def process_dataset(name, cfg):
    """对单个数据集执行粗对准+精对准"""
    print(f"\n{'#'*70}")
    print(f"# Processing: {name}")
    print(f"{'#'*70}")

    imu_file = os.path.join(DATA_DIR, cfg['gtimu'])
    ref_file = os.path.join(DATA_DIR, cfg['gpfpd'])

    # 1. Load
    print(f"\n[1] Loading {cfg['n_rows']} rows from {cfg['gtimu']}...")
    imu_data = load_imu(imu_file, cfg['n_rows'])
    t_norm = imu_data[:, 6] - imu_data[0, 6]
    print(f"    {len(imu_data)} pts, {t_norm[0]:.1f}s ~ {t_norm[-1]:.1f}s ({t_norm[-1]:.1f}s)")

    # 2. Coarse
    print(f"\n[2] Coarse alignment (t1={cfg['t1']}, t2={cfg['t2']})...")
    C_nb, att_coarse = coarse_align(imu_data, cfg['t1'], cfg['t2'])
    print(f"    Coarse: R={att_coarse[0]:.4f} P={att_coarse[1]:.4f} H={att_coarse[2]:.4f}")

    # 3. Fine
    print(f"\n[3] KF Fine alignment...")
    fine_aligner = FineAligner(latitude_deg=LAT, imu_err=(0.002, 20.0, 0.001, 10.0))
    att_fine, X_hist, P_hist = fine_aligner.run(
        imu_data, C_nb,
        reset_feedback=True,
        init_cov_config={'phi_deg': 1.0, 'phi_yaw_deg': 5.0, 'dv_mps': 0.1,
                         'gyro_bias_dps': 0.01, 'acc_bias_mps2': 0.0001, 'vel_noise_mps': 0.5}
    )
    gyro_bias = fine_aligner.get_gyro_bias_estimate()
    acc_bias = fine_aligner.get_acc_bias_estimate()
    print(f"    Fine:   R={att_fine[0]:.4f} P={att_fine[1]:.4f} H={att_fine[2]:.4f}")

    # 4. Compare
    ref = cfg['ref']
    print(f"\n[4] Error analysis")
    print(f"    Reference: R={ref[0]:.4f} P={ref[1]:.4f} H={ref[2]:.4f}")

    coarse_comp = ComparisonAnalyzer.compare_alignment(att_coarse, ref)
    fine_comp = ComparisonAnalyzer.compare_alignment(att_fine, ref)

    coarse_err = np.array(att_coarse) - np.array(ref)
    fine_err = np.array(att_fine) - np.array(ref)

    print(f"\n    {'':<12} {'Coarse':<14} {'Fine':<14} {'Improvement':<14}")
    print(f"    {'-'*54}")
    for i, label in enumerate(['Roll', 'Pitch', 'Heading']):
        imp = abs(coarse_err[i]) - abs(fine_err[i])
        print(f"    {label:<12} {abs(coarse_err[i]):<14.4f} {abs(fine_err[i]):<14.4f} {imp:<+14.4f}")
    print(f"    {'L2 Error':<12} {coarse_comp['diff_norm']:<14.4f} {fine_comp['diff_norm']:<14.4f} {coarse_comp['diff_norm']-fine_comp['diff_norm']:<+14.4f}")

    return {
        'imu_data': imu_data,
        'att_coarse': att_coarse,
        'att_fine': att_fine,
        'X_hist': X_hist,
        'P_hist': P_hist,
        'gyro_bias': gyro_bias,
        'acc_bias': acc_bias,
        'coarse_comp': coarse_comp,
        'fine_comp': fine_comp,
        'att_history': np.array(fine_aligner.att_history),
    }


def plot_results(results, name, ref):
    """生成精对准结果图表"""
    save_dir = os.path.join(RESULTS_DIR, f'fine_align_{name}')
    os.makedirs(save_dir, exist_ok=True)

    r = results
    att_c = np.array(r['att_coarse'])
    att_f = np.array(r['att_fine'])
    ref_arr = np.array(ref)
    X = r['X_hist']
    P = r['P_hist']
    att_hist = r['att_history']
    t_hist = np.arange(len(att_hist)) * 100 * 0.005  # every 100 steps * dt

    # ----- 图1: 粗对准 vs 精对准 vs 参考 -----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(3); w = 0.22
    labels = ['Roll', 'Pitch', 'Heading']
    for i, (ax, title) in enumerate([(ax1, 'Coarse Align'), (ax2, 'Fine Align')]):
        vals = [att_c, att_f][i]
        ax.bar(x - w/2, vals, w, color=['#E63946', '#2A9D8F', '#264653'],
               label='Computed', edgecolor='white', linewidth=1.2)
        ax.bar(x + w/2, ref_arr, w, color=['#F4A261', '#E9C46A', '#E76F51'],
               alpha=0.7, label='Reference', edgecolor='white', linewidth=1.2)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel('Angle (deg)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.2, axis='y')
    fig.suptitle(f'Coarse vs Fine Alignment — {name}', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '01_coarse_vs_fine.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # ----- 图2: KF 12维状态估计收敛曲线 -----
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    t = np.arange(X.shape[1]) * 0.005
    # 失准角
    ax = axes[0]
    for i, lb in enumerate([r'$\phi_E$', r'$\phi_N$', r'$\phi_U$']):
        ax.plot(t, np.rad2deg(X[i]), label=lb, linewidth=0.6)
    ax.set_ylabel('Misalignment (deg)'); ax.legend(ncol=3); ax.grid(True, alpha=0.3)
    ax.set_title('Attitude Misalignment Angles')
    # 速度误差
    ax = axes[1]
    for i, lb in enumerate([r'$\delta v_E$', r'$\delta v_N$', r'$\delta v_U$']):
        ax.plot(t, X[3+i], label=lb, linewidth=0.6)
    ax.set_ylabel('Velocity Error (m/s)'); ax.legend(ncol=3); ax.grid(True, alpha=0.3)
    ax.set_title('Velocity Errors')
    # 陀螺零偏
    ax = axes[2]
    for i, lb in enumerate([r'$\varepsilon_x$', r'$\varepsilon_y$', r'$\varepsilon_z$']):
        ax.plot(t, np.rad2deg(X[6+i]), label=lb, linewidth=0.6)
    ax.set_ylabel('Gyro Bias (deg/s)'); ax.legend(ncol=3); ax.grid(True, alpha=0.3)
    ax.set_title('Gyroscope Bias Estimates')
    # 加计零偏
    ax = axes[3]
    for i, lb in enumerate([r'$\nabla_x$', r'$\nabla_y$', r'$\nabla_z$']):
        ax.plot(t, X[9+i], label=lb, linewidth=0.6)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Acc Bias (m/s^2)')
    ax.legend(ncol=3); ax.grid(True, alpha=0.3)
    ax.set_title('Accelerometer Bias Estimates')
    fig.suptitle(f'KF 12-State Estimation — {name}', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '02_kf_states.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # ----- 图3: 协方差收敛 -----
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    cov_labels = [
        ('Misalignment Cov', [r'$P_\phi E$', r'$P_\phi N$', r'$P_\phi U$'], [0,1,2], 'deg^2'),
        ('Velocity Cov', [r'$P_{\delta v}E$', r'$P_{\delta v}N$', r'$P_{\delta v}U$'], [3,4,5], 'm^2/s^2'),
        ('Gyro Bias Cov', [r'$P_\varepsilon x$', r'$P_\varepsilon y$', r'$P_\varepsilon z$'], [6,7,8], '(rad/s)^2'),
        ('Acc Bias Cov', [r'$P_\nabla x$', r'$P_\nabla y$', r'$P_\nabla z$'], [9,10,11], '(m/s^2)^2'),
    ]
    for ax, (title, lbs, idx, unit) in zip(axes, cov_labels):
        for i, lb in zip(idx, lbs):
            ax.semilogy(t, P[i], label=lb, linewidth=0.6)
        ax.set_ylabel(f'Covariance ({unit})'); ax.legend(ncol=3); ax.grid(True, alpha=0.3)
        ax.set_title(title)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'KF Covariance Convergence — {name}', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '03_kf_covariance.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # ----- 图4: 姿态收敛曲线 -----
    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    for i, (ax, lb, rv) in enumerate(zip(axes, ['Roll', 'Pitch', 'Heading'], ref)):
        ax.plot(t_hist[:len(att_hist)], att_hist[:len(t_hist), i], 'b-', linewidth=1, label='Fine Align')
        ax.axhline(y=rv, color='#E76F51', linestyle='--', linewidth=1.2, label=f'Ref: {rv:.3f}')
        ax.set_ylabel(f'{lb} (deg)', fontsize=11); ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Attitude Convergence — {name}', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '04_attitude_convergence.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # ----- 图5: 综合仪表盘 -----
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.arange(3); w = 0.2
    ax1.bar(x - 1.5*w, att_c, w, color='#90CAF9', label='Coarse', edgecolor='white')
    ax1.bar(x - 0.5*w, att_f, w, color='#1565C0', label='Fine', edgecolor='white')
    ax1.bar(x + 0.5*w, ref_arr, w, color='#F4A261', alpha=0.7, label='Ref', edgecolor='white')
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylabel('Angle (deg)'); ax1.set_title('Alignment Comparison'); ax1.legend()
    ax1.grid(True, alpha=0.2, axis='y')
    for i in range(3):
        ax1.text(i-1.5*w, att_c[i]+0.03, f'{att_c[i]:.3f}', ha='center', fontsize=8)
        ax1.text(i-0.5*w, att_f[i]+0.03, f'{att_f[i]:.3f}', ha='center', fontsize=8)

    ax2 = fig.add_subplot(gs[0, 2]); ax2.axis('off')
    c_err = np.abs(att_c - ref_arr); f_err = np.abs(att_f - ref_arr)
    td = [['Param', 'Coarse Err', 'Fine Err', 'Improve'],
          ['Roll', f'{c_err[0]:.4f}', f'{f_err[0]:.4f}', f'{c_err[0]-f_err[0]:+.4f}'],
          ['Pitch', f'{c_err[1]:.4f}', f'{f_err[1]:.4f}', f'{c_err[1]-f_err[1]:+.4f}'],
          ['Heading', f'{c_err[2]:.4f}', f'{f_err[2]:.4f}', f'{c_err[2]-f_err[2]:+.4f}'],
          ['', '', '', ''],
          ['L2', f'{np.linalg.norm(att_c-ref_arr):.4f}', f'{np.linalg.norm(att_f-ref_arr):.4f}', '']]
    tbl = ax2.table(cellText=td[1:], cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for k, c in tbl.get_celld().items():
        if k[0] == 0: c.set_facecolor('#2C3E50'); c.set_text_props(color='white', fontweight='bold')
    ax2.set_title('Error Summary', fontsize=13, fontweight='bold')

    ax3 = fig.add_subplot(gs[1, :])
    for i, lb in enumerate([r'$\phi_E$', r'$\phi_N$', r'$\phi_U$']):
        ax3.plot(t, np.rad2deg(X[i]), label=lb, linewidth=0.6)
    ax3.set_ylabel('Misalignment (deg)'); ax3.legend(ncol=3)
    ax3.set_title('KF Misalignment Angle Convergence'); ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[2, :]); ax4.axis('off')
    gb = r['gyro_bias']; ab = r['acc_bias']
    info = (f'Fine Alignment Report — {name} | Lat={LAT} deg\n'
            f'Gyro Bias Est: [{np.rad2deg(gb[0]):.6f}, {np.rad2deg(gb[1]):.6f}, {np.rad2deg(gb[2]):.6f}] deg/s\n'
            f'Acc Bias Est:  [{ab[0]:.6f}, {ab[1]:.6f}, {ab[2]:.6f}] m/s^2\n'
            f'Coarse L2={np.linalg.norm(att_c-ref_arr):.4f} deg -> Fine L2={np.linalg.norm(att_f-ref_arr):.4f} deg')
    ax4.text(0.5, 0.5, info, transform=ax4.transAxes, fontsize=11, va='center', ha='center',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#F0F3F5', edgecolor='#ADB5BD', alpha=0.9))
    fig.suptitle(f'Fine Alignment Summary — {name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '05_dashboard.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # CSV
    with open(os.path.join(save_dir, 'error_analysis.csv'), 'w') as f:
        f.write('Method,Angle,Computed(deg),Reference(deg),Error(deg)\n')
        for method, vals in [('Coarse', att_c), ('Fine', att_f)]:
            for aname, v, rv in zip(['Roll', 'Pitch', 'Heading'], vals, ref_arr):
                f.write(f'{method},{aname},{v:.6f},{rv:.6f},{v-rv:+.6f}\n')
        f.write(f'\nCoarse_L2,{np.linalg.norm(att_c-ref_arr):.6f}\n')
        f.write(f'Fine_L2,{np.linalg.norm(att_f-ref_arr):.6f}\n')

    print(f"    Plots saved to: {save_dir}")
    return save_dir


def main():
    print("=" * 70)
    print("KF Fine Alignment — Complete Pipeline")
    print("=" * 70)

    all_results = {}
    for name, cfg in DATASETS.items():
        all_results[name] = process_dataset(name, cfg)

    # Generate plots
    for name, cfg in DATASETS.items():
        print(f"\n[Plot] Generating figures for {name}...")
        plot_results(all_results[name], name, cfg['ref'])

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':<12} {'Method':<10} {'Roll Err':<12} {'Pitch Err':<12} {'Heading Err':<12} {'L2 Err':<12}")
    print("-" * 70)
    for name, cfg in DATASETS.items():
        r = all_results[name]
        ref = np.array(cfg['ref'])
        for method, att in [('Coarse', r['att_coarse']), ('Fine', r['att_fine'])]:
            err = np.abs(np.array(att) - ref)
            l2 = np.linalg.norm(np.array(att) - ref)
            print(f"{name:<12} {method:<10} {err[0]:<12.4f} {err[1]:<12.4f} {err[2]:<12.4f} {l2:<12.4f}")

    print(f"\nAll results saved to: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
