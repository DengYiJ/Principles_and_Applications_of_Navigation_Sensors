"""
精对准 30_0_0 全量数据 (78472行, ~392s) + 完整作图
==================================================
8-state KF: 粗对准航向 + KF Roll/Pitch + 零偏估计
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from src.data_io import DataLoader
from src.alignment import FineAligner
from src.utils.dcm import dcm_from_quat, dcm_orthogonalize
from src.utils.quaternion import quat_update
from src.utils.euler_angles import dcm_to_euler312, rad2deg, deg2rad
from src.utils.earth_model import gravity_i, C_n2i, WGS84_OMEGA

LAT = 45.734501; WIE = 7.292115e-5; G = 9.7803267714
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', u'初始对准')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'fine_align_30_0_0_full')
os.makedirs(RESULTS_DIR, exist_ok=True)
REF = (0.003, -0.017, 330.642)
N_COARSE = 14901  # 粗对准用前14901行


def coarse_align_first_n(imu_data, n_use=N_COARSE, t1=5.0, t2=70.0, n_avg=5):
    sub = imu_data[:n_use]
    gyro = sub[:, 0:3]; acc = sub[:, 3:6]
    time_raw = sub[:, 6]; time = time_raw - time_raw[0]
    N = len(time); lat_rad = np.deg2rad(LAT)
    dt = np.median(np.diff(time)[np.diff(time) > 1e-12])

    q_b2i = np.array([1.,0.,0.,0.]); vi = np.zeros(3); ri = np.zeros(3)
    vh = np.zeros((N,3)); rh = np.zeros((N,3))
    for k in range(N):
        q_b2i = quat_update(q_b2i, gyro[k], dt)
        C_b2i = dcm_from_quat(q_b2i)
        vi += C_b2i @ acc[k] * dt
        ri -= gravity_i(time[k], lat_rad, WIE, G) * dt
        vh[k] = vi.copy(); rh[k] = ri.copy()

    dur = time[-1]; margin = min(5.0, dur*0.1)
    C_list = []
    for p in range(n_avg):
        t1k = np.clip(t1 + p*(t2-t1)/n_avg, margin, dur-margin)
        t2k = np.clip(t2 + p*(t2-t1)/n_avg, t1k+margin, dur-margin)
        i1 = np.argmin(np.abs(time-t1k)); i2 = np.argmin(np.abs(time-t2k))
        if abs(i2-i1)*dt < max(5.0, dur*0.2): continue
        Mv = np.column_stack([vh[i1], vh[i2], np.cross(vh[i1], vh[i2])])
        Mr = np.column_stack([rh[i1], rh[i2], np.cross(rh[i1], rh[i2])])
        if np.linalg.cond(Mr) > 1e12: continue
        C_list.append(dcm_orthogonalize(Mv @ np.linalg.inv(Mr)))

    C_bi0 = dcm_orthogonalize(np.mean(C_list, axis=0))
    C_ni0 = C_n2i(time[0], lat_rad, WIE)
    C_nb = dcm_orthogonalize(C_ni0 @ C_bi0.T)
    r, p, y = dcm_to_euler312(C_nb)
    rd, pd, yd = rad2deg(r), rad2deg(p), rad2deg(y)
    if abs(rd) > 90: rd -= np.sign(rd)*180; yd = -yd
    return C_nb, (rd, pd, yd % 360)


def main():
    print("=" * 70)
    print("Fine Alignment — gtimu_30_0_0.log FULL (78472 lines)")
    print("=" * 70)

    imu_file = os.path.join(DATA_DIR, 'gtimu_30_0_0.log')
    imu_raw = DataLoader.load_imu(imu_file, scale_acc=9.7803267714)
    imu_data = np.column_stack([imu_raw['gyro'], imu_raw['acc'], imu_raw['time']])
    t_norm = imu_data[:, 6] - imu_data[0, 6]
    print(f"    {len(imu_data)} pts, {t_norm[0]:.1f}s ~ {t_norm[-1]:.1f}s ({t_norm[-1]:.1f}s)")

    # Coarse
    C_nb, att_c = coarse_align_first_n(imu_data)
    print(f"    Coarse: R={att_c[0]:.4f} P={att_c[1]:.4f} H={att_c[2]:.4f}")

    # Fine
    fa = FineAligner(latitude_deg=LAT, imu_err=(0.002, 20.0, 0.001, 10.0))
    att_f, X, P = fa.run(imu_data, C_nb, reset_feedback=True,
        init_cov_config={'phi_deg': 1.0, 'dv_mps': 0.1, 'gyro_bias_dps': 0.01,
                         'acc_bias_mps2': 0.0001, 'vel_noise_mps': 0.01})

    # Combined: coarse heading + fine roll/pitch
    att_comb = (att_f[0], att_f[1], att_c[2])
    gyro_bias = fa.get_gyro_bias_estimate()
    acc_bias = fa.get_acc_bias_estimate()

    ref = np.array(REF)
    c_err = np.abs(np.array(att_c) - ref)
    f_err = np.abs(np.array(att_comb) - ref)
    c_l2 = np.linalg.norm(np.array(att_c) - ref)
    f_l2 = np.linalg.norm(np.array(att_comb) - ref)

    print(f"\n    {'':<12} {'Coarse':<14} {'Fine*':<14} {'Improve':<12}")
    print(f"    {'-'*52}")
    for i, lb in enumerate(['Roll', 'Pitch', 'Heading']):
        print(f"    {lb:<12} {c_err[i]:<14.4f} {f_err[i]:<14.4f} {c_err[i]-f_err[i]:<+12.4f}")
    print(f"    {'L2':<12} {c_l2:<14.4f} {f_l2:<14.4f} {c_l2-f_l2:<+12.4f}")
    print(f"    * Heading from coarse alignment")
    print(f"    Gyro bias: {np.rad2deg(gyro_bias)} deg/s")

    # ====== Plots ======
    t_kf = np.arange(X.shape[1]) * 0.005
    att_hist = np.array(fa.att_history)
    t_att = np.arange(len(att_hist)) * 100 * 0.005

    # Fig 1: 姿态收敛曲线
    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    for i, (ax, lb, rv) in enumerate(zip(axes, ['Roll', 'Pitch', 'Heading'], REF)):
        ax.plot(t_att[:len(att_hist)], att_hist[:len(t_att), i], 'b-', linewidth=0.8, label='Fine Align')
        ax.axhline(y=rv, color='#E76F51', linestyle='--', linewidth=1.2, alpha=0.8, label=f'Ref: {rv:.3f}')
        ax.set_ylabel(f'{lb} (deg)', fontsize=11); ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    fig.suptitle('Fine Alignment Attitude Convergence — 30_0_0 (392s)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '01_attitude_convergence.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Fig 2: KF 8-state error parameters
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    for i, lb in enumerate([r'$\phi_E$', r'$\phi_N$']):
        axes[0].plot(t_kf, np.rad2deg(X[i]), label=lb, linewidth=0.8)
    axes[0].set_ylabel('Misalignment (deg)', fontsize=11); axes[0].legend(ncol=2)
    axes[0].grid(True, alpha=0.3); axes[0].set_title('Horizontal Misalignment Angles')
    for i, lb in enumerate([r'$\delta v_E$', r'$\delta v_N$']):
        axes[1].plot(t_kf, X[2+i], label=lb, linewidth=0.8)
    axes[1].set_ylabel('Velocity Error (m/s)', fontsize=11); axes[1].legend(ncol=2)
    axes[1].grid(True, alpha=0.3); axes[1].set_title('Horizontal Velocity Errors')
    for i, lb in enumerate([r'$\varepsilon_x$', r'$\varepsilon_y$']):
        axes[2].plot(t_kf, np.rad2deg(X[4+i]), label=lb, linewidth=0.8)
    axes[2].set_ylabel('Gyro Bias (deg/s)', fontsize=11); axes[2].legend(ncol=2)
    axes[2].grid(True, alpha=0.3); axes[2].set_title('Gyroscope Bias Estimates')
    for i, lb in enumerate([r'$\nabla_x$', r'$\nabla_y$']):
        axes[3].plot(t_kf, X[6+i], label=lb, linewidth=0.8)
    axes[3].set_xlabel('Time (s)', fontsize=12); axes[3].set_ylabel('Acc Bias (m/s^2)', fontsize=11)
    axes[3].legend(ncol=2); axes[3].grid(True, alpha=0.3)
    axes[3].set_title('Accelerometer Bias Estimates')
    fig.suptitle('KF 8-State Error Parameters — 30_0_0 (392s)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '02_kf_states.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Fig 3: 协方差收敛
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    cov_groups = [
        ('Misalignment Covariance', [r'$P_{\phi E}$', r'$P_{\phi N}$'], [0,1]),
        ('Velocity Error Covariance', [r'$P_{\delta v E}$', r'$P_{\delta v N}$'], [2,3]),
        ('Gyro Bias Covariance', [r'$P_{\epsilon x}$', r'$P_{\epsilon y}$'], [4,5]),
        ('Acc Bias Covariance', [r'$P_{\nabla x}$', r'$P_{\nabla y}$'], [6,7]),
    ]
    for ax, (title, lbs, idx) in zip(axes, cov_groups):
        for i, lb in zip(idx, lbs):
            ax.semilogy(t_kf, P[i], label=lb, linewidth=0.8)
        ax.set_ylabel('Covariance', fontsize=11); ax.legend(ncol=2)
        ax.grid(True, alpha=0.3); ax.set_title(title)
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    fig.suptitle('KF Covariance Convergence — 30_0_0 (392s)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '03_kf_covariance.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Fig 4: 综合仪表盘
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.arange(3); w = 0.22
    for j, (vals, lbl, clr) in enumerate([(att_c, 'Coarse', '#90CAF9'), (att_comb, 'Fine*', '#1565C0')]):
        ax1.bar(x + (j-1)*w, vals, w, color=clr, label=lbl, edgecolor='white')
    ax1.bar(x + 0.5*w, ref, w, color='#F4A261', alpha=0.7, label='Ref', edgecolor='white')
    ax1.set_xticks(x); ax1.set_xticklabels(['Roll', 'Pitch', 'Heading'])
    ax1.set_ylabel('Angle (deg)'); ax1.set_title('Coarse vs Fine* (30_0_0 Full)')
    ax1.legend(); ax1.grid(True, alpha=0.2, axis='y')

    ax2 = fig.add_subplot(gs[0, 2]); ax2.axis('off')
    td = [['', 'Coarse', 'Fine*', 'Imprv'],
          ['Roll', f'{c_err[0]:.4f}', f'{f_err[0]:.4f}', f'{c_err[0]-f_err[0]:+.4f}'],
          ['Pitch', f'{c_err[1]:.4f}', f'{f_err[1]:.4f}', f'{c_err[1]-f_err[1]:+.4f}'],
          ['Head', f'{c_err[2]:.4f}', f'{f_err[2]:.4f}', '-'],
          ['L2', f'{c_l2:.4f}', f'{f_l2:.4f}', f'{c_l2-f_l2:+.4f}']]
    tbl = ax2.table(cellText=td[1:], cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for k, c in tbl.get_celld().items():
        if k[0] == 0: c.set_facecolor('#2C3E50'); c.set_text_props(color='white', fontweight='bold')
    ax2.set_title('Error Summary', fontsize=13, fontweight='bold')

    ax3 = fig.add_subplot(gs[1, :])
    for i, lb in enumerate([r'$\phi_E$', r'$\phi_N$']):
        ax3.plot(t_kf, np.rad2deg(X[i]), label=lb, linewidth=0.8)
    ax3.set_ylabel('Misalignment (deg)'); ax3.legend(ncol=2)
    ax3.set_title('KF Horizontal Misalignment (8-state)'); ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[2, :]); ax4.axis('off')
    info = (f'Fine Alignment (8-state KF) — 30_0_0 Full (78472 rows, 392s) | Lat={LAT}°\n'
            f'Coarse: L2={c_l2:.4f}° | Fine*: L2={f_l2:.4f}° (*heading from coarse)\n'
            f'Gyro Bias: {np.rad2deg(gyro_bias)} deg/s | Acc Bias: {acc_bias} m/s^2\n'
            f'States: [φ_E,φ_N,δv_E,δv_N,ε_x,ε_y,∇_x,∇_y] — heading locked to coarse alignment')
    ax4.text(0.5, 0.5, info, transform=ax4.transAxes, fontsize=11, va='center', ha='center',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#F0F3F5', edgecolor='#ADB5BD', alpha=0.9))
    fig.suptitle('Fine Alignment Summary — 30_0_0 Full 392s', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '04_dashboard.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # CSV
    with open(os.path.join(RESULTS_DIR, 'error_analysis.csv'), 'w') as f:
        f.write('Method,Angle,Computed(deg),Reference(deg),Error(deg)\n')
        for m, vals in [('Coarse', att_c), ('Fine_combined', att_comb)]:
            for an, v, rv in zip(['Roll','Pitch','Heading'], vals, REF):
                f.write(f'{m},{an},{v:.6f},{rv:.6f},{v-rv:+.6f}\n')
        f.write(f'\nCoarse_L2,{c_l2:.6f}\nFine_L2,{f_l2:.6f}\n')

    print(f"\n    Plots saved to: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
