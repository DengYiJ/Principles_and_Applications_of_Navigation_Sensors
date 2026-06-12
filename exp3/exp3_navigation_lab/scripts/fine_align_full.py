"""
精对准 — gtimu_0_0_0.log 全量数据 (67792行, ~339s)
=====================================================
粗对准仍用前9500行，精对准用全部数据
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
from src.analysis import ComparisonAnalyzer
from src.utils.dcm import dcm_from_quat, dcm_orthogonalize
from src.utils.quaternion import quat_update
from src.utils.euler_angles import dcm_to_euler312, rad2deg, deg2rad
from src.utils.earth_model import gravity_i, C_n2i, WGS84_OMEGA

LAT = 45.734501; WIE = 7.292115e-5; G = 9.7803267714
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', u'初始对准')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'fine_align_0_0_0_full')
os.makedirs(RESULTS_DIR, exist_ok=True)
REF = (0.001, -0.018, 0.770)


def load_full_imu(filepath):
    """加载全部IMU数据"""
    imu_raw = DataLoader.load_imu(filepath, scale_acc=9.7803267714)
    gyro = imu_raw['gyro']; acc = imu_raw['acc']; t = imu_raw['time']
    return np.column_stack([gyro, acc, t])


def coarse_align_first_n(imu_data, n_use=9500, t1=5.0, t2=40.0, n_avg=5):
    """用前n行做粗对准"""
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
    print("Fine Alignment — gtimu_0_0_0.log FULL (67792 lines, ~339s)")
    print("=" * 70)

    # Load full data
    imu_file = os.path.join(DATA_DIR, 'gtimu_0_0_0.log')
    print(f"\n[1] Loading full IMU data...")
    imu_data = load_full_imu(imu_file)
    t_norm = imu_data[:, 6] - imu_data[0, 6]
    print(f"    {len(imu_data)} pts, {t_norm[0]:.1f}s ~ {t_norm[-1]:.1f}s ({t_norm[-1]:.1f}s)")

    # Coarse align (first 9500 rows)
    print(f"\n[2] Coarse alignment (first 9500 rows)...")
    C_nb, att_c = coarse_align_first_n(imu_data)
    print(f"    Coarse: R={att_c[0]:.4f} P={att_c[1]:.4f} H={att_c[2]:.4f}")

    # Fine align (all data)
    print(f"\n[3] KF Fine alignment (ALL {len(imu_data)} rows)...")
    fa = FineAligner(latitude_deg=LAT, imu_err=(0.002, 20.0, 0.001, 10.0))
    # 航向锁定: 粗对准航向可靠(~0.1°), KF只估计水平失准角+零偏
    # phi_yaw_deg 设极小 = 航向几乎不修正
    att_f, X_hist, P_hist = fa.run(
        imu_data, C_nb, reset_feedback=True,
        init_cov_config={'phi_deg': 1.0, 'dv_mps': 0.1,
                         'gyro_bias_dps': 0.01, 'acc_bias_mps2': 0.0001, 'vel_noise_mps': 0.01}
    )
    gyro_bias = fa.get_gyro_bias_estimate()
    print(f"    Fine:   R={att_f[0]:.4f} P={att_f[1]:.4f} H={att_f[2]:.4f}")

    # Compare
    ref = np.array(REF)
    c_err = np.abs(np.array(att_c) - ref)
    f_err = np.abs(np.array(att_f) - ref)
    c_l2 = np.linalg.norm(np.array(att_c) - ref)
    f_l2 = np.linalg.norm(np.array(att_f) - ref)

    print(f"\n[4] Results")
    print(f"    {'':<12} {'Coarse (9.5k)':<18} {'Fine (67.8k)':<18} {'Improvement':<14}")
    print(f"    {'-'*62}")
    for i, lb in enumerate(['Roll', 'Pitch', 'Heading']):
        imp = c_err[i] - f_err[i]
        print(f"    {lb:<12} {c_err[i]:<18.4f} {f_err[i]:<18.4f} {imp:<+14.4f}")
    print(f"    {'L2 Error':<12} {c_l2:<18.4f} {f_l2:<18.4f} {c_l2-f_l2:<+14.4f}")
    print(f"\n    Gyro bias est:  {np.rad2deg(gyro_bias)} deg/s")
    print(f"    Acc bias est:   {fa.get_acc_bias_estimate()} m/s^2")

    # ---- Plots ----
    X = X_hist; P = P_hist
    t_kf = np.arange(X.shape[1]) * 0.005
    att_hist = np.array(fa.att_history)
    t_att = np.arange(len(att_hist)) * 100 * 0.005

    # KF states (8-dim)
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    for i, lb in enumerate([r'$\phi_E$', r'$\phi_N$']):
        axes[0].plot(t_kf, np.rad2deg(X[i]), label=lb, linewidth=0.8)
    axes[0].set_ylabel('Misalignment (deg)'); axes[0].legend(ncol=2); axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Horizontal Misalignment (full 339s) [$\phi_U$ locked to coarse]')
    for i, lb in enumerate([r'$\delta v_E$', r'$\delta v_N$']):
        axes[1].plot(t_kf, X[2+i], label=lb, linewidth=0.8)
    axes[1].set_ylabel('Vel Error (m/s)'); axes[1].legend(ncol=2); axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Horizontal Velocity Errors')
    for i, lb in enumerate([r'$\varepsilon_x$', r'$\varepsilon_y$']):
        axes[2].plot(t_kf, np.rad2deg(X[4+i]), label=lb, linewidth=0.8)
    axes[2].set_ylabel('Gyro Bias (deg/s)'); axes[2].legend(ncol=2); axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Gyroscope Bias Estimates [$\varepsilon_z$=0, unobservable]')
    for i, lb in enumerate([r'$\nabla_x$', r'$\nabla_y$']):
        axes[3].plot(t_kf, X[6+i], label=lb, linewidth=0.8)
    axes[3].set_xlabel('Time (s)'); axes[3].set_ylabel('Acc Bias (m/s^2)')
    axes[3].legend(ncol=2); axes[3].grid(True, alpha=0.3)
    axes[3].set_title('Accelerometer Bias Estimates [$\nabla_z$=0, unobservable]')
    fig.suptitle('KF 8-State Estimation — 0_0_0 Full Dataset (339s)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'kf_states_full.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Attitude convergence
    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    for i, (ax, lb, rv) in enumerate(zip(axes, ['Roll', 'Pitch', 'Heading'], REF)):
        ax.plot(t_att[:len(att_hist)], att_hist[:len(t_att), i], 'b-', linewidth=0.8, label='Fine Align')
        ax.axhline(y=rv, color='#E76F51', linestyle='--', linewidth=1.2, label=f'Ref: {rv:.3f}')
        ax.set_ylabel(f'{lb} (deg)'); ax.legend(); ax.grid(True, alpha=0.3, linestyle='--')
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Attitude Convergence — Full 339s', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'attitude_convergence_full.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Dashboard
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.arange(3); w = 0.22
    for i, (vals, lbl, clr) in enumerate([(att_c, 'Coarse(9.5k)', '#90CAF9'),
                                            (att_f, 'Fine(67.8k)', '#1565C0')]):
        ax1.bar(x + (i-1)*w, vals, w, color=clr, label=lbl, edgecolor='white')
    ax1.bar(x + 0.5*w, ref, w, color='#F4A261', alpha=0.7, label='Ref', edgecolor='white')
    ax1.set_xticks(x); ax1.set_xticklabels(['Roll', 'Pitch', 'Heading'])
    ax1.set_ylabel('Angle (deg)'); ax1.set_title('Coarse vs Fine (Full Data)')
    ax1.legend(); ax1.grid(True, alpha=0.2, axis='y')

    ax2 = fig.add_subplot(gs[0, 2]); ax2.axis('off')
    td = [['', 'Coarse', 'Fine', 'Imprv'],
          ['Roll', f'{c_err[0]:.4f}', f'{f_err[0]:.4f}', f'{c_err[0]-f_err[0]:+.4f}'],
          ['Pitch', f'{c_err[1]:.4f}', f'{f_err[1]:.4f}', f'{c_err[1]-f_err[1]:+.4f}'],
          ['Head', f'{c_err[2]:.4f}', f'{f_err[2]:.4f}', f'{c_err[2]-f_err[2]:+.4f}'],
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
    ax3.set_title('KF Horizontal Misalignment (8-state, 339s, heading locked)'); ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[2, :]); ax4.axis('off')
    info = (f'Fine Alignment (8-state KF) — Full Dataset (67792 rows, 339s) | Lat={LAT}°\n'
            f'Coarse (9.5k rows): L2={c_l2:.4f}° | Fine (67.8k rows): L2={f_l2:.4f}°\n'
            f'Gyro Bias: {np.rad2deg(gyro_bias)} deg/s | Acc Bias: {fa.get_acc_bias_estimate()} m/s^2\n'
            f'States: [φ_E,φ_N,δv_E,δv_N,ε_x,ε_y,∇_x,∇_y] — heading locked to coarse. φ_U,δv_U,ε_z,∇_z removed (unobservable in static).')
    ax4.text(0.5, 0.5, info, transform=ax4.transAxes, fontsize=11, va='center', ha='center',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#F0F3F5', edgecolor='#ADB5BD', alpha=0.9))
    fig.suptitle('Fine Alignment Summary — 0_0_0 Full 339s', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'dashboard_full.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # CSV
    with open(os.path.join(RESULTS_DIR, 'error_analysis.csv'), 'w') as f:
        f.write('Method,Angle,Computed(deg),Reference(deg),Error(deg)\n')
        for m, vals in [('Coarse_9.5k', att_c), ('Fine_67.8k', att_f)]:
            for an, v, rv in zip(['Roll','Pitch','Heading'], vals, REF):
                f.write(f'{m},{an},{v:.6f},{rv:.6f},{v-rv:+.6f}\n')
        f.write(f'\nCoarse_L2,{c_l2:.6f}\nFine_L2,{f_l2:.6f}\n')

    print(f"\n    Plots saved to: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
