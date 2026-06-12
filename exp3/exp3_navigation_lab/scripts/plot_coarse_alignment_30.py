"""
粗对准可视化 — gtimu_30_0_0.log 前14901行
===========================================
参考: Heading=330.642, Pitch=-0.017, Roll=0.003
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data_io import DataLoader
from src.utils.dcm import dcm_from_quat, dcm_orthogonalize
from src.utils.quaternion import quat_update
from src.utils.euler_angles import dcm_to_euler312, rad2deg, deg2rad
from src.utils.earth_model import gravity_i, C_n2i, WGS84_OMEGA

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', u'初始对准')
GTIMU_FILE = os.path.join(DATA_DIR, 'gtimu_30_0_0.log')
GPFPD_FILE = os.path.join(DATA_DIR, 'gpfpd_30_0_0.log')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'coarse_align_30_0_0')
N_ROWS = 14901

LAT = 45.734501; WIE = 7.292115e-5; G = 9.7803267714
REF = np.array([0.003, -0.017, 330.642])  # Roll, Pitch, Heading

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_coarse_align(imu_data, t1=5.0, t2=70.0):
    """双矢量法粗对准 (单组取点)"""
    gyro = imu_data[:, 0:3]; acc = imu_data[:, 3:6]
    time_raw = imu_data[:, 6]; time = time_raw - time_raw[0]
    N = len(time); lat_rad = np.deg2rad(LAT)
    dt = np.median(np.diff(time)[np.diff(time) > 1e-12])
    if np.isnan(dt): dt = 0.005

    q_b2i = np.array([1., 0., 0., 0.]); v_i = np.zeros(3); r_i = np.zeros(3)
    v_hist = np.zeros((N, 3)); r_hist = np.zeros((N, 3))
    for k in range(N):
        q_b2i = quat_update(q_b2i, gyro[k], dt)
        C_b2i = dcm_from_quat(q_b2i)
        v_i += C_b2i @ acc[k] * dt
        r_i -= gravity_i(time[k], lat_rad, WIE, G) * dt
        v_hist[k] = v_i.copy(); r_hist[k] = r_i.copy()

    i1 = np.argmin(np.abs(time - t1)); i2 = np.argmin(np.abs(time - t2))
    V1 = v_hist[i1]; V2 = v_hist[i2]; R1 = r_hist[i1]; R2 = r_hist[i2]
    Mv = np.column_stack([V1, V2, np.cross(V1, V2)])
    Mr = np.column_stack([R1, R2, np.cross(R1, R2)])
    if np.linalg.cond(Mr) > 1e12: return None

    C_bi0 = dcm_orthogonalize(Mv @ np.linalg.inv(Mr))
    C_ni0 = C_n2i(time[0], lat_rad, WIE)
    C_nb = dcm_orthogonalize(C_ni0 @ C_bi0.T)
    r, p, y = dcm_to_euler312(C_nb)
    rd, pd, yd = rad2deg(r), rad2deg(p), rad2deg(y)
    if abs(rd) > 90: rd -= np.sign(rd) * 180; yd = -yd
    yd = yd % 360
    return np.array([rd, pd, yd])


def load_data():
    print("[1] Loading data...")
    lines = []
    with open(GTIMU_FILE, 'r') as f:
        for i, l in enumerate(f):
            if i >= N_ROWS: break
            lines.append(l)
    tmp = GTIMU_FILE + '.tmp30p'
    with open(tmp, 'w') as f: f.writelines(lines)
    imu_raw = DataLoader.load_imu(tmp, scale_acc=9.7803267714)
    os.remove(tmp)
    gyro = imu_raw['gyro']; acc = imu_raw['acc']; t = imu_raw['time']
    imu_data = np.column_stack([gyro, acc, t])

    ref_raw = DataLoader.load_gpfpd(GPFPD_FILE)
    print(f"    IMU: {len(gyro)} pts, {t[0]:.1f}s ~ {t[-1]:.1f}s")
    print(f"    GPFPD: {len(ref_raw['time'])} pts")
    return imu_data, ref_raw


def plot_imu_raw(imu_data):
    print("[2] IMU raw data...")
    t = imu_data[:, 6] - imu_data[0, 6]
    gyro = np.rad2deg(imu_data[:, 0:3]); acc = imu_data[:, 3:6]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    for i, (l, c) in enumerate(zip(['X', 'Y', 'Z'], ['#E63946', '#2A9D8F', '#264653'])):
        ax1.plot(t, gyro[:, i], color=c, label=l, linewidth=0.6, alpha=0.85)
        ax2.plot(t, acc[:, i], color=c, label=l, linewidth=0.6, alpha=0.85)
    ax1.set_ylabel('Angular Rate (deg/s)', fontsize=12)
    ax1.set_title('Gyroscope', fontsize=13, fontweight='bold')
    ax1.legend(); ax1.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Specific Force (m/s^2)', fontsize=12)
    ax2.set_title('Accelerometer', fontsize=13, fontweight='bold')
    ax2.legend(); ax2.grid(True, alpha=0.3, linestyle='--')
    fig.suptitle('GTIMU Raw Data (gtimu_30_0_0.log, first 14901 lines)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '01_imu_raw.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_result_bar(att):
    print("[3] Result bar chart...")
    labels = ['Roll\n(roll)', 'Pitch\n(pitch)', 'Heading\n(heading)']
    errors = att - REF

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(3); w = 0.32

    ax1.bar(x - w/2, att, w, color=['#E63946', '#2A9D8F', '#264653'],
            label='Computed', edgecolor='white', linewidth=1.2)
    ax1.bar(x + w/2, REF, w, color=['#F4A261', '#E9C46A', '#E76F51'],
            alpha=0.7, label='Reference', edgecolor='white', linewidth=1.2)
    for i, (vc, vr) in enumerate(zip(att, REF)):
        ax1.text(i - w/2, vc + 0.03, f'{vc:.4f}', ha='center', fontsize=10, fontweight='bold')
        ax1.text(i + w/2, vr + 0.03, f'{vr:.3f}', ha='center', fontsize=10)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylabel('Angle (deg)', fontsize=12)
    ax1.set_title('Coarse Alignment vs Reference', fontsize=13, fontweight='bold')
    ax1.legend(); ax1.grid(True, alpha=0.2, axis='y')

    bar_c = ['#2ECC71' if abs(e) < 0.05 else '#E74C3C' if abs(e) > 0.3 else '#F39C12' for e in errors]
    ax2.bar(x, np.abs(errors), color=bar_c, edgecolor='white', linewidth=1.2)
    for i, e in enumerate(errors):
        ax2.text(i, abs(e) + 0.003, f'{e:+.4f}', ha='center', fontsize=11, fontweight='bold',
                 color='#C0392B' if abs(e) > 0.05 else '#27AE60')
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_ylabel('Absolute Error (deg)', fontsize=12)
    ax2.set_title('Error Analysis', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.2, axis='y')
    l2 = np.linalg.norm(errors)
    ax2.text(0.98, 0.95, f'L2 Error: {l2:.4f} deg\nHeading Err: {abs(errors[2]):.4f} deg',
             transform=ax2.transAxes, fontsize=10, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='#F8F9FA', edgecolor='#ADB5BD'))

    fig.suptitle('Coarse Alignment — Result & Error (gtimu_30_0_0.log)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '02_result_bar.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_convergence(imu_data):
    print("[4] Convergence curve...")
    t_norm = imu_data[:, 6] - imu_data[0, 6]
    dur = t_norm[-1]
    t2_vals = np.linspace(10.0, dur - 2, 25)
    results = []
    for t2 in t2_vals:
        r = run_coarse_align(imu_data, t1=5.0, t2=t2)
        if r is not None: results.append([t2, r[0], r[1], r[2]])
    results = np.array(results)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    labels = ['Roll (deg)', 'Pitch (deg)', 'Heading (deg)']
    colors = ['#E63946', '#2A9D8F', '#264653']
    for i, (ax, lb, c, rv) in enumerate(zip(axes, labels, colors, REF)):
        ax.plot(results[:, 0], results[:, i+1], 'o-', color=c, linewidth=1.5,
                markersize=6, markerfacecolor='white', markeredgewidth=1.5)
        ax.axhline(y=rv, color='#E76F51', linestyle='--', linewidth=1.5,
                   alpha=0.8, label=f'Ref: {rv:.3f}')
        ax.annotate(f'{results[0, i+1]:.3f}', (results[0, 0], results[0, i+1]),
                    xytext=(0, -20), textcoords='offset points', ha='center', fontsize=9, color=c)
        ax.annotate(f'{results[-1, i+1]:.3f}', (results[-1, 0], results[-1, i+1]),
                    xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9, color=c)
        ax.set_ylabel(lb, fontsize=11); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, linestyle='--')
    axes[-1].set_xlabel('t2 (s) [t1=5s fixed]', fontsize=12)
    fig.suptitle('Convergence — Attitude vs Integration Window', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '03_convergence.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_gpfpd_ref(ref_raw):
    print("[5] GPFPD reference...")
    t = ref_raw['time']; h = ref_raw['heading']; p = ref_raw['pitch']; r = ref_raw['roll']
    n_show = min(3000, len(t)); step = max(1, len(t) // n_show)
    idx = slice(0, len(t), step); ts = t[idx] - t[0]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    data = [('Heading', h[idx], '#264653'), ('Pitch', p[idx], '#2A9D8F'), ('Roll', r[idx], '#E63946')]
    for ax, (lb, vals, c) in zip(axes, data):
        mu = np.mean(vals)
        ax.plot(ts, vals, color=c, linewidth=0.6, alpha=0.85)
        ax.axhline(y=mu, color='#E76F51', linestyle='--', linewidth=1.2, alpha=0.7, label=f'Mean: {mu:.4f}')
        ax.set_ylabel('Angle (deg)', fontsize=11)
        ax.set_title(lb, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, linestyle='--')
    axes[-1].set_xlabel('Time (s, relative)', fontsize=12)
    fig.suptitle('$GPFPD Reference Attitude (gpfpd_30_0_0.log)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '04_gpfpd_ref.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_dashboard(att, imu_data):
    print("[6] Summary dashboard...")
    errors = att - REF
    labels = ['Roll\n(roll)', 'Pitch\n(pitch)', 'Heading\n(heading)']

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    # Attitude bar
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.arange(3); w = 0.3
    ax1.bar(x - w/2, att, w, color=['#E63946', '#2A9D8F', '#264653'],
            label='Computed', edgecolor='white', linewidth=1.5)
    ax1.bar(x + w/2, REF, w, color=['#F4A261', '#E9C46A', '#E76F51'],
            alpha=0.7, label='Reference', edgecolor='white', linewidth=1.5)
    for i, (vc, vr) in enumerate(zip(att, REF)):
        ax1.text(i - w/2, vc + 0.05, f'{vc:.4f}', ha='center', fontsize=10, fontweight='bold')
        ax1.text(i + w/2, vr + 0.05, f'{vr:.3f}', ha='center', fontsize=10)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylabel('Angle (deg)', fontsize=11)
    ax1.set_title('Attitude Comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10); ax1.grid(True, alpha=0.2, axis='y')

    # Error table
    ax2 = fig.add_subplot(gs[0, 2]); ax2.axis('off')
    td = [['Param', 'Computed', 'Reference', 'Error'],
          ['Roll', f'{att[0]:.4f} deg', f'{REF[0]:.3f} deg', f'{errors[0]:+.4f} deg'],
          ['Pitch', f'{att[1]:.4f} deg', f'{REF[1]:.3f} deg', f'{errors[1]:+.4f} deg'],
          ['Heading', f'{att[2]:.4f} deg', f'{REF[2]:.3f} deg', f'{errors[2]:+.4f} deg'],
          ['', '', '', ''],
          ['L2 Error', '', '', f'{np.linalg.norm(errors):.4f} deg'],
          ['Max Error', '', '', f'{np.max(np.abs(errors)):.4f} deg']]
    tbl = ax2.table(cellText=td[1:], cellLoc='center', loc='center', colWidths=[0.22, 0.28, 0.22, 0.28])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    for k, c in tbl.get_celld().items():
        if k[0] == 0: c.set_facecolor('#2C3E50'); c.set_text_props(color='white', fontweight='bold')
        elif k[1] == 3: c.set_text_props(fontweight='bold')
    ax2.set_title('Error Statistics', fontsize=13, fontweight='bold', y=1.05)

    # Gyro
    ax3 = fig.add_subplot(gs[1, :2])
    tn = imu_data[:, 6] - imu_data[0, 6]
    gyro_d = np.rad2deg(imu_data[:, 0:3])
    for i, (lb, c) in enumerate(zip(['Gx', 'Gy', 'Gz'], ['#E63946', '#2A9D8F', '#264653'])):
        ax3.plot(tn, gyro_d[:, i], color=c, label=lb, linewidth=0.5, alpha=0.8)
    ax3.set_ylabel('Angular Rate (deg/s)', fontsize=11)
    ax3.set_title('Gyroscope Measurements', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9, ncol=3); ax3.grid(True, alpha=0.3, linestyle='--')

    # Acc Z
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(tn, imu_data[:, 5], color='#264653', linewidth=0.5, alpha=0.8, label='Acc Z')
    ax4.set_ylabel('Specific Force (m/s^2)', fontsize=11)
    ax4.set_title('Accelerometer Z-axis', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3, linestyle='--')

    # Info
    ax5 = fig.add_subplot(gs[2, :]); ax5.axis('off')
    info = (f'Algorithm: Dual-Vector Coarse Alignment\n'
            f'Data: gtimu_30_0_0.log (first {N_ROWS} lines) | Lat: {LAT} deg | 200Hz | {tn[-1]:.1f}s\n'
            f'Params: t1=5.0s, t2=70.0s | g={G:.4f} m/s^2 | w_ie={WIE:.6e} rad/s\n'
            f'Result: L2 Error = {np.linalg.norm(errors):.4f} deg | Heading Error = {abs(errors[2]):.4f} deg')
    ax5.text(0.5, 0.5, info, transform=ax5.transAxes, fontsize=11, va='center', ha='center',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#F0F3F5', edgecolor='#ADB5BD', alpha=0.9))

    fig.suptitle('Coarse Alignment — Summary Report (gtimu_30_0_0.log)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '05_dashboard.png'), dpi=200, bbox_inches='tight')
    plt.close()


def save_csv(att):
    errors = att - REF
    with open(os.path.join(RESULTS_DIR, 'error_analysis.csv'), 'w') as f:
        f.write('Angle,Computed(deg),Reference(deg),Error(deg),AbsError(deg)\n')
        for n, c, r, e in zip(['Roll', 'Pitch', 'Heading'], att, REF, errors):
            f.write(f'{n},{c:.6f},{r:.6f},{e:+.6f},{abs(e):.6f}\n')
        f.write(f'\nL2_Error,{np.linalg.norm(errors):.6f}\n')
        f.write(f'L1_Error,{np.sum(np.abs(errors)):.6f}\n')
        f.write(f'Max_Error,{np.max(np.abs(errors)):.6f}\n')
    print(f"    CSV saved")


def main():
    print("=" * 70)
    print("Coarse Alignment Visualization — gtimu_30_0_0.log")
    print("=" * 70)

    imu_data, ref_raw = load_data()

    print("\n[0] Running coarse alignment...")
    att = run_coarse_align(imu_data)
    print(f"    Roll={att[0]:.4f}  Pitch={att[1]:.4f}  Heading={att[2]:.4f}")

    plot_imu_raw(imu_data)
    plot_result_bar(att)
    plot_convergence(imu_data)
    plot_gpfpd_ref(ref_raw)
    plot_dashboard(att, imu_data)
    save_csv(att)

    print(f"\n{'=' * 70}")
    print(f"All plots saved to: {RESULTS_DIR}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
