"""
粗对准解算 — gtimu_30_0_0.log 前14901行
=========================================
参考: Heading=330.642, Pitch=-0.017, Roll=0.003 (deg)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.data_io import DataLoader
from src.analysis import ComparisonAnalyzer
from src.utils.dcm import dcm_from_quat, dcm_orthogonalize
from src.utils.quaternion import quat_update
from src.utils.euler_angles import dcm_to_euler312, rad2deg, deg2rad
from src.utils.earth_model import gravity_i, C_n2i, WGS84_OMEGA

DATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', u'初始对准', 'gtimu_30_0_0.log')
N_ROWS = 14901
REF = (0.003, -0.017, 330.642)  # Roll, Pitch, Heading (deg)
LAT = 45.734501
WIE = 7.292115e-5
G = 9.7803267714


def run_coarse_align(imu_data, t1=5.0, t2=70.0, n_avg=5):
    """双矢量法粗对准 (修正版)"""
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

    dur = time[-1]; margin = min(5.0, dur * 0.1)
    C_list = []
    for p in range(n_avg):
        t1k = np.clip(t1 + p * (t2 - t1) / max(n_avg, 1), margin, dur - margin)
        t2k = np.clip(t2 + p * (t2 - t1) / max(n_avg, 1), t1k + margin, dur - margin)
        i1 = np.argmin(np.abs(time - t1k)); i2 = np.argmin(np.abs(time - t2k))
        if abs(i2 - i1) * dt < max(5.0, dur * 0.2): continue
        V1 = v_hist[i1]; V2 = v_hist[i2]; R1 = r_hist[i1]; R2 = r_hist[i2]
        Mv = np.column_stack([V1, V2, np.cross(V1, V2)])
        Mr = np.column_stack([R1, R2, np.cross(R1, R2)])
        if np.linalg.cond(Mr) > 1e12: continue
        C_list.append(dcm_orthogonalize(Mv @ np.linalg.inv(Mr)))

    if not C_list: raise RuntimeError("All pairs failed")
    C_bi0 = dcm_orthogonalize(np.mean(C_list, axis=0))
    C_ni0 = C_n2i(time[0], lat_rad, WIE)
    C_nb = dcm_orthogonalize(C_ni0 @ C_bi0.T)
    r, p, y = dcm_to_euler312(C_nb)
    rd, pd, yd = rad2deg(r), rad2deg(p), rad2deg(y)
    if abs(rd) > 90: rd -= np.sign(rd) * 180; yd = -yd
    # Normalize heading to [0, 360) to match reference convention
    yd = yd % 360
    return (rd, pd, yd), C_nb


def main():
    print("=" * 70)
    print(f"粗对准解算 — gtimu_30_0_0.log 前{N_ROWS}行")
    print("=" * 70)

    # Load
    print(f"\n[1] Loading first {N_ROWS} lines...")
    lines = [];
    with open(DATA_FILE, 'r') as f:
        for i, l in enumerate(f):
            if i >= N_ROWS: break
            lines.append(l)
    tmp = DATA_FILE + '.tmp30'
    with open(tmp, 'w') as f: f.writelines(lines)
    imu_raw = DataLoader.load_imu(tmp, scale_acc=9.7803267714)
    os.remove(tmp)

    gyro = imu_raw['gyro']; acc = imu_raw['acc']; t = imu_raw['time']
    print(f"    Points: {len(gyro)}, time: {t[0]:.1f}s ~ {t[-1]:.1f}s ({t[-1]-t[0]:.1f}s)")
    print(f"    Gyro mean (rad/s): {np.mean(gyro, axis=0)}")
    print(f"    Acc mean (m/s^2):  {np.mean(acc, axis=0)}")

    imu_data = np.column_stack([gyro, acc, t])

    # Align
    print(f"\n[2] Coarse alignment (t1=5s, t2=70s, avg=5)...")
    att, Cnb = run_coarse_align(imu_data)
    rd, pd, yd = att
    print(f"    Roll={rd:.4f} deg, Pitch={pd:.4f} deg, Heading={yd:.4f} deg")

    # Compare
    print(f"\n[3] Error analysis")
    comp = ComparisonAnalyzer.compare_alignment(att, REF)
    ComparisonAnalyzer.print_comparison("Coarse Align vs Reference", comp)

    errors = np.abs(np.array(att) - np.array(REF))
    print(f"\n{'Angle':<14} {'Computed':<14} {'Reference':<14} {'AbsError':<14}")
    print("-" * 56)
    for name, c, r, e in zip(['Roll', 'Pitch', 'Heading'], att, REF, errors):
        print(f"{name:<14} {c:<14.4f} {r:<14.4f} {e:<14.4f}")
    print(f"\nL2 Error: {comp['diff_norm']:.4f} deg")
    print(f"L1 Error: {errors.sum():.4f} deg")

    print("\n" + "=" * 70)
    print("Done")
    return att, comp


if __name__ == '__main__':
    main()
