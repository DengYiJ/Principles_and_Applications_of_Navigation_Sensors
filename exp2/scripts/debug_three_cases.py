"""
三组定位对比验证
================
Case1: 所有卫星 (含PRN9)
Case2: 删除PRN9
Case3: 仅保留 Elevation > 10°

输出每颗卫星的:
  PRN, Elevation, Raw pseudorange, Sat clock corr(m), Corrected pseudorange, Residual
"""

import sys, os, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import configs.constants as const
from algorithms.extractor import extract_obs_time, extract_ephemeris_matrix, extract_pseudorange_matrix
from algorithms.satellite import compute_satellite_position
from algorithms.corrections import apply_clock_correction, apply_earth_rotation_correction
from algorithms.solver import least_squares_solution
from algorithms.transform import ecef_to_geodetic
from evaluation.accuracy import validate_accuracy, parse_bestposa
from visualization.plot_earth import compute_elev_azim

with open('exp2/data.txt', 'r') as f:
    raw = f.read()

week, t_obs = extract_obs_time(raw)
eph, vprn = extract_ephemeris_matrix(raw, week, t_obs)
pr = extract_pseudorange_matrix(raw, week, vprn)
ref_lon, ref_lat, ref_h = parse_bestposa(raw)

N = pr.shape[0]

# 哈尔滨参考ECEF
rx_lat_r = ref_lat * const.D2R
rx_lon_r = ref_lon * const.D2R
sl = math.sin(rx_lat_r)
cl = math.cos(rx_lat_r)
N_rx = const.A_WGS84 / math.sqrt(1.0 - const.E2_WGS84 * sl * sl)
rx_ecef = (
    (N_rx + ref_h) * cl * math.cos(rx_lon_r),
    (N_rx + ref_h) * cl * math.sin(rx_lon_r),
    (N_rx * (1.0 - const.E2_WGS84) + ref_h) * sl
)

# =====================================================================
# 计算每颗卫星的详细信息
# =====================================================================
print("=" * 110)
print("SATELLITE DETAILS (relative to Harbin reference)")
print("=" * 110)

sv_data = []
for i in range(N):
    prn = int(eph[i, 0])
    Xs, Ys, Zs, Ek, A, e = compute_satellite_position(eph[i], t_obs)
    dt_s = apply_clock_correction(eph[i], t_obs, Ek, A, e)
    raw_pr = pr[i, 1]
    clock_corr_m = const.C * dt_s
    corrected_pr = raw_pr - clock_corr_m
    Xc, Yc, Zc = apply_earth_rotation_correction(Xs, Ys, Zs, corrected_pr)
    elev, azim = compute_elev_azim((Xc, Yc, Zc), rx_ecef)

    sv_data.append({
        'prn': prn, 'elev': elev, 'azim': azim,
        'raw_pr': raw_pr, 'dt_s': dt_s, 'clock_corr_m': clock_corr_m,
        'corrected_pr': corrected_pr,
        'X': Xc, 'Y': Yc, 'Z': Zc
    })

print(f"\n{'PRN':>4s} {'Elev(°)':>8s} {'Azim(°)':>8s} {'RawPr(m)':>16s} {'ClkCorr(m)':>12s} {'CorrPr(m)':>16s}")
print("  " + "-" * 68)
for sv in sv_data:
    print(f"{sv['prn']:4d} {sv['elev']:>8.2f} {sv['azim']:>8.2f} {sv['raw_pr']:>16.3f} {sv['clock_corr_m']:>12.3f} {sv['corrected_pr']:>16.3f}")

# =====================================================================
# 定位函数
# =====================================================================
def run_positioning(sv_indices, label):
    n = len(sv_indices)
    print(f"\n{'='*70}")
    print(f"CASE: {label} ({n} satellites)")
    print(f"{'='*70}")

    if n < 4:
        print(f"  INSUFFICIENT SATELLITES")
        return None

    sat_pos = np.zeros((n, 3))
    rho = np.zeros(n)
    for j, i in enumerate(sv_indices):
        sat_pos[j] = [sv_data[i]['X'], sv_data[i]['Y'], sv_data[i]['Z']]
        rho[j] = sv_data[i]['corrected_pr']

    X = least_squares_solution(sat_pos, rho)
    lon, lat, h = ecef_to_geodetic(X[0], X[1], X[2])

    # 残差
    dx = sat_pos[:,0] - X[0]
    dy = sat_pos[:,1] - X[1]
    dz = sat_pos[:,2] - X[2]
    rho_hat = np.sqrt(dx*dx + dy*dy + dz*dz)
    residuals = rho - (rho_hat + X[3])
    rms_res = math.sqrt(np.mean(residuals**2))

    dE, dN, dU, r2d, r3d = validate_accuracy(lon, lat, h, ref_lon, ref_lat, ref_h)

    print(f"\n  {'PRN':>4s} {'Elev(°)':>8s} {'CorrectedPr':>16s} {'rho_hat':>16s} {'Residual':>16s}")
    print("  " + "-" * 62)
    for j, i in enumerate(sv_indices):
        print(f"  {sv_data[i]['prn']:4d} {sv_data[i]['elev']:>8.2f} {rho[j]:>16.3f} {rho_hat[j]:>16.3f} {residuals[j]:>16.3f}")

    print(f"\n  ECEF: X={X[0]:.4f}  Y={X[1]:.4f}  Z={X[2]:.4f}")
    print(f"  cdt = {X[3]:.4f} m = {X[3]/const.C:.6e} s")
    print(f"  RMS Residual = {rms_res:.1f} m")
    print(f"  3D Error = {r3d:.1f} m")

    return {
        'Xecef': X.copy(),
        'cdt': X[3],
        'lon': lon, 'lat': lat, 'h': h,
        'rms_res': rms_res,
        'r2d': r2d, 'r3d': r3d,
        'n_sats': n
    }

# =====================================================================
# Case 1: 所有卫星
# =====================================================================
res1 = run_positioning(list(range(N)), "ALL 9 SATELLITES")

# =====================================================================
# Case 2: 删除 PRN 9
# =====================================================================
mask_no9 = [i for i in range(N) if sv_data[i]['prn'] != 9]
res2 = run_positioning(mask_no9, "EXCLUDE PRN 9")

# =====================================================================
# Case 3: 仅保留 Elevation > 10°
# =====================================================================
mask_high = [i for i in range(N) if sv_data[i]['elev'] > 10.0]
res3 = run_positioning(mask_high, "ELEV > 10°")

# =====================================================================
# 对比汇总
# =====================================================================
print("\n" + "=" * 90)
print("COMPARISON SUMMARY")
print("=" * 90)

results = [("CASE 1: ALL 9 SVs", res1),
           ("CASE 2: EXCLUDE PRN9", res2),
           ("CASE 3: ELEV > 10°", res3)]

print(f"\n{'':>25s} {'CASE 1':>20s} {'CASE 2':>20s} {'CASE 3':>20s}")
print(f"{'Sats':>25s} {res1['n_sats'] if res1 else '-':>20d} {res2['n_sats'] if res2 else '-':>20d} {res3['n_sats'] if res3 else '-':>20d}")
comparison_fields = [
    ('Xecef', 0, 'X (m)'), ('Xecef', 1, 'Y (m)'), ('Xecef', 2, 'Z (m)'),
    ('cdt', None, 'cdt (m)'), ('rms_res', None, 'RMS Res (m)'),
    ('r2d', None, '2D RMS (m)'), ('r3d', None, '3D Error (m)')
]
for key, idx, label in comparison_fields:
    if idx is not None:
        v1 = res1[key][idx] if res1 else '-'
        v2 = res2[key][idx] if res2 else '-'
        v3 = res3[key][idx] if res3 else '-'
    else:
        v1 = res1[key] if res1 else '-'
        v2 = res2[key] if res2 else '-'
        v3 = res3[key] if res3 else '-'
    print(f"{label:>25s} {v1:>20.1f} {v2:>20.1f} {v3:>20.1f}")

print(f"\n{'='*90}")
print(f"VERDICT")
print(f"{'='*90}")
print(f"  PRN9 causes solution crash: {'YES' if res1 and res2 and res1['r3d'] / res2['r3d'] > 10 else 'NO'}")
print(f"  Elevation filter helps:      {'YES' if res3 and res3['r3d'] < 1000 else 'NO'}")
print(f"  Best case: {min((r['r3d'], name) for r, name in [(res2,'CASE2'),(res3,'CASE3')] if r)[1]} with {min((r['r3d'], r['n_sats']) for r in [res2,res3] if r)[0]:.1f}m error")