"""
验证: 剔除PRN 9后定位精度
============================
剔除异常卫星 PRN 9，重新运行定位解算。
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

with open('exp2/data.txt', 'r') as f:
    raw = f.read()

week, t_obs = extract_obs_time(raw)
eph, vprn = extract_ephemeris_matrix(raw, week, t_obs)
pr = extract_pseudorange_matrix(raw, week, vprn)

ref_lon, ref_lat, ref_h = parse_bestposa(raw)

print("=" * 70)
print("EXCLUDE PRN 9 VERIFICATION")
print("=" * 70)

# 原始结果 (9颗星)
print("\n--- ALL 9 SATELLITES ---")
N_all = pr.shape[0]
sat_pos_all = np.zeros((N_all, 3))
rho_corr_all = np.zeros(N_all)
for i in range(N_all):
    Xs, Ys, Zs, Ek, A, e = compute_satellite_position(eph[i], t_obs)
    sat_pos_all[i] = [Xs, Ys, Zs]
    dt_s = apply_clock_correction(eph[i], t_obs, Ek, A, e)
    rho_corr_all[i] = pr[i, 1] - const.C * dt_s
    Xc, Yc, Zc = apply_earth_rotation_correction(*sat_pos_all[i], rho_corr_all[i])
    sat_pos_all[i] = [Xc, Yc, Zc]

X_all = least_squares_solution(sat_pos_all, rho_corr_all)
lon_all, lat_all, h_all = ecef_to_geodetic(X_all[0], X_all[1], X_all[2])
print(f"  Result: lon={lon_all:.8f}  lat={lat_all:.8f}  h={h_all:.1f}m")
print(f"  Clock bias: {X_all[3]/const.C:.6e}s ({X_all[3]/1000:.1f}km)")
dE, dN, dU, r2d, r3d = validate_accuracy(lon_all, lat_all, h_all, ref_lon, ref_lat, ref_h)

# === 排除 PRN 9 ===
print("\n--- EXCLUDING PRN 9 ---")
mask = [i for i in range(N_all) if int(eph[i,0]) != 9]
N_filt = len(mask)
print(f"  Removed satellites: {N_all - N_filt}")
print(f"  Remaining PRNs: {[int(eph[i,0]) for i in mask]}")

sat_pos_filt = np.zeros((N_filt, 3))
rho_corr_filt = np.zeros(N_filt)
for j, i in enumerate(mask):
    Xs, Ys, Zs, Ek, A, e = compute_satellite_position(eph[i], t_obs)
    sat_pos_filt[j] = [Xs, Ys, Zs]
    dt_s = apply_clock_correction(eph[i], t_obs, Ek, A, e)
    rho_corr_filt[j] = pr[i, 1] - const.C * dt_s
    Xc, Yc, Zc = apply_earth_rotation_correction(*sat_pos_filt[j], rho_corr_filt[j])
    sat_pos_filt[j] = [Xc, Yc, Zc]

X_filt = least_squares_solution(sat_pos_filt, rho_corr_filt)
lon_filt, lat_filt, h_filt = ecef_to_geodetic(X_filt[0], X_filt[1], X_filt[2])
print(f"\n  Result: lon={lon_filt:.8f}  lat={lat_filt:.8f}  h={h_filt:.1f}m")
print(f"  Clock bias: {X_filt[3]/const.C:.6e}s ({X_filt[3]:.1f}m)")
dE2, dN2, dU2, r2d2, r3d2 = validate_accuracy(lon_filt, lat_filt, h_filt, ref_lon, ref_lat, ref_h)

# 对比
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"\n{'':>20s} {'ALL 9 SVs':>20s} {'EXCL PRN9':>20s}")
print(f"  {'-'*60}")
print(f"{'Lon(°)':>20s} {lon_all:>20.8f} {lon_filt:>20.8f}")
print(f"{'Lat(°)':>20s} {lat_all:>20.8f} {lat_filt:>20.8f}")
print(f"{'H(m)':>20s} {h_all:>20.1f} {h_filt:>20.1f}")
print(f"{'cdt(s)':>20s} {X_all[3]/const.C:>20.6e} {X_filt[3]/const.C:>20.6e}")
print(f"{'ΔE(m)':>20s} {dE:>20.1f} {dE2:>20.1f}")
print(f"{'ΔN(m)':>20s} {dN:>20.1f} {dN2:>20.1f}")
print(f"{'ΔU(m)':>20s} {dU:>20.1f} {dU2:>20.1f}")
print(f"{'2D RMS(m)':>20s} {r2d:>20.1f} {r2d2:>20.1f}")
print(f"{'3D Error(m)':>20s} {r3d:>20.1f} {r3d2:>20.1f}")

print(f"\n{'='*70}")
print(f"JUDGMENT")
print(f"{'='*70}")
if r3d2 < 100:
    print(f"  ✅ EXCLUDING PRN 9 REDUCES 3D ERROR FROM {r3d:.0f}m TO {r3d2:.1f}m")
    print(f"  → Achieving normal SPP accuracy (~10-50m)")
elif r3d2 < 1000:
    print(f"  ⚠️  EXCLUDING PRN 9 REDUCES 3D ERROR FROM {r3d:.0f}m TO {r3d2:.1f}m")
    print(f"  → Significant improvement, but residual error remains")
else:
    print(f"  ❌ PRN 9 is NOT the primary cause")
    print(f"  → 3D Error remains at {r3d2:.1f}m even after exclusion")