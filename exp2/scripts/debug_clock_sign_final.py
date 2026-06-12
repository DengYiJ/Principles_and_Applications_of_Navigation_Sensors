"""
最终钟差符号验证
================
验证修正公式: rho_corr = P + c·dt_s  (不是 P - c·dt_s)

GPS观测方程: P = ρ + c·dt_r - c·dt^s
→ P + c·dt^s = ρ + c·dt_r  ← 这才是正确的修正伪距
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

# BESTPOSA 接收机 ECEF (真值)
rx_lat_r = ref_lat * const.D2R
rx_lon_r = ref_lon * const.D2R
sl = math.sin(rx_lat_r)
N_rx = const.A_WGS84 / math.sqrt(1.0 - const.E2_WGS84 * sl * sl)
rx_ecef = np.array([
    (N_rx + ref_h) * math.cos(rx_lat_r) * math.cos(rx_lon_r),
    (N_rx + ref_h) * math.cos(rx_lat_r) * math.sin(rx_lon_r),
    (N_rx * (1.0 - const.E2_WGS84) + ref_h) * sl
])

print("=" * 100)
print("FINAL CLOCK CORRECTION SIGN VERIFICATION")
print("=" * 100)
print(f"\nRef RX ECEF: {rx_ecef}")
print(f"\nGPS Observation Equation: P = ρ + c(dt_r - dt^s)")
print(f"  => P + c·dt^s = ρ + c·dt_r  (corrected pseudorange = geometric range + receiver clock)")

# 计算每颗卫星的两种修正方式
print(f"\n{'PRN':>4s} {'P(raw)':>14s} {'dt_s':>14s} {'GeoRange':>14s} {'P-c·dt_s':>16s} {'-Geo':>10s} {'P+c·dt_s':>16s} {'+Geo':>10s}")
print("  " + "-" * 100)

results_no9 = []
for i in range(N):
    prn = int(eph[i,0])
    Xs, Ys, Zs, Ek, A, e = compute_satellite_position(eph[i], t_obs)
    dt_s = apply_clock_correction(eph[i], t_obs, Ek, A, e)
    raw_pr = pr[i,1]
    
    # 地球自转
    tau = raw_pr / const.C
    Xc = Xs + const.OMEGA_E * tau * Ys
    Yc = Ys - const.OMEGA_E * tau * Xs
    Zc = Zs
    geo = np.linalg.norm(np.array([Xc,Yc,Zc]) - rx_ecef)
    
    minus = raw_pr - const.C * dt_s  # Case A (当前代码)
    plus  = raw_pr + const.C * dt_s  # Case B (正确公式)
    
    d_minus = minus - geo
    d_plus  = plus - geo
    
    status = "OK" if abs(d_plus) < 5000 else "?"
    print(f"{prn:4d} {raw_pr:>14.3f} {dt_s:>14.6e} {geo:>14.3f} {minus:>16.3f} {d_minus:>10.1f} {plus:>16.3f} {d_plus:>10.1f}  [{status}]")
    
    if prn != 9:
        results_no9.append((prn, Xc, Yc, Zc, plus, d_plus))

# =====================================================================
# CASE: P + c·dt_s, 剔除 PRN9 => 预期 ~10m 精度
# =====================================================================
print("\n" + "=" * 100)
print("FIXED SOLUTION: P + c·dt_s, excluding PRN 9")
print("=" * 100)

n = len(results_no9)
sat_pos = np.zeros((n, 3))
rho = np.zeros(n)
for j, (prn, Xc, Yc, Zc, plus, _) in enumerate(results_no9):
    sat_pos[j] = [Xc, Yc, Zc]
    rho[j] = plus

X = least_squares_solution(sat_pos, rho)
lon, lat, h = ecef_to_geodetic(X[0], X[1], X[2])
dE, dN, dU, r2d, r3d = validate_accuracy(lon, lat, h, ref_lon, ref_lat, ref_h)

dx = sat_pos[:,0] - X[0]
dy = sat_pos[:,1] - X[1]
dz = sat_pos[:,2] - X[2]
rho_hat = np.sqrt(dx*dx+dy*dy+dz*dz)
res = rho - (rho_hat + X[3])
rms_res = math.sqrt(np.mean(res**2))

print(f"\n  {'PRN':>4s} {'Elev(°)':>8s} {'CorrectedPr':>16s} {'rho_hat':>16s} {'Residual':>16s}")
print("  " + "-" * 60)
for j, (prn, Xc, Yc, Zc, plus, d_plus) in enumerate(results_no9):
    elev, azim = compute_elev_azim(sat_pos[j], rx_ecef)
    print(f"  {prn:4d} {elev:>8.2f} {rho[j]:>16.3f} {rho_hat[j]:>16.3f} {res[j]:>16.3f}")

print(f"\n  ECEF: X={X[0]:.4f}  Y={X[1]:.4f}  Z={X[2]:.4f}")
print(f"  cdt = {X[3]:.4f} m = {X[3]/const.C:.6e} s")
print(f"  RMS Residual = {rms_res:.1f} m")
print(f"  3D Error = {r3d:.1f} m")

print(f"\n{'='*100}")
print(f"VERDICT")
print(f"{'='*100}")

if r3d < 100:
    print(f"  ✅ FIX CONFIRMED: P + c·dt_s reduces 3D Error to {r3d:.1f} m")
    print(f"  → The current code has a SIGN ERROR in the clock correction")
    print(f"\n  In main.py line 140:")
    print(f"    rho_corr[i] = pr[i, 1] - const.C * dt_s   ← WRONG (subtract)")
    print(f"    should be:")
    print(f"    rho_corr[i] = pr[i, 1] + const.C * dt_s   ← CORRECT (add)")
else:
    print(f"  ❌ Not fixed, 3D Error still {r3d:.1f} m")