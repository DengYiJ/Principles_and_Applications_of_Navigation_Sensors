"""
GNSS Residual Analysis Mode
============================
分析 Case 2 (剔除 PRN 9) 收敛后的残差分布。
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

# 构建 Case 2 数据 (剔除 PRN 9)
sv_list = []
for i in range(N):
    prn = int(eph[i, 0])
    if prn == 9:
        continue
    Xs, Ys, Zs, Ek, A, e = compute_satellite_position(eph[i], t_obs)
    dt_s = apply_clock_correction(eph[i], t_obs, Ek, A, e)
    raw_pr = pr[i, 1]
    corrected_pr = raw_pr - const.C * dt_s
    Xc, Yc, Zc = apply_earth_rotation_correction(Xs, Ys, Zs, corrected_pr)
    elev, azim = compute_elev_azim((Xc, Yc, Zc), rx_ecef)
    sv_list.append({'prn': prn, 'elev': elev, 'azim': azim,
                    'corrected_pr': corrected_pr, 'X': Xc, 'Y': Yc, 'Z': Zc})

n = len(sv_list)
sat_pos = np.zeros((n, 3))
rho = np.zeros(n)
for j, sv in enumerate(sv_list):
    sat_pos[j] = [sv['X'], sv['Y'], sv['Z']]
    rho[j] = sv['corrected_pr']

# LSQ
X = least_squares_solution(sat_pos, rho)

# 残差
dx = sat_pos[:,0] - X[0]
dy = sat_pos[:,1] - X[1]
dz = sat_pos[:,2] - X[2]
rho_hat = np.sqrt(dx*dx + dy*dy + dz*dz)
residuals = rho - (rho_hat + X[3])
rms_res = math.sqrt(np.mean(residuals**2))

print("=" * 100)
print("GNSS RESIDUAL ANALYSIS MODE — Case 2 (exclude PRN 9)")
print("=" * 100)

# --- 1. 每颗卫星残差 (按 PRN 升序) ---
print("\n--- 1. SATELLITE RESIDUALS (sorted by PRN) ---")
print(f"\n{'PRN':>4s} {'Elev(°)':>8s} {'CorrPr(m)':>16s} {'rho_hat(m)':>16s} {'Residual(m)':>16s}")
print("  " + "-" * 62)
for j in range(n):
    print(f"{sv_list[j]['prn']:4d} {sv_list[j]['elev']:>8.2f} {rho[j]:>16.3f} {rho_hat[j]:>16.3f} {residuals[j]:>16.3f}")

# --- 2. 按 |residual| 从大到小排序 ---
print("\n--- 2. SATELLITE RESIDUALS (sorted by |residual| DESC) ---")
order = sorted(range(n), key=lambda j: abs(residuals[j]), reverse=True)
print(f"\n{'Rank':>4s} {'PRN':>4s} {'Elev(°)':>8s} {'Residual(m)':>16s} {'|Residual|(m)':>16s} {'Cumul(%)':>10s}")
print("  " + "-" * 60)
total_abs = sum(abs(residuals[j]) for j in range(n))
cumul = 0.0
for rank, j in enumerate(order, 1):
    cumul += abs(residuals[j])
    pct = cumul / total_abs * 100
    print(f"{rank:4d} {sv_list[j]['prn']:4d} {sv_list[j]['elev']:>8.2f} {residuals[j]:>16.3f} {abs(residuals[j]):>16.3f} {pct:>9.1f}")

# --- 3. 统计 ---
print("\n--- 3. RESIDUAL STATISTICS ---")
res_arr = residuals
print(f"\n  Max residual:     {np.max(res_arr):>12.3f} m")
print(f"  Min residual:     {np.min(res_arr):>12.3f} m")
print(f"  Mean residual:    {np.mean(res_arr):>12.3f} m  (should be ~0 from LSQ)")
print(f"  Std residual:     {np.std(res_arr, ddof=1):>12.3f} m")
print(f"  RMS residual:     {rms_res:>12.3f} m")
print(f"  # satellites:     {n:>12d}")

# --- 4. 判断 ---
print("\n--- 4. JUDGMENT ---")

# 找到最大残差的PRN
max_idx = np.argmax(np.abs(res_arr))
max_prn = sv_list[max_idx]['prn']
max_res = res_arr[max_idx]

# 检查占主导的卫星
top3_abs_sum = sum(sorted([abs(r) for r in res_arr], reverse=True)[:3])
top3_ratio = top3_abs_sum / total_abs

print(f"\n  Top 3 |residual| account for {top3_ratio*100:.1f}% of total |residual|")

all_positive = all(r > 0 for r in res_arr)
all_negative = all(r < 0 for r in res_arr)
mixed_signs = not (all_positive or all_negative)

print(f"  All residuals same sign: {'YES (systematic bias)' if not mixed_signs else 'NO (mixed signs)'}")
print(f"  Residual range: {np.min(res_arr):.0f} to {np.max(res_arr):.0f} m")

if top3_ratio > 0.7:
    print(f"\n  ==> TYPE B: FEW SATELLITES DOMINATE RMS")
    print(f"      Top 3 account for {top3_ratio*100:.1f}% of total absolute residual")
    print(f"      Dominant PRN: {max_prn} with |res|={abs(max_res):.0f} m")
else:
    print(f"\n  ==> TYPE A: ALL SATELLITES SHARE COMMON OFFSET")
    print(f"      Top 3 only account for {top3_ratio*100:.1f}% of total")
    print(f"      Residuals spread evenly across all {n} satellites")

print(f"\n  Diagnosis: The {int(rms_res/1000)}km RMS residual is distributed across ALL satellites,")
if top3_ratio < 0.6:
    print(f"  not dominated by 1-2 outliers. This indicates a SYSTEMATIC error")
    print(f"  affecting ALL measurements equally — likely pseudorange bias or")
    print(f"  incorrect field mapping (all rho_corr shifted by a common offset).")
else:
    print(f"  but with {top3_ratio*100:.0f}% concentrated in top 3 satellites.")