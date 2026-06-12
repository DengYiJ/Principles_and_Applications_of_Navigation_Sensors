"""
验证卫星钟差修正符号。

GPS观测方程:
    P = ρ + c(dt_r - dt^s)

其中:
    P    = 测量伪距
    ρ    = 真实几何距离
    dt_r = 接收机钟差 (正=接收机钟快)
    dt^s = 卫星钟差 (正=卫星钟快)

推导:
    ρ = P - c·dt_r + c·dt^s
    P - c·dt^s = ρ + c·dt_r  ← 用于最小二乘的修正伪距

测试两种符号:
    Case A (当前): rho_corr = P - c·dt_s
    Case B (备选): rho_corr = P + c·dt_s
"""

import sys, os, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

import configs.constants as const
from algorithms.extractor import extract_obs_time, extract_ephemeris_matrix, extract_pseudorange_matrix
from algorithms.satellite import compute_satellite_position
from algorithms.corrections import apply_clock_correction, apply_earth_rotation_correction
from algorithms.solver import least_squares_solution
from algorithms.transform import ecef_to_geodetic
from evaluation.accuracy import validate_accuracy, parse_bestposa

# =====================================================================
# 读取数据
# =====================================================================
with open('exp2/data.txt', 'r') as f:
    raw = f.read()

week, t_obs = extract_obs_time(raw)
eph, vprn = extract_ephemeris_matrix(raw, week, t_obs)
pr = extract_pseudorange_matrix(raw, week, vprn)

N = pr.shape[0]
assert N == eph.shape[0]

# Reference
ref_lon, ref_lat, ref_h = parse_bestposa(raw)

# =====================================================================
# 计算卫星位置 (共用部分)
# =====================================================================
sat_pos = np.zeros((N, 3), dtype=np.float64)
ek_list = np.zeros(N, dtype=np.float64)
e_list = np.zeros(N, dtype=np.float64)
a_list = np.zeros(N, dtype=np.float64)

print("=" * 78)
print("CLOCK CORRECTION SIGN VERIFICATION")
print("=" * 78)

for i in range(N):
    Xs, Ys, Zs, Ek, A, e = compute_satellite_position(eph[i], t_obs)
    sat_pos[i] = [Xs, Ys, Zs]
    ek_list[i] = Ek
    e_list[i] = e
    a_list[i] = A

# 打印钟差修正值的符号
print("\n--- Raw clock corrections ---")
for i in range(N):
    dt_s = apply_clock_correction(eph[i], t_obs, ek_list[i], a_list[i], e_list[i])
    print(f"  PRN {int(eph[i,0]):2d}: dt_s = {dt_s:.6e} s  {'(positive = satellite clock AHEAD)' if dt_s > 0 else '(negative = satellite clock BEHIND)'}")

# =====================================================================
# Case A: rho_corr = P - C * dt_s  (当前代码)
# =====================================================================
print("\n" + "=" * 78)
print("CASE A: rho_corr = P - C * dt_s  (CURRENT CODE)")
print("=" * 78)

rho_corr_a = np.zeros(N, dtype=np.float64)
print(f"\n  {'PRN':>4s} | {'P(raw)':>15s} {'dt_s':>15s} {'C*dt_s':>15s} {'P - C*dt_s':>15s}")
print("  " + "-" * 67)
for i in range(N):
    dt_s = apply_clock_correction(eph[i], t_obs, ek_list[i], a_list[i], e_list[i])
    rho_corr_a[i] = pr[i, 1] - const.C * dt_s
    print(f"  {int(eph[i,0]):4d} | {pr[i,1]:15.3f} {dt_s:15.6e} {const.C*dt_s:15.3f} {rho_corr_a[i]:15.3f}")

# 地球自转修正
sat_pos_corr_a = np.zeros_like(sat_pos)
for i in range(N):
    Xc, Yc, Zc = apply_earth_rotation_correction(sat_pos[i,0], sat_pos[i,1], sat_pos[i,2], rho_corr_a[i])
    sat_pos_corr_a[i] = [Xc, Yc, Zc]

X_rx_a = least_squares_solution(sat_pos_corr_a, rho_corr_a)
lon_a, lat_a, h_a = ecef_to_geodetic(X_rx_a[0], X_rx_a[1], X_rx_a[2])
dE_a, dN_a, dU_a, r2d_a, r3d_a = validate_accuracy(lon_a, lat_a, h_a, ref_lon, ref_lat, ref_h)

# =====================================================================
# Case B: rho_corr = P + C * dt_s  (备选符号)
# =====================================================================
print("\n" + "=" * 78)
print("CASE B: rho_corr = P + C * dt_s  (ALTERNATIVE SIGN)")
print("=" * 78)

rho_corr_b = np.zeros(N, dtype=np.float64)
print(f"\n  {'PRN':>4s} | {'P(raw)':>15s} {'dt_s':>15s} {'C*dt_s':>15s} {'P + C*dt_s':>15s}")
print("  " + "-" * 67)
for i in range(N):
    dt_s = apply_clock_correction(eph[i], t_obs, ek_list[i], a_list[i], e_list[i])
    rho_corr_b[i] = pr[i, 1] + const.C * dt_s
    print(f"  {int(eph[i,0]):4d} | {pr[i,1]:15.3f} {dt_s:15.6e} {const.C*dt_s:15.3f} {rho_corr_b[i]:15.3f}")

# 地球自转修正
sat_pos_corr_b = np.zeros_like(sat_pos)
for i in range(N):
    Xc, Yc, Zc = apply_earth_rotation_correction(sat_pos[i,0], sat_pos[i,1], sat_pos[i,2], rho_corr_b[i])
    sat_pos_corr_b[i] = [Xc, Yc, Zc]

X_rx_b = least_squares_solution(sat_pos_corr_b, rho_corr_b)
lon_b, lat_b, h_b = ecef_to_geodetic(X_rx_b[0], X_rx_b[1], X_rx_b[2])
dE_b, dN_b, dU_b, r2d_b, r3d_b = validate_accuracy(lon_b, lat_b, h_b, ref_lon, ref_lat, ref_h)

# =====================================================================
# 对比
# =====================================================================
print("\n" + "=" * 78)
print("COMPARISON SUMMARY")
print("=" * 78)

print(f"""
{'':>30s} {'CASE A (P - C*dt)':>25s} {'CASE B (P + C*dt)':>25s}
{'':>30s} {'='*25} {'='*25}
{'ECEF X (m)':>30s} {X_rx_a[0]:>25.4f} {X_rx_b[0]:>25.4f}
{'ECEF Y (m)':>30s} {X_rx_a[1]:>25.4f} {X_rx_b[1]:>25.4f}
{'ECEF Z (m)':>30s} {X_rx_a[2]:>25.4f} {X_rx_b[2]:>25.4f}
{'Clock bias (s)':>30s} {X_rx_a[3]/const.C:>25.4e} {X_rx_b[3]/const.C:>25.4e}
{'Lon (deg)':>30s} {lon_a:>25.8f} {lon_b:>25.8f}
{'Lat (deg)':>30s} {lat_a:>25.8f} {lat_b:>25.8f}
{'H (m)':>30s} {h_a:>25.4f} {h_b:>25.4f}
{'ΔE (m)':>30s} {dE_a:>25.4f} {dE_b:>25.4f}
{'ΔN (m)':>30s} {dN_a:>25.4f} {dN_b:>25.4f}
{'ΔU (m)':>30s} {dU_a:>25.4f} {dU_b:>25.4f}
{'2D RMS (m)':>30s} {r2d_a:>25.4f} {r2d_b:>25.4f}
{'3D Error (m)':>30s} {r3d_a:>25.4f} {r3d_b:>25.4f}
""")

# 结论
print("CONCLUSION:")
print(f"  CASE A (P - C*dt): 2D RMS = {r2d_a:.1f} m, 3D Error = {r3d_a:.1f} m")
print(f"  CASE B (P + C*dt): 2D RMS = {r2d_b:.1f} m, 3D Error = {r3d_b:.1f} m")

better_case = "A" if r3d_a < r3d_b else "B"
sign_str = "-" if better_case == "A" else "+"
print(f"\n  ==> Case {better_case} produces SMALLER positioning error")
print(f"  ==> Recommended formula: rho_corr = P {sign_str} C * dt_s")
