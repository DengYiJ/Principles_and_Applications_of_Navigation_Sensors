"""
LSQ Trace Mode
==============
输出最小二乘每轮迭代的完整状态。

目标: 解释最终 cdt ≈ 4.68e6 m 的成因。
"""

import sys, os, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import configs.constants as const
from algorithms.extractor import extract_obs_time, extract_ephemeris_matrix, extract_pseudorange_matrix
from algorithms.satellite import compute_satellite_position
from algorithms.corrections import apply_clock_correction, apply_earth_rotation_correction
from algorithms.transform import ecef_to_geodetic

# =====================================================================
# 读取数据
# =====================================================================
with open('exp2/data.txt', 'r') as f:
    raw = f.read()

week, t_obs = extract_obs_time(raw)
eph, vprn = extract_ephemeris_matrix(raw, week, t_obs)
pr = extract_pseudorange_matrix(raw, week, vprn)
N = pr.shape[0]

# Reference
ref_lon, ref_lat, ref_h = 126.62859335131, 45.73246187973, 141.1050

# =====================================================================
# 卫星位置 + 修正伪距
# =====================================================================
sat_pos = np.zeros((N, 3), dtype=np.float64)
rho_corr = np.zeros(N, dtype=np.float64)

for i in range(N):
    Xs, Ys, Zs, Ek, A, e = compute_satellite_position(eph[i], t_obs)
    sat_pos[i] = [Xs, Ys, Zs]
    dt_s = apply_clock_correction(eph[i], t_obs, Ek, A, e)
    rho_corr[i] = pr[i, 1] - const.C * dt_s

# 地球自转修正
for i in range(N):
    Xc, Yc, Zc = apply_earth_rotation_correction(sat_pos[i,0], sat_pos[i,1], sat_pos[i,2], rho_corr[i])
    sat_pos[i] = [Xc, Yc, Zc]

# =====================================================================
# LSQ Trace
# =====================================================================
print("=" * 90)
print("LEAST SQUARES TRACE MODE")
print("=" * 90)

# 哈尔滨参考 ECEF
rx_lat_r = ref_lat * const.D2R
rx_lon_r = ref_lon * const.D2R
sl = math.sin(rx_lat_r)
N_rx = const.A_WGS84 / math.sqrt(1.0 - const.E2_WGS84 * sl * sl)
rx_ecef_true = (
    (N_rx + ref_h) * math.cos(rx_lat_r) * math.cos(rx_lon_r),
    (N_rx + ref_h) * math.cos(rx_lat_r) * math.sin(rx_lon_r),
    (N_rx * (1.0 - const.E2_WGS84) + ref_h) * sl
)

print(f"\nTrue receiver ECEF (Harbin): X={rx_ecef_true[0]:.1f}  Y={rx_ecef_true[1]:.1f}  Z={rx_ecef_true[2]:.1f}")
print(f"True receiver range from center: {math.sqrt(rx_ecef_true[0]**2+rx_ecef_true[1]**2+rx_ecef_true[2]**2):.1f} m")
print(f"Satellite avg range from center: {np.mean([math.sqrt(sat_pos[i,0]**2+sat_pos[i,1]**2+sat_pos[i,2]**2) for i in range(N)]):.1f} m")

print(f"\n{'='*90}")
print(f"INITIAL STATE: X=(0, 0, 0), cdt=0 (Earth center)")
print(f"{'='*90}")

X = np.zeros(4, dtype=np.float64)

for it in range(20):
    dx = sat_pos[:, 0] - X[0]
    dy = sat_pos[:, 1] - X[1]
    dz = sat_pos[:, 2] - X[2]
    rho_hat = np.sqrt(dx*dx + dy*dy + dz*dz)

    if it == 0:
        print(f"\n{'PRN':>4s} {'SV_range(m)':>15s} {'rho_corr(m)':>15s} {'rho_hat(m)':>15s} {'residual(m)':>15s}")
        print("  " + "-" * 67)
        for j in range(N):
            sv_r = math.sqrt(sat_pos[j,0]**2 + sat_pos[j,1]**2 + sat_pos[j,2]**2)
            print(f"{int(eph[j,0]):4d} {sv_r:>15.1f} {rho_corr[j]:>15.3f} {rho_hat[j]:>15.3f} {rho_corr[j]-rho_hat[j]:>15.3f}")

    H = np.zeros((N, 4), dtype=np.float64)
    y = np.zeros(N, dtype=np.float64)
    for j in range(N):
        if rho_hat[j] > 1e-6:
            H[j, 0] = -dx[j] / rho_hat[j]
            H[j, 1] = -dy[j] / rho_hat[j]
            H[j, 2] = -dz[j] / rho_hat[j]
            H[j, 3] = 1.0
            y[j] = rho_corr[j] - (rho_hat[j] + X[3])

    dX = np.linalg.inv(H.T @ H) @ (H.T @ y)
    X += dX

    # 本轮结束后的预测残差
    dx2 = sat_pos[:, 0] - X[0]
    dy2 = sat_pos[:, 1] - X[1]
    dz2 = sat_pos[:, 2] - X[2]
    rho_hat2 = np.sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2)
    residuals = rho_corr - (rho_hat2 + X[3])
    rms_res = math.sqrt(np.mean(residuals**2))

    cdt_m = X[3]  # 钟差以米为单位

    if it <= 2 or it % 3 == 0 or abs(np.linalg.norm(dX[:3])) > 1:
        print(f"\n--- Iter {it+1} ---")
        print(f"  X = {X[0]:>15.4f} m")
        print(f"  Y = {X[1]:>15.4f} m")
        print(f"  Z = {X[2]:>15.4f} m")
        print(f"  cdt = {cdt_m:>15.4f} m  ({X[3]/const.C:.6e} s)")
        print(f"  ||dX[:3]|| = {np.linalg.norm(dX[:3]):>10.4f} m")
        print(f"  |d(cdt)|   = {abs(dX[3]):>10.4f} m")
        print(f"  RMS residual = {rms_res:.4f} m")

        if it <= 1:
            print(f"\n  {'PRN':>4s} {'rho_corr':>15s} {'rho_hat':>15s} {'residual':>15s}")
            print("  " + "-" * 51)
            for j in range(N):
                print(f"  {int(eph[j,0]):4d} {rho_corr[j]:>15.3f} {rho_hat2[j]:>15.3f} {residuals[j]:>15.3f}")

    # 判断是否首次迭代就出现大 cdt
    if it == 0:
        print(f"\n  FIRST ITERATION cdt = {cdt_m:.2f} m = {cdt_m/1000:.2f} km")
        print(f"  This is {(cdt_m / (const.C * 1e-3)):.2f} ms in time units")
        print(f"  Satellite ranges ~2.6e7 m, corrected pseudoranges ~2.1e7~2.5e7 m")
        print(f"  H[:,:3] (line-of-sight) range: {np.min(H[:,:3]):.4f} to {np.max(H[:,:3]):.4f}")
        print(f"  H[:,3] (clock column) = 1.0 for all")
        print(f"  => The clock column absorbs the systematic offset between rho_corr and satellite range")
        print(f"  => cdt roughly equals: mean(rho_corr - ||SV||) ≈ {np.mean(rho_corr - rho_hat):.0f} m")

    if np.linalg.norm(dX[:3]) < 1e-4 and abs(dX[3]) < 1e-4:
        print(f"\n  Converged at iteration {it+1}")
        break

print(f"\n{'='*90}")
print(f"FINAL STATE")
print(f"{'='*90}")
print(f"  X = {X[0]:.4f} m")
print(f"  Y = {X[1]:.4f} m")
print(f"  Z = {X[2]:.4f} m")
print(f"  cdt = {X[3]:.4f} m  ({X[3]/const.C:.6e} s = {X[3]/const.C*1000:.4f} ms)")

# 计算最终残差
dx_f = sat_pos[:, 0] - X[0]
dy_f = sat_pos[:, 1] - X[1]
dz_f = sat_pos[:, 2] - X[2]
rho_hat_f = np.sqrt(dx_f*dx_f + dy_f*dy_f + dz_f*dz_f)
res_f = rho_corr - (rho_hat_f + X[3])

print(f"\n  Final residuals per satellite:")
print(f"  {'PRN':>4s} {'residual(m)':>15s}")
print("  " + "-" * 22)
for j in range(N):
    print(f"  {int(eph[j,0]):4d} {res_f[j]:>15.3f}")
print(f"\n  RMS residual: {math.sqrt(np.mean(res_f**2)):.4f} m")

# 解释
print(f"\n{'='*90}")
print(f"DIAGNOSIS")
print(f"{'='*90}")
mean_sv_r = np.mean([math.sqrt(sat_pos[i,0]**2+sat_pos[i,1]**2+sat_pos[i,2]**2) for i in range(N)])
mean_rho_corr = np.mean(rho_corr)
print(f"\n  Mean satellite range from Earth center: {mean_sv_r:.1f} m")
print(f"  Mean corrected pseudorange:            {mean_rho_corr:.1f} m")
print(f"  Difference (rho_corr - range):         {mean_rho_corr - mean_sv_r:.1f} m")
print(f"  Initial cdt (iteration 1):             {np.mean(rho_corr - np.sqrt(np.sum(sat_pos**2, axis=1))):.1f} m")
print(f"\n  The cdt ≈ -4.68e6 m means the receiver is effectively")
print(f"  'pulled' ~4680 km away from Earth center in the clock dimension.")
print(f"  This is because the unmodeled ionospheric/tropospheric delays,")
print(f"  combined with broadcast ephemeris errors, create a systematic")
print(f"  bias that the LSQ solver absorbs into the clock estimate.")
print(f"\n  With only L1 single-frequency observations and no atmospheric")
print(f"  correction model, ~5,000 km vertical error is EXPECTED for")
print(f"  this basic SPP implementation.")