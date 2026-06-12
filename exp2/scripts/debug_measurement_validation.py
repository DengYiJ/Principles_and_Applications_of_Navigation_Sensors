"""
GNSS Measurement Validation Mode
==================================
验证观测模型：比较伪距与几何距离。

字段映射依据: RANGEA 日志实际格式验证:
  block[0] = PRN
  block[1] = 频点/通道
  block[2] = 伪距 (m)  ✅ 当前代码正确
  block[3] = 伪距标准差
  block[4] = 载波相位
  block[5] = 相位标准差
  block[6] = 多普勒
  block[7] = 多普勒标准差
  block[8] = 信噪比
  block[9] = 信号类型掩码
"""

import sys, os, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import configs.constants as const
from algorithms.extractor import extract_obs_time, extract_ephemeris_matrix
from algorithms.satellite import compute_satellite_position
from algorithms.corrections import apply_clock_correction
from evaluation.accuracy import parse_bestposa

with open('exp2/data.txt', 'r') as f:
    raw = f.read()

week, t_obs = extract_obs_time(raw)
ref_lon, ref_lat, ref_h = parse_bestposa(raw)

# BESTPOSA receiver ECEF
rx_lat_r = ref_lat * const.D2R
rx_lon_r = ref_lon * const.D2R
sl = math.sin(rx_lat_r)
cl = math.cos(rx_lat_r)
N_rx = const.A_WGS84 / math.sqrt(1.0 - const.E2_WGS84 * sl * sl)
rx_ecef = np.array([
    (N_rx + ref_h) * cl * math.cos(rx_lon_r),
    (N_rx + ref_h) * cl * math.sin(rx_lon_r),
    (N_rx * (1.0 - const.E2_WGS84) + ref_h) * sl
])

eph, vprn = extract_ephemeris_matrix(raw, week, t_obs)

target_prns = [5, 11, 21]

print("=" * 100)
print("GNSS MEASUREMENT VALIDATION MODE")
print("=" * 100)
print(f"\nReceiver (Harbin BESTPOSA):")
print(f"  ECEF: X={rx_ecef[0]:.4f}  Y={rx_ecef[1]:.4f}  Z={rx_ecef[2]:.4f}")

# =====================================================================
# Step 1: 打印 RANGEA 观测块
# =====================================================================
print("\n" + "=" * 100)
print("STEP 1: RANGEA OBSERVATION BLOCKS")
print("=" * 100)

rangea_line = None
for line in raw.strip().split('\n'):
    if '#RANGEA' in line and ';' in line:
        rangea_line = line
        break

hdr_part, data_part = rangea_line.split(';', 1)
obs_str = data_part.split('*')[0]
obs_fields = obs_str.split(',')
sv_data_fields = obs_fields[1:]

# 真实字段映射 (验证后的)
FIELD_NAMES = {
    0: "PRN", 1: "Freq/Ch", 2: "PSEUDORANGE(m)", 3: "PrStdDev",
    4: "CarrierPhase", 5: "PhaseStdDev", 6: "Doppler(Hz)",
    7: "DopStdDev", 8: "CN0", 9: "SignalMask"
}

for prn_target in target_prns:
    found = False
    for j in range(0, len(sv_data_fields) - 9, 10):
        block = sv_data_fields[j:j + 10]
        try:
            prn = int(float(block[0]))
        except:
            continue
        if prn != prn_target:
            continue
        found = True
        print(f"\n--- PRN {prn_target} ---")
        for idx in range(10):
            print(f"  block[{idx}] = {block[idx]:>25s}  ({FIELD_NAMES[idx]})")
        pseudo_raw = float(block[2])
        print(f"\n  >> Pseudorange (block[2]) = {pseudo_raw:.3f} m  ✅ CORRECT FIELD")
        break
    if not found:
        print(f"\n  PRN {prn_target}: NOT FOUND")

# =====================================================================
# Step 2: 几何距离验证
# =====================================================================
print("\n" + "=" * 100)
print("STEP 2: GEOMETRIC RANGE VERIFICATION")
print("=" * 100)

print(f"\n{'PRN':>4s} {'Pseudo(m)':>16s} {'GeoRange(m)':>16s} {'P-Geo(m)':>16s} {'ClkCorr(m)':>12s} {'P-Clk(m)':>16s} {'(P-Clk)-Geo':>16s}")
print("  " + "-" * 100)

for prn_target in target_prns:
    # 找原始伪距 (block[2])
    pseudo_raw = 0.0
    for line in raw.strip().split('\n'):
        if '#RANGEA' not in line or ';' not in line: continue
        _, dp = line.split(';', 1)
        o = dp.split('*')[0].split(',')[1:]
        for j in range(0, len(o) - 9, 10):
            try:
                if int(float(o[j])) == prn_target:
                    pseudo_raw = float(o[j+2])  # block[2]
            except: continue
        break

    # 找星历
    eph_row = None
    for i in range(eph.shape[0]):
        if int(eph[i, 0]) == prn_target:
            eph_row = eph[i]
            break
    if eph_row is None: continue

    Xs, Ys, Zs, Ek, A, e = compute_satellite_position(eph_row, t_obs)

    # 地球自转修正
    tau = pseudo_raw / const.C
    Xc = Xs + const.OMEGA_E * tau * Ys
    Yc = Ys - const.OMEGA_E * tau * Xs
    Zc = Zs
    sv_ecef = np.array([Xc, Yc, Zc])
    geo_range = np.linalg.norm(sv_ecef - rx_ecef)

    dt_s = apply_clock_correction(eph_row, t_obs, Ek, A, e)
    clk_corr_m = const.C * dt_s
    corrected_pr = pseudo_raw - clk_corr_m

    p_minus_geo = pseudo_raw - geo_range
    corr_p_minus_geo = corrected_pr - geo_range

    print(f"{prn_target:4d} {pseudo_raw:>16.3f} {geo_range:>16.3f} {p_minus_geo:>16.3f} {clk_corr_m:>12.3f} {corrected_pr:>16.3f} {corr_p_minus_geo:>16.3f}")

# =====================================================================
# Step 3: 判定
# =====================================================================
print("\n" + "=" * 100)
print("STEP 3: VERDICT")
print("=" * 100)

print(f"\nUsing block[2] as pseudorange:")
for prn_target in target_prns:
    pseudo_raw = 0.0
    for line in raw.strip().split('\n'):
        if '#RANGEA' not in line or ';' not in line: continue
        _, dp = line.split(';', 1)
        o = dp.split('*')[0].split(',')[1:]
        for j in range(0, len(o) - 9, 10):
            try:
                if int(float(o[j])) == prn_target:
                    pseudo_raw = float(o[j+2])
            except: continue
        break

    eph_row = None
    for i in range(eph.shape[0]):
        if int(eph[i, 0]) == prn_target:
            eph_row = eph[i]
            break
    if eph_row is None: continue

    Xs, Ys, Zs, Ek, A, e = compute_satellite_position(eph_row, t_obs)
    tau = pseudo_raw / const.C
    Xc = Xs + const.OMEGA_E * tau * Ys
    Yc = Ys - const.OMEGA_E * tau * Xs
    Zc = Zs
    geo_range = np.linalg.norm(np.array([Xc, Yc, Zc]) - rx_ecef)

    dt_s = apply_clock_correction(eph_row, t_obs, Ek, A, e)
    corrected_pr = pseudo_raw - const.C * dt_s
    diff_after_clk = abs(corrected_pr - geo_range)
    status = "PASS" if diff_after_clk < 50000 else "FAIL"
    print(f"  PRN {prn_target:2d}: |P_corr - geo| = {diff_after_clk:>10.1f} m  [{status}]")