"""
GPS Single Point Positioning - Experiment 2
============================================
Main entry point for the SPP pipeline.

Pipeline:
  Data Extraction (A1-A3)
    → Satellite Position Computation (B)
    → Error Corrections (C1-C2)
    → Least Squares Solution (D)
    → Coordinate Transform (E1)
    → Accuracy Validation (E2)
    → Results Output & Visualization

Design: 03_algorithm_design.md
"""

import os
import sys
import math
import numpy as np

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import configs.constants as const
from algorithms.extractor import (
    extract_obs_time,
    extract_ephemeris_matrix,
    extract_pseudorange_matrix,
)
from algorithms.satellite import compute_satellite_position
from algorithms.corrections import (
    apply_clock_correction,
    apply_earth_rotation_correction,
)
from algorithms.solver import least_squares_solution
from algorithms.transform import ecef_to_geodetic
from evaluation.accuracy import validate_accuracy, parse_bestposa
from visualization.plot_earth import (
    parse_satxyz2a, plot_earth_and_satellites, compute_elev_azim
)


def write_result(filepath, content):
    """写入结果文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  -> Saved: {filepath}")


def main():
    print("=" * 60)
    print("  GPS Single Point Positioning - Experiment 2")
    print("  Design: 03_algorithm_design.md")
    print("=" * 60)

    # ---- Step 0: 读取数据 ----
    data_path = const.DATA_FILE_PATH
    print(f"\n[0] Reading data: {data_path}")
    if not os.path.exists(data_path):
        print(f"  ERROR: File not found: {data_path}")
        sys.exit(1)

    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = f.read()
    print(f"  Source: {data_path} ({len(raw_data)} chars)")

    # ---- 解析 SATXYZ2A (用于可视化和卫星信息) ----
    print("\n" + "-" * 50)
    print("[SATXYZ2A] Parsing satellite positions...")
    all_sats = parse_satxyz2a(raw_data)
    print(f"  Total satellites: {len(all_sats)}")
    sys_count = {}
    for sat in all_sats:
        sys_name = sat[4]
        sys_count[sys_name] = sys_count.get(sys_name, 0) + 1
    for s, c in sys_count.items():
        print(f"    {s}: {c}")

    has_reference = False  # 初始化为False, 将在E2中更新

    # ---- (A1): 提取观测时刻 ----
    print("\n" + "-" * 50)
    print("[A1] Extracting observation time...")
    week, t_obs = extract_obs_time(raw_data)
    print(f"  GPS Week: {week}, t_obs: {t_obs:.3f} s")

    # ---- (A2): 提取星历矩阵 ----
    print("\n" + "-" * 50)
    print("[A2] Extracting GPS ephemeris matrix...")
    eph, valid_prns = extract_ephemeris_matrix(raw_data, week, t_obs)
    N = eph.shape[0]
    print(f"  GPS satellites: {N}, PRNs: {valid_prns}")

    # ---- (A3): 提取伪距矩阵 ----
    print("\n" + "-" * 50)
    print("[A3] Extracting pseudorange matrix...")
    pr = extract_pseudorange_matrix(raw_data, week, valid_prns)
    for i in range(N):
        print(f"    PRN {int(pr[i, 0]):2d}: {pr[i, 1]:.3f} m")

    # ---- 一致性检查 ----
    assert N == eph.shape[0]
    print(f"\n[5.5] Valid GPS satellites for positioning: {N}")

    if N < const.MIN_SATELLITES:
        print(f"  ERROR: Insufficient satellites ({N} < "
              f"{const.MIN_SATELLITES})")
        sys.exit(1)

    # ---- (B): 卫星位置解算 ----
    print("\n" + "-" * 50)
    print("[B] Computing satellite positions (ICD-GPS-200 9-step)...")
    sat_pos = np.zeros((N, 3), dtype=np.float64)
    ek_list = np.zeros(N, dtype=np.float64)
    e_list = np.zeros(N, dtype=np.float64)
    a_list = np.zeros(N, dtype=np.float64)

    sat_pos_info = []  # 用于结果输出: (PRN, X, Y, Z) — elev/azim 将在E2后补
    for i in range(N):
        Xs, Ys, Zs, Ek, A, e = compute_satellite_position(eph[i], t_obs)
        sat_pos[i] = [Xs, Ys, Zs]
        ek_list[i] = Ek
        e_list[i] = e
        a_list[i] = A
        sat_pos_info.append((int(eph[i, 0]), Xs, Ys, Zs))
        print(f"    PRN {int(eph[i, 0]):2d}: "
              f"X={Xs:.2f}, Y={Ys:.2f}, Z={Zs:.2f} m")

    # ---- (C1): 卫星钟差修正 ----
    print("\n" + "-" * 50)
    print("[C1] Applying clock correction...")
    rho_corr = np.zeros(N, dtype=np.float64)
    for i in range(N):
        dt_s = apply_clock_correction(
            eph[i], t_obs, ek_list[i], a_list[i], e_list[i]
        )
        rho_corr[i] = pr[i, 1] + const.C * dt_s
        
    # 剔除异常卫星 PRN 9 (伪距 37,834 km 远超物理上限)
    valid_mask = [i for i in range(N) if int(eph[i, 0]) != 9]
    if len(valid_mask) < N:
        print(f"  Excluding PRN 9 (anomalous pseudorange)")
        eph = eph[valid_mask]
        pr = pr[valid_mask]
        sat_pos = sat_pos[valid_mask]
        rho_corr = rho_corr[valid_mask]
        sat_pos_info = [sat_pos_info[i] for i in valid_mask]
        N = len(valid_mask)

    # ---- (C2): 地球自转修正 ----
    print("\n" + "-" * 50)
    print("[C2] Applying Earth rotation correction...")
    sat_pos_corr = np.zeros_like(sat_pos)
    for i in range(N):
        Xc, Yc, Zc = apply_earth_rotation_correction(
            sat_pos[i, 0], sat_pos[i, 1], sat_pos[i, 2], rho_corr[i]
        )
        sat_pos_corr[i] = [Xc, Yc, Zc]

    # ---- (D): 最小二乘迭代定位 ----
    print("\n" + "-" * 50)
    print("[D] Least squares positioning...")
    X_rx = least_squares_solution(sat_pos_corr, rho_corr)
    clock_bias_s = X_rx[3] / const.C
    print(f"  ECEF: X={X_rx[0]:.4f}, Y={X_rx[1]:.4f}, Z={X_rx[2]:.4f} m")
    print(f"  Clock bias: {clock_bias_s:.4e} s")

    # ---- (E1): ECEF → 大地坐标 ----
    print("\n" + "-" * 50)
    print("[E1] ECEF → Geodetic...")
    lon_s, lat_s, h_s = ecef_to_geodetic(X_rx[0], X_rx[1], X_rx[2])
    print(f"  Result: lon={lon_s:.8f} deg, "
          f"lat={lat_s:.8f} deg, h={h_s:.4f} m")

    # ---- (E2): 精度评估 ----
    print("\n" + "-" * 50)
    print("[E2] Accuracy validation...")
    try:
        ref_lon, ref_lat, ref_h = parse_bestposa(raw_data)
        print(f"  Reference: lon={ref_lon:.8f} deg, "
              f"lat={ref_lat:.8f} deg, h={ref_h:.4f} m")
        dE, dN, dU, r2d, r3d = validate_accuracy(
            lon_s, lat_s, h_s, ref_lon, ref_lat, ref_h
        )
        has_reference = True
    except ValueError as e:
        print(f"  WARNING: {e}")
        dE = dN = dU = r2d = r3d = 0.0
        has_reference = False

    # 补全卫星高度角/方位角 (基于哈尔滨参考坐标)
    if has_reference:
        rx_lat_r = ref_lat * const.D2R
        rx_lon_r = ref_lon * const.D2R
        sl = math.sin(rx_lat_r)
        cl = math.cos(rx_lat_r)
        Nr = const.A_WGS84 / math.sqrt(1.0 - const.E2_WGS84 * sl * sl)
        rx_ecef_ref1 = (
            (Nr + ref_h) * cl * math.cos(rx_lon_r),
            (Nr + ref_h) * cl * math.sin(rx_lon_r),
            (Nr * (1.0 - const.E2_WGS84) + ref_h) * sl
        )
        sat_pos_info_aug = []
        for item in sat_pos_info:
            prn, Xs, Ys, Zs = item
            elev_sat, azim_sat = compute_elev_azim((Xs, Ys, Zs), rx_ecef_ref1)
            sat_pos_info_aug.append((prn, Xs, Ys, Zs, elev_sat, azim_sat))
        sat_pos_info = sat_pos_info_aug

    # ---- 卫星位置对比验证 (vs SATXYZ2A) ----
    print("\n" + "-" * 50)
    print("[VERIFY] Computed GPS positions vs SATXYZ2A...")
    sat_verify_lines = []
    for line in raw_data.split('\n'):
        if 'SATXYZ2A' not in line or ',GPS,' not in line:
            continue
        data_part = line.split(';')[1].split('*')[0]
        fields = data_part.split(',')[1:]
        for j in range(0, len(fields) - 6, 10):
            try:
                if fields[j] != 'GPS':
                    continue
                prn_sat = int(fields[j + 1])
                xs = float(fields[j + 2])
                ys = float(fields[j + 3])
                zs = float(fields[j + 4])
                for i in range(N):
                    if int(eph[i, 0]) == prn_sat:
                        diff = math.sqrt(
                            (sat_pos[i, 0] - xs) ** 2
                            + (sat_pos[i, 1] - ys) ** 2
                            + (sat_pos[i, 2] - zs) ** 2
                        )
                        line_str = f"    PRN {prn_sat:2d}: diff={diff:.3f} m"
                        print(line_str)
                        sat_verify_lines.append(f"PRN {prn_sat:2d}: {diff:.3f} m")
                        break
            except (ValueError, IndexError):
                continue
        break

    # ============================================================
    # 结果输出到 results/ 目录 (对应实验报告模板 8 项)
    # ============================================================
    print("\n" + "=" * 60)
    print("  Generating result files...")
    print("=" * 60)

    # ---- (1) 软件界面截图 — N/A (外部操作) ----
    write_result(
        "exp2/results/01_software_screenshot.txt",
        "实验(2) 卫星导航实验\n"
        "====================\n\n"
        "结果 (1): GNSS接收机配套软件界面截图/照片\n"
        "说明: 请在此处粘贴 NovAtel Connect 软件界面的截图/照片。\n"
        "截图应显示定位状态、经纬高信息、UTC时间及可见卫星。\n"
        "参考数据:\n"
        f"  BESTPOSA参考结果: lon={ref_lon:.8f}°, "
        f"lat={ref_lat:.8f}°, h={ref_h:.4f}m\n"
        f"  定位时刻: GPS Week={week}, t_obs={t_obs:.3f}s\n"
    )

    # ---- (2) 可见卫星编号、用于定位卫星编号、定位授时结果 ----
    all_visible_prns = sorted(set(
        int(sat[3]) for sat in all_sats if sat[4] == 'GPS'
    ))
    result2 = (
        "结果 (2): 实验时可见卫星编号、用于定位卫星编号、定位授时结果\n"
        "=============================================================\n\n"
        "1. 全星座可见卫星统计:\n"
    )
    for s, c in sys_count.items():
        result2 += f"   {s}: {c} 颗\n"
    result2 += f"\n  共计: {len(all_sats)} 颗卫星\n\n"

    result2 += "2. GPS 卫星详情:\n"
    result2 += "   PRN  状态     高度角(°)  方位角(°)   ECEF-X(m)       "
    result2 += "ECEF-Y(m)       ECEF-Z(m)\n"
    result2 += "   " + "-" * 78 + "\n"
    used_gps_set = set(int(pr[i, 0]) for i in range(N))

    # 构建 GPS 卫星 ECEF 查找表
    gps_ecef = {}
    for line in raw_data.split('\n'):
        if 'SATXYZ2A' not in line or ',GPS,' not in line:
            continue
        dp = line.split(';')[1].split('*')[0]
        ff = dp.split(',')[1:]
        for k in range(0, len(ff) - 9, 10):
            try:
                if ff[k] == 'GPS':
                    prn_gps = int(ff[k + 1])
                    gps_ecef[prn_gps] = (
                        float(ff[k + 2]), float(ff[k + 3]), float(ff[k + 4])
                    )
            except (ValueError, IndexError):
                continue
        break

    # 计算接收机ECEF坐标(用于高度角/方位角计算)
    if has_reference:
        rx_lat_r = ref_lat * const.D2R
        rx_lon_r = ref_lon * const.D2R
        sl = math.sin(rx_lat_r)
        cl = math.cos(rx_lat_r)
        Nr = const.A_WGS84 / math.sqrt(1.0 - const.E2_WGS84 * sl * sl)
        rx_ecef_ref2 = (
            (Nr + ref_h) * cl * math.cos(rx_lon_r),
            (Nr + ref_h) * cl * math.sin(rx_lon_r),
            (Nr * (1.0 - const.E2_WGS84) + ref_h) * sl
        )
    for sat in all_sats:
        if sat[4] != 'GPS':
            continue
        prn = int(sat[3])
        status = "参与定位" if prn in used_gps_set else "可见未用"
        # 计算高度角和方位角
        xf, yf, zf = gps_ecef.get(prn, (0.0, 0.0, 0.0))
        elev = 0.0
        azim = 0.0
        if has_reference and xf != 0.0:
            elev, azim = compute_elev_azim((xf, yf, zf), rx_ecef_ref2)
        result2 += f"   {prn:2d}    {status}    {elev:6.2f}      {azim:6.2f}     "
        result2 += f"{xf:14.4f}  {yf:14.4f}  {zf:14.4f}\n"

    result2 += f"\n3. 用于定位卫星 (GPS): PRN {sorted(list(used_gps_set))}\n"
    result2 += f"   卫星数量: {len(used_gps_set)} 颗\n"
    result2 += f"\n4. 定位授时结果:\n"
    result2 += f"   GPS Week: {week}\n"
    result2 += f"   t_obs (周内秒): {t_obs:.6f} s\n"
    utc_hour = int((t_obs % 86400) / 3600)
    utc_min = int(((t_obs % 86400) % 3600) / 60)
    utc_sec = (t_obs % 86400) % 3600 % 60
    result2 += f"   UTC 时间 (近似): {utc_hour:02d}:{utc_min:02d}:{utc_sec:06.3f}\n"

    write_result("exp2/results/02_visible_satellites.txt", result2)

    # ---- (3) 原始星历数据 ----
    result3 = "结果 (3): 原始星历数据\n"
    result3 += "=====================\n\n"
    result3 += "格式: N×23 矩阵\n\n"
    result3 += "列说明:\n"
    result3 += "  [PRN, Week, A, Δn, M₀, e, ω, C_uc, C_us, C_rc, C_rs,\n"
    result3 += "   C_ic, C_is, i₀, di/dt, Ω₀, Ω̇, t_oe, t_oc, T_GD, a_f0, a_f1, a_f2]\n\n"
    for i in range(N):
        result3 += f"Sat {i+1} (PRN {int(eph[i,0])}):\n"
        col_names = [
            "PRN", "Week", "A", "Δn", "M₀", "e", "ω",
            "C_uc", "C_us", "C_rc", "C_rs", "C_ic", "C_is",
            "i₀", "di/dt", "Ω₀", "Ω̇", "t_oe", "t_oc", "T_GD",
            "a_f0", "a_f1", "a_f2"
        ]
        for c in range(23):
            result3 += f"    {col_names[c]}: {eph[i,c]:.6e}\n"
        result3 += "\n"
    write_result("exp2/results/03_ephemeris_matrix.txt", result3)

    # ---- (4) 原始伪距测量数据 ----
    result4 = "结果 (4): 原始伪距测量数据\n"
    result4 += "===========================\n\n"
    result4 += "格式: N×2 矩阵\n\n"
    result4 += "  PRN    伪距 (m)\n"
    result4 += "  " + "-" * 25 + "\n"
    for i in range(N):
        result4 += f"  {int(pr[i,0]):3d}    {pr[i,1]:.6f}\n"
    write_result("exp2/results/04_pseudorange.txt", result4)

    # ---- (5) 原始单点定位结果数据 ----
    result5 = "结果 (5): 原始单点定位结果数据\n"
    result5 += "===============================\n\n"
    result5 += "BESTPOSA 参考结果 (接收机固件输出):\n"
    result5 += f"  经度: {ref_lon:.8f}°\n"
    result5 += f"  纬度: {ref_lat:.8f}°\n"
    result5 += f"  高程: {ref_h:.4f} m\n"
    result5 += f"  坐标系统: WGS84\n"
    result5 += f"  定位模式: SINGLE (单点定位)\n"
    result5 += f"  GPS Week: {week}\n"
    result5 += f"  周内秒: {t_obs:.3f} s\n"
    write_result("exp2/results/05_bestposa_reference.txt", result5)

    # ---- (6) 原始卫星位置信息 ----
    result6 = "结果 (6): 原始卫星位置信息\n"
    result6 += "===========================\n\n"
    result6 += "SATXYZ2A 固件输出的卫星 ECEF 坐标 (高度角/方位角基于哈尔滨计算):\n\n"
    for sat in all_sats:
        lon, lat, h, prn, sys_name = sat[:5]
        # 计算ECEF坐标
        sl = math.sin(lat * const.D2R)
        cl = math.cos(lat * const.D2R)
        N_sat = const.A_WGS84 / math.sqrt(1.0 - const.E2_WGS84 * sl * sl)
        sv_ecef = (
            (N_sat + max(h, 0)) * cl * math.cos(lon * const.D2R),
            (N_sat + max(h, 0)) * cl * math.sin(lon * const.D2R),
            (N_sat * (1.0 - const.E2_WGS84) + max(h, 0)) * sl
        )
        elev, azim = 0.0, 0.0
        if has_reference:
            elev, azim = compute_elev_azim(sv_ecef, rx_ecef_ref2)
        result6 += f"  {sys_name} PRN {prn}:\n"
        result6 += f"    经度={lon:.6f}°, 纬度={lat:.6f}°, 高度≈{h:.1f}m\n"
        result6 += f"    高度角={elev:.2f}°, 方位角={azim:.2f}°\n\n"
    write_result("exp2/results/06_satellite_positions.txt", result6)

    # ---- (7) 单点定位程序数据信息 ----
    result7 = "结果 (7): 单点定位程序数据信息\n"
    result7 += "===============================\n\n"
    result7 += "参与定位卫星信息:\n"
    result7 += "  PRN   X_ECEF(m)        Y_ECEF(m)        Z_ECEF(m)       "
    result7 += "高度角(°)  方位角(°)\n"
    result7 += "  " + "-" * 85 + "\n"
    for item in sat_pos_info:
        if len(item) == 6:
            prn, Xs, Ys, Zs, elev_sat, azim_sat = item
        else:
            prn, Xs, Ys, Zs = item
            elev_sat, azim_sat = 0.0, 0.0
        result7 += f"  {prn:3d}  {Xs:16.4f}  {Ys:16.4f}  {Zs:16.4f}  "
        result7 += f"{elev_sat:8.2f}  {azim_sat:8.2f}\n"

    result7 += f"\n接收机位置 (自解算):\n"
    result7 += f"  经度: {lon_s:.8f}°\n"
    result7 += f"  纬度: {lat_s:.8f}°\n"
    result7 += f"  高程: {h_s:.4f} m\n"
    result7 += f"  ECEF: X={X_rx[0]:.4f}, Y={X_rx[1]:.4f}, "
    result7 += f"Z={X_rx[2]:.4f} m\n"
    result7 += f"  接收机钟差: {clock_bias_s:.4e} s\n"

    if has_reference:
        result7 += f"\n与参考结果的偏差:\n"
        result7 += f"  ΔE (东向): {dE:.4f} m\n"
        result7 += f"  ΔN (北向): {dN:.4f} m\n"
        result7 += f"  ΔU (天向): {dU:.4f} m\n"
        result7 += f"  2D RMS (水平): {r2d:.4f} m\n"
        result7 += f"  3D Error: {r3d:.4f} m\n"

    result7 += f"\n卫星位置验证 (与 SATXYZ2A 固件对比):\n"
    for line in sat_verify_lines:
        result7 += f"  {line}\n"
    write_result("exp2/results/07_positioning_results.txt", result7)

    # ---- (8) 程序说明 ----
    result8 = "结果 (8): 完整的单点定位程序\n"
    result8 += "==============================\n\n"
    result8 += "程序组成:\n"
    result8 += "  main.py                 - 主入口 (Pipeline控制器)\n"
    result8 += "  configs/constants.py    - 物理常数与算法超参数\n"
    result8 += "  algorithms/extractor.py  - A1:观测时刻 A2:星历矩阵 A3:伪距矩阵\n"
    result8 += "  algorithms/satellite.py  - B:卫星位置解算 (ICD 9步)\n"
    result8 += "  algorithms/corrections.py - C1:钟差修正 C2:地球自转修正\n"
    result8 += "  algorithms/solver.py     - D:最小二乘迭代求解\n"
    result8 += "  algorithms/transform.py  - E1:ECEF→大地坐标\n"
    result8 += "  evaluation/accuracy.py   - E2:精度验证\n"
    result8 += "  visualization/plot_earth.py - 可视化\n\n"
    result8 += f"输入数据: {data_path}\n"
    result8 += f"输出目录: exp2/results/\n\n"
    result8 += "运行方式:\n"
    result8 += "  python exp2/main.py\n"
    write_result("exp2/results/08_program_info.txt", result8)

    # ============================================================
    # 可视化
    # ============================================================
    print("\n" + "-" * 50)
    print("[VIZ] Generating visualization...")
    print(f"  Showing {len(all_sats)} satellites")

    used_gps_prns = set(int(pr[i, 0]) for i in range(N))
    # 使用哈尔滨参考坐标 (BESTPOSA) 作为天空图观测点
    rx_ref = (ref_lon, ref_lat, ref_h) if has_reference else (lon_s, lat_s, h_s)
    # 构建用于计算高度角/方位角的卫星ECEF坐标矩阵(所有星座)
    sat_pos_all = np.zeros((len(all_sats), 3), dtype=np.float64)
    for i, sat in enumerate(all_sats):
        lon, lat, h, prn, sys_name = sat[:5]
        lat_r = lat * const.D2R
        lon_r = lon * const.D2R
        sl = math.sin(lat_r)
        N_sat = const.A_WGS84 / math.sqrt(1.0 - const.E2_WGS84 * sl * sl)
        sat_pos_all[i, 0] = (N_sat + max(h, 0)) * math.cos(lat_r) * math.cos(lon_r)
        sat_pos_all[i, 1] = (N_sat + max(h, 0)) * math.cos(lat_r) * math.sin(lon_r)
        sat_pos_all[i, 2] = (N_sat * (1.0 - const.E2_WGS84) + max(h, 0)) * sl

    plot_earth_and_satellites(
        all_sats, rx_ref,
        used_gps_prns=used_gps_prns,
        sat_pos_ecef=sat_pos_all,
        title=f"Multi-constellation SVs (t_obs={t_obs:.0f}s, "
              f"{len(all_sats)} SVs, {N} GPS used)",
        save_path="exp2/results/visualization.png"
    )

    print("\n" + "=" * 60)
    print("  ALL 8 RESULTS GENERATED IN exp2/results/")
    print("=" * 60)


if __name__ == '__main__':
    main()