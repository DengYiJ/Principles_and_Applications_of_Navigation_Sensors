"""
卫星位置解算层 — SV Position Solver
=====================================
Module B: SatellitePositionSolver

基于广播星历参数，按 ICD-GPS-200 9 步标准算法计算卫星 ECEF 坐标。
每个子步骤对应算法设计文档 Stage 1 流程图中的 Step 1-9。

设计依据: 03_algorithm_design.md Stage 2 Module B
"""

from typing import Tuple
import math
import numpy as np
import configs.constants as const


def compute_satellite_position(
    eph_row: np.ndarray,
    t_obs: float
) -> Tuple[float, float, float, float, float, float]:
    """计算单颗卫星的 ECEF 坐标 (Module B)

    按 ICD-GPS-200 20.3.3.4.3 标准的 9 步算法:
        Step 1: 计算相对于星历参考时刻的时间差 tk
        Step 2: 计算平均角速度 n
        Step 3: 计算平近点角 Mk
        Step 4: 开普勒方程迭代求解偏近点角 Ek
        Step 5: 计算真近点角 νk
        Step 6: 计算升交角距 Φk
        Step 7: 摄动改正 δu, δr, δi
        Step 8: 计算升交点经度 Ωk
        Step 9: 轨道坐标 → ECEF 坐标旋转

    Args:
        eph_row: 长度为 23 的星历参数向量
            [PRN, week, A, Δn, M₀, e, ω, C_uc, C_us, C_rc, C_rs,
             C_ic, C_is, i₀, di/dt, Ω₀, Ω̇, t_oe, t_oc, T_GD, a_f0, a_f1, a_f2]
        t_obs: 观测时刻 (GPS 周内秒) [s]

    Returns:
        (X, Y, Z, Ek, A, e)
        - X, Y, Z: ECEF 坐标 [m]
        - Ek: 偏近点角 [rad] (供钟差修正使用)
        - A: 轨道半长轴 [m] (供钟差修正使用)
        - e: 偏心率 (供钟差修正使用)

    Reference:
        ICD-GPS-200 20.3.3.4.3, 03_algorithm_design.md Stage 4
    """
    # ---- 解包星历参数 ----
    A = eph_row[2]       # 轨道半长轴 [m]
    dn = eph_row[3]      # Δn — 平均角速度修正 [rad/s]
    M0 = eph_row[4]      # M₀ — 参考时刻平近点角 [rad]
    e = eph_row[5]       # e — 偏心率
    om = eph_row[6]      # ω — 近地点角距 [rad]
    Cuc = eph_row[7]     # C_uc
    Cus = eph_row[8]     # C_us
    Crc = eph_row[9]     # C_rc
    Crs = eph_row[10]    # C_rs
    Cic = eph_row[11]    # C_ic
    Cis = eph_row[12]    # C_is
    i0 = eph_row[13]     # i₀ — 参考时刻轨道倾角 [rad]
    didt = eph_row[14]   # di/dt — 轨道倾角变化率 [rad/s]
    O0 = eph_row[15]     # Ω₀ — 参考时刻升交点赤经 [rad]
    Odot = eph_row[16]   # Ω̇ — 升交点赤经变化率 [rad/s]
    toe = eph_row[17]    # t_oe — 星历参考时刻 [s]

    # ---- Step 1: 时间差计算 ----
    tk = t_obs - toe

    # 处理周切换 (半周 302400 秒)
    if tk > 302400.0:
        tk -= 604800.0
    elif tk < -302400.0:
        tk += 604800.0

    # ---- Step 2: 平均角速度 ----
    n = math.sqrt(const.GM / (A * A * A)) + dn

    # ---- Step 3: 平近点角 ----
    Mk = M0 + n * tk

    # ---- Step 4: 偏近点角 (开普勒方程迭代) ----
    Ek = Mk
    for _ in range(const.KEPLER_MAX_ITER):
        En = Mk + e * math.sin(Ek)
        if abs(En - Ek) < const.KEPLER_CONV_TOL:
            Ek = En
            break
        Ek = En

    # ---- Step 5: 真近点角 ----
    nu_k = math.atan2(
        math.sqrt(1.0 - e * e) * math.sin(Ek),
        math.cos(Ek) - e
    )

    # ---- Step 6: 升交角距 ----
    Phi_k = nu_k + om

    # ---- Step 7: 摄动改正 ----
    delta_u = Cus * math.sin(2.0 * Phi_k) + Cuc * math.cos(2.0 * Phi_k)
    delta_r = Crs * math.sin(2.0 * Phi_k) + Crc * math.cos(2.0 * Phi_k)
    delta_i = Cis * math.sin(2.0 * Phi_k) + Cic * math.cos(2.0 * Phi_k)

    uk = Phi_k + delta_u      # 改正后升交角距
    rk = A * (1.0 - e * math.cos(Ek)) + delta_r  # 改正后向径
    ik = i0 + delta_i + didt * tk  # 改正后轨道倾角

    # ---- Step 8: 升交点经度 ----
    Omega_k = O0 + (Odot - const.OMEGA_E) * tk - const.OMEGA_E * toe

    # ---- Step 9: 轨道→ECEF 坐标旋转 ----
    xp = rk * math.cos(uk)   # 轨道平面 x 坐标
    yp = rk * math.sin(uk)   # 轨道平面 y 坐标

    cO = math.cos(Omega_k)
    sO = math.sin(Omega_k)
    ci = math.cos(ik)
    si = math.sin(ik)

    Xs = xp * cO - yp * ci * sO
    Ys = xp * sO + yp * ci * cO
    Zs = yp * si

    return Xs, Ys, Zs, Ek, A, e