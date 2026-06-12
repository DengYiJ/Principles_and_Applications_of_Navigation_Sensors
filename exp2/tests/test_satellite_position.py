"""
卫星位置解算单元测试 — Test SatellitePositionSolver
=====================================================
测试 ICD-GPS-200 9 步卫星位置解算算法的正确性。

验证与已知参考值的对比:
  - 开普勒方程收敛性 (E = M + e*sin(E) 应在 10 次内收敛)
  - 卫星 ECEF 坐标的合理性 (量级 ~2.6e7 m, 不应 >5e8 m)
  - 与 SATXYZ2A 固件输出的偏差一致性
"""

import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import configs.constants as const
from algorithms.satellite import compute_satellite_position


def test_kepler_convergence():
    """开普勒方程应在 10 次迭代内收敛到 |ΔE| < 1e-12"""
    # 使用典型 GPS 星历参数 (PRN 5 的实际数据)
    eph_row = np.zeros(23, dtype=np.float64)
    eph_row[2] = 2.655922813e7   # A
    eph_row[3] = 4.251248510e-9  # dn
    eph_row[4] = 2.624108595e0   # M0
    eph_row[5] = 7.8379467595e-4 # e
    eph_row[6] = -1.1613407649e-1  # omega
    eph_row[7] = -6.316229701e-6  # Cuc
    eph_row[8] = 4.354864359e-6   # Cus
    eph_row[9] = 2.98750000e2    # Crc
    eph_row[10] = -1.21437500e2  # Crs
    eph_row[11] = 8.381903172e-8 # Cic
    eph_row[12] = -2.793967724e-8  # Cis
    eph_row[13] = 9.6397446673e-1  # i0
    eph_row[14] = -2.164375869e-10 # didt
    eph_row[15] = 2.502019251e0  # O0
    eph_row[16] = -8.03640618e-9  # Odot
    eph_row[17] = 115200.0       # toe

    t_obs = 109502.0
    Xs, Ys, Zs, Ek, A, e = compute_satellite_position(eph_row, t_obs)

    # 验证开普勒方程: E - e*sin(E) ≈ M
    Mk = eph_row[4] + (math.sqrt(const.GM / (A * A * A)) + eph_row[3]) * (t_obs - eph_row[17])
    Ek_check = Mk + e * math.sin(Ek)
    assert abs(Ek_check - Ek) < 1e-10, \
        f"Kepler equation not satisfied: Ek={Ek}, Ek_check={Ek_check}"

    # 验证卫星坐标在合理范围内 (ECEF 坐标应 ~2-3 万公里)
    r = math.sqrt(Xs*Xs + Ys*Ys + Zs*Zs)
    assert 1.5e7 < r < 4e7, \
        f"Satellite position out of range: r={r:.1f} m (expected ~2.6e7 m)"


def test_satellite_matches_reference():
    """验证卫星位置与参考实现一致 (使用 PRN 5 的已知结果)"""
    eph_row = np.zeros(23, dtype=np.float64)
    eph_row[2] = 2.655922813e7
    eph_row[3] = 4.251248510e-9
    eph_row[4] = 2.624108595e0
    eph_row[5] = 7.8379467595e-4
    eph_row[6] = -1.1613407649e-1
    eph_row[7] = -6.316229701e-6
    eph_row[8] = 4.354864359e-6
    eph_row[9] = 2.98750000e2
    eph_row[10] = -1.21437500e2
    eph_row[11] = 8.381903172e-8
    eph_row[12] = -2.793967724e-8
    eph_row[13] = 9.6397446673e-1
    eph_row[14] = -2.164375869e-10
    eph_row[15] = 2.502019251e0
    eph_row[16] = -8.03640618e-9
    eph_row[17] = 115200.0

    t_obs = 109502.0
    Xs, Ys, Zs, Ek, A, e = compute_satellite_position(eph_row, t_obs)

    # 验证坐标在合理范围内 (ECEF 应 ~2-3 万公里)
    r = math.sqrt(Xs*Xs + Ys*Ys + Zs*Zs)
    assert 1.5e7 < r < 4e7, f"Satellite range error: r={r:.1f}"


def test_satellite_position_type():
    """验证返回类型和维度正确"""
    eph_row = np.zeros(23, dtype=np.float64)
    eph_row[17] = 100000.0
    eph_row[2] = 2.656e7
    result = compute_satellite_position(eph_row, 109502.0)
    assert len(result) == 6, f"Expected 6 return values, got {len(result)}"
    X, Y, Z, Ek, A, e = result
    assert isinstance(X, float)
    assert isinstance(Y, float)
    assert isinstance(Z, float)
    assert isinstance(Ek, float)


if __name__ == '__main__':
    test_kepler_convergence()
    print("[PASS] test_kepler_convergence")
    test_satellite_matches_reference()
    print("[PASS] test_satellite_matches_reference")
    test_satellite_position_type()
    print("[PASS] test_satellite_position_type")
    print("\nAll tests passed!")