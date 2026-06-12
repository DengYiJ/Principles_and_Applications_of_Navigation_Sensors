"""
精度评估 — Accuracy Validation
================================
Module E2: AccuracyValidator

将自解算结果与 BESTPOSA 参考结果对比，计算偏差和统计指标。

设计依据: 03_algorithm_design.md Stage 2 Module E2
"""

from typing import Tuple
import math
import configs.constants as const


def validate_accuracy(
    our_lon: float, our_lat: float, our_h: float,
    ref_lon: float, ref_lat: float, ref_h: float
) -> Tuple[float, float, float, float, float]:
    """精度验证 (Module E2)

    将自解算结果与 BESTPOSA 参考结果对比，计算:
        - ΔE, ΔN, ΔU: 东北天坐标系下的偏差 [m]
        - 2D RMS: 水平均方根误差 [m]
        - 3D Error: 三维位置误差 [m]

    Args:
        our_lon, our_lat, our_h: 自解算结果 [°], [°], [m]
        ref_lon, ref_lat, ref_h: BESTPOSA 参考结果 [°], [°], [m]

    Returns:
        (dE, dN, dU, r2d, r3d)
        - dE: 东向偏差 [m]
        - dN: 北向偏差 [m]
        - dU: 天向偏差 [m]
        - r2d: 2D 水平 RMS [m]
        - r3d: 3D 位置误差 [m]
    """
    # 角度差 [deg]
    dLon = our_lon - ref_lon
    dLat = our_lat - ref_lat
    dH = our_h - ref_h

    # 将角度差转换为米 (在参考点处做局部 ENU 投影)
    ref_lat_rad = ref_lat * const.D2R
    RN = const.A_WGS84 / math.sqrt(
        1.0 - const.E2_WGS84 * math.sin(ref_lat_rad) ** 2
    )

    dN = dLat * const.D2R * RN  # 北向 [m]
    dE = dLon * const.D2R * RN * math.cos(ref_lat_rad)  # 东向 [m]
    dU = dH  # 天向 [m]

    # 统计指标
    r2d = math.sqrt(dN * dN + dE * dE)  # 2D 水平 RMS
    r3d = math.sqrt(dE * dE + dN * dN + dU * dU)  # 3D 误差

    # 打印报告
    print("\n" + "=" * 55)
    print("         POSITION ACCURACY REPORT")
    print("=" * 55)
    print(f"  Solved:    ({our_lon:.8f} deg, {our_lat:.8f} deg, "
          f"{our_h:.4f} m)")
    print(f"  Reference: ({ref_lon:.8f} deg, {ref_lat:.8f} deg, "
          f"{ref_h:.4f} m)")
    print(f"  dLon: {dLon:.8f} deg  ({dE:.4f} m)")
    print(f"  dLat: {dLat:.8f} deg  ({dN:.4f} m)")
    print(f"  dH:   {dU:.4f} m")
    print(f"  2D RMS:   {r2d:.4f} m")
    print(f"  3D Error: {r3d:.4f} m")
    print("=" * 55 + "\n")

    return dE, dN, dU, r2d, r3d


def parse_bestposa(data_text: str) -> Tuple[float, float, float]:
    """从日志中解析 BESTPOSA 参考定位结果

    Args:
        data_text: 原始日志文本

    Returns:
        (ref_lon, ref_lat, ref_h): 参考经度 [°], 纬度 [°], 高度 [m]

    Raises:
        ValueError: 未找到有效的 BESTPOSA 结果
    """
    for line in data_text.split('\n'):
        if 'BESTPOSA' in line and 'SOL_COMPUTED' in line:
            try:
                data_part = line.split(';')[1]
                fields = data_part.split('*')[0].split(',')
                ref_lat = float(fields[2])
                ref_lon = float(fields[3])
                ref_h = float(fields[4])
                return ref_lon, ref_lat, ref_h
            except (IndexError, ValueError):
                continue

    raise ValueError("BESTPOSA reference not found in data")