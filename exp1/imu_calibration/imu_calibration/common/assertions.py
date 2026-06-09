"""
物理边界断言函数
对应 02_algorithm_design.md Stage 6 ⚡ 物理边界断言
"""
import numpy as np
from .constants import GYRO_RANGE_DEGPS, ACCEL_RANGE_G


def assert_physical_bounds(
    gyro_data: np.ndarray = None,
    accel_data: np.ndarray = None,
    K_a: np.ndarray = None,
    K_g: np.ndarray = None,
    D_a: np.ndarray = None,
    D_g: np.ndarray = None,
    D_g_unit: str = "deg/s",
    label: str = ""
):
    """
    执行所有物理边界断言。
    当数据超出物理合理范围时立即抛出 AssertionError。
    """
    prefix = f"[Assertion {label}]" if label else "[Assertion]"

    # —— 角速率数值物理合理性 ——
    if gyro_data is not None and gyro_data.size > 0:
        abs_max = np.max(np.abs(gyro_data))
        assert abs_max < GYRO_RANGE_DEGPS * 1.1, \
            f"{prefix} Gyro rate exceeds bound: max |ω|={abs_max:.4f} °/s, limit={GYRO_RANGE_DEGPS} °/s"

    # —— 加速度数值物理合理性 ——
    if accel_data is not None and accel_data.size > 0:
        abs_max = np.max(np.abs(accel_data))
        # 数据单位可能是 g 或 m/s²，保守采用阈值 30（对应m/s²）或 3（对应g）
        assert abs_max < 30.0, \
            f"{prefix} Accel magnitude exceeds bound: max |a|={abs_max:.4f}"

    # —— 综合误差矩阵对角线应在合理标度因数范围内 ——
    if K_a is not None:
        diag = np.diag(K_a)
        # 数据单位为g时标度因数约0.1左右（取决于是否归一化），放宽下界
        assert np.all(diag > 0.001) and np.all(diag < 2.0), \
            f"{prefix} K_a diagonal (scalefactors) out of range [0.001,2.0]: {diag}"

    if K_g is not None:
        diag = np.diag(K_g)
        assert np.all((diag > 0.5) & (diag < 1.5)), \
            f"{prefix} K_g diagonal (scalefactors) out of range [0.5,1.5]: {diag}"

    # —— 零偏应在合理范围内 ——
    if D_a is not None:
        assert np.all(np.abs(D_a) < 10), \
            f"{prefix} D_a magnitude > 10 m/s²: {D_a}"

    if D_g is not None:
        if D_g_unit == "deg/s":
            assert np.all(np.abs(D_g) < 1.0), \
                f"{prefix} D_g magnitude > 1.0 °/s: {D_g}"
        elif D_g_unit == "deg/h":
            assert np.all(np.abs(D_g) < 360), \
                f"{prefix} D_g magnitude > 360 °/h: {D_g}"


def check_condition_number(A: np.ndarray, threshold: float = 1e8) -> float:
    """
    计算矩阵条件数并返回。
    若超过阈值，输出WARNING（不抛错，由上层决定是否继续）。
    """
    from numpy.linalg import cond
    c = cond(A)
    if c > threshold:
        import warnings
        warnings.warn(f"Ill-conditioned matrix: cond={c:.2e} > threshold={threshold:.2e}")
    return c


def data_shape_monitor(data: np.ndarray, expected_ndim: int, name: str = "data"):
    """监测数据shape是否符合预期"""
    print(f"[Monitor] {name}: shape={data.shape}, dtype={data.dtype}, "
          f"range=[{data.min():.6f}, {data.max():.6f}]")
    assert data.ndim == expected_ndim, \
        f"{name}: expected ndim={expected_ndim}, got ndim={data.ndim}"