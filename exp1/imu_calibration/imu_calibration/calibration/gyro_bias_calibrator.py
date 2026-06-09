"""
陀螺仪八位置零偏标定模块
对应 02_algorithm_design.md → Pseudocode 3

核心算法：D_g^(i) = mean_i - K_g · ω_earth^(i)  →  8位置取均值
"""
from typing import Dict
import numpy as np
from ..common.types import GyroBiasCalibResult
from ..common.pose_tables import build_gyro_bias_pose_table_8
from ..common.constants import EARTH_ROTATION_RATE_DEGH
from ..common.assertions import assert_physical_bounds


class GyroBiasCalibrator:
    """
    陀螺仪八位置零偏标定器

    流程：
    Step 1: 利用 K_g 和地球自转投影计算每个位置的零偏估计
    Step 2: 8个位置取均值得到 D_g
    Step 3: 计算标准差评估一致性
    """

    def __init__(self):
        pass

    def calibrate(self, gyro_means: Dict[str, np.ndarray],
                  K_g: np.ndarray,
                  pose_table: np.ndarray = None) -> GyroBiasCalibResult:
        """
        执行陀螺仪八位置零偏标定

        参数:
            gyro_means: key=pose_id, value=ndarray[3] 每个位置的陀螺均值 (°/s)
            K_g: (3,3) 陀螺综合误差矩阵（来自速率标定）
            pose_table: shape=(8,3), 8个位置的地球自转理论投影 (°/s)

        返回:
            GyroBiasCalibResult
        """
        if pose_table is None:
            pose_table = build_gyro_bias_pose_table_8()

        pose_ids = sorted(gyro_means.keys())
        n_poses = len(pose_ids)
        if n_poses < 1:
            raise ValueError("Need at least 1 pose for bias calibration")

        bias_estimates = np.zeros((n_poses, 3), dtype=np.float64)

        for i, pid in enumerate(pose_ids):
            # 式(22): D_g^(i) = mean_i - K_g · ω_earth^(i)
            mean_i = gyro_means[pid]
            omega_earth_i = pose_table[i]
            predicted = K_g @ omega_earth_i  # K_g * ω_earth
            bias_estimates[i] = mean_i - predicted

        # 式(23): D_g = (1/n) Σ D_g^(i)
        D_g = np.mean(bias_estimates, axis=0)
        D_g_std = np.std(bias_estimates, axis=0)

        # 转换为°/h
        D_g_deg_h = D_g * 3600.0

        # 输出每个位置的零偏估计
        print("[GyroBias] Bias estimates per pose:")
        for i, pid in enumerate(pose_ids):
            print(f"  {pid}: D_g={bias_estimates[i]*3600.0} (°/h)")

        print(f"[GyroBias] Final D_g = {D_g} °/s = {D_g_deg_h} °/h")
        print(f"[GyroBias] D_g_std = {D_g_std} °/s = {D_g_std*3600.0} °/h")

        # 物理边界断言
        assert_physical_bounds(D_g=D_g, D_g_unit="deg/s", label="GyroBiasCalibrator")

        return GyroBiasCalibResult(
            D_g=D_g,
            D_g_std=D_g_std,
            bias_per_pose=bias_estimates,
            D_g_deg_h=D_g_deg_h,
        )