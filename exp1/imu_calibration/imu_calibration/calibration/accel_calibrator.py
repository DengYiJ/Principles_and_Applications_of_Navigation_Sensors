"""
加速度计六位置标定模块
对应 02_algorithm_design.md → Pseudocode 1

核心算法：最小二乘法辨识综合误差矩阵 K_a(3×3) 和零偏 D_a(3×1)
"""
from typing import Dict, List, Tuple
import numpy as np
from numpy.linalg import lstsq
from ..common.types import AccelCalibResult
from ..common.pose_tables import build_accel_pose_table_6
from ..common.assertions import assert_physical_bounds, check_condition_number


class AccelCalibrator:
    """
    加速度计六位置标定器
    
    流程：
    Step 1: 从6个位置的重力分量构建输入矩阵 A(6×4)
    Step 2: 对X/Y/Z轴分别执行最小二乘 X = (A^T A)^{-1} A^T · b
    Step 3: 组装 K_a(3×3) 和 D_a(3×1)
    Step 4: 计算拟合残差与矩阵条件数
    """

    def __init__(self, cond_threshold: float = 1e8):
        self.cond_threshold = cond_threshold

    def calibrate(self, acc_means: Dict[str, np.ndarray],
                  pose_table: np.ndarray = None) -> AccelCalibResult:
        """
        执行加速度计六位置标定

        参数:
            acc_means: key=pose_id, value=ndarray[3] 每个位置的加速度均值
            pose_table: shape=(6,3), 6个位置的理论重力分量

        返回:
            AccelCalibResult
        """
        if pose_table is None:
            pose_table = build_accel_pose_table_6()

        # 将acc_means按pose_id排序成(6,3)矩阵
        pose_ids = sorted(acc_means.keys())
        n_poses = len(pose_ids)
        if n_poses < 4:
            raise ValueError(f"Need at least 4 poses for calibration, got {n_poses}")

        # 构建输入矩阵 A (n_poses × 4)
        # 每行 = [ax, ay, az, 1]
        A = np.ones((n_poses, 4), dtype=np.float64)
        for i, pid in enumerate(pose_ids):
            A[i, :3] = pose_table[i]

        # 检查条件数
        check_condition_number(A, self.cond_threshold)

        # 观测值矩阵 B (n_poses × 3)
        B = np.zeros((n_poses, 3), dtype=np.float64)
        for i, pid in enumerate(pose_ids):
            B[i] = acc_means[pid]

        # 最小二乘求解: X = (A^T A)^{-1} A^T B
        # X shape = (4, 3): 每列对应 X/Y/Z 轴的4个参数
        X, residuals_sum, rank, s = lstsq(A, B, rcond=None)

        # 组装 K_a 和 D_a
        # X[0:3, :] 转置 → K_a(3,3)
        # X[3, :] → D_a(3,)
        K_a = X[:3, :].T  # shape (3, 3)
        D_a = X[3, :]     # shape (3,)

        # 计算拟合残差: B_pred = A @ X
        B_pred = A @ X
        residuals = B - B_pred  # shape (n_poses, 3)
        reprojection_error = np.sqrt(np.mean(residuals ** 2))

        # 物理边界断言
        assert_physical_bounds(K_a=K_a, D_a=D_a, label="AccelCalibrator")

        return AccelCalibResult(
            K_a=K_a,
            D_a=D_a,
            residuals=residuals.T,  # (3, n_poses)
            condition_number=float(check_condition_number(A, self.cond_threshold)),
            reprojection_error=float(reprojection_error),
        )

    @staticmethod
    def compute_compensated(acc_measurements: np.ndarray,
                            K_a: np.ndarray, D_a: np.ndarray) -> np.ndarray:
        """
        使用标定参数补偿加速度计测量值
        a_true = K_a^{-1} · (a_measured - D_a)

        参数:
            acc_measurements: shape=(N,3) 或 (3,), 加速度计原始测量值 (m/s²)
            K_a: (3,3) 综合误差矩阵
            D_a: (3,) 零偏向量

        返回:
            acc_compensated: 补偿后的加速度 (m/s²)
        """
        K_a_inv = np.linalg.inv(K_a)
        if acc_measurements.ndim == 1:
            return K_a_inv @ (acc_measurements - D_a)
        else:
            return (K_a_inv @ (acc_measurements - D_a).T).T