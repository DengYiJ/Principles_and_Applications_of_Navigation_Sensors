"""
定位解算引擎 — Positioning Engine
===================================
Module D: LeastSquaresSolver

通过迭代最小二乘估计接收机位置和钟差。

设计依据: 03_algorithm_design.md Stage 2 Module D
"""

import numpy as np
import configs.constants as const


def least_squares_solution(
    sat_pos: np.ndarray,
    rho_corr: np.ndarray,
    max_iter: int = const.LSQ_MAX_ITER,
    tol: float = const.LSQ_CONV_TOL
) -> np.ndarray:
    """最小二乘迭代定位 (Module D)

    通过迭代最小二乘估计接收机位置 (X, Y, Z) 和钟差 c·δt。
    初始状态: 地心 (0, 0, 0) + 钟差 0
    迭代: ΔX = (HᵀH)⁻¹Hᵀy

    Args:
        sat_pos: N×3 矩阵, 卫星 ECEF 坐标 [m]
        rho_corr: N 维向量, 修正后伪距 [m]
        max_iter: 最大迭代次数
        tol: 位置收敛阈值 [m]

    Returns:
        state: 4 维向量 [X_rx, Y_rx, Z_rx, cδt]
            - X_rx, Y_rx, Z_rx: 接收机 ECEF 坐标 [m]
            - cδt: 接收机钟差 [m] (距离单位)

    Raises:
        np.linalg.LinalgError: 观测矩阵 H 奇异 (卫星几何分布病态)
        ValueError: 卫星数不足

    Reference:
        指导书 3.4 节, 03_algorithm_design.md Stage 4 Step 8
    """
    N = sat_pos.shape[0]

    if N < const.MIN_SATELLITES:
        raise ValueError(
            f"Insufficient satellites: {N} < {const.MIN_SATELLITES}"
        )

    # 初始状态: 地心 + 零钟差
    X = np.zeros(4, dtype=np.float64)
    converged = False

    for it in range(max_iter):
        # 计算几何距离和方向余弦
        dx = sat_pos[:, 0] - X[0]
        dy = sat_pos[:, 1] - X[1]
        dz = sat_pos[:, 2] - X[2]
        rho_hat = np.sqrt(dx * dx + dy * dy + dz * dz)

        # 构造观测矩阵 H 和残差向量 y
        H = np.zeros((N, 4), dtype=np.float64)
        y = np.zeros(N, dtype=np.float64)

        for j in range(N):
            if rho_hat[j] < 1e-6:
                continue

            # 方向余弦 (视线向量取负) + 钟差列
            H[j, 0] = -dx[j] / rho_hat[j]
            H[j, 1] = -dy[j] / rho_hat[j]
            H[j, 2] = -dz[j] / rho_hat[j]
            H[j, 3] = 1.0

            # 残差 = 修正伪距 - 预测伪距
            y[j] = rho_corr[j] - (rho_hat[j] + X[3])

        # 最小二乘解: ΔX = (HᵀH)⁻¹Hᵀy
        try:
            HtH = H.T @ H
            HtH_inv = np.linalg.inv(HtH)
            dX = HtH_inv @ (H.T @ y)
        except np.linalg.LinAlgError:
            # 退化为伪逆
            dX = np.linalg.pinv(H.T @ H) @ (H.T @ y)

        X += dX

        # 收敛判断 (仅位置分量)
        pos_norm = np.linalg.norm(dX[:3])
        if pos_norm < tol:
            print(f"    Converged: iteration {it + 1}, ||dX||={pos_norm:.2e} m")
            converged = True
            break

    if not converged:
        print(f"    WARNING: LSQ not converged after {max_iter} iterations, "
              f"last ||dX||={np.linalg.norm(dX[:3]):.2e} m")

    return X