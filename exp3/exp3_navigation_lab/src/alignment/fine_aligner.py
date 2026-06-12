"""
卡尔曼滤波精对准模块 (8维简化版)
===============================
8维状态: X = [φ_E, φ_N, δv_E, δv_N, ε_x, ε_y, ∇_x, ∇_y]

静基座下不可观测的状态被移除:
  φ_U (航向误差) — 仅在 ω_ie 耦合下微弱可观测
  δv_U (垂速误差) — 由 f^n+g^n≈2g 主导，不含对准信息
  ε_z, ∇_z — 不影响水平通道

量测: z = [δv_E, δv_N] (仅水平速度)
"""

import numpy as np
from typing import Tuple, Optional
from src.utils.quaternion import quat_update
from src.utils.dcm import dcm_from_quat, dcm_to_quat, skew, dcm_orthogonalize
from src.utils.euler_angles import dcm_to_euler312, rad2deg, deg2rad
from src.utils.earth_model import gravity_n, earth_rate_n, WGS84_OMEGA
from src.utils.kalman_filter import LinearKF
from src.utils.assert_helpers import assert_dcm_valid


def build_F8(Cbn: np.ndarray, f_n: np.ndarray, wie_n: np.ndarray) -> np.ndarray:
    """
    构建8维SINS误差状态系统矩阵 F (8×8)
    状态: X = [φ_E, φ_N | δv_E, δv_N | ε_x, ε_y | ∇_x, ∇_y]

    F[0:2, 0:2] = 地球自转耦合 (φ → φ)  — 仅水平分量
    F[0:2, 2:4] = I₂                   — δv → φ (单位阵)
    F[0:2, 4:6] = Cbn[0:2, 0:2]       — ε → φ (水平陀螺零偏)
    F[2:4, 0:2] = -skew(f_n)[h]       — φ → δv (比力耦合)
    F[2:4, 6:8] = Cbn[1:3, 0:2]       — ∇ → δv (水平加计零偏)
    """
    F = np.zeros((8, 8))

    # F[0:2, 0:2]: Earth rate coupling for horizontal φ
    # -skew(wie_n) = [[0, w_U, -w_N], [-w_U, 0, w_E], [w_N, -w_E, 0]]
    # Horizontal (φ_E, φ_N): [[0, w_U], [-w_U, 0]]
    F[0, 1] = wie_n[2]   # φ_E → φ_N coupling
    F[1, 0] = -wie_n[2]  # φ_N → φ_E coupling

    # F[0:2, 2:4] = I₂
    F[0:2, 2:4] = np.eye(2)

    # F[0:2, 4:6] = Cbn[0:2, 0:2] (ε_x, ε_y → φ_E, φ_N)
    F[0:2, 4:6] = Cbn[0:2, 0:2]

    # F[2:4, 0:2]: specific force coupling
    # -skew(f_n)[1:3, 0:2] where f_n = [f_E, f_N, f_U]
    # -skew = [[0, f_U, -f_N], [-f_U, 0, f_E], [f_N, -f_E, 0]]
    # Rows for δv_E, δv_N (rows 1,2): [[-f_U, 0], [0, -f_U]] for cols φ_E, φ_N
    F[2:4, 0:2] = np.array([[0, f_n[2]], [-f_n[2], 0]])

    # F[2:4, 6:8] = Cbn[1:3, 0:2] (∇_x, ∇_y → δv_E, δv_N)
    F[2:4, 6:8] = Cbn[1:3, 0:2]

    return F


def build_H8() -> np.ndarray:
    """量测矩阵 H (2×8): 仅观测水平速度 δv_E, δv_N"""
    H = np.zeros((2, 8))
    H[0:2, 2:4] = np.eye(2)
    return H


def discretize_F8(F: np.ndarray, dt: float) -> np.ndarray:
    """F离散化: Phi ≈ I + F·dt + 0.5·(F·dt)^2"""
    Fdt = F * dt
    return np.eye(8) + Fdt + 0.5 * Fdt @ Fdt


class FineAligner:
    """
    精对准器 — 8维线性卡尔曼滤波

    状态量: X = [φ_E, φ_N, δv_E, δv_N, ε_x, ε_y, ∇_x, ∇_y] (8,)
    量测量: z = [δv_E, δv_N] (静基座真实速度=0)

    航向角 φ_U 锁定于粗对准结果，ε_z, ∇_z, δv_U 因不可观测而移除
    """

    def __init__(self, latitude_deg: float,
                 imu_err: Tuple[float, float, float, float] = (0.002, 20.0, 0.001, 10.0)):
        self.lat_rad = np.deg2rad(latitude_deg)
        self.gyro_bias, self.acc_bias, self.gyro_arw, self.acc_vrw = imu_err

        self.kf = LinearKF(dim_x=8, dim_z=2)
        self.kf.H = build_H8()

        self.att_history = []
        self.vel_history = []
        self._gyro_bias_est = np.zeros(3)   # 完整3轴 (ε_z=0)
        self._acc_bias_est = np.zeros(3)    # 完整3轴 (∇_z=0)

        print(f"[FineAligner] 8-state KF: latitude={latitude_deg}°")

    def _init_covariance(self, init_cov_config: Optional[dict] = None):
        """初始化 P₀, Q, R"""
        if init_cov_config is None:
            init_cov_config = {}

        phi_deg = init_cov_config.get('phi_deg', 1.0)
        dv_mps = init_cov_config.get('dv_mps', 0.1)
        gyro_bias_dps = init_cov_config.get('gyro_bias_dps', 0.01)
        acc_bias_mps2 = init_cov_config.get('acc_bias_mps2', 0.0001)

        phi_rad = deg2rad(phi_deg)
        gyro_bias_rad = deg2rad(gyro_bias_dps)

        # 8维 P₀
        P_diag = np.array([
            phi_rad**2, phi_rad**2,              # φ_E, φ_N
            dv_mps**2, dv_mps**2,                # δv_E, δv_N
            gyro_bias_rad**2, gyro_bias_rad**2,  # ε_x, ε_y
            acc_bias_mps2**2, acc_bias_mps2**2   # ∇_x, ∇_y
        ])
        self.kf.P = np.diag(P_diag)

        # 8维 Q (连续形式)
        arw_rad = deg2rad(self.gyro_arw / 60.0)
        vrw_m_s = self.acc_vrw / 60.0
        Q_diag = np.array([
            arw_rad**2, arw_rad**2,     # φ 角度随机游走
            vrw_m_s**2, vrw_m_s**2,     # δv 速度随机游走
            0.0, 0.0,                   # ε (不驱动)
            0.0, 0.0                    # ∇ (不驱动)
        ])
        self.kf.Q = np.diag(Q_diag)

        # 量测噪声 R (2×2)
        vel_noise = init_cov_config.get('vel_noise_mps', 0.01)
        self.kf.R = np.eye(2) * vel_noise**2

    def run(self, imu_data: np.ndarray,
            Cnb0: np.ndarray,
            reset_feedback: bool = True,
            init_cov_config: Optional[dict] = None
            ) -> Tuple[Tuple[float, float, float], np.ndarray, np.ndarray]:
        """
        执行8维KF精对准
        输入:
            imu_data — [N×7]
            Cnb0 — [3×3] 粗对准姿态矩阵 C_n^b
        输出:
            (roll, pitch, yaw) — 精对准姿态角 (°)
            X_history — [8×T] 状态历史
            P_history — [8×T] 协方差对角历史
        """
        assert_dcm_valid(Cnb0, name="Cnb0")
        self._init_covariance(init_cov_config)

        gyro = imu_data[:, 0:3]
        acc = imu_data[:, 3:6]
        time = imu_data[:, 6]
        dt = float(np.mean(np.diff(time)))
        N = len(time)

        # 初始姿态: Cnb0 = C_n^b → Cbn = C_b^n
        Cbn = Cnb0.T.copy()
        q_nb = dcm_to_quat(Cbn)  # body->nav quaternion (consistent with dcm_from_quat)
        v_n = np.zeros(3)

        # 自检
        f0_n = Cbn @ acc[0]
        g_n = gravity_n(self.lat_rad)
        g_bias = np.linalg.norm(f0_n + g_n)
        if g_bias > 1.0:
            print(f"[FineAligner] Note: |f^n+g^n| = {g_bias:.2f} m/s^2 (normal for static)")

        print(f"[FineAligner] KF start: {N} pts, dt={dt:.4f}s")

        for k in range(N):
            # ====== INS 解算 ======
            wie_n = earth_rate_n(self.lat_rad)

            # 陀螺补偿: ω_nb^b = ω_ib^b - ε - C_n^b·ω_in^n
            # 将8维状态映射到完整3轴: ε = [ε_x, ε_y, 0], ∇ = [∇_x, ∇_y, 0]
            eps_full = np.array([self.kf.x[4], self.kf.x[5], 0.0])
            omega_nb_b = gyro[k] - eps_full - Cbn.T @ wie_n

            q_nb = quat_update(q_nb, omega_nb_b, dt)
            Cbn = dcm_from_quat(q_nb)

            # 比力投影
            nabla_full = np.array([self.kf.x[6], self.kf.x[7], 0.0])
            f_n = Cbn @ (acc[k] - nabla_full)

            # 速度更新
            v_n = v_n + (f_n + g_n) * dt

            # ====== KF 预测 ======
            F = build_F8(Cbn, f_n, wie_n)
            self.kf.Phi = discretize_F8(F, dt)
            # Q_d = Q_c * dt
            self.kf.Q_saved = self.kf.Q.copy()
            self.kf.Q = self.kf.Q_saved * dt
            self.kf.predict()
            self.kf.Q = self.kf.Q_saved

            # ====== KF 更新 ======
            z = v_n[0:2]  # 仅水平速度
            self.kf.update(z)

            # ====== 状态反馈 ======
            if reset_feedback:
                # 失准角修正 (仅水平, φ_U=0 锁定航向)
                phi = np.array([self.kf.x[0], self.kf.x[1], 0.0])
                Cbn = (np.eye(3) - skew(phi)) @ Cbn
                Cbn = dcm_orthogonalize(Cbn)
                q_nb = dcm_to_quat(Cbn)  # restore body->nav quaternion
                self.kf.x[0:2] = 0.0

                # 速度修正
                v_n[0:2] -= self.kf.x[2:4]
                self.kf.x[2:4] = 0.0

            # ====== 记录 ======
            self.kf.record()

            if k % 100 == 0:
                roll_k, pitch_k, yaw_k = dcm_to_euler312(Cbn.T)
                rd, yd = rad2deg(roll_k), rad2deg(yaw_k)
                if abs(rd) > 90: rd -= np.sign(rd) * 180; yd = -yd
                self.att_history.append([rd, rad2deg(pitch_k), yd % 360])
                self.vel_history.append(v_n.copy())

        # ====== 结果提取 ======
        # Cbn = C_b^n, dcm_to_euler312 需要 C_n^b = Cbn.T
        roll, pitch, yaw = dcm_to_euler312(Cbn.T)
        roll_deg = rad2deg(roll)
        pitch_deg = rad2deg(pitch)
        yaw_deg = rad2deg(yaw)

        if abs(roll_deg) > 90.0:
            roll_deg -= np.sign(roll_deg) * 180.0
            yaw_deg = -yaw_deg
        yaw_deg = yaw_deg % 360

        self._gyro_bias_est = np.array([self.kf.x[4], self.kf.x[5], 0.0])
        self._acc_bias_est = np.array([self.kf.x[6], self.kf.x[7], 0.0])

        print(f"[FineAligner] Done: R={roll_deg:.4f} P={pitch_deg:.4f} H={yaw_deg:.4f}")
        print(f"  Gyro bias: [{np.rad2deg(self._gyro_bias_est[0]):.6f}, "
              f"{np.rad2deg(self._gyro_bias_est[1]):.6f}, 0] deg/s")
        print(f"  Acc bias:  [{self._acc_bias_est[0]:.6f}, "
              f"{self._acc_bias_est[1]:.6f}, 0] m/s^2")

        X_history = self.kf.get_state_history()
        P_history = self.kf.get_diag_history()

        return (roll_deg, pitch_deg, yaw_deg), X_history, P_history

    def get_gyro_bias_estimate(self) -> np.ndarray:
        return self._gyro_bias_est

    def get_acc_bias_estimate(self) -> np.ndarray:
        return self._acc_bias_est
