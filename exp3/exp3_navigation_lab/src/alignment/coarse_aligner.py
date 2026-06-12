"""
双矢量法粗对准模块
==================
实现静基座下基于双矢量定姿的初始姿态粗对准算法
参考指导书公式(8)-(13)和PSINS的aligni0vn
"""

import numpy as np
from typing import Tuple
from src.utils.quaternion import quat_update, quat_normalize
from src.utils.dcm import dcm_from_quat, dcm_orthogonalize, dcm_is_valid, skew
from src.utils.euler_angles import dcm_to_euler312, rad2deg
from src.utils.earth_model import gravity_i, C_n2i, WGS84_OMEGA
from src.utils.assert_helpers import assert_dcm_valid, check_singularity


class CoarseAligner:
    """
    双矢量法粗对准器
    利用重力矢量在惯性空间中的投影作为参考，
    通过比力积分构造观测向量，解算初始姿态矩阵
    
    公式参考:
        Vⁱ(t) = ∫C_bⁱ(τ)·fᵇ(τ)dτ  (观测速度)
        Rⁱ(t) = -∫gⁱ(τ)dτ          (参考速度)
        C_bⁱ(0) = [V₁,V₂,V₁×V₂]·[R₁,R₂,R₁×R₂]⁻¹
        C_bⁿ(0) = C_nⁱ(0)ᵀ·C_bⁱ(0)
    """
    
    def __init__(self, latitude_deg: float, 
                 wie: float = WGS84_OMEGA,
                 g: float = 9.7803267714):
        """
        初始化粗对准器
        
        输入:
            latitude_deg — 当地纬度 (°)
            wie — 地球自转角速度 (rad/s)
            g — 当地重力加速度 (m/s^2)
        """
        self.lat_rad = np.deg2rad(latitude_deg)
        self.wie = wie
        self.g = g
        
        print(f"[CoarseAligner] 初始化: 纬度={latitude_deg}°, g={g:.4f} m/s^2")
    
    def run(self, imu_data: np.ndarray,
            t1: float = 10.0,
            t2: float = 100.0,
            n_avg_pairs: int = 5) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        执行双矢量法粗对准
        
        输入:
            imu_data — [N×7] = [gyroX,gyroY,gyroZ,accX,accY,accZ,time]
            t1 — 第一个取点时刻 (s) 默认10s
            t2 — 第二个取点时刻 (s) 默认100s
            n_avg_pairs — 多组取点平均的组数 (默认5)
        
        输出:
            Cnb0 — [3×3] 初始姿态矩阵 C_n^b
            (roll, pitch, yaw) — 初始姿态角 (°)
        """
        # ----- 步骤1: 提取数据并归一化时间（从0开始）-----
        gyro = imu_data[:, 0:3]   # 陀螺仪 (rad/s)
        acc = imu_data[:, 3:6]    # 加速度计 (m/s^2)
        time_raw = imu_data[:, 6] # 原始时间戳 (GPS s)
        time = time_raw - time_raw[0]  # 归一化到从0开始
        N = len(time)
        
        assert N > max(t1, t2) * 200 + 100, \
            f"数据点数不足: {N} (需要至少{max(t1, t2)*200+100}个点)"
        
        # ----- 步骤2: 计算时间间隔（三重鲁棒方案）-----
        time_diff = np.diff(time)
        pos_diffs = time_diff[time_diff > 1e-12]
        if len(pos_diffs) > 0:
            dt = np.median(pos_diffs)
        elif time[-1] - time[0] > 0:
            dt = (time[-1] - time[0]) / (N - 1)
        else:
            dt = 0.005  # 默认200Hz采样
        print(f"[CoarseAligner] 采样间隔 dt={dt:.6f}s, 等效频率={1/dt:.1f}Hz")
        
        # ----- 步骤3: 初始化积分变量 -----
        q_b2i = np.array([1.0, 0.0, 0.0, 0.0])  # 四元数: b系→i系
        v_i = np.zeros(3)  # 观测速度向量 Vⁱ(t)
        r_i = np.zeros(3)  # 参考速度向量 Rⁱ(t)
        
        # 存储所有时刻的V和R
        N_stored = N
        v_history = np.zeros((N_stored, 3))
        r_history = np.zeros((N_stored, 3))
        
        # ----- 步骤4: 主循环 — 对每个采样时刻积分 -----
        for k in range(N):
            # 4.1 陀螺角速度
            omega = gyro[k]
            
            # 4.2 四元数更新 (b系→i系)
            q_b2i = quat_update(q_b2i, omega, dt)
            
            # 4.3 姿态矩阵 C_b2i
            C_b2i = dcm_from_quat(q_b2i)
            
            # 4.4 比力投影到i系并积分
            f_i = C_b2i @ acc[k]
            v_i = v_i + f_i * dt
            
            # 4.5 重力在i系的投影并积分
            t_k = time[k]
            g_i_t = gravity_i(t_k, self.lat_rad, self.wie, self.g)
            r_i = r_i - g_i_t * dt
            
            # 4.6 记录
            v_history[k] = v_i.copy()
            r_history[k] = r_i.copy()
        
        # ----- 步骤5-6: 双矢量定姿（支持多组平均）-----
        C_b2i0_list = []
        
        # 确保取点在数据范围内（自适应边界）
        data_duration = time[-1] - time[0]
        safe_margin = min(5.0, data_duration * 0.1)  # 动态边界
        t_min_data = time[0] + safe_margin
        t_max_data = time[-1] - safe_margin
        
        for pair_idx in range(n_avg_pairs):
            # 多组t1,t2组合，取平均提高精度
            t1_k = t1 + pair_idx * (t2 - t1) / max(n_avg_pairs, 1)
            t2_k = t2 + pair_idx * (t2 - t1) / max(n_avg_pairs, 1)
            
            # 确保在数据范围内
            t1_k = np.clip(t1_k, t_min_data, t_max_data)
            t2_k = np.clip(t2_k, t1_k + safe_margin, t_max_data)
            
            # 找到对应时刻的索引
            idx1 = np.argmin(np.abs(time - t1_k))
            idx2 = np.argmin(np.abs(time - t2_k))
            
            # 检查间隔（自适应：至少数据时长的20%，不低于5s）
            min_interval = max(5.0, data_duration * 0.2)
            interval = abs(idx2 - idx1) * dt
            if interval < min_interval:
                print(f"[CoarseAligner] 警告: 取点间隔{interval:.1f}s < {min_interval:.1f}s，跳过")
                continue
            
            V1 = v_history[idx1]  # 观测速度 Vⁱ(t₁)
            V2 = v_history[idx2]  # 观测速度 Vⁱ(t₂)
            R1 = r_history[idx1]  # 参考速度 Rⁱ(t₁)
            R2 = r_history[idx2]  # 参考速度 Rⁱ(t₂)
            
            # 双矢量定姿: C_b2i0 = Mv @ inv(Mr)
            Mv = np.column_stack([V1, V2, np.cross(V1, V2)])
            Mr = np.column_stack([R1, R2, np.cross(R1, R2)])
            
            # 检查Mr是否奇异
            cond_Mr = check_singularity(Mr, "Mr")
            if cond_Mr > 1e12:
                print(f"[CoarseAligner] 警告: Mr条件数过大({cond_Mr:.2e})，跳过此组")
                continue
            
            C_b2i0_k = Mv @ np.linalg.inv(Mr)
            
            # SVD正交化投影到SO(3)
            C_b2i0_k = dcm_orthogonalize(C_b2i0_k)
            C_b2i0_list.append(C_b2i0_k)
        
        if len(C_b2i0_list) == 0:
            raise RuntimeError("粗对准失败：所有取点组合均无效")
        
        # 多组结果取平均（矩阵平均后重新正交化）
        C_b2i0_avg = np.mean(C_b2i0_list, axis=0)
        C_b2i0 = dcm_orthogonalize(C_b2i0_avg)
        
        # ----- 步骤7: 计算初始n系→i系变换 -----
        t0 = time[0]
        C_n2i0 = C_n2i(t0, self.lat_rad, self.wie)

        # ----- 步骤8: 计算初始姿态矩阵 C_n^b -----
        # 修正: C_n^b = C_n^i @ C_b^i^T (正确表示 nav-to-body)
        # 旧公式 C_n^i^T @ C_b^i = C_b^n (body-to-nav) 有误
        Cnb0 = C_n2i0 @ C_b2i0.T
        Cnb0 = dcm_orthogonalize(Cnb0)

        # 断言DCM有效
        assert_dcm_valid(Cnb0, name="Cnb0")

        # ----- 步骤9: 提取欧拉角 (3-1-2转序) -----
        roll, pitch, yaw = dcm_to_euler312(Cnb0)

        # 转换为度
        roll_deg = rad2deg(roll)
        pitch_deg = rad2deg(pitch)
        yaw_deg = rad2deg(yaw)

        # 欧拉角解包裹: 3-1-2转序下 roll=±180° 等价于 roll=0° 但 heading 反向
        if abs(roll_deg) > 90.0:
            roll_deg = roll_deg - np.sign(roll_deg) * 180.0
            yaw_deg = -yaw_deg

        print(f"[CoarseAligner] 粗对准完成:")
        print(f"  横滚角: {roll_deg:.4f}°")
        print(f"  俯仰角: {pitch_deg:.4f}°")
        print(f"  航向角: {yaw_deg:.4f}°")

        return Cnb0, (roll_deg, pitch_deg, yaw_deg)
    
    def run_multiple_pairs(self, imu_data: np.ndarray,
                           t_start: float = 10.0,
                           t_end: float = 190.0,
                           n_pairs: int = 10) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        使用多组取点并平均（提高鲁棒性）
        
        输入:
            imu_data — [N×7]
            t_start — 起始取点时间
            t_end — 终止取点时间
            n_pairs — 取点组数
        
        输出: (Cnb0, (roll, pitch, yaw))
        """
        time = imu_data[:, 6]
        t_max = time[-1] - 5.0
        t_end = min(t_end, t_max)
        
        Cnb0_list = []
        for i in range(n_pairs):
            t1 = t_start + i * (t_end - t_start) / n_pairs
            t2 = t1 + (t_end - t_start) / (2 * n_pairs)
            try:
                Cnb0_k, att_k = self.run(imu_data, t1=t1, t2=t2, n_avg_pairs=1)
                Cnb0_list.append(Cnb0_k)
            except Exception as e:
                print(f"[CoarseAligner] 第{i}组取点失败: {e}")
                continue
        
        if len(Cnb0_list) == 0:
            raise RuntimeError("所有取点组合均失败")
        
        # 平均并正交化
        Cnb0_avg = np.mean(Cnb0_list, axis=0)
        Cnb0 = dcm_orthogonalize(Cnb0_avg)
        
        roll, pitch, yaw = dcm_to_euler312(Cnb0)
        roll_deg = rad2deg(roll)
        pitch_deg = rad2deg(pitch)
        yaw_deg = rad2deg(yaw)

        # 欧拉角解包裹
        if abs(roll_deg) > 90.0:
            roll_deg = roll_deg - np.sign(roll_deg) * 180.0
            yaw_deg = -yaw_deg

        print(f"[CoarseAligner] 多组平均粗对准完成 (共{len(Cnb0_list)}组成功):")
        print(f"  横滚角: {roll_deg:.4f}°")
        print(f"  俯仰角: {pitch_deg:.4f}°")
        print(f"  航向角: {yaw_deg:.4f}°")

        return Cnb0, (roll_deg, pitch_deg, yaw_deg)
