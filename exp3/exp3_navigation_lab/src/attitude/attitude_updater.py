"""
姿态更新器
==========
基于四元数微分方程数值积分，实时递推载体姿态
参考指导书图2和《捷联惯导算法与组合导航原理》
"""

import numpy as np
from typing import Tuple
from src.utils.quaternion import quat_update, quat_normalize
from src.utils.dcm import dcm_from_quat, dcm_to_quat, dcm_orthogonalize, dcm_is_valid
from src.utils.euler_angles import euler312_to_dcm, dcm_to_euler312, rad2deg
from src.utils.earth_model import earth_rate_n, WGS84_OMEGA
from src.utils.assert_helpers import assert_dcm_valid, assert_attitude_range


class AttitudeUpdater:
    """
    姿态更新器
    
    基于精对准结果提供的初始姿态和零偏补偿后的陀螺数据，
    通过四元数微分方程积分实时更新载体姿态
    """
    
    def __init__(self, latitude_deg: float,
                 integration_method: str = 'picard_2nd',
                 normalize_quat: bool = True):
        """
        初始化姿态更新器
        
        输入:
            latitude_deg — 当地纬度 (°)
            integration_method — 积分方法: 'picard_2nd' / 'rk4'
            normalize_quat — 每步是否归一化四元数
        """
        self.lat_rad = np.deg2rad(latitude_deg)
        self.method = integration_method
        self.normalize_quat = normalize_quat
        
        print(f"[AttitudeUpdater] 初始化: 纬度={latitude_deg}°, 方法={integration_method}")
    
    def run(self, imu_data: np.ndarray,
            att_init_deg: Tuple[float, float, float],
            gyro_bias: Tuple[float, float, float] = (0.0, 0.0, 0.0)
            ) -> np.ndarray:
        """
        执行姿态更新
        
        输入:
            imu_data — [N×7] = [gyroX,gyroY,gyroZ,accX,accY,accZ,time]
            att_init_deg — (roll, pitch, yaw) 初始姿态角 (°)
            gyro_bias — (bx, by, bz) 陀螺零偏 (rad/s)
        
        输出:
            att_history — [N×3] [roll, pitch, yaw] 随时间变化序列 (°)
        """
        # 断言初始姿态角范围
        roll0, pitch0, yaw0 = att_init_deg
        assert_attitude_range(roll0, pitch0, yaw0)
        
        # 提取数据
        gyro = imu_data[:, 0:3]
        time = imu_data[:, 6]
        N = len(time)
        dt = np.mean(np.diff(time))
        
        # 从初始姿态构建初始四元数
        roll0_rad = np.deg2rad(roll0)
        pitch0_rad = np.deg2rad(pitch0)
        yaw0_rad = np.deg2rad(yaw0)
        
        Cbn_init = euler312_to_dcm(roll0_rad, pitch0_rad, yaw0_rad)
        q_nb = dcm_to_quat(Cbn_init)
        
        # 零偏
        bias_vec = np.array(gyro_bias)
        
        # 记录姿态历史
        att_history = np.zeros((N, 3))
        
        print(f"[AttitudeUpdater] 开始姿态更新 ({N}个点, dt={dt:.6f}s)")
        print(f"  初始姿态: roll={roll0:.4f}°, pitch={pitch0:.4f}°, yaw={yaw0:.4f}°")
        
        # 主循环
        for k in range(N):
            # 3.1 扣除零偏
            omega_ib_b = gyro[k] - bias_vec
            
            # 3.2 计算导航系相对惯性系的角速度在b系投影
            # ω_nb^b = ω_ib^b - C_n^bᵀ · ω_ie^n
            wie_n = earth_rate_n(self.lat_rad)
            Cbn = dcm_from_quat(q_nb)
            omega_nb_b = omega_ib_b - Cbn.T @ wie_n
            
            # 3.3 四元数更新
            q_nb = quat_update(q_nb, omega_nb_b, dt, method=self.method)
            if self.normalize_quat:
                q_nb = quat_normalize(q_nb)
            
            # 3.4 更新姿态矩阵
            Cbn = dcm_from_quat(q_nb)
            
            # 3.5 提取欧拉角 (3-1-2转序)
            roll_k, pitch_k, yaw_k = dcm_to_euler312(Cbn)
            
            # 3.6 记录 (度)
            att_history[k] = [
                rad2deg(roll_k),
                rad2deg(pitch_k),
                rad2deg(yaw_k)
            ]
        
        # 最终检查
        final_roll, final_pitch, final_yaw = att_history[-1]
        assert_attitude_range(final_roll, final_pitch, final_yaw)
        
        print(f"[AttitudeUpdater] 姿态更新完成")
        print(f"  最终姿态: roll={final_roll:.4f}°, pitch={final_pitch:.4f}°, yaw={final_yaw:.4f}°")
        
        return att_history