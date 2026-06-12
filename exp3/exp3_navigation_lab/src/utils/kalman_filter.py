"""
卡尔曼滤波核心运算库
====================
实现线性卡尔曼滤波器的预测与更新步骤
"""

import numpy as np
from .dcm import skew


class LinearKF:
    """
    线性卡尔曼滤波器
    支持标准形式: 
      状态方程: x_{k|k-1} = Phi * x_{k-1} + w
      量测方程: z = H * x + v
    """
    
    def __init__(self, dim_x: int, dim_z: int):
        """
        初始化KF
        输入: dim_x — 状态维度
              dim_z — 量测维度
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        
        # 核心变量
        self.x = np.zeros(dim_x)        # 状态向量
        self.P = np.eye(dim_x)          # 协方差矩阵
        
        # KF矩阵
        self.Phi = np.eye(dim_x)        # 状态转移矩阵
        self.H = np.zeros((dim_z, dim_x))  # 量测矩阵
        self.Q = np.eye(dim_x) * 1e-6   # 系统噪声协方差
        self.R = np.eye(dim_z) * 1e-3   # 量测噪声协方差
        
        # 中间变量
        self.K = np.zeros((dim_x, dim_z))  # 卡尔曼增益
        self.S = np.zeros((dim_z, dim_z))  # 新息协方差
        self.z = np.zeros(dim_z)           # 量测残差
        
        # 日志
        self.x_history = []
        self.P_diag_history = []
    
    def predict(self):
        """KF预测步骤"""
        # 状态预测
        self.x = self.Phi @ self.x
        # 协方差预测
        self.P = self.Phi @ self.P @ self.Phi.T + self.Q
    
    def update(self, z: np.ndarray):
        """
        KF更新步骤
        输入: z — shape (dim_z,) 量测值
        """
        # 量测残差
        self.z = z - self.H @ self.x
        
        # 新息协方差
        self.S = self.H @ self.P @ self.H.T + self.R
        
        # 卡尔曼增益
        self.K = self.P @ self.H.T @ np.linalg.inv(self.S)
        
        # 状态更新
        self.x = self.x + self.K @ self.z
        
        # 协方差更新（Joseph形式，保证对称正定）
        I_KH = np.eye(self.dim_x) - self.K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + self.K @ self.R @ self.K.T
    
    def record(self):
        """记录当前状态和协方差对角"""
        self.x_history.append(self.x.copy())
        self.P_diag_history.append(np.diag(self.P).copy())
    
    def get_state_history(self) -> np.ndarray:
        """获取状态历史，shape = (dim_x, T)"""
        if len(self.x_history) == 0:
            return np.zeros((self.dim_x, 1))
        return np.array(self.x_history).T
    
    def get_diag_history(self) -> np.ndarray:
        """获取协方差对角历史，shape = (dim_x, T)"""
        if len(self.P_diag_history) == 0:
            return np.zeros((self.dim_x, 1))
        return np.array(self.P_diag_history).T
    
    def reset_state(self):
        """重置状态和协方差到初始值"""
        self.x.fill(0.0)
        self.P = np.eye(self.dim_x)
        self.x_history.clear()
        self.P_diag_history.clear()


def build_F_matrix(Cbn: np.ndarray, f_n: np.ndarray, 
                   wie_n: np.ndarray) -> np.ndarray:
    """
    构建12维SINS误差状态方程系统矩阵F
    状态: X = [φ(3), δv(3), ε(3), ∇(3)]^T
    
    输入: Cbn — shape (3,3) 姿态矩阵
          f_n — shape (3,) 比力在n系投影 (m/s^2)
          wie_n — shape (3,) 地球自转在n系投影 (rad/s)
    输出: F — shape (12,12) 系统矩阵
    
    F = | -(wie_n×)    I₃×₃      Cbn       0₃×₃ |
        |  -(f_n×)     0₃×₃      0₃×₃     Cbn  |
        |   0₃×₃       0₃×₃      0₃×₃     0₃×₃ |
        |   0₃×₃       0₃×₃      0₃×₃     0₃×₃ |
    """
    F = np.zeros((12, 12))
    
    # F[0:3, 0:3] = -(wie_n ×)
    F[0:3, 0:3] = -skew(wie_n)
    
    # F[0:3, 3:6] = I₃×₃
    F[0:3, 3:6] = np.eye(3)
    
    # F[0:3, 6:9] = Cbn (陀螺零偏对失准角的影响)
    F[0:3, 6:9] = Cbn
    
    # F[3:6, 0:3] = -(f_n ×)
    F[3:6, 0:3] = -skew(f_n)
    
    # F[3:6, 9:12] = Cbn (加速度计零偏对速度误差的影响)
    F[3:6, 9:12] = Cbn
    
    return F


def discretize_F(F: np.ndarray, dt: float) -> np.ndarray:
    """
    连续系统矩阵F离散化为状态转移矩阵Phi
    Phi = I + F·dt + 0.5·(F·dt)^2 (二阶近似)
    
    输入: F — shape (12,12) 连续系统矩阵
          dt — 采样间隔 (s)
    输出: Phi — shape (12,12) 离散状态转移矩阵
    """
    Fdt = F * dt
    Phi = np.eye(F.shape[0]) + Fdt + 0.5 * Fdt @ Fdt
    return Phi


def build_H_matrix(dim_z: int = 3) -> np.ndarray:
    """
    构建量测矩阵H (速度误差观测量)
    dim_z=3: H = [0_3x3, I_3x3, 0_3x6]
    dim_z=2: 仅水平速度 H = [0_2x3, [I_2x2,0], 0_2x6]

    输出: H — shape (dim_z, 12)
    """
    H = np.zeros((dim_z, 12))
    H[0:dim_z, 3:3+dim_z] = np.eye(dim_z)
    return H
