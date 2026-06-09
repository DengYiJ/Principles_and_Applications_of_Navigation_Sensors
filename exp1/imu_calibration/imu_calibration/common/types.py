"""
数据结构定义 (Data Types)
对应 02_algorithm_design.md Stage 6 接口规范
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class RawDataBundle:
    """原始数据捆绑包——DataLoader产出"""
    gyro: np.ndarray          # shape=(N, 3), dtype=float64, 陀螺三轴输出(°/s)
    accel: np.ndarray         # shape=(N, 3), dtype=float64, 加速度三轴输出(m/s²)
    timestamps: np.ndarray    # shape=(N,), dtype=float64, GPS时间(s)
    scenario_tags: List[str]  # N个场景标签
    metadata: Dict = field(default_factory=dict)


@dataclass
class ProcessedData:
    """预处理数据——Preprocessor产出"""
    acc_means: Dict[str, np.ndarray]      # key=pose_id, value=ndarray[3], m/s²
    gyro_means: Dict[str, np.ndarray]     # key=pose_id, value=ndarray[3], °/s
    gyro_integrals: Dict[str, Tuple[np.ndarray, np.ndarray]]  # key=rate_setting, value=(J_pos[3], J_neg[3]), °
    static_gyro: np.ndarray               # shape=(M, 3), 原始静态陀螺数据(不均值)
    static_timestamps: np.ndarray         # shape=(M,)


@dataclass
class AccelCalibResult:
    """加速度计标定结果"""
    K_a: np.ndarray           # shape=(3, 3), dtype=float64
    D_a: np.ndarray           # shape=(3,), dtype=float64, 单位m/s²
    residuals: np.ndarray     # shape=(3, 6), dtype=float64, 每轴每位置拟合残差
    condition_number: float   # 输入矩阵条件数
    reprojection_error: float # 重投影RMS误差


@dataclass
class GyroRateCalibResult:
    """陀螺速率标定结果"""
    K_g: np.ndarray           # shape=(3, 3), dtype=float64
    K_g_std: np.ndarray       # shape=(3, 3), dtype=float64, 多转速标准差
    K_g_per_rate: np.ndarray  # shape=(N_rate, 3, 3), dtype=float64, 每个转速的K_g
    earth_rate_corrected: bool = True


@dataclass
class GyroBiasCalibResult:
    """陀螺零偏标定结果"""
    D_g: np.ndarray           # shape=(3,), dtype=float64, 单位°/s
    D_g_std: np.ndarray       # shape=(3,), dtype=float64, 8位置标准差
    bias_per_pose: np.ndarray # shape=(8, 3), dtype=float64, 每个位置的零偏估计
    D_g_deg_h: np.ndarray     # shape=(3,), 单位°/h（转换后）


@dataclass
class AllanResult:
    """Allan方差分析结果"""
    tau: np.ndarray           # shape=(P,), dtype=float64, 相关时间(s)
    sigma: np.ndarray         # shape=(P, 3), dtype=float64, Allan标准差(°/h)
    ARW: np.ndarray           # shape=(3,), dtype=float64, 角度随机游走(°/√h)
    BI: np.ndarray            # shape=(3,), dtype=float64, 零偏不稳定性(°/h)
    fitted_log_slopes: np.ndarray  # shape=(3,), ARW拟合区间log-log斜率


@dataclass
class CalibrationReport:
    """完整标定报告——ResultAssembler产出"""
    accel: AccelCalibResult
    gyro_rate: GyroRateCalibResult
    gyro_bias: GyroBiasCalibResult
    allan: Optional[AllanResult] = None
    metadata: Dict = field(default_factory=dict)
    quality_flags: Dict[str, bool] = field(default_factory=dict)