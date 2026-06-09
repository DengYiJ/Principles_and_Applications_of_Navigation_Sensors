"""
物理常数与单位转换常量
"""
import numpy as np

# 圆周率
PI = np.pi

# 重力加速度 (北京当地, m/s²)
# 北京重力加速度 ≈ 9.801 m/s² (纬度约39.9°N)
# 注意：加速度计原始数据单位为"倍g"，需在预处理时乘以该值转换为m/s²
G_MAGNITUDE = 9.801  # m/s² (北京地区)

# 地球自转角速率
EARTH_ROTATION_RATE_RADPS = 7.2921150e-5   # rad/s
EARTH_ROTATION_RATE_DEGPS = EARTH_ROTATION_RATE_RADPS * 180.0 / PI  # °/s
EARTH_ROTATION_RATE_DEGH = EARTH_ROTATION_RATE_DEGPS * 3600.0       # °/h

# 单位转换
DEG2RAD = PI / 180.0
RAD2DEG = 180.0 / PI

# 默认本地纬度 (实验数据来源: 哈尔滨工业大学, 约45.73°N)
DEFAULT_LATITUDE_DEG = 45.7345

# GTIMU采样率
GTIMU_SAMPLE_RATE = 200.0  # Hz
GTIMU_DT = 1.0 / GTIMU_SAMPLE_RATE  # 0.005s

# 物理边界
GYRO_RANGE_DEGPS = 500.0   # 陀螺最大量程 °/s
ACCEL_RANGE_G = 3.0        # 加速度计最大量程 (g)