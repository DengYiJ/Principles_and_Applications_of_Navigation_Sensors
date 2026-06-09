# 报告资源清单

## 图片资源

| 图号 | 文件名 | 说明 | 对应实验报告章节 |
|:-----|:-------|:-----|:----------------|
| 图1 | gyro_rate_calibration.png | 陀螺仪组合系统速率标定实验数据曲线 | (1) | ✅ |
| 图2 | gyro_bias_calibration.png | 陀螺仪组合系统位置标定实验数据曲线 | (2) | ✅ |
| 图3 | accel_calibration.png | 加速度计组合系统位置标定实验数据曲线 | (4) | ✅ |
| 图4 | allan_variance.png | 陀螺仪Allan方差曲线分析结果 | (6) | ✅ |

## 数值结果

| 编号 | 内容 | 文件 |
|:-----|:-----|:-----|
| (3) | 陀螺仪标定结果（K_g, D_g） | metrics.json / report.yaml |
| (5) | 加速度计标定结果（K_a, D_a） | metrics.json / report.yaml |
| (6) | ARW / BI 参数 | metrics.json / report.yaml |
| (7) | 陀螺仪标定程序 | `imu_calibration/calibration/gyro_rate_calibrator.py`, `gyro_bias_calibrator.py` |
| (8) | 加速度计标定程序 | `imu_calibration/calibration/accel_calibrator.py` |
| (9) | Allan方差分析程序 | `imu_calibration/analysis/allan_variance_analyzer.py` |
