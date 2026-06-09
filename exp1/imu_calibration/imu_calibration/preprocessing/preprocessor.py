"""
数据预处理模块
职责：均值滤波、异常值剔除、角度增量积分、场景数据分组
"""
from typing import Dict, List, Tuple
import numpy as np
from ..common.types import RawDataBundle, ProcessedData
from ..common.constants import GTIMU_DT, GTIMU_SAMPLE_RATE, G_MAGNITUDE
from ..common.assertions import data_shape_monitor
from .outlier_filter import OutlierFilter
from .integrator import Integrator


class Preprocessor:
    """
    数据预处理器
    将 RawDataBundle 按场景标签分组，提取均值/积分/静态数据
    """

    def __init__(self, outlier_sigma: float = 3.0, integration_method: str = "trapezoidal"):
        self.outlier_sigma = outlier_sigma
        self.integration_method = integration_method

    def process(self, raw_data: RawDataBundle) -> ProcessedData:
        """
        执行完整预处理流程

        参数:
            raw_data: DataLoader产出的原始数据

        返回:
            ProcessedData: 包含各场景的统计量和静态数据
        """
        # shape监控
        data_shape_monitor(raw_data.gyro, 2, "Preprocessor.input.gyro")
        data_shape_monitor(raw_data.accel, 2, "Preprocessor.input.accel")

        # 1. 按场景标签分组
        acc_means = self._compute_acc_means(raw_data)
        gyro_means = self._compute_gyro_means(raw_data)
        gyro_integrals = self._compute_gyro_integrals(raw_data)
        static_gyro, static_time = self._extract_static(raw_data)

        processed = ProcessedData(
            acc_means=acc_means,
            gyro_means=gyro_means,
            gyro_integrals=gyro_integrals,
            static_gyro=static_gyro,
            static_timestamps=static_time,
        )

        return processed

    def _compute_acc_means(self, raw: RawDataBundle) -> Dict[str, np.ndarray]:
        """
        按场景标签计算加速度均值
        先3σ滤波，再取算术平均，最后转换单位为 倍g → m/s²
        
        注意：原始数据单位为"倍g"，乘以北京当地重力加速度 G_MAGNITUDE (m/s²)
        转换为 m/s²，以与姿态表中的理论重力分量(g值)单位一致。
        """
        tags = list(set(raw.scenario_tags))
        result = {}
        for tag in tags:
            mask = [t == tag for t in raw.scenario_tags]
            acc_segment = raw.accel[mask]

            # 3σ异常值剔除
            acc_clean = OutlierFilter.filter_3sigma(
                acc_segment, sigma_threshold=self.outlier_sigma, verbose=False
            )

            # 将均值从"倍g"转换为 m/s²
            mean_val_g = np.mean(acc_clean, axis=0)      # 单位: g
            mean_val_ms2 = mean_val_g * G_MAGNITUDE       # 单位: m/s²
            result[tag] = mean_val_ms2

            print(f"[Preprocessor] {tag}: acc_mean={mean_val_g} (g) = {mean_val_ms2} (m/s²), "
                  f"samples={len(acc_segment)}→{len(acc_clean)}")

        return result

    def _compute_gyro_means(self, raw: RawDataBundle) -> Dict[str, np.ndarray]:
        """按场景标签计算陀螺均值"""
        tags = list(set(raw.scenario_tags))
        result = {}
        for tag in tags:
            mask = [t == tag for t in raw.scenario_tags]
            gyro_segment = raw.gyro[mask]

            gyro_clean = OutlierFilter.filter_3sigma(
                gyro_segment, sigma_threshold=self.outlier_sigma, verbose=False
            )

            mean_val = np.mean(gyro_clean, axis=0)
            result[tag] = mean_val

            print(f"[Preprocessor] {tag}: gyro_mean={mean_val}, "
                  f"samples={len(gyro_segment)}→{len(gyro_clean)}")

        return result

    def _compute_gyro_integrals(self, raw: RawDataBundle) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        计算速率标定场景的角度增量
        查找标签中含有"rate"的场景，按正反转分组积分

        注意：速率标定的积分需要特殊处理
        此处为通用接口，具体积分计算在 GyroRateCalibrator 中完成
        """
        # 只对标签中包含"rate"的场景进行积分处理
        # 实际积分在 GyroRateCalibrator 内完成
        return {}

    def _extract_static(self, raw: RawDataBundle) -> Tuple[np.ndarray, np.ndarray]:
        """提取静态数据（用于Allan方差）"""
        static_mask = ["static" in tag.lower() for tag in raw.scenario_tags]
        if any(static_mask):
            static_gyro = raw.gyro[static_mask]
            static_time = raw.timestamps[static_mask]
            print(f"[Preprocessor] static data: {len(static_gyro)} samples, "
                  f"duration={static_time[-1]-static_time[0]:.1f}s")
            return static_gyro, static_time
        # 无static标签时返回空
        return np.array([], dtype=np.float64).reshape(0, 3), np.array([], dtype=np.float64)