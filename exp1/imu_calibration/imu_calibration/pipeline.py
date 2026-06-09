"""
标定流水线编排器 — 组合所有模块完成端到端标定
"""
from typing import Dict, Optional
from pathlib import Path
import numpy as np

from .common.types import CalibrationReport
from .io.data_loader import IMUDataLoader
from .preprocessing.preprocessor import Preprocessor
from .calibration.accel_calibrator import AccelCalibrator
from .calibration.gyro_rate_calibrator import GyroRateCalibrator
from .calibration.gyro_bias_calibrator import GyroBiasCalibrator
from .analysis.allan_variance_analyzer import AllanVarianceAnalyzer
from .assembly.result_assembler import ResultAssembler


class CalibrationPipeline:
    """
    标定流水线
    按依赖顺序执行: DataLoader → Preprocessor → (AccelCalibrator ∥ GyroRateCalibrator → GyroBiasCalibrator) → ResultAssembler
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.data_loader = IMUDataLoader()
        self.preprocessor = Preprocessor(
            outlier_sigma=self.config.get("outlier_sigma", 3.0)
        )
        self.accel_calibrator = AccelCalibrator(
            cond_threshold=self.config.get("cond_threshold", 1e8)
        )
        self.gyro_rate_calibrator = GyroRateCalibrator(
            integration_method=self.config.get("integration_method", "trapezoidal")
        )
        self.gyro_bias_calibrator = GyroBiasCalibrator()
        self.allan_analyzer = AllanVarianceAnalyzer(
            tau_base=self.config.get("allan_tau_base", 2.0),
            min_cluster_count=self.config.get("allan_min_cluster_count", 9),
            use_overlapping=self.config.get("allan_use_overlapping", True),
        )
        self.assembler = ResultAssembler(
            output_dir=self.config.get("output_dir", "./calibration_results")
        )

    def run_accel_calibration(self, data_dir: str) -> CalibrationReport:
        """仅运行加速度计标定"""
        print("\n" + "=" * 50)
        print("  加速度计六位置标定")
        print("=" * 50)

        raw = self.data_loader.load_accel_six_pose(data_dir)
        processed = self.preprocessor.process(raw)
        result = self.accel_calibrator.calibrate(processed.acc_means)

        print(f"  K_a = \n{result.K_a}")
        print(f"  D_a = {result.D_a} m/s²")
        print(f"  Reprojection error = {result.reprojection_error:.6f} m/s²")

        return result

    def run_gyro_rate_calibration(self, rate_dir: str) -> CalibrationReport:
        """仅运行陀螺速率标定"""
        print("\n" + "=" * 50)
        print("  陀螺仪速率标定（综合误差矩阵 K_g）")
        print("=" * 50)

        result = self.gyro_rate_calibrator.load_and_calibrate(rate_dir)

        print(f"  K_g = \n{result.K_g}")
        print(f"  K_g_std = \n{result.K_g_std}")

        return result

    def run_gyro_bias_calibration(self, data_dir: str, K_g: np.ndarray) -> CalibrationReport:
        """仅运行陀螺零偏标定"""
        print("\n" + "=" * 50)
        print("  陀螺仪八位置零偏标定")
        print("=" * 50)

        raw = self.data_loader.load_gyro_eight_pose(data_dir)
        processed = self.preprocessor.process(raw)
        result = self.gyro_bias_calibrator.calibrate(processed.gyro_means, K_g)

        return result

    def run_allan_analysis(self, static_file: str, fs: float = 200.0):
        """运行Allan方差分析"""
        print("\n" + "=" * 50)
        print("  Allan方差分析")
        print("=" * 50)

        gyro, time_arr = self.data_loader.load_static_data(static_file)
        result = self.allan_analyzer.analyze(gyro, fs)

        return result

    def run_full_calibration(self, accel_dir: str, rate_dir: str,
                             gyro_bias_dir: str,
                             static_file: Optional[str] = None,
                             output_name: str = "calibration_report") -> CalibrationReport:
        """
        执行完整标定流程

        参数:
            accel_dir: 加速度计六位置数据目录
            rate_dir: 速率标定数据根目录
            gyro_bias_dir: 陀螺八位置零偏数据目录
            static_file: Allan方差静态数据文件（可选）

        返回:
            CalibrationReport
        """
        metadata = {
            "accel_dir": accel_dir,
            "rate_dir": rate_dir,
            "gyro_bias_dir": gyro_bias_dir,
        }

        # Step 1: 加速度计标定（独立链路）
        accel_result = self.run_accel_calibration(accel_dir)
        metadata["accel_completed"] = True

        # Step 2: 陀螺速率标定（独立链路）
        gyro_rate_result = self.run_gyro_rate_calibration(rate_dir)
        metadata["gyro_rate_completed"] = True

        # Step 3: 陀螺零偏标定（依赖 K_g）
        gyro_bias_result = self.run_gyro_bias_calibration(gyro_bias_dir, gyro_rate_result.K_g)
        metadata["gyro_bias_completed"] = True

        # Step 4: Allan方差分析（可选独立链路）
        allan_result = None
        if static_file and Path(static_file).exists():
            allan_result = self.run_allan_analysis(static_file)
            metadata["allan_completed"] = True

        # Step 5: 组装结果
        report = self.assembler.assemble(
            accel_result, gyro_rate_result, gyro_bias_result,
            allan_result, metadata=metadata
        )

        # 输出
        self.assembler.print_summary(report)
        self.assembler.save_report(report, f"{output_name}.yaml")
        self.assembler.save_report_json(report, f"{output_name}.json")

        return report