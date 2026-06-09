"""
结果组装与输出模块
将各标定模块的输出组装为统一标定报告
"""
from typing import Dict, Optional
import json
import yaml
import numpy as np
from pathlib import Path
from ..common.types import (
    AccelCalibResult, GyroRateCalibResult, GyroBiasCalibResult,
    AllanResult, CalibrationReport
)


class ResultAssembler:
    """
    标定结果组装器
    职责：汇总标定参数，计算质量指标，序列化输出
    """

    def __init__(self, output_dir: str = "./calibration_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def assemble(self, accel_result: AccelCalibResult,
                 gyro_rate_result: GyroRateCalibResult,
                 gyro_bias_result: GyroBiasCalibResult,
                 allan_result: Optional[AllanResult] = None,
                 metadata: Optional[Dict] = None) -> CalibrationReport:
        """组装完整标定报告"""
        quality_flags = self._evaluate_quality(accel_result, gyro_rate_result, gyro_bias_result)
        report = CalibrationReport(
            accel=accel_result,
            gyro_rate=gyro_rate_result,
            gyro_bias=gyro_bias_result,
            allan=allan_result,
            metadata=metadata or {},
            quality_flags=quality_flags,
        )
        return report

    def save_report(self, report: CalibrationReport, filename: str = "calibration_report.yaml"):
        """保存标定报告为YAML格式"""
        report_dict = self._report_to_dict(report)
        fpath = self.output_dir / filename
        with open(fpath, 'w', encoding='utf-8') as f:
            yaml.dump(report_dict, f, default_flow_style=False, indent=2, allow_unicode=True)
        print(f"[ResultAssembler] Report saved to {fpath}")
        return str(fpath)

    def save_report_json(self, report: CalibrationReport, filename: str = "calibration_report.json"):
        """保存标定报告为JSON格式"""
        report_dict = self._report_to_dict(report)
        fpath = self.output_dir / filename
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        print(f"[ResultAssembler] Report saved to {fpath}")
        return str(fpath)

    def print_summary(self, report: CalibrationReport):
        """打印标定结果摘要"""
        print("\n" + "=" * 60)
        print("          IMU 标定结果摘要")
        print("=" * 60)
        print("\n【加速度计标定结果】")
        print(f"  K_a =")
        for row in report.accel.K_a:
            print(f"       [{row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}]")
        print(f"  D_a = [{report.accel.D_a[0]:.6f}  {report.accel.D_a[1]:.6f}  {report.accel.D_a[2]:.6f}] m/s²")
        print(f"  重投影误差 = {report.accel.reprojection_error:.6f} m/s²")
        print(f"  条件数 = {report.accel.condition_number:.2e}")
        print("\n【陀螺仪速率标定结果】")
        print(f"  K_g =")
        for row in report.gyro_rate.K_g:
            print(f"       [{row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}]")
        print(f"  K_g_std =")
        for row in report.gyro_rate.K_g_std:
            print(f"           [{row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}]")
        print("\n【陀螺仪零偏标定结果】")
        print(f"  D_g = [{report.gyro_bias.D_g[0]:.6f}  {report.gyro_bias.D_g[1]:.6f}  {report.gyro_bias.D_g[2]:.6f}] °/s")
        print(f"  D_g = [{report.gyro_bias.D_g_deg_h[0]:.2f}  {report.gyro_bias.D_g_deg_h[1]:.2f}  {report.gyro_bias.D_g_deg_h[2]:.2f}] °/h")
        print(f"  标准差 = [{report.gyro_bias.D_g_std[0]:.6f}  {report.gyro_bias.D_g_std[1]:.6f}  {report.gyro_bias.D_g_std[2]:.6f}] °/s")
        if report.allan:
            print("\n【Allan方差分析结果】")
            for i, axis in enumerate(['X', 'Y', 'Z']):
                print(f"  {axis}轴: ARW={report.allan.ARW[i]:.6f} °/√h, BI={report.allan.BI[i]:.6f} °/h")
        print("\n【质量标志】")
        for flag, val in report.quality_flags.items():
            print(f"  {flag}: {'✅' if val else '❌'}")
        print("=" * 60 + "\n")

    def _evaluate_quality(self, accel: AccelCalibResult,
                           gyro_rate: GyroRateCalibResult,
                           gyro_bias: GyroBiasCalibResult) -> Dict[str, bool]:
        flags = {}
        flags["accel_residual_ok"] = accel.reprojection_error < 0.5
        flags["accel_condition_ok"] = accel.condition_number < 1e8
        flags["gyro_bias_consistent"] = bool(np.all(gyro_bias.D_g_std * 3600.0 < 0.5))
        return flags

    @staticmethod
    def _report_to_dict(report: CalibrationReport) -> Dict:
        def arr(v):
            if hasattr(v, 'tolist'):
                return v.tolist()
            return v

        d = {
            "metadata": report.metadata,
            "quality_flags": report.quality_flags,
            "accel": {
                "K_a": arr(report.accel.K_a),
                "D_a_m_s2": arr(report.accel.D_a),
                "reprojection_error_m_s2": float(report.accel.reprojection_error),
                "condition_number": float(report.accel.condition_number),
            },
            "gyro_rate": {
                "K_g": arr(report.gyro_rate.K_g),
                "K_g_std": arr(report.gyro_rate.K_g_std),
            },
            "gyro_bias": {
                "D_g_deg_s": arr(report.gyro_bias.D_g),
                "D_g_deg_h": arr(report.gyro_bias.D_g_deg_h),
                "D_g_std_deg_s": arr(report.gyro_bias.D_g_std),
                "bias_per_pose_deg_s": arr(report.gyro_bias.bias_per_pose),
            },
        }
        if report.allan:
            d["allan"] = {
                "ARW_deg_sqrt_h": arr(report.allan.ARW),
                "BI_deg_h": arr(report.allan.BI),
                "fitted_slopes": arr(report.allan.fitted_log_slopes),
            }
        return d