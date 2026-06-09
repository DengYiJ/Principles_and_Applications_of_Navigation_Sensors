"""
一键全标定脚本
输出实验报告所需的9项内容：
(1) 陀螺仪组合系统速率标定实验数据曲线
(2) 陀螺仪组合系统位置标定实验数据曲线
(3) 陀螺仪组合系统标定结果（标度因数矩阵、零偏向量）
(4) 加速度计组合系统位置标定实验数据曲线
(5) 加速度计组合系统标定结果（标度因数矩阵、零偏向量）
(6) 陀螺仪Allan方差曲线分析结果
(7) 陀螺仪组合系统标定程序
(8) 加速度计组合系统标定程序
(9) Allan方差分析处理程序
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path

from imu_calibration.pipeline import CalibrationPipeline
from imu_calibration.common.pose_tables import (
    build_accel_pose_table_6, build_gyro_bias_pose_table_8
)
from visualization.plot_calibration import CalibrationPlotter
from visualization.plot_allan import AllanPlotter


def main():
    # ============================================================
    # 数据路径配置（根据实际位置调整）
    # ============================================================
    EXP_DIR = Path(__file__).resolve().parents[2]  # exp1目录

    # 加速度计六位置标定数据
    ACCEL_DIR = str(EXP_DIR / "实验一标定(1)" / "实验一标定" / "加速度计标定")

    # 陀螺速率标定数据
    RATE_DIR = str(EXP_DIR / "实验一标定(1)" / "实验一标定" / "速率标定")

    # 陀螺八位置零偏数据
    GYRO_BIAS_DIR = str(EXP_DIR / "实验一标定(1)" / "实验一标定" / "零偏多位置")

    # Allan方差静态数据
    STATIC_FILE = str(EXP_DIR / "实验一标定(1)" / "实验一标定" / "gtimu_3.5h.log")

    # 输出目录
    RESULT_DIR = str(EXP_DIR / "imu_calibration" / "result")

    print("=" * 60)
    print("  惯性导航传感器标定系统 — 全流程执行")
    print("=" * 60)
    print(f"\n数据路径:")
    print(f"  加速度计: {ACCEL_DIR}")
    print(f"  速率标定: {RATE_DIR}")
    print(f"  零偏标定: {GYRO_BIAS_DIR}")
    print(f"  静态数据: {STATIC_FILE}")
    print(f"  结果输出: {RESULT_DIR}")

    # 初始化流水线
    pipeline = CalibrationPipeline(config={
        "outlier_sigma": 3.0,
        "cond_threshold": 1e8,
        "integration_method": "trapezoidal",
        "output_dir": str(EXP_DIR / "imu_calibration" / "calibration_results"),
    })

    # 初始化绘图器
    plotter = CalibrationPlotter(output_dir=RESULT_DIR)
    allan_plotter = AllanPlotter(output_dir=RESULT_DIR)

    # ============================================================
    # (4) + (5) 加速度计位置标定 + 结果
    # ============================================================
    print("\n" + "─" * 50)
    print("  加速度计六位置标定")
    print("─" * 50)

    accel_result = pipeline.run_accel_calibration(ACCEL_DIR)

    # 重现加速度均值（用于绘图）
    raw = pipeline.data_loader.load_accel_six_pose(ACCEL_DIR)
    processed = pipeline.preprocessor.process(raw)
    pose_table = build_accel_pose_table_6()

    # 绘图
    plotter.plot_accel_calibration(accel_result, processed.acc_means, pose_table)

    print(f"\n加速度计标定结果 (5):")
    print(f"  K_a =")
    for row in accel_result.K_a:
        print(f"       [{row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}]")
    print(f"  D_a = {accel_result.D_a} m/s²")

    # ============================================================
    # (1) + (3) 陀螺仪速率标定 + 结果
    # ============================================================
    print("\n" + "─" * 50)
    print("  陀螺仪速率标定")
    print("─" * 50)

    gyro_rate_result = pipeline.run_gyro_rate_calibration(RATE_DIR)

    # 绘图
    plotter.plot_gyro_rate_calibration(gyro_rate_result)

    print(f"\n陀螺速率标定结果 (3):")
    print(f"  K_g =")
    for row in gyro_rate_result.K_g:
        print(f"       [{row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}]")

    # ============================================================
    # (2) + (3) 陀螺仪位置标定（零偏）+ 结果
    # ============================================================
    print("\n" + "─" * 50)
    print("  陀螺仪八位置零偏标定")
    print("─" * 50)

    gyro_bias_result = pipeline.run_gyro_bias_calibration(GYRO_BIAS_DIR, gyro_rate_result.K_g)

    # 绘图
    plotter.plot_gyro_bias_calibration(gyro_bias_result)

    print(f"\n陀螺零偏标定结果 (3):")
    print(f"  D_g = {gyro_bias_result.D_g} °/s = {gyro_bias_result.D_g_deg_h} °/h")

    # ============================================================
    # (6) Allan方差分析
    # ============================================================
    print("\n" + "─" * 50)
    print("  Allan方差分析")
    print("─" * 50)

    if Path(STATIC_FILE).exists():
        allan_result = pipeline.run_allan_analysis(STATIC_FILE)

        # 绘图
        allan_plotter.plot_allan_curve(allan_result)

        print(f"\nAllan方差分析结果 (6):")
        for i, axis in enumerate(['X', 'Y', 'Z']):
            print(f"  {axis}轴: ARW={allan_result.ARW[i]:.6f} °/√h, "
                  f"BI={allan_result.BI[i]:.6f} °/h")
    else:
        print(f"  [WARN] 静态数据文件不存在: {STATIC_FILE}")
        print("  跳过Allan方差分析")
        allan_result = None

    # ============================================================
    # 组装完整报告并输出
    # ============================================================
    print("\n" + "─" * 50)
    print("  标定结果汇总输出")
    print("─" * 50)

    report = pipeline.assembler.assemble(
        accel_result, gyro_rate_result, gyro_bias_result,
        allan_result,
        metadata={
            "实验": "导航传感器原理与应用-实验1-惯性导航传感器测试与标定",
            "加速度计数据": ACCEL_DIR,
            "速率标定数据": RATE_DIR,
            "零偏标定数据": GYRO_BIAS_DIR,
        }
    )

    pipeline.assembler.print_summary(report)
    pipeline.assembler.save_report(report, "full_calibration_report.yaml")
    pipeline.assembler.save_report_json(report, "full_calibration_report.json")

    # ============================================================
    # 输出文件清单
    # ============================================================
    print("\n" + "=" * 60)
    print("  输出文件清单")
    print("=" * 60)
    result_path = Path(RESULT_DIR)
    print(f"\n📊 图片 (存入 result/):")
    for f in result_path.glob("*.png"):
        print(f"  ✅ {f.name}")
    print(f"\n📄 报告 (存入 calibration_results/):")
    for f in (EXP_DIR / "imu_calibration" / "calibration_results").glob("*"):
        print(f"  ✅ {f.name}")
    print(f"\n📁 程序代码 (存入 imu_calibration/):")
    print(f"  ✅ accel_calibrator.py    → 加速度计组合系统标定程序 (8)")
    print(f"  ✅ gyro_rate_calibrator.py → 陀螺仪速率标定程序 (7)")
    print(f"  ✅ gyro_bias_calibrator.py → 陀螺仪零偏标定程序 (7)")
    print(f"  ✅ allan_variance_analyzer.py → Allan方差分析处理程序 (9)")

    print("\n" + "=" * 60)
    print("  标定完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()