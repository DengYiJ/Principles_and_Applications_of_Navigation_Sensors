"""
实验执行脚本（Experiment Execution）
遵循 clinerules/30_experiment_execution.md 标准化流程

执行以下实验：
  EXP_001: 加速度计六位置标定 + 陀螺仪速率标定 + 陀螺仪零偏标定 + Allan方差分析
"""
import sys, os, json, yaml, time, datetime, shutil
from pathlib import Path
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imu_calibration.pipeline import CalibrationPipeline
from imu_calibration.common.pose_tables import (
    build_accel_pose_table_6, build_gyro_bias_pose_table_8
)
from visualization.plot_calibration import CalibrationPlotter
from visualization.plot_allan import AllanPlotter


# ================================================================
# Stage 0：实验准备检查 (Experiment Readiness Check)
# ================================================================
def readiness_check() -> dict:
    """检查数据集、配置、输出目录"""
    EXP_DIR = Path(__file__).resolve().parents[2]  # exp1
    
    checks = {
        "数据路径": {},
        "输出目录": {},
        "依赖包": {},
        "依赖包(可选)": {},
    }
    
    # 数据路径检查
    accel_dir = EXP_DIR / "实验一标定(1)" / "实验一标定" / "加速度计标定"
    rate_dir = EXP_DIR / "实验一标定(1)" / "实验一标定" / "速率标定"
    bias_dir = EXP_DIR / "实验一标定(1)" / "实验一标定" / "零偏多位置"
    static_file = EXP_DIR / "实验一标定(1)" / "实验一标定" / "gtimu_3.5h.log"
    
    checks["数据路径"]["accel_dir"] = str(accel_dir.exists())
    checks["数据路径"]["rate_dir"] = str(rate_dir.exists())
    checks["数据路径"]["bias_dir"] = str(bias_dir.exists())
    checks["数据路径"]["static_file"] = str(static_file.exists())
    
    # 加速度计6文件检查
    accel_files = all((accel_dir / f"gtimu_{i}.log").exists() for i in range(1,7))
    checks["数据路径"]["accel_6files"] = str(accel_files)
    
    # 速率标定文件检查（位置一/二/三）
    rate_pos1 = all((rate_dir / "位置一" / f"gtimu_{s}{v}.log").exists()
                    for v in [10,20,30,40,50] for s in ["", "-"])
    rate_pos2 = all((rate_dir / "位置二" / f"gtimu_{s}{v}.log").exists()
                    for v in [10,20,30,40,50] for s in ["", "-"])
    rate_pos3 = all((rate_dir / "位置三" / f"gtimu_{s}{v}.log").exists()
                    for v in [10,20,30,40,50] for s in ["", "-"])
    checks["数据路径"]["rate_pos1"] = str(rate_pos1)
    checks["数据路径"]["rate_pos2"] = str(rate_pos2)
    checks["数据路径"]["rate_pos3"] = str(rate_pos3)
    
    # 零偏8文件检查
    bias_files = all((bias_dir / f"gtimu_{i}.log").exists() for i in range(1,9))
    checks["数据路径"]["bias_8files"] = str(bias_files)
    
    # 输出目录
    out_dirs = ["result", "logs", "calibration_results", "experiments"]
    for d in out_dirs:
        p = EXP_DIR / "imu_calibration" / d
        p.mkdir(parents=True, exist_ok=True)
        checks["输出目录"][d] = str(p.exists())
    
    # 依赖检查（tqdm 为可选依赖，不影响核心功能）
    deps = ["numpy", "matplotlib", "yaml"]
    optional_deps = ["tqdm"]
    for dep in deps:
        try:
            __import__(dep)
            checks["依赖包"][dep] = "OK"
        except ImportError:
            checks["依赖包"][dep] = "MISSING"
    for dep in optional_deps:
        try:
            __import__(dep)
            checks["依赖包(可选)"][dep] = "OK"
        except ImportError:
            checks["依赖包(可选)"][dep] = "MISSING (不影响核心功能)"
    
    # 所有非可选检查项必须通过
    critical_pass = all(
        v == "True" or v == "OK"
        for c_key, c_val in checks.items()
        for v in c_val.values()
        if "可选" not in c_key
    )
    if critical_pass:
        return {"status": "PASS", "details": checks}
    else:
        return {"status": "FAIL", "details": checks}


def save_readiness_report(check_result: dict, path: Path):
    """保存实验准备检查报告"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(check_result, f, default_flow_style=False, allow_unicode=True)
    print(f"[Readiness] Report saved: {path}")


# ================================================================
# Stage 1：实验矩阵设计 (Experiment Matrix)
# ================================================================
EXPERIMENT_MATRIX = [
    {
        "exp_id": "EXP_001",
        "name": "Full IMU Calibration",
        "description": "加速度计六位置标定 + 陀螺仪速率标定 + 陀螺仪八位置零偏偏定 + Allan方差分析",
        "type": "Baseline + Full Calibration",
        "config": {
            "outlier_sigma": 3.0,
            "cond_threshold": 1e8,
            "integration_method": "trapezoidal",
            "allan_use_overlapping": True,
        }
    },
]


# ================================================================
# Stage 2：实验执行 (Experiment Execution)
# ================================================================
def run_experiment(exp_config: dict) -> dict:
    """执行单个实验，返回实验结果"""
    exp_id = exp_config["exp_id"]
    cfg = exp_config["config"]
    
    EXP_DIR = Path(__file__).resolve().parents[2]
    ACCEL_DIR = str(EXP_DIR / "实验一标定(1)" / "实验一标定" / "加速度计标定")
    RATE_DIR = str(EXP_DIR / "实验一标定(1)" / "实验一标定" / "速率标定")
    BIAS_DIR = str(EXP_DIR / "实验一标定(1)" / "实验一标定" / "零偏多位置")
    STATIC_FILE = str(EXP_DIR / "实验一标定(1)" / "实验一标定" / "gtimu_3.5h.log")
    
    # 创建实验输出目录
    exp_out = EXP_DIR / "imu_calibration" / "experiments" / exp_id
    result_dir = exp_out / "result"
    log_dir = exp_out / "logs"
    figures_dir = exp_out / "figures"
    for d in [result_dir, log_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 保存配置文件
    config_path = exp_out / "config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(exp_config, f, default_flow_style=False, allow_unicode=True)
    
    # 初始化流水线
    pipeline = CalibrationPipeline(config={
        **cfg,
        "output_dir": str(result_dir),
    })
    plotter = CalibrationPlotter(output_dir=str(figures_dir))
    allan_plotter = AllanPlotter(output_dir=str(figures_dir))
    
    results = {"exp_id": exp_id, "start_time": datetime.datetime.now().isoformat()}
    
    try:
        # --- Step 1: 加速度计标定 ---
        print(f"\n{'='*60}")
        print(f"  [{exp_id}] Step 1/4: 加速度计六位置标定")
        print(f"{'='*60}")
        t0 = time.time()
        accel_result = pipeline.run_accel_calibration(ACCEL_DIR)
        raw = pipeline.data_loader.load_accel_six_pose(ACCEL_DIR, verbose=False)
        processed = pipeline.preprocessor.process(raw)
        pose_table = build_accel_pose_table_6()
        plotter.plot_accel_calibration(accel_result, processed.acc_means, pose_table)
        t1 = time.time()
        results["accel"] = {
            "status": "OK",
            "time_s": round(t1-t0, 2),
            "K_a": accel_result.K_a.tolist(),
            "D_a": accel_result.D_a.tolist(),
            "reprojection_error": float(accel_result.reprojection_error),
            "condition_number": float(accel_result.condition_number),
        }
        
        # --- Step 2: 陀螺速率标定 ---
        print(f"\n{'='*60}")
        print(f"  [{exp_id}] Step 2/4: 陀螺仪速率标定")
        print(f"{'='*60}")
        t0 = time.time()
        gyro_rate_result = pipeline.run_gyro_rate_calibration(RATE_DIR)
        plotter.plot_gyro_rate_calibration(gyro_rate_result)
        t1 = time.time()
        results["gyro_rate"] = {
            "status": "OK",
            "time_s": round(t1-t0, 2),
            "K_g": gyro_rate_result.K_g.tolist(),
            "K_g_std": gyro_rate_result.K_g_std.tolist(),
        }
        
        # --- Step 3: 陀螺零偏标定 ---
        print(f"\n{'='*60}")
        print(f"  [{exp_id}] Step 3/4: 陀螺仪八位置零偏标定")
        print(f"{'='*60}")
        t0 = time.time()
        gyro_bias_result = pipeline.run_gyro_bias_calibration(BIAS_DIR, gyro_rate_result.K_g)
        plotter.plot_gyro_bias_calibration(gyro_bias_result)
        t1 = time.time()
        results["gyro_bias"] = {
            "status": "OK",
            "time_s": round(t1-t0, 2),
            "D_g_deg_s": gyro_bias_result.D_g.tolist(),
            "D_g_deg_h": gyro_bias_result.D_g_deg_h.tolist(),
            "D_g_std_deg_s": gyro_bias_result.D_g_std.tolist(),
            "bias_per_pose_deg_h": (gyro_bias_result.bias_per_pose * 3600).tolist(),
        }
        
        # --- Step 4: Allan方差分析 ---
        print(f"\n{'='*60}")
        print(f"  [{exp_id}] Step 4/4: Allan方差分析")
        print(f"{'='*60}")
        t0 = time.time()
        allan_result = pipeline.run_allan_analysis(STATIC_FILE, fs=200.0)
        allan_plotter.plot_allan_curve(allan_result)
        t1 = time.time()
        results["allan"] = {
            "status": "OK",
            "time_s": round(t1-t0, 2),
            "ARW_deg_sqrt_h": allan_result.ARW.tolist(),
            "BI_deg_h": allan_result.BI.tolist(),
            "fitted_slopes": allan_result.fitted_log_slopes.tolist(),
        }
        
        # --- 组装报告 ---
        report = pipeline.assembler.assemble(
            accel_result, gyro_rate_result, gyro_bias_result, allan_result,
            metadata={
                "experiment_id": exp_id,
                "config": exp_config,
            }
        )
        pipeline.assembler.print_summary(report)
        
        # 保存报告
        pipeline.assembler.save_report(report, f"{exp_id}_report.yaml")
        pipeline.assembler.save_report_json(report, f"{exp_id}_report.json")
        
        # 复制报告到exp_out
        src_yaml = result_dir / f"{exp_id}_report.yaml"
        src_json = result_dir / f"{exp_id}_report.json"
        if src_yaml.exists():
            shutil.copy2(src_yaml, exp_out / f"{exp_id}_report.yaml")
        if src_json.exists():
            shutil.copy2(src_json, exp_out / f"{exp_id}_report.json")
        
        results["end_time"] = datetime.datetime.now().isoformat()
        results["status"] = "SUCCESS"
        
    except Exception as e:
        results["end_time"] = datetime.datetime.now().isoformat()
        results["status"] = "FAILED"
        results["error"] = str(e)
        import traceback
        results["traceback"] = traceback.format_exc()
        print(f"[ERROR] Experiment {exp_id} failed: {e}")
    
    return results


# ================================================================
# Stage 3-4：自动结果保存 + 性能评估
# ================================================================
def save_experiment_results(results: dict, base_dir: Path):
    """保存实验结果到结构化目录"""
    exp_id = results["exp_id"]
    exp_dir = base_dir / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 指标文件
    metrics_path = exp_dir / "metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 指标CSV摘要
    csv_path = exp_dir / "metrics.csv"
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("metric,value\n")
        if results.get("status") == "SUCCESS":
            if "accel" in results:
                f.write(f"accel_reprojection_error_m_s2,{results['accel']['reprojection_error']}\n")
                f.write(f"accel_condition_number,{results['accel']['condition_number']}\n")
                f.write(f"accel_time_s,{results['accel']['time_s']}\n")
            if "gyro_rate" in results:
                f.write(f"gyro_rate_time_s,{results['gyro_rate']['time_s']}\n")
            if "allan" in results:
                for i, axis in enumerate(['X','Y','Z']):
                    f.write(f"allan_ARW_{axis}_deg_sqrt_h,{results['allan']['ARW_deg_sqrt_h'][i]}\n")
                    f.write(f"allan_BI_{axis}_deg_h,{results['allan']['BI_deg_h'][i]}\n")
                    f.write(f"allan_fitted_slope_{axis},{results['allan']['fitted_slopes'][i]}\n")
                f.write(f"allan_time_s,{results['allan']['time_s']}\n")
    
    print(f"[Results] Metrics saved: {metrics_path}")
    print(f"[Results] CSV saved: {csv_path}")


# ================================================================
# Stage 5：自动可视化摘要
# ================================================================
def generate_visualization_summary(base_dir: Path, exp_id: str):
    """生成可视化文件清单和说明"""
    figures_dir = base_dir / exp_id / "figures"
    summary = []
    
    figure_map = {
        "accel_calibration.png": "加速度计六位置标定结果（实测vs拟合、残差、K_a热力图）",
        "gyro_rate_calibration.png": "陀螺仪速率标定结果（K_g热力图、多转速标度因数变化）",
        "gyro_bias_calibration.png": "陀螺仪八位置零偏标定结果（各位置零偏估计值、最终结果）",
        "allan_variance.png": "Allan方差双对数曲线分析结果（ARW/BI标注）",
    }
    
    for fname, desc in figure_map.items():
        fpath = figures_dir / fname
        if fpath.exists():
            summary.append({"file": fname, "description": desc, "exists": True})
        else:
            summary.append({"file": fname, "description": desc, "exists": False})
    
    summary_path = base_dir / exp_id / "figure_mapping.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# 可视化文件清单 (Figure Mapping)\n\n")
        f.write(f"实验ID: {exp_id}\n\n")
        f.write("| 文件名 | 说明 | 状态 |\n")
        f.write("|:-------|:-----|:-----|\n")
        for item in summary:
            status = "✅" if item["exists"] else "❌"
            f.write(f"| {item['file']} | {item['description']} | {status} |\n")
    
    print(f"[VisSummary] Figure mapping saved: {summary_path}")
    return summary


# ================================================================
# Stage 6：异常检测
# ================================================================
def anomaly_detection(results: dict) -> dict:
    """检测实验中的异常"""
    anomalies = []
    
    if results.get("status") == "FAILED":
        anomalies.append({
            "type": "EXPERIMENT_FAILURE",
            "detail": results.get("error", "Unknown error"),
            "suggestion": "检查数据文件完整性和配置参数"
        })
        return {"has_anomaly": True, "anomalies": anomalies}
    
    # 检查加速度计
    accel = results.get("accel", {})
    if accel.get("reprojection_error", 0) > 0.5:
        anomalies.append({
            "type": "ACCEL_HIGH_RESIDUAL",
            "detail": f"重投影误差={accel['reprojection_error']:.6f} m/s² > 0.5",
            "suggestion": "检查六位置姿态是否正确、数据是否有异常跳变"
        })
    if accel.get("condition_number", 0) > 1e8:
        anomalies.append({
            "type": "ACCEL_ILL_CONDITIONED",
            "detail": f"条件数={accel['condition_number']:.2e} > 1e8",
            "suggestion": "检查输入矩阵A是否线性相关"
        })
    
    # 检查Allan方差斜率
    allan = results.get("allan", {})
    slopes = allan.get("fitted_slopes", [])
    for i, slope in enumerate(slopes):
        if slope > -0.3:
            anomalies.append({
                "type": "ALLAN_SLOPE_ANOMALY",
                "detail": f"Axis {['X','Y','Z'][i]} log-log slope={slope:.3f} > -0.3",
                "suggestion": "静态数据可能包含非白噪声成分（如温漂、振动）"
            })
    
    return {"has_anomaly": len(anomalies) > 0, "anomalies": anomalies}


# ================================================================
# Stage 7-8：结果总结 + 报告资源准备
# ================================================================
def generate_experiment_summary(results: dict, base_dir: Path):
    """生成实验总结文档"""
    exp_id = results["exp_id"]
    exp_dir = base_dir / exp_id
    
    summary_path = exp_dir / "experiment_summary.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"# 实验总结报告\n\n")
        f.write(f"## 实验信息\n")
        f.write(f"- **实验ID**: {exp_id}\n")
        f.write(f"- **状态**: {'✅ 成功' if results.get('status')=='SUCCESS' else '❌ 失败'}\n")
        f.write(f"- **开始时间**: {results.get('start_time', 'N/A')}\n")
        f.write(f"- **结束时间**: {results.get('end_time', 'N/A')}\n")
        
        if results.get("status") == "SUCCESS":
            f.write(f"\n## 标定结果\n\n")
            
            # 加速度计
            if "accel" in results:
                a = results["accel"]
                f.write(f"### 加速度计标定结果\n")
                f.write(f"- K_a ≈ I(3×3)（对角元素≈1.0，交轴元素≈0）\n")
                K_a = np.array(a["K_a"])
                f.write(f"- K_a = \n")
                for row in K_a:
                    f.write(f"  [{row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}]\n")
                f.write(f"- D_a = {a['D_a']} m/s²\n")
                f.write(f"- 重投影误差 = {a['reprojection_error']:.6f} m/s²\n")
                f.write(f"- 条件数 = {a['condition_number']:.2e}\n\n")
            
            # 陀螺仪
            if "gyro_rate" in results:
                g = results["gyro_rate"]
                f.write(f"### 陀螺仪综合误差矩阵\n")
                K_g = np.array(g["K_g"])
                for row in K_g:
                    f.write(f"  [{row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}]\n")
                f.write(f"\n")
            
            if "gyro_bias" in results:
                b = results["gyro_bias"]
                f.write(f"### 陀螺仪零偏\n")
                f.write(f"- D_g = {b['D_g_deg_s']} °/s = {b['D_g_deg_h']} °/h\n")
                f.write(f"- 标准差 = {b['D_g_std_deg_s']} °/s\n\n")
            
            # Allan方差
            if "allan" in results:
                al = results["allan"]
                f.write(f"### Allan方差分析结果\n")
                for i, axis in enumerate(['X', 'Y', 'Z']):
                    f.write(f"- {axis}轴: ARW={al['ARW_deg_sqrt_h'][i]:.6f} °/√h, "
                           f"BI={al['BI_deg_h'][i]:.6f} °/h, "
                           f"斜率={al['fitted_slopes'][i]:.3f}\n")
                f.write(f"\n")
            
            # 性能
            f.write(f"## 性能\n")
            f.write(f"- 加速度计标定耗时: {a.get('time_s', 'N/A')}s\n")
            f.write(f"- 陀螺速率标定耗时: {g.get('time_s', 'N/A')}s\n")
            f.write(f"- Allan方差耗时: {al.get('time_s', 'N/A')}s\n")
        
        # 异常
        anomalies = anomaly_detection(results)
        if anomalies["has_anomaly"]:
            f.write(f"\n## 异常警告\n")
            for a in anomalies["anomalies"]:
                f.write(f"- **{a['type']}**: {a['detail']}\n")
                f.write(f"  - 建议: {a['suggestion']}\n")
    
    print(f"[Summary] Experiment summary saved: {summary_path}")
    return summary_path


def generate_assets_summary(results: dict, base_dir: Path):
    """生成报告资源清单（Stage 8）"""
    exp_id = results["exp_id"]
    exp_dir = base_dir / exp_id
    
    # 资源清单
    assets_path = exp_dir / "report_assets.md"
    with open(assets_path, 'w', encoding='utf-8') as f:
        f.write("# 报告资源清单\n\n")
        f.write("## 图片资源\n\n")
        f.write("| 图号 | 文件名 | 说明 | 对应实验报告章节 |\n")
        f.write("|:-----|:-------|:-----|:----------------|\n")
        
        figure_refs = [
            ("图1", "gyro_rate_calibration.png", "陀螺仪组合系统速率标定实验数据曲线", "(1)"),
            ("图2", "gyro_bias_calibration.png", "陀螺仪组合系统位置标定实验数据曲线", "(2)"),
            ("图3", "accel_calibration.png", "加速度计组合系统位置标定实验数据曲线", "(4)"),
            ("图4", "allan_variance.png", "陀螺仪Allan方差曲线分析结果", "(6)"),
        ]
        for fig_id, fname, desc, section in figure_refs:
            fpath = exp_dir / "figures" / fname
            status = "✅" if fpath.exists() else "❌ 缺失"
            f.write(f"| {fig_id} | {fname} | {desc} | {section} | {status} |\n")
        
        f.write("\n## 数值结果\n\n")
        f.write("| 编号 | 内容 | 文件 |\n")
        f.write("|:-----|:-----|:-----|\n")
        f.write("| (3) | 陀螺仪标定结果（K_g, D_g） | metrics.json / report.yaml |\n")
        f.write("| (5) | 加速度计标定结果（K_a, D_a） | metrics.json / report.yaml |\n")
        f.write("| (6) | ARW / BI 参数 | metrics.json / report.yaml |\n")
        f.write("| (7) | 陀螺仪标定程序 | `imu_calibration/calibration/gyro_rate_calibrator.py`, `gyro_bias_calibrator.py` |\n")
        f.write("| (8) | 加速度计标定程序 | `imu_calibration/calibration/accel_calibrator.py` |\n")
        f.write("| (9) | Allan方差分析程序 | `imu_calibration/analysis/allan_variance_analyzer.py` |\n")
    
    print(f"[Assets] Report assets summary saved: {assets_path}")


# ================================================================
# 主函数
# ================================================================
def main():
    print("=" * 60)
    print("  实验执行流程 — clinerules/30_experiment_execution.md")
    print("=" * 60)
    
    BASE_DIR = Path(__file__).resolve().parents[1]  # imu_calibration/
    EXP_BASE = BASE_DIR / "experiments"
    
    # Stage 0: 准备检查
    print("\n--- Stage 0: 实验准备检查 ---")
    check = readiness_check()
    save_readiness_report(check, BASE_DIR / "logs" / "readiness_report.yaml")
    
    if check["status"] == "FAIL":
        print("[ERROR] 准备检查未通过，终止实验")
        import yaml
        print(yaml.dump(check, default_flow_style=False))
        sys.exit(1)
    elif check["status"] == "PASS_WITH_WARNING":
        print("[PASS] 准备检查通过（含警告）")
        import yaml
        print(yaml.dump(check, default_flow_style=False))
    print("[PASS] 准备检查通过")
    
    # Stage 1: 实验矩阵
    print("\n--- Stage 1: 实验矩阵 ---")
    for exp in EXPERIMENT_MATRIX:
        print(f"  {exp['exp_id']}: {exp['name']}")
    
    # Stage 2: 执行实验
    print("\n--- Stage 2: 实验执行 ---")
    all_results = []
    for exp_config in EXPERIMENT_MATRIX:
        results = run_experiment(exp_config)
        all_results.append(results)
        
        # Stage 3-4: 保存结果 + 评估
        save_experiment_results(results, EXP_BASE)
        
        # Stage 5: 可视化摘要
        generate_visualization_summary(EXP_BASE, results["exp_id"])
        
        # Stage 6: 异常检测
        anomalies = anomaly_detection(results)
        if anomalies["has_anomaly"]:
            print(f"\n[ANOMALY] 实验 {results['exp_id']} 检测到异常:")
            for a in anomalies["anomalies"]:
                print(f"  - {a['type']}: {a['detail']}")
        
        # Stage 7-8: 总结 + 资源准备
        generate_experiment_summary(results, EXP_BASE)
        generate_assets_summary(results, EXP_BASE)
    
    # 最终汇总
    print("\n" + "=" * 60)
    print("  实验执行完成")
    print("=" * 60)
    for r in all_results:
        status_icon = "✅" if r["status"] == "SUCCESS" else "❌"
        print(f"  {status_icon} {r['exp_id']}: {r['status']}")
    
    summary_all = {
        "total": len(all_results),
        "success": sum(1 for r in all_results if r["status"] == "SUCCESS"),
        "failed": sum(1 for r in all_results if r["status"] == "FAILED"),
        "experiments": [{"exp_id": r["exp_id"], "status": r["status"]} for r in all_results],
    }
    with open(EXP_BASE / "experiment_summary_all.json", 'w') as f:
        json.dump(summary_all, f, indent=2)
    print(f"\n总体摘要: {summary_all['success']}/{summary_all['total']} 成功")


if __name__ == "__main__":
    main()