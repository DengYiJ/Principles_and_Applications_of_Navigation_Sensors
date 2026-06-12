"""
实验(3) 批量运行脚本
====================
对初始对准(2个姿态)和姿态更新(3个姿态)数据集运行完整流水线
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time as time_module

from src.data_io import DataLoader, DataSaver
from src.preprocessing import Preprocessor
from src.alignment import CoarseAligner, FineAligner
from src.attitude import AttitudeUpdater
from src.analysis import ComparisonAnalyzer, Plotter
from src.utils.euler_angles import rad2deg, deg2rad
from src.utils.earth_model import gravity_n, earth_rate_n

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def process_alignment(imu_file, gpfpd_file, attitude_name, config):
    """处理单个初始对准姿态"""
    print(f"\n{'='*60}")
    print(f"处理初始对准: {attitude_name}")
    print(f"  IMU: {os.path.basename(imu_file)}")
    print(f"  GPFPD: {os.path.basename(gpfpd_file)}")
    print(f"{'='*60}")
    
    # 加载数据
    imu_raw, ref_raw = DataLoader.load_dataset(
        imu_file, gpfpd_file,
        scale_gyro=config['data']['scale_gyro'],
        scale_acc=config['data']['scale_acc']
    )
    
    # 预处理
    pre_config = config.get('preprocess', {})
    imu_data = Preprocessor.process(
        imu_raw,
        outlier_threshold=pre_config.get('outlier_threshold_sigma', 5.0),
        interpolation=pre_config.get('interpolation', 'linear')
    )
    
    ref_aligned = Preprocessor.align_reference(ref_raw, imu_data[:, 6])
    
    # 粗对准
    print(f"\n--- 双矢量法粗对准 ---")
    coarse_aligner = CoarseAligner(
        latitude_deg=config['earth']['latitude_deg'],
        wie=config['earth']['wie'],
        g=config['earth']['g']
    )
    
    ca_config = config['coarse_align']
    Cnb0, att_coarse = coarse_aligner.run(
        imu_data,
        t1=ca_config['t1'],
        t2=ca_config['t2'],
        n_avg_pairs=ca_config.get('n_avg_pairs', 5)
    )
    
    # 精对准
    print(f"\n--- 卡尔曼滤波精对准 ---")
    imu_err_tuple = (
        config['imu_err']['gyro_bias_deg_h'],
        config['imu_err']['acc_bias_mg'],
        config['imu_err']['gyro_arw_deg_sqrth'],
        config['imu_err']['acc_vrw_m_s_sqrth']
    )
    
    fine_aligner = FineAligner(
        latitude_deg=config['earth']['latitude_deg'],
        imu_err=imu_err_tuple
    )
    
    fa_config = config.get('fine_align', {})
    init_cov = fa_config.get('init_cov', {})
    init_cov['vel_noise_mps'] = fa_config.get('R', {}).get('vel_noise_mps', 0.01)
    
    att_fine, X_history, P_diag = fine_aligner.run(
        imu_data, Cnb0,
        reset_feedback=fa_config.get('feedback', {}).get('reset_phi', True),
        init_cov_config=init_cov
    )
    
    gyro_bias_est = fine_aligner.get_gyro_bias_estimate()
    acc_bias_est = fine_aligner.get_acc_bias_estimate()
    
    # 对比参考
    if len(ref_aligned) > 0:
        ref_att = tuple(ref_aligned[-1])  # 取最后收敛时刻
        coarse_comp = ComparisonAnalyzer.compare_alignment(att_coarse, ref_att)
        fine_comp = ComparisonAnalyzer.compare_alignment(att_fine, ref_att)
        ComparisonAnalyzer.print_comparison("粗对准", coarse_comp)
        ComparisonAnalyzer.print_comparison("精对准", fine_comp)
        
        # 保存结果
        result_dir = os.path.join(RESULTS_DIR, attitude_name)
        os.makedirs(result_dir, exist_ok=True)
        
        DataSaver.save_alignment_result(
            os.path.join(result_dir, 'coarse_align_result.csv'),
            '粗对准',
            att_coarse[0], att_coarse[1], att_coarse[2],
            ref_att[0], ref_att[1], ref_att[2]
        )
        DataSaver.save_alignment_result(
            os.path.join(result_dir, 'fine_align_result.csv'),
            '精对准',
            att_fine[0], att_fine[1], att_fine[2],
            ref_att[0], ref_att[1], ref_att[2]
        )
        
        # 生成图表
        plotter = Plotter(
            save_dir=os.path.join(result_dir, 'plots'),
            dpi=config['output'].get('plot_dpi', 300),
            fmt=config['output'].get('plot_format', 'png')
        )
        plotter.plot_imu_raw_data(imu_data, f'初始对准-{attitude_name}',
                                   f'imu_raw_{attitude_name}')
        plotter.plot_kf_error_params(X_history, filename=f'kf_params_{attitude_name}')
    
    return {
        'Cnb0': Cnb0,
        'att_coarse': att_coarse,
        'att_fine': att_fine,
        'gyro_bias': gyro_bias_est,
        'acc_bias': acc_bias_est
    }


def process_attitude_update(imu_file, gpfpd_file, attitude_name, config,
                            att_init_deg, gyro_bias=(0,0,0)):
    """处理单个姿态更新姿态"""
    print(f"\n{'='*60}")
    print(f"处理姿态更新: {attitude_name} (初始: {att_init_deg})")
    print(f"  IMU: {os.path.basename(imu_file)}")
    print(f"  GPFPD: {os.path.basename(gpfpd_file)}")
    print(f"{'='*60}")
    
    # 加载数据
    imu_raw, ref_raw = DataLoader.load_dataset(
        imu_file, gpfpd_file,
        scale_gyro=config['data']['scale_gyro'],
        scale_acc=config['data']['scale_acc']
    )
    
    # 预处理
    pre_config = config.get('preprocess', {})
    imu_data = Preprocessor.process(
        imu_raw,
        outlier_threshold=pre_config.get('outlier_threshold_sigma', 5.0),
        interpolation=pre_config.get('interpolation', 'linear')
    )
    
    ref_seq = Preprocessor.extract_reference_for_compare(ref_raw)
    
    # 姿态更新
    updater = AttitudeUpdater(
        latitude_deg=config['earth']['latitude_deg'],
        integration_method=config['attitude_update'].get('integration_method', 'picard_2nd')
    )
    
    att_history = updater.run(imu_data, att_init_deg, gyro_bias=gyro_bias)
    
    # 对比
    comp = ComparisonAnalyzer.compare_attitude_sequence(att_history, ref_seq)
    ComparisonAnalyzer.print_comparison(f"姿态更新-{attitude_name}", comp)
    
    # 保存
    result_dir = os.path.join(RESULTS_DIR, f'update_{attitude_name}')
    os.makedirs(result_dir, exist_ok=True)
    
    plotter = Plotter(
        save_dir=os.path.join(result_dir, 'plots'),
        dpi=config['output'].get('plot_dpi', 300),
        fmt=config['output'].get('plot_format', 'png')
    )
    plotter.plot_attitude_update_comparison(
        att_history, ref_seq,
        filename=f'attitude_update_{attitude_name}'
    )
    
    # 保存误差统计
    error_stats = ComparisonAnalyzer.compute_all_metrics(
        att_history[:min(len(att_history), len(ref_seq))],
        ref_seq[:min(len(att_history), len(ref_seq))]
    )
    DataSaver.save_error_statistics(
        os.path.join(result_dir, 'error_statistics.csv'),
        {k: v for k, v in error_stats.items()}
    )
    
    return {'att_history': att_history, 'comparison': comp}


def main():
    import yaml
    
    print("=" * 70)
    print("实验(3) 惯性导航实验 — 真实数据批量处理")
    print("=" * 70)
    
    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'exp3_config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # ====== Phase 1: 初始对准 ======
    print(f"\n{'#'*70}")
    print("# Phase 1: 初始对准 (粗对准 + 精对准)")
    print(f"{'#'*70}")
    
    alignment_results = {}
    
    # 姿态1: [0°,0°,0°]
    r1 = process_alignment(
        os.path.join(DATA_DIR, '初始对准', 'gtimu_0_0_0.log'),
        os.path.join(DATA_DIR, '初始对准', 'gpfpd_0_0_0.log'),
        '对准_0_0_0', config
    )
    alignment_results['0_0_0'] = r1
    
    # 姿态2: [30°,0°,0°]
    r2 = process_alignment(
        os.path.join(DATA_DIR, '初始对准', 'gtimu_30_0_0.log'),
        os.path.join(DATA_DIR, '初始对准', 'gpfpd_30_0_0.log'),
        '对准_30_0_0', config
    )
    alignment_results['30_0_0'] = r2
    
    # ====== Phase 2: 姿态更新 ======
    print(f"\n{'#'*70}")
    print("# Phase 2: 姿态更新")
    print(f"{'#'*70}")
    
    update_results = {}
    
    # 使用精对准结果的陀螺零偏（取姿态1的估计值）
    gyro_bias = tuple(alignment_results['0_0_0']['gyro_bias'])
    print(f"\n使用陀螺零偏估计: {np.rad2deg(gyro_bias[0]):.6f}, "
          f"{np.rad2deg(gyro_bias[1]):.6f}, {np.rad2deg(gyro_bias[2]):.6f} °/s")
    
    update_results['0_20_0'] = process_attitude_update(
        os.path.join(DATA_DIR, '姿态更新', 'gtimu_0_20_0.log'),
        os.path.join(DATA_DIR, '姿态更新', 'gpfpd_0_20_0.log'),
        '0_20_0', config, (0, 20, 0), gyro_bias
    )
    
    update_results['0_0_90'] = process_attitude_update(
        os.path.join(DATA_DIR, '姿态更新', 'gtimu_0_0_90.log'),
        os.path.join(DATA_DIR, '姿态更新', 'gpfpd_0_0_90.log'),
        '0_0_90', config, (0, 0, 90), gyro_bias
    )
    
    update_results['-30_-20_180'] = process_attitude_update(
        os.path.join(DATA_DIR, '姿态更新', 'gtimu_-30_-20_180.log'),
        os.path.join(DATA_DIR, '姿态更新', 'gpfpd_-30_-20_180.log'),
        '-30_-20_180', config, (-30, -20, 180), gyro_bias
    )
    
    # ====== 汇总 ======
    print(f"\n{'='*70}")
    print("汇总报告")
    print(f"{'='*70}")
    
    print("\n[初始对准结果]")
    print(f"{'姿态':<15} {'方法':<10} {'横滚(°)':<12} {'俯仰(°)':<12} {'航向(°)':<12}")
    print("-" * 61)
    for name, r in alignment_results.items():
        print(f"{name:<15} {'粗对准':<10} {r['att_coarse'][0]:<12.4f} {r['att_coarse'][1]:<12.4f} {r['att_coarse'][2]:<12.4f}")
        print(f"{name:<15} {'精对准':<10} {r['att_fine'][0]:<12.4f} {r['att_fine'][1]:<12.4f} {r['att_fine'][2]:<12.4f}")
    
    print(f"\n[姿态更新误差统计]")
    print(f"{'姿态':<15} {'RMSE_横滚':<12} {'RMSE_俯仰':<12} {'RMSE_航向':<12}")
    print("-" * 51)
    for name, r in update_results.items():
        m = r['comparison']['metrics']
        print(f"{name:<15} {m['RMSE'][0]:<12.4f} {m['RMSE'][1]:<12.4f} {m['RMSE'][2]:<12.4f}")
    
    print(f"\n结果已保存至: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()