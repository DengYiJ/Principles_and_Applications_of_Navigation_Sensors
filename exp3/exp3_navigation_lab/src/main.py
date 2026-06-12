"""
主入口程序
==========
实验（3）惯性导航实验的完整执行流程

用法:
    python src/main.py --imu_path <IMU数据文件> --ref_path <参考数据文件>
"""

import numpy as np
import argparse
import os
import sys
import yaml
import logging
from typing import Dict, Optional

# 确保src包可以从项目根目录导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_io import DataLoader, DataSaver
from src.preprocessing import Preprocessor
from src.alignment import CoarseAligner, FineAligner
from src.attitude import AttitudeUpdater
from src.analysis import ComparisonAnalyzer, Plotter
from src.utils.euler_angles import rad2deg


def setup_logging(config: dict) -> logging.Logger:
    """配置日志"""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))
    log_file = log_config.get('log_file', './logs/exp3_run.log')
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('exp3')


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"[Config] 已加载配置: {config_path}")
    return config


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='实验(3) 惯性导航实验 - 捷联惯导解算系统')
    parser.add_argument('--config', type=str, default='configs/exp3_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--imu_path', type=str, help='IMU数据文件路径')
    parser.add_argument('--ref_path', type=str, help='参考数据文件路径 ($GPFPD)')
    parser.add_argument('--mode', type=str, default='alignment',
                        choices=['alignment', 'update', 'full'],
                        help='运行模式: alignment=仅对准, update=仅姿态更新, full=全部')
    parser.add_argument('--save_results', action='store_true', default=True,
                        help='是否保存结果')
    return parser.parse_args()


def run_alignment_pipeline(imu_data: np.ndarray,
                           ref_aligned: np.ndarray,
                           config: dict,
                           logger: logging.Logger) -> Dict:
    """
    对准流水线: 粗对准 -> 精对准
    """
    latitude_deg = config['earth']['latitude_deg']
    imu_err_tuple = (
        config['imu_err']['gyro_bias_deg_h'],
        config['imu_err']['acc_bias_mg'],
        config['imu_err']['gyro_arw_deg_sqrth'],
        config['imu_err']['acc_vrw_m_s_sqrth']
    )
    
    results = {}
    
    # ====== 粗对准 ======
    logger.info("=" * 50)
    logger.info("开始双矢量法粗对准")
    logger.info("=" * 50)
    
    coarse_aligner = CoarseAligner(
        latitude_deg=latitude_deg,
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
    
    results['Cnb0'] = Cnb0
    results['att_coarse'] = att_coarse
    
    # ====== 精对准 ======
    logger.info("=" * 50)
    logger.info("开始卡尔曼滤波精对准")
    logger.info("=" * 50)
    
    fine_aligner = FineAligner(
        latitude_deg=latitude_deg,
        imu_err=imu_err_tuple
    )
    
    fa_config = config.get('fine_align', {})
    init_cov = fa_config.get('init_cov', {})
    init_cov['vel_noise_mps'] = fa_config.get('R', {}).get('vel_noise_mps', 0.01)
    
    att_fine, X_history, P_diag_history = fine_aligner.run(
        imu_data,
        Cnb0,
        reset_feedback=fa_config.get('feedback', {}).get('reset_phi', True),
        init_cov_config=init_cov
    )
    
    gyro_bias_est = fine_aligner.get_gyro_bias_estimate()
    acc_bias_est = fine_aligner.get_acc_bias_estimate()
    
    results['att_fine'] = att_fine
    results['X_history'] = X_history
    results['P_diag_history'] = P_diag_history
    results['gyro_bias_est'] = gyro_bias_est
    results['acc_bias_est'] = acc_bias_est
    
    # ====== 对准对比 ======
    if len(ref_aligned) > 0:
        ref_att = tuple(ref_aligned[0])
        coarse_comp = ComparisonAnalyzer.compare_alignment(att_coarse, ref_att)
        fine_comp = ComparisonAnalyzer.compare_alignment(att_fine, ref_att)
        
        results['coarse_comparison'] = coarse_comp
        results['fine_comparison'] = fine_comp
        
        ComparisonAnalyzer.print_comparison("粗对准", coarse_comp)
        ComparisonAnalyzer.print_comparison("精对准", fine_comp)
    
    return results


def run_attitude_update(imu_data: np.ndarray,
                        ref_aligned: np.ndarray,
                        config: dict,
                        att_init_deg: tuple,
                        gyro_bias: tuple = (0.0, 0.0, 0.0),
                        logger: logging.Logger = None) -> Dict:
    """姿态更新流水线"""
    if logger:
        logger.info("=" * 50)
        logger.info("开始姿态更新")
        logger.info("=" * 50)
    
    latitude_deg = config['earth']['latitude_deg']
    au_config = config.get('attitude_update', {})
    
    updater = AttitudeUpdater(
        latitude_deg=latitude_deg,
        integration_method=au_config.get('integration_method', 'picard_2nd'),
        normalize_quat=au_config.get('normalize_quat', True)
    )
    
    att_history = updater.run(imu_data, att_init_deg, gyro_bias=gyro_bias)
    
    results = {'att_history': att_history}
    
    if ref_aligned is not None and len(ref_aligned) > 0:
        update_comp = ComparisonAnalyzer.compare_attitude_sequence(att_history, ref_aligned)
        results['update_comparison'] = update_comp
        ComparisonAnalyzer.print_comparison("姿态更新", update_comp)
    
    return results


def main():
    """主入口"""
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logging(config)
    
    logger.info("=" * 60)
    logger.info("实验(3) 惯性导航实验 -- 捷联惯导解算系统")
    logger.info("=" * 60)
    
    # 加载数据
    imu_file = args.imu_path
    ref_file = args.ref_path
    if not imu_file:
        imu_file = input("请输入IMU数据文件路径: ").strip()
    if not ref_file:
        ref_file = input("请输入参考数据文件路径($GPFPD): ").strip()
    
    logger.info(f"加载IMU数据: {imu_file}")
    logger.info(f"加载参考数据: {ref_file}")
    
    imu_raw, ref_raw = DataLoader.load_dataset(
        imu_file, ref_file,
        scale_gyro=config['data']['scale_gyro'],
        scale_acc=config['data']['scale_acc']
    )
    
    logger.info("预处理IMU数据...")
    pre_config = config.get('preprocess', {})
    imu_data = Preprocessor.process(
        imu_raw,
        outlier_threshold=pre_config.get('outlier_threshold_sigma', 5.0),
        interpolation=pre_config.get('interpolation', 'linear')
    )
    
    ref_aligned = Preprocessor.align_reference(ref_raw, imu_data[:, 6])
    ref_seq = Preprocessor.extract_reference_for_compare(ref_raw)
    
    results = {}
    
    if args.mode in ['alignment', 'full']:
        align_results = run_alignment_pipeline(imu_data, ref_aligned, config, logger)
        results.update(align_results)
    
    if args.mode in ['update', 'full']:
        if 'att_fine' in results:
            att_init = results['att_fine']
            gyro_bias = tuple(results.get('gyro_bias_est', np.zeros(3)))
        elif 'att_coarse' in results:
            att_init = results['att_coarse']
            gyro_bias = (0.0, 0.0, 0.0)
            logger.warning("未找到精对准结果，使用粗对准作为姿态更新初始值")
        else:
            logger.error("无可用初始姿态，请先执行对准")
            return
        
        update_results = run_attitude_update(imu_data, ref_seq, config, att_init, gyro_bias, logger)
        results.update(update_results)
    
    # 保存结果
    if args.save_results:
        save_path = config['output'].get('save_path', './results/')
        os.makedirs(save_path, exist_ok=True)
        
        if 'att_coarse' in results:
            DataSaver.save_alignment_result(
                os.path.join(save_path, 'coarse_align_result.csv'),
                '粗对准',
                results['att_coarse'][0], results['att_coarse'][1], results['att_coarse'][2],
                0, 0, 0
            )
        
        if 'att_fine' in results:
            DataSaver.save_alignment_result(
                os.path.join(save_path, 'fine_align_result.csv'),
                '精对准',
                results['att_fine'][0], results['att_fine'][1], results['att_fine'][2],
                0, 0, 0
            )
        
        if 'X_history' in results:
            DataSaver.save_kf_estimates(
                os.path.join(save_path, 'kf_estimates.csv'),
                results['X_history'],
                results.get('P_diag_history', np.zeros((12, 1))),
                imu_data[:results['X_history'].shape[1], 6]
            )
        
        plotter = Plotter(
            save_dir=os.path.join(save_path, 'plots'),
            dpi=config['output'].get('plot_dpi', 300),
            fmt=config['output'].get('plot_format', 'png')
        )
        
        plotter.plot_all_results(
            imu_data,
            {'att_history': ref_aligned if args.mode in ['alignment', 'full'] else None},
            {'att_history': ref_aligned if args.mode in ['alignment', 'full'] else None,
             'X_history': results.get('X_history')},
            {'att_history': results.get('att_history'),
             'ref_aligned': ref_seq}
        )
    
    logger.info("=" * 60)
    logger.info("实验(3) 处理完成")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()