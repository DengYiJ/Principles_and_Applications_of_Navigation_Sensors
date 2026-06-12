"""
对比分析器
==========
计算各算法结果与参考姿态的误差统计
"""

import numpy as np
from typing import Dict, Tuple, Optional


class ComparisonAnalyzer:
    """误差统计与对比分析"""
    
    @staticmethod
    def compute_rmse(estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        计算均方根误差 (RMSE)
        输入: estimate — [N×3] 解算值
              reference — [N×3] 参考值
        输出: [3] [rmse_roll, rmse_pitch, rmse_yaw]
        """
        diff = estimate - reference
        return np.sqrt(np.mean(diff**2, axis=0))
    
    @staticmethod
    def compute_mae(estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        计算平均绝对误差 (MAE)
        输出: [3]
        """
        return np.mean(np.abs(estimate - reference), axis=0)
    
    @staticmethod
    def compute_max_error(estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """计算最大绝对误差"""
        return np.max(np.abs(estimate - reference), axis=0)
    
    @staticmethod
    def compute_std_error(estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """计算误差标准差"""
        return np.std(estimate - reference, axis=0)
    
    @staticmethod
    def compute_all_metrics(estimate: np.ndarray, 
                            reference: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算全部误差指标
        返回: Dict with keys: 'RMSE', 'MAE', 'MaxError', 'StdError'
        """
        return {
            'RMSE': ComparisonAnalyzer.compute_rmse(estimate, reference),
            'MAE': ComparisonAnalyzer.compute_mae(estimate, reference),
            'MaxError': ComparisonAnalyzer.compute_max_error(estimate, reference),
            'StdError': ComparisonAnalyzer.compute_std_error(estimate, reference)
        }
    
    @staticmethod
    def compare_alignment(att_result: Tuple[float, float, float],
                          att_ref: Tuple[float, float, float]) -> Dict:
        """
        对准结果对比
        
        输入:
            att_result — (roll, pitch, yaw) 解算结果 (°)
            att_ref — (roll, pitch, yaw) 参考值 (°)
        
        返回: Dict 包含差值
        """
        result = np.array(att_result)
        ref = np.array(att_ref)
        diff = result - ref
        
        return {
            'result': att_result,
            'reference': att_ref,
            'diff': tuple(diff),
            'diff_norm': np.linalg.norm(diff)
        }
    
    @staticmethod
    def compare_attitude_sequence(att_history: np.ndarray,
                                   ref_aligned: np.ndarray) -> Dict:
        """
        姿态更新序列对比
        
        输入:
            att_history — [N×3] 解算姿态序列 (°)
            ref_aligned — [M×3] 参考姿态序列 (°)
        
        返回: Dict 包含全部误差指标
        """
        # 确保长度一致
        min_len = min(len(att_history), len(ref_aligned))
        est = att_history[:min_len]
        ref = ref_aligned[:min_len]
        
        metrics = ComparisonAnalyzer.compute_all_metrics(est, ref)
        
        return {
            'metrics': metrics,
            'error_sequence': est - ref,
            'estimate': est,
            'reference': ref
        }
    
    @staticmethod
    def print_comparison(name: str, result: Dict):
        """打印对比结果"""
        print(f"\n{'='*60}")
        print(f" {name} 对比结果")
        print(f"{'='*60}")
        
        if 'result' in result:
            print(f"  解算:  roll={result['result'][0]:.4f}°  pitch={result['result'][1]:.4f}°  yaw={result['result'][2]:.4f}°")
            print(f"  参考:  roll={result['reference'][0]:.4f}°  pitch={result['reference'][1]:.4f}°  yaw={result['reference'][2]:.4f}°")
            print(f"  误差:  roll={result['diff'][0]:.4f}°  pitch={result['diff'][1]:.4f}°  yaw={result['diff'][2]:.4f}°")
            print(f"  总误差范数: {result['diff_norm']:.4f}°")
        
        if 'metrics' in result:
            m = result['metrics']
            print(f"\n  误差统计:")
            print(f"    RMSE:     roll={m['RMSE'][0]:.4f}°  pitch={m['RMSE'][1]:.4f}°  yaw={m['RMSE'][2]:.4f}°")
            print(f"    MAE:      roll={m['MAE'][0]:.4f}°  pitch={m['MAE'][1]:.4f}°  yaw={m['MAE'][2]:.4f}°")
            print(f"    MaxError: roll={m['MaxError'][0]:.4f}°  pitch={m['MaxError'][1]:.4f}°  yaw={m['MaxError'][2]:.4f}°")