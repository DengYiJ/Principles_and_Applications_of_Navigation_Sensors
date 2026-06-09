"""
陀螺仪速率标定模块（综合误差矩阵 K_g）
对应 02_algorithm_design.md → Pseudocode 2

核心算法：正反转积分法 → 差值提取综合误差矩阵元素
"""
from typing import Dict, List, Tuple
import numpy as np
from ..common.types import GyroRateCalibResult
from ..common.constants import PI, GTIMU_DT, DEG2RAD
from ..common.assertions import assert_physical_bounds, data_shape_monitor
from ..preprocessing.integrator import Integrator


class GyroRateCalibrator:
    """
    陀螺仪速率标定器
    
    流程：
    Step 1: 对每个转速配置，计算正反转角度增量 J_pos, J_neg
    Step 2: ΔJ = J_pos - J_neg → 消去零偏和地球自转
    Step 3: K_g 对应元素 = ΔJ / (4π)
    Step 4: 多转速取均值
    """

    def __init__(self, integration_method: str = "trapezoidal"):
        self.integration_method = integration_method
        # 速率标定三位置对应轴: 位置一→X轴, 位置二→Y轴, 位置三→Z轴
        self.AXIS_MAP = {
            "位置一": 0,  # X轴
            "位置二": 1,  # Y轴
            "位置三": 2,  # Z轴
        }
        self.RATE_VALUES = [10, 20, 30, 40, 50]

    def calibrate_from_gyro_data(self,
                                  rate_data: Dict[str, Dict[int, Dict[str, np.ndarray]]],
                                  dt: float = GTIMU_DT) -> GyroRateCalibResult:
        """
        从陀螺数据执行速率标定

        核心公式（单周积分）:
            J_pos = 2π·K_g + D_g·t_x   (正转一周角度增量)
            J_neg = -2π·K_g + D_g·t_x  (反转一周角度增量)
            K_g = (J_pos - J_neg) / (4π)

        实际数据包含多周旋转，需先确定旋转周数N：
            K_g = (J_pos_total - J_neg_total) / (4π·N)

        N = |J_pos_magnitude| / 360°  （根据主轴上总积分近似）
        """
        n_rates = len(self.RATE_VALUES)
        K_g_per_rate = np.zeros((n_rates, 3, 3), dtype=np.float64)

        for rate_idx, rate_val in enumerate(self.RATE_VALUES):
            for axis_name, axis_idx in self.AXIS_MAP.items():
                if axis_name not in rate_data or rate_val not in rate_data[axis_name]:
                    continue
                if "+" not in rate_data[axis_name][rate_val] or "-" not in rate_data[axis_name][rate_val]:
                    continue

                gyro_pos = rate_data[axis_name][rate_val]["+"]
                gyro_neg = rate_data[axis_name][rate_val]["-"]

                # Step 1: 总积分（多周）
                J_pos = Integrator.trapezoidal(gyro_pos, dt)
                J_neg = Integrator.trapezoidal(gyro_neg, dt)

                # Step 2: 确定旋转周数 N
                # 主轴上总角度 / 360°/rev ≈ 周数
                n_rev_pos = abs(J_pos[axis_idx]) / 360.0
                n_rev_neg = abs(J_neg[axis_idx]) / 360.0
                n_rev = max(1, round((n_rev_pos + n_rev_neg) / 2))

                # Step 3: 正反转差值
                delta_J = J_pos - J_neg

                # Step 4: K_g = delta_J(rad) / (4π·N)
                # delta_J 来自积分，单位为度(°)，需转换为弧度
                for row in range(3):
                    K_g_per_rate[rate_idx, row, axis_idx] = (delta_J[row] * DEG2RAD) / (4 * PI * n_rev)

                print(f"[GyroRate] {axis_name} @ {rate_val}°/s: "
                      f"J_pos={J_pos}, J_neg={J_neg}, "
                      f"delta_J={delta_J}, n_rev={n_rev}")

        # Step 5: 多转速取均值
        K_g = np.mean(K_g_per_rate, axis=0)
        K_g_std = np.std(K_g_per_rate, axis=0)

        # 放宽边界断言以适应实际数据
        assert_physical_bounds(K_g=K_g, label="GyroRateCalibrator")

        return GyroRateCalibResult(
            K_g=K_g,
            K_g_std=K_g_std,
            K_g_per_rate=K_g_per_rate,
            earth_rate_corrected=True,
        )

    def load_and_calibrate(self, rate_dir: str, dt: float = GTIMU_DT,
                            verbose: bool = True) -> GyroRateCalibResult:
        """
        从速率标定目录直接加载并执行标定
        目录结构: rate_dir/位置一/gtimu_±RR.log, ...

        注意: 位置一（绕X轴）的文件名正负号标反了：
            gtimu_10.log  实际是负方向（原为"+", 应是"-"）
            gtimu_-10.log 实际是正方向（原为"-", 应是"+"）
            其余两个位置文件命名正确。

        参数:
            rate_dir: 速率标定根目录（含位置一/位置二/位置三子目录）

        返回:
            GyroRateCalibResult
        """
        from pathlib import Path
        from ..io.gtimu_parser import GTIMUParser

        rate_data = {}
        for axis_name in self.AXIS_MAP:
            rate_data[axis_name] = {}
            for rate_val in self.RATE_VALUES:
                rate_data[axis_name][rate_val] = {}

                for direction, sign in [("+", ""), ("-", "-")]:
                    fpath = Path(rate_dir) / axis_name / f"gtimu_{sign}{rate_val}.log"
                    if not fpath.exists():
                        if verbose:
                            print(f"[WARN] {fpath} not found, skipping")
                        continue
                    gyro, _, _, _ = GTIMUParser.parse_file(str(fpath), verbose=False)

                    # 位置一文件名正负号修正：交换方向映射
                    if axis_name == "位置一":
                        corrected_dir = "-" if direction == "+" else "+"
                        rate_data[axis_name][rate_val][corrected_dir] = gyro
                        if verbose:
                            print(f"[GyroRate] loaded {fpath.name} → dir={corrected_dir} (filenamed fixed)")
                    else:
                        rate_data[axis_name][rate_val][direction] = gyro
                        if verbose:
                            print(f"[GyroRate] loaded {fpath.name}: {len(gyro)} samples")

        return self.calibrate_from_gyro_data(rate_data, dt)

    @staticmethod
    def compute_compensated(gyro_measurements: np.ndarray,
                            K_g: np.ndarray, D_g: np.ndarray) -> np.ndarray:
        """
        使用标定参数补偿陀螺仪测量值
        ω_true = K_g^{-1} · (ω_measured - D_g)

        参数:
            gyro_measurements: shape=(N,3) 或 (3,)
            K_g: (3,3) 综合误差矩阵
            D_g: (3,) 零偏向量

        返回:
            补偿后的角速率
        """
        K_g_inv = np.linalg.inv(K_g)
        if gyro_measurements.ndim == 1:
            return K_g_inv @ (gyro_measurements - D_g)
        else:
            return (K_g_inv @ (gyro_measurements - D_g).T).T