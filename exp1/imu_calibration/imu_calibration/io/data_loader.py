"""
数据加载器 — 将原始文件解析并按实验场景分组标记
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from .gtimu_parser import GTIMUParser, GPFPDParser
from ..common.types import RawDataBundle
from ..common.assertions import assert_physical_bounds, data_shape_monitor


class IMUDataLoader:
    """
    数据加载器
    职责：加载GTIMU/GPFPD文件，按实验场景分组标记，构建RawDataBundle
    """

    def __init__(self, data_root: str = None):
        self.data_root = Path(data_root) if data_root else None

    def load_gtimu_files(self, file_paths: List[str],
                         scenario_tags: List[str] = None,
                         max_lines: int = None,
                         verbose: bool = True) -> RawDataBundle:
        """
        加载多个GTIMU文件并合并，带场景标签

        参数:
            file_paths: GTIMU文件路径列表
            scenario_tags: 每个文件对应的场景标签（如pos1,pos2,...）
            max_lines: 每个文件最大读取行数(None=全部)

        返回:
            RawDataBundle
        """
        all_gyro, all_acc, all_time = [], [], []
        all_tags = []

        for i, fpath in enumerate(file_paths):
            tag = scenario_tags[i] if scenario_tags else f"file{i}"

            gyro, acc, time_arr, n_valid = GTIMUParser.parse_file(
                fpath, max_lines=max_lines, verbose=verbose
            )

            all_gyro.append(gyro)
            all_acc.append(acc)
            all_time.append(time_arr)
            all_tags.extend([tag] * len(gyro))

            if verbose:
                print(f"  [{tag}] loaded {n_valid} lines from {Path(fpath).name}")

        # 合并
        gyro_all = np.vstack(all_gyro) if len(all_gyro) > 1 else all_gyro[0]
        acc_all = np.vstack(all_acc) if len(all_acc) > 1 else all_acc[0]
        time_all = np.concatenate(all_time) if len(all_time) > 1 else all_time[0]

        # 物理边界断言
        assert_physical_bounds(gyro_data=gyro_all, accel_data=acc_all, label="IMUDataLoader.load_gtimu_files")

        # shape监控
        data_shape_monitor(gyro_all, 2, "gyro_all")
        data_shape_monitor(acc_all, 2, "acc_all")

        return RawDataBundle(
            gyro=gyro_all,
            accel=acc_all,
            timestamps=time_all,
            scenario_tags=all_tags,
            metadata={"sources": [str(Path(p).name) for p in file_paths]}
        )

    def load_accel_six_pose(self, base_dir: str,
                             file_names: List[str] = None,
                             verbose: bool = True) -> RawDataBundle:
        """
        加载加速度计六位置标定数据
        默认文件名: gtimu_1.log ~ gtimu_6.log
        """
        if file_names is None:
            file_names = [f"gtimu_{i}.log" for i in range(1, 7)]
        file_paths = [str(Path(base_dir) / fn) for fn in file_names]
        tags = [f"acc_pos{i}" for i in range(1, 7)]
        return self.load_gtimu_files(file_paths, tags, verbose=verbose)

    def load_gyro_eight_pose(self, base_dir: str,
                              file_names: List[str] = None,
                              verbose: bool = True) -> RawDataBundle:
        """
        加载陀螺仪八位置零偏标定数据
        默认文件名: gtimu_1.log ~ gtimu_8.log
        """
        if file_names is None:
            file_names = [f"gtimu_{i}.log" for i in range(1, 9)]
        file_paths = [str(Path(base_dir) / fn) for fn in file_names]
        tags = [f"gyro_bias_pos{i}" for i in range(1, 9)]
        return self.load_gtimu_files(file_paths, tags, verbose=verbose)

    def load_static_data(self, file_path: str,
                         max_lines: int = None,
                         verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载Allan方差所需的长时间静态GTIMU数据
        返回: (gyro_static[M,3], time_static[M])
        """
        gyro, _, time_arr, n = GTIMUParser.parse_file(
            file_path, max_lines=max_lines, verbose=verbose
        )
        return gyro, time_arr

    @staticmethod
    def load_gyro_rate_gtimu(rate_dir: str, axis_name: str,
                              rate_value: int, direction: str,
                              verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        加载单个速率标定的GTIMU文件

        参数:
            rate_dir: 速率标定目录路径
            axis_name: "位置一"/"位置二"/"位置三" 对应X/Y/Z轴
            rate_value: 转速值 10~50
            direction: "+" 或 "-"
        返回:
            (gyro[N,3], acc[N,3], time[N])
        """
        sign = direction
        fname = f"gtimu_{sign}{rate_value}.log"
        fpath = Path(rate_dir) / axis_name / fname
        gyro, acc, time_arr, n = GTIMUParser.parse_file(
            str(fpath), verbose=verbose
        )
        return gyro, acc, time_arr