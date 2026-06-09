"""
GTIMU / GPFPD 语句解析器
对应 03_data_format_specification.md 的解析器契约
"""
from typing import Optional, List, Dict, Tuple
import numpy as np


# GTIMU 字段索引映射表（严格按 03_data_format_specification.md 定义）
GTIMU_FIELD_MAP: Dict[str, Dict] = {
    "GPSWeek": {"index": 0, "dtype": int},
    "GPSTime": {"index": 1, "dtype": float},
    "GyroX":   {"index": 2, "dtype": float, "range": (-500, 500)},
    "GyroY":   {"index": 3, "dtype": float, "range": (-500, 500)},
    "GyroZ":   {"index": 4, "dtype": float, "range": (-500, 500)},
    "AccX":    {"index": 5, "dtype": float, "range": (-30, 30)},
    "AccY":    {"index": 6, "dtype": float, "range": (-30, 30)},
    "AccZ":    {"index": 7, "dtype": float, "range": (-30, 30)},
    "Tpr":     {"index": 8, "dtype": float, "range": (-40, 85)},
}

GPFPD_FIELD_MAP: Dict[str, Dict] = {
    "GPSWeek": {"index": 0, "dtype": int},
    "GPSTime": {"index": 1, "dtype": float},
    "Roll":    {"index": 2, "dtype": float, "range": (-180, 180)},
    "Pitch":   {"index": 3, "dtype": float, "range": (-90, 90)},
    "Heading": {"index": 4, "dtype": float, "range": (0, 360)},
    "Lat":     {"index": 5, "dtype": float, "range": (-90, 90)},
    "Lon":     {"index": 6, "dtype": float, "range": (-180, 180)},
    "Height":  {"index": 7, "dtype": float, "range": (-500, 10000)},
    "Ve":      {"index": 8, "dtype": float},
    "Vn":      {"index": 9, "dtype": float},
    "Vu":      {"index": 10, "dtype": float},
    "Status":  {"index": 11, "dtype": int, "range": (0, 6)},
    "NSV":     {"index": 12, "dtype": int, "range": (0, 32)},
    "Age":     {"index": 13, "dtype": float, "range": (0, 999)},
}

# 字段数硬规约（Block Size 校验）
GTIMU_FIELD_COUNT = 9
GPFPD_FIELD_COUNT = 14


class GTIMUParser:
    """$GTIMU 语句解析器"""

    @staticmethod
    def parse_line(line: str) -> Optional[Dict[str, float]]:
        """
        解析单行 GTIMU 语句
        返回 dict 或 None（解析失败/无效行）
        
        Assertions:
        - 数据字段数必须 == 9 (不计消息头 "$GTIMU")
        - 数值字段必须在物理范围内
        
        实际数据格式（10个逗号分隔字段）:
            $GTIMU, GPSWeek, GPSTime, GyroX, GyroY, GyroZ, AccX, AccY, AccZ, Tpr*CS
        """
        line = line.strip()
        if not line:
            return None

        # 去除校验和后缀（如果存在）
        if '*' in line:
            line = line.split('*')[0]

        # 逗号分割
        tokens = line.split(',')

        # 格式1: 10字段, tokens[0] = "$GTIMU" or "GTIMU"
        # 格式2: 9字段, 不含消息头
        if len(tokens) == 10:
            # 去掉消息头, 取后9个数据字段
            data_tokens = tokens[1:]
        elif len(tokens) == GTIMU_FIELD_COUNT:
            data_tokens = tokens
        else:
            return None

        # Block Size 校验
        if len(data_tokens) != GTIMU_FIELD_COUNT:
            return None

        result = {}
        try:
            for name, spec in GTIMU_FIELD_MAP.items():
                idx = spec["index"]
                val = spec["dtype"](data_tokens[idx])
                # 物理范围断言
                if "range" in spec:
                    lo, hi = spec["range"]
                    if val < lo or val > hi:
                        return None  # 超出物理范围，标记无效行
                result[name] = val
        except (ValueError, IndexError):
            return None

        return result

    @staticmethod
    def parse_file(file_path: str, max_lines: int = None,
                   verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        解析 GTIMU 文件
        返回: (gyro_arr[N,3], acc_arr[N,3], time_arr[N], valid_count)
        """
        gyro_list, acc_list, time_list = [], [], []
        valid_count, invalid_count = 0, 0

        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                result = GTIMUParser.parse_line(line)
                if result:
                    gyro_list.append([result["GyroX"], result["GyroY"], result["GyroZ"]])
                    acc_list.append([result["AccX"], result["AccY"], result["AccZ"]])
                    time_list.append(result["GPSTime"])
                    valid_count += 1
                else:
                    invalid_count += 1

        if verbose and invalid_count > 0:
            print(f"[Parser] {file_path}: {valid_count} valid, {invalid_count} invalid lines")

        if valid_count == 0:
            raise ValueError(f"No valid GTIMU data in {file_path}")

        return (np.array(gyro_list, dtype=np.float64),
                np.array(acc_list, dtype=np.float64),
                np.array(time_list, dtype=np.float64),
                valid_count)


class GPFPDParser:
    """$GPFPD 语句解析器"""

    @staticmethod
    def parse_line(line: str) -> Optional[Dict[str, float]]:
        """解析单行 GPFPD 语句"""
        line = line.strip()
        if not line:
            return None
        if '*' in line:
            line = line.split('*')[0]

        tokens = line.split(',')

        # 格式1: 15字段, tokens[0] = "$GPFPD" or "GPFPD"
        # 格式2: 14字段, 不含消息头
        if len(tokens) == GPFPD_FIELD_COUNT + 1:  # 有消息头
            data_tokens = tokens[1:]
        elif len(tokens) == GPFPD_FIELD_COUNT:
            data_tokens = tokens
        else:
            return None

        result = {}
        try:
            for name, spec in GPFPD_FIELD_MAP.items():
                idx = spec["index"]
                val = spec["dtype"](data_tokens[idx])
                if "range" in spec:
                    lo, hi = spec["range"]
                    if val < lo or val > hi:
                        return None
                result[name] = val
        except (ValueError, IndexError):
            return None
        return result

    @staticmethod
    def parse_file(file_path: str, max_lines: int = None,
                   verbose: bool = True) -> Tuple[np.ndarray, int]:
        """
        解析 GPFPD 文件
        返回: (data_arr[N,14], valid_count)
        """
        data_list = []
        valid_count, invalid_count = 0, 0

        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                result = GPFPDParser.parse_line(line)
                if result:
                    row = [result[name] for name in GPFPD_FIELD_MAP.keys()]
                    data_list.append(row)
                    valid_count += 1
                else:
                    invalid_count += 1

        if verbose and invalid_count > 0:
            print(f"[Parser] {file_path}: {valid_count} valid, {invalid_count} invalid lines")
        if valid_count == 0:
            raise ValueError(f"No valid GPFPD data in {file_path}")

        return np.array(data_list, dtype=np.float64), valid_count