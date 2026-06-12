"""
数据提取层 — Data Extractor
=============================
Module A1: ExtractObsTime
Module A2: ExtractEphemerisMatrix
Module A3: ExtractPseudorangeMatrix

从 NovAtel ASCII 日志中提取观测时刻、星历矩阵、伪距矩阵。

设计依据: 03_algorithm_design.md Stage 2 Module A1/A2/A3
"""

from typing import List, Tuple
import numpy as np
import configs.constants as const


# =============================================================================
# Module A1: ExtractObsTime
# =============================================================================

def extract_obs_time(data_text: str) -> Tuple[int, float]:
    """解析观测时刻 (Module A1)

    从 #RANGEA 日志头段中解析 GPS 周计数和周内秒。
    格式示例:
        #RANGEA,COM1,0,68.0,FINESTEERING,2421,109502.000,...
                                    ^^^^  ^^^^^^^^^
                                    week  t_obs (周内秒)

    Args:
        data_text: 原始日志文本

    Returns:
        (week, t_obs) — GPS 周计数 [int] 和周内秒 [float]

    Raises:
        ValueError: 未找到 RANGEA 日志头

    Reference:
        指导书 4.1 节
    """
    for line in data_text.split('\n'):
        line = line.strip()
        if '#RANGEA' not in line or ';' not in line:
            continue

        # 提取分号前的头段
        header = line.split(';')[0]

        # 处理可能的 ']#' 前缀 (如 "[COM1]#RANGEA,...")
        if ']#' in header:
            header = header.split(']#')[1]
        elif '#RANGEA' in header:
            header = header[header.find('#RANGEA'):]

        fields = header.split(',')
        try:
            week = int(fields[5])
            t_obs = float(fields[6])
            return week, t_obs
        except (IndexError, ValueError):
            continue

    raise ValueError("RANGEA header not found in data")


# =============================================================================
# Module A2: ExtractEphemerisMatrix
# =============================================================================

def extract_ephemeris_matrix(
    data_text: str,
    obs_week: int,
    t_obs: float
) -> Tuple[np.ndarray, List[int]]:
    """提取 GPS 星历矩阵 (Module A2)

    从 #GPSEPHEMA 日志行中解析星历参数，按 ICD-GPS-200 标准筛选。
    GPS 星历数据段格式 (28 字段):
        字段 1: PRN
        字段 7: TOE (星历参考时刻)
        字段 8-28: 轨道参数
    映射到 eph 矩阵的 23 列 (列 0=PRN, 列 1=week, 列 2-22=轨道参数)。

    Args:
        data_text: 原始日志文本
        obs_week: GPS 周计数 (来自 extract_obs_time)
        t_obs: 观测时刻 [s] (来自 extract_obs_time)

    Returns:
        (eph_matrix, valid_prns)
        - eph_matrix: N×23 np.float64, 每行对应一颗卫星
        - valid_prns: N 个有效 PRN 的列表 (升序)

    Raises:
        ValueError: 没有有效 GPS 星历

    Reference:
        指导书 4.2 节, ICD-GPS-200 20.3.3.4.3
    """
    # ICD-GPS-200 字段索引映射 (数据段 0-based 索引 → eph 列)
    # 列 0=PRN, 列 1=week, 列 2-22 对应:
    COLUMN_MAP = {
        2: 8,    # A (轨道半长轴, 数据段索引 8, 0-based)
        3: 9,    # Δn (平均角速度修正)
        4: 10,   # M₀ (参考时刻平近点角)
        5: 11,   # e (偏心率)
        6: 12,   # ω (近地点角距)
        7: 13,   # C_uc (纬度幅角余弦修正)
        8: 14,   # C_us (纬度幅角正弦修正)
        9: 15,   # C_rc (轨道半径余弦修正)
        10: 16,  # C_rs (轨道半径正弦修正)
        11: 17,  # C_ic (轨道倾角余弦修正)
        12: 18,  # C_is (轨道倾角正弦修正)
        13: 19,  # i₀ (参考时刻轨道倾角)
        14: 20,  # di/dt (轨道倾角变化率)
        15: 21,  # Ω₀ (参考时刻升交点赤经)
        16: 22,  # Ω̇ (升交点赤经变化率)
        17: 7,   # t_oe (星历参考时刻, 数据段索引 7, 0-based)
        18: 24,  # t_oc (星钟参考时刻)
        19: 25,  # T_GD (群延迟)
        20: 26,  # a_f0 (卫星钟差系数 0)
        21: 27,  # a_f1 (卫星钟差系数 1)
        22: 28,  # a_f2 (卫星钟差系数 2)
    }

    eph_dict = {}  # PRN → (TOE, data_fields)

    for line in data_text.split('\n'):
        line = line.strip()
        if not line or '#GPSEPHEMA' not in line or ';' not in line:
            continue

        # 分离头段和数据段
        header_part, data_part = line.split(';', 1)

        # 提取头段字段
        if ']#' in header_part:
            header_part = header_part.split(']#')[1]
        elif '#GPSEPHEMA' in header_part:
            header_part = header_part[header_part.find('#GPSEPHEMA'):]

        hdr_fields = header_part.split(',')

        # 检查周计数
        try:
            if int(hdr_fields[5]) != obs_week:
                continue
        except (IndexError, ValueError):
            continue

        # 提取数据段字段 (去掉校验和)
        data_str = data_part.split('*')[0]
        data_fields = data_str.split(',')

        try:
            prn = int(float(data_fields[0]))
        except (ValueError, IndexError):
            continue

        # PRN 范围检查
        if prn < const.GPS_PRN_MIN or prn > const.GPS_PRN_MAX:
            continue

        # TOE 字段 (数据段索引 7, 0-based)
        try:
            toe = float(data_fields[7])
        except (ValueError, IndexError):
            continue

        # TOE 时差检查
        if abs(toe - t_obs) > const.EPHEMERIS_MAX_TIME_DIFF:
            continue

        # 保留最新的星历 (按 TOE)
        if prn not in eph_dict or toe > eph_dict[prn][0]:
            eph_dict[prn] = (toe, data_fields)

    if not eph_dict:
        raise ValueError("No valid GPS ephemeris found in data")

    valid_prns = sorted(eph_dict.keys())
    N = len(valid_prns)
    eph = np.zeros((N, 23), dtype=np.float64)

    for i, prn in enumerate(valid_prns):
        _, data = eph_dict[prn]
        eph[i, 0] = prn
        eph[i, 1] = obs_week

        for col, idx in COLUMN_MAP.items():
            try:
                val = float(data[idx].strip())
                eph[i, col] = val
            except (ValueError, IndexError):
                eph[i, col] = 0.0

    return eph, valid_prns


# =============================================================================
# Module A3: ExtractPseudorangeMatrix
# =============================================================================

def extract_pseudorange_matrix(
    data_text: str,
    obs_week: int,
    valid_prns: List[int]
) -> np.ndarray:
    """提取伪距矩阵 (Module A3)

    从 #RANGEA 日志行中提取伪距测量值。
    筛选规则:
        1. 周计数匹配
        2. PRN 属于 valid_prns
        3. 信号掩码标识为 L1 C/A 码 (末2位 04/0b/0c 且低4位 ∈ {0x04, 0x08, 0x0C})

    Args:
        data_text: 原始日志文本
        obs_week: GPS 周计数
        valid_prns: 有效 PRN 列表 (来自 extract_ephemeris_matrix)

    Returns:
        pr_matrix: N×2 np.float64
            第 0 列: PRN
            第 1 列: 伪距 [m]
            行顺序与 valid_prns 一致

    Reference:
        指导书 4.3 节
    """
    prn_set = set(valid_prns)
    pr_data = {}  # PRN → pseudorange

    for line in data_text.split('\n'):
        line = line.strip()
        if not line or '#RANGEA' not in line or ';' not in line:
            continue

        # 分离头段和数据段
        header_part, data_part = line.split(';', 1)

        # 提取头段字段
        if ']#' in header_part:
            header_part = header_part.split(']#')[1]
        elif '#RANGEA' in header_part:
            header_part = header_part[header_part.find('#RANGEA'):]

        hdr_fields = header_part.split(',')

        # 检查周计数
        try:
            if int(hdr_fields[5]) != obs_week:
                continue
        except (IndexError, ValueError):
            continue

        # 提取观测块字段 (去掉校验和)
        obs_str = data_part.split('*')[0]
        obs_fields = obs_str.split(',')[1:]  # 跳过第一个空字段(分号后)

        # 每 RANGEA_BLOCK_SIZE 个字段为一个观测块
        for j in range(0, len(obs_fields) - const.RANGEA_BLOCK_SIZE + 1,
                       const.RANGEA_BLOCK_SIZE):
            block = obs_fields[j:j + const.RANGEA_BLOCK_SIZE]
            if len(block) < const.RANGEA_BLOCK_SIZE:
                continue

            try:
                prn = int(float(block[0]))
            except (ValueError, IndexError):
                continue

            if prn not in prn_set:
                continue

            # 检查信号类型掩码 (block[9], 十六进制)
            try:
                sig_mask = int(block[9], 16) & 0x0F
                if sig_mask not in const.L1_SIGNAL_MASKS:
                    continue
            except (ValueError, IndexError):
                continue

            # 提取伪距 (block[2])
            try:
                pseudorange = float(block[2])
            except (ValueError, IndexError):
                continue

            # 保留第一个遇到的 L1 伪距
            if prn not in pr_data:
                pr_data[prn] = pseudorange

    N = len(valid_prns)
    pr = np.zeros((N, 2), dtype=np.float64)

    for i, prn in enumerate(valid_prns):
        pr[i, 0] = prn
        if prn in pr_data:
            pr[i, 1] = pr_data[prn]
        else:
            print(f"  WARNING: PRN {prn}: no L1 pseudorange found")
            pr[i, 1] = 0.0

    return pr