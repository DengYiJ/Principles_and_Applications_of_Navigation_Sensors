"""
数据加载器
==========
解析$GTIMU（IMU原始数据）和$GPFPD（惯导参考输出）协议文件
"""

import numpy as np
import re
import os
from typing import Dict, Optional, Tuple


class DataLoader:
    """
    实验数据加载器
    支持$GTIMU（IMU）和$GPFPD（参考姿态）两种NMEA协议格式
    """
    
    @staticmethod
    def load_imu(filepath: str, 
                 scale_gyro: float = 0.01745329252,
                 scale_acc: float = 9.7803267714) -> Dict[str, np.ndarray]:
        """
        解析$GTIMU协议文件
        格式: $GTIMU,GPSWeek,GPSTime,GyroX,GyroY,GyroZ,AccX,AccY,AccZ,Tpr*cs
        
        输入:
            filepath — 文件路径
            scale_gyro — 陀螺单位转换系数 °/s → rad/s (默认deg2rad)
            scale_acc — 加速度计单位转换系数 g → m/s^2
        
        返回:
            Dict:
                'gyro' — [N×3] 陀螺角速率 (rad/s)
                'acc'  — [N×3] 加速度计比力 (m/s^2)
                'time' — [N] 时间戳 (s)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"IMU数据文件不存在: {filepath}")
        
        gyro_list = []
        acc_list = []
        time_list = []
        skip_count = 0
        total_count = 0
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                total_count += 1
                
                # 格式: $GTIMU,week,time,gx,gy,gz,ax,ay,az,tpr*cs
                # 去掉$GTIMU前缀和校验和部分
                try:
                    # 移除$前缀和*cs部分
                    if line.startswith('$GTIMU'):
                        line = line[7:]  # 去掉"$GTIMU," (注意: 6=$ 7=G 8=T 9=I 10=M 11=U 12=,)
                    else:
                        skip_count += 1
                        continue

                    # 拆分字段
                    parts = line.split(',')
                    if len(parts) < 9:
                        skip_count += 1
                        continue

                    # 如果包含校验和，去掉
                    last_field = parts[-1]
                    if '*' in last_field:
                        parts[-1] = last_field.split('*')[0]

                    # 解析字段
                    # parts[0]=GPSWeek, parts[1]=GPSTime
                    gps_time = float(parts[1])
                    
                    # parts[2:5]=GyroX/Y/Z
                    gx = float(parts[2]) * scale_gyro
                    gy = float(parts[3]) * scale_gyro
                    gz = float(parts[4]) * scale_gyro
                    
                    # parts[5:8]=AccX/Y/Z
                    ax = float(parts[5]) * scale_acc
                    ay = float(parts[6]) * scale_acc
                    az = float(parts[7]) * scale_acc
                    
                    gyro_list.append([gx, gy, gz])
                    acc_list.append([ax, ay, az])
                    time_list.append(gps_time)
                    
                except (ValueError, IndexError) as e:
                    skip_count += 1
                    continue
        
        if len(gyro_list) == 0:
            raise ValueError(f"未成功解析任何有效数据行，总行数={total_count}，跳过行数={skip_count}")
        
        # 转换为numpy数组
        result = {
            'gyro': np.array(gyro_list, dtype=np.float64),
            'acc': np.array(acc_list, dtype=np.float64),
            'time': np.array(time_list, dtype=np.float64)
        }
        
        # 记录解析统计
        skip_ratio = skip_count / max(total_count, 1) * 100
        if skip_ratio > 10:
            import warnings
            warnings.warn(f"数据解析跳过率 {skip_ratio:.1f}% (>10%)，请检查文件格式: {filepath}")
        
        print(f"[DataLoader] 已加载 {len(gyro_list)} 条IMU数据 (跳过 {skip_count}/{total_count} 行)")
        return result
    
    @staticmethod
    def load_gpfpd(filepath: str) -> Dict[str, np.ndarray]:
        """
        解析$GPFPD协议文件（惯导系统参考输出）
        格式: $GPFPD,GPSWeek,GPSTime,Heading,Pitch,Roll,Lat,Lon,Alt,Ve,Vn,Vu,Baseline,NSV1,NSV2,Status*cs
        
        输入:
            filepath — 文件路径
        
        返回:
            Dict:
                'heading' — [N] 航向角 (°)
                'pitch'   — [N] 俯仰角 (°)
                'roll'    — [N] 横滚角 (°)
                'lat'     — [N] 纬度 (°)
                'lon'     — [N] 经度 (°)
                'alt'     — [N] 高度 (m)
                'Ve'      — [N] 东向速度 (m/s)
                'Vn'      — [N] 北向速度 (m/s)
                'Vu'      — [N] 天向速度 (m/s)
                'time'    — [N] 时间戳 (s)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"参考数据文件不存在: {filepath}")
        
        data_dict = {
            'heading': [], 'pitch': [], 'roll': [],
            'lat': [], 'lon': [], 'alt': [],
            'Ve': [], 'Vn': [], 'Vu': [],
            'time': []
        }
        
        skip_count = 0
        total_count = 0
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                total_count += 1
                
                try:
                    if line.startswith('$GPFPD'):
                        line = line[7:]  # 去掉"$GPFPD,"
                    else:
                        skip_count += 1
                        continue
                    
                    parts = line.split(',')
                    if len(parts) < 15:
                        skip_count += 1
                        continue
                    
                    # 处理校验和
                    last_field = parts[-1]
                    if '*' in last_field:
                        parts[-1] = last_field.split('*')[0]
                    
                    # 解析字段
                    gps_time = float(parts[1])
                    heading = float(parts[2])  # 航向 (°)
                    pitch = float(parts[3])    # 俯仰 (°)
                    roll = float(parts[4])     # 横滚 (°)
                    lat = float(parts[5])      # 纬度 (°)
                    lon = float(parts[6])      # 经度 (°)
                    alt = float(parts[7])      # 高度 (m)
                    ve = float(parts[8])       # 东速 (m/s)
                    vn = float(parts[9])       # 北速 (m/s)
                    vu = float(parts[10])      # 天速 (m/s)
                    
                    data_dict['heading'].append(heading)
                    data_dict['pitch'].append(pitch)
                    data_dict['roll'].append(roll)
                    data_dict['lat'].append(lat)
                    data_dict['lon'].append(lon)
                    data_dict['alt'].append(alt)
                    data_dict['Ve'].append(ve)
                    data_dict['Vn'].append(vn)
                    data_dict['Vu'].append(vu)
                    data_dict['time'].append(gps_time)
                    
                except (ValueError, IndexError) as e:
                    skip_count += 1
                    continue
        
        if len(data_dict['heading']) == 0:
            raise ValueError(f"未成功解析任何有效$GPFPD数据行，总行数={total_count}，跳过行数={skip_count}")
        
        result = {k: np.array(v, dtype=np.float64) for k, v in data_dict.items()}
        
        print(f"[DataLoader] 已加载 {len(result['heading'])} 条$GPFPD参考数据 (跳过 {skip_count}/{total_count} 行)")
        return result
    
    @staticmethod
    def load_dataset(imu_path: str, ref_path: str,
                     scale_gyro: float = 0.01745329252,
                     scale_acc: float = 0.0097803267714) -> Tuple[Dict, Dict]:
        """
        同时加载IMU数据和参考数据
        
        返回: (imu_raw_dict, ref_raw_dict)
        """
        imu_data = DataLoader.load_imu(imu_path, scale_gyro, scale_acc)
        ref_data = DataLoader.load_gpfpd(ref_path)
        return imu_data, ref_data