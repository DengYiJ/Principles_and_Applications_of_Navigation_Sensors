"""
卫星分布可视化 — Satellite Visualization
=========================================
3D 地球 + 卫星分布图 + 极坐标星空图。

颜色语义:
  - 绿色: 运行良好的卫星 (参与定位或可见)
  - 红色: 不可用或未参与定位的 GPS 卫星

形状语义 (对应卫星系统):
  - 圆形 (o):     GPS
  - 正方形 (s):   GLONASS
  - 五边形 (p):   BeiDou
  - 三角形 (^):   QZSS
  - 四角星 (*):   SBAS

设计依据: 03_algorithm_design.md
"""

from typing import List, Tuple, Set
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
import configs.constants as const

# 形状映射: 卫星系统 → matplotlib marker
SYS_MARKERS = {
    'GPS':      'o',   # 圆形
    'GLONASS':  's',   # 正方形
    'BEIDOU':   'p',   # 五边形
    'QZSS':     '^',   # 三角形
    'SBAS':     '*',   # 四角星
    'GALILEO':  'D',   # 菱形
}


def parse_satxyz2a(data_text: str) -> List[Tuple]:
    """从 SATXYZ2A 日志中解析所有卫星位置

    SATXYZ2A 格式 (每卫星 10 个字段):
        系统, PRN, X, Y, Z, 卫星钟差, 电离层延迟, 对流层延迟, 保留, 保留

    Args:
        data_text: 原始日志文本

    Returns:
        [(lon, lat, h, prn_str, system), ...] 列表
        注: 此函数仅解析卫星的 ECEF 坐标和系统类型。
            高度角/方位角需通过接收机位置计算，此处不包含。
    """
    sats = []
    for line in data_text.split('\n'):
        if 'SATXYZ2A' not in line or ';' not in line:
            continue

        data_part = line.split(';')[1].split('*')[0]
        fields = data_part.split(',')

        # fields[0] = 卫星总数, fields[1:] = 卫星数据
        sv_count = int(fields[0])
        sv_fields = fields[1:]

        # 每 10 个字段为一个卫星块
        # (system, PRN, X, Y, Z, clock_bias, iono_delay, tropo_delay, reserved1, reserved2)
        for j in range(0, len(sv_fields) - 9, 10):
            block = sv_fields[j:j + 10]
            try:
                system = block[0]
                prn = block[1]
                X = float(block[2])
                Y = float(block[3])
                Z = float(block[4])
                # 不包含高度角/方位角信息
                lon = math.atan2(Y, X) * const.R2D
                p = math.sqrt(X*X + Y*Y)
                lat = math.atan2(Z, p) * const.R2D
                h = math.sqrt(X*X + Y*Y + Z*Z) - const.A_WGS84
                sats.append((lon, lat, h, prn, system))
            except (ValueError, IndexError):
                continue
        break  # 只处理第一条 SATXYZ2A
    return sats


def compute_elev_azim(sv_ecef, rx_ecef):
    """计算单颗卫星相对于接收机的仰角和方位角

    Args:
        sv_ecef: (X, Y, Z) 卫星 ECEF 坐标 [m]
        rx_ecef: (X, Y, Z) 接收机 ECEF 坐标 [m]

    Returns:
        (elevation_deg, azimuth_deg): 仰角 [°], 方位角 [°] (从北顺时针)
    """
    dx = sv_ecef[0] - rx_ecef[0]
    dy = sv_ecef[1] - rx_ecef[1]
    dz = sv_ecef[2] - rx_ecef[2]

    # 接收机经纬度用于 ENU 旋转
    rx_lon = math.atan2(rx_ecef[1], rx_ecef[0])
    rx_lat = math.atan2(
        rx_ecef[2],
        math.sqrt(rx_ecef[0]**2 + rx_ecef[1]**2)
    )

    slat = math.sin(rx_lat)
    clat = math.cos(rx_lat)
    slon = math.sin(rx_lon)
    clon = math.cos(rx_lon)

    # ECEF → ENU 旋转矩阵
    # [e]   [-sin(lon)         cos(lon)         0  ] [dx]
    # [n] = [-sin(lat)cos(lon) -sin(lat)sin(lon) cos(lat)] [dy]
    # [u]   [ cos(lat)cos(lon)  cos(lat)sin(lon) sin(lat)] [dz]
    e = -slon * dx + clon * dy
    n = -slat * clon * dx - slat * slon * dy + clat * dz
    u = clat * clon * dx + clat * slon * dy + slat * dz

    # 仰角
    hor_dist = math.sqrt(e*e + n*n)
    elev = math.atan2(u, hor_dist) * const.R2D

    # 方位角 (从北顺时针)
    azim = math.atan2(e, n) * const.R2D
    if azim < 0:
        azim += 360.0

    return elev, azim


def plot_earth_and_satellites(
    all_sats: List[Tuple],
    rx_llh: Tuple[float, float, float],
    used_gps_prns: Set[int] = None,
    title: str = "Multi-constellation Satellite Distribution",
    save_path: str = "exp2/satellite_visualization.png",
    sat_pos_ecef: np.ndarray = None,
):
    """绘制 3D 地球 + 卫星分布图 + 极坐标星空图

    Args:
        all_sats: [(lon, lat, h, prn_str, system, ...), ...]
        rx_llh: (lon, lat, h) 接收机位置
        used_gps_prns: 参与定位的 GPS PRN 集合
        title: 图表标题
        save_path: 图片保存路径
        sat_pos_ecef: N×3 卫星 ECEF 坐标数组 (用于计算高度角/方位角)
    """
    if used_gps_prns is None:
        used_gps_prns = set()

    # 计算接收机 ECEF 坐标
    rx_lat_r = rx_llh[1] * const.D2R
    rx_lon_r = rx_llh[0] * const.D2R
    sin_lat = math.sin(rx_lat_r)
    cos_lat = math.cos(rx_lat_r)
    N = const.A_WGS84 / math.sqrt(1.0 - const.E2_WGS84 * sin_lat * sin_lat)
    rx_ecef = (
        (N + rx_llh[2]) * cos_lat * math.cos(rx_lon_r),
        (N + rx_llh[2]) * cos_lat * math.sin(rx_lon_r),
        (N * (1.0 - const.E2_WGS84) + rx_llh[2]) * sin_lat
    )

    fig = plt.figure(figsize=(18, 9))

    # ---- 左图: 3D 视图 ----
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Earth & Satellites 3D", fontsize=12, fontweight='bold')

    # 地球球体
    Re = const.A_WGS84
    u = np.linspace(0, 2 * const.PI, 50)
    v = np.linspace(0, const.PI, 50)
    xe = Re * np.outer(np.cos(u), np.sin(v))
    ye = Re * np.outer(np.sin(u), np.sin(v))
    ze = Re * np.outer(np.ones(50), np.cos(v))
    ax1.plot_surface(xe, ye, ze, color='lightblue', alpha=0.25,
                     edgecolor='gray', linewidth=0.1)

    # 绘制卫星
    for i, sat in enumerate(all_sats):
        lon, lat, h, prn, sys_name = sat[:5]
        lat_r = lat * const.D2R
        lon_r = lon * const.D2R
        r = Re + max(h, 0)

        xs = r * math.cos(lat_r) * math.cos(lon_r)
        ys = r * math.cos(lat_r) * math.sin(lon_r)
        zs = r * math.sin(lat_r)

        # 确定颜色状态
        try:
            prn_int = int(prn)
        except ValueError:
            prn_int = -1

        if sys_name == 'GPS' and prn_int not in used_gps_prns:
            color = 'red'
        else:
            color = 'green'

        marker = SYS_MARKERS.get(sys_name, 'o')

        if sys_name == 'GPS' and prn_int not in used_gps_prns:
            ax1.scatter(xs, ys, zs, c='red', s=50, marker=marker,
                        edgecolors='red', linewidths=0.5, alpha=0.5)
        else:
            ax1.scatter(xs, ys, zs, c=color, s=50, marker=marker,
                        edgecolors='black', linewidths=0.3)

        label = f'{prn}'
        ax1.text(xs, ys, zs, label, fontsize=6, color=color)

    # 绘制接收机
    rr = Re + rx_llh[2]
    ax1.scatter(
        rr * math.cos(rx_lat_r) * math.cos(rx_lon_r),
        rr * math.cos(rx_lat_r) * math.sin(rx_lon_r),
        rr * math.sin(rx_lat_r),
        c='black', s=120, marker='o', edgecolors='white', linewidths=0.5
    )
    ax1.text(
        rr * math.cos(rx_lat_r) * math.cos(rx_lon_r),
        rr * math.cos(rx_lat_r) * math.sin(rx_lon_r),
        rr * math.sin(rx_lat_r),
        'RX', fontsize=11, color='black', fontweight='bold'
    )

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_box_aspect([1, 1, 1])

    # 图例
    present = set(s[4] for s in all_sats)
    legend_elements = [
        Patch(facecolor=SYS_COLORS.get(s, 'gray'), label=s,
              edgecolor='black', linewidth=0.5)
        for s in present
    ]
    legend_elements.extend([
        Patch(facecolor='green', label='Active / Good Signal',
              edgecolor='black', linewidth=0.5),
        Patch(facecolor='red', label='GPS Not Used for Positioning',
              edgecolor='red', linewidth=0.5, alpha=0.5),
    ])
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=7)

    # ---- 右图: 星空图 ----
    ax2 = fig.add_subplot(122, projection='polar')
    ax2.set_title("Satellite Sky Plot", fontsize=12, fontweight='bold')

    for i, sat in enumerate(all_sats):
        lon, lat, h, prn, sys_name = sat[:5]

        # 使用 ECEF 坐标计算高度角和方位角
        if sat_pos_ecef is not None and i < sat_pos_ecef.shape[0]:
            sv_xyz = sat_pos_ecef[i]
        else:
            # 从经纬高算回 ECEF (近似)
            lat_r = lat * const.D2R
            lon_r = lon * const.D2R
            sl = math.sin(lat_r)
            N_sat = const.A_WGS84 / math.sqrt(1.0 - const.E2_WGS84 * sl * sl)
            sv_xyz = (
                (N_sat + h) * math.cos(lat_r) * math.cos(lon_r),
                (N_sat + h) * math.cos(lat_r) * math.sin(lon_r),
                (N_sat * (1.0 - const.E2_WGS84) + h) * sl
            )

        elev, azim = compute_elev_azim(sv_xyz, rx_ecef)

        # 颜色状态 (与 3D 图一致)
        try:
            prn_int = int(prn)
        except ValueError:
            prn_int = -1

        if sys_name == 'GPS' and prn_int not in used_gps_prns:
            color = 'red'
        else:
            color = 'green'

        marker = SYS_MARKERS.get(sys_name, 'o')

        # 极坐标: 角度=方位角, 半径=90-仰角(天顶距)
        az_rad = azim * const.D2R
        r = 90.0 - max(elev, 0.5)

        if sys_name == 'GPS' and prn_int not in used_gps_prns:
            ax2.scatter(az_rad, r, c='red', s=35, marker=marker,
                        edgecolors='red', linewidths=0.5, alpha=0.4)
        else:
            ax2.scatter(az_rad, r, c='green', s=35, marker=marker,
                        edgecolors='black', linewidths=0.3, alpha=0.7)

        ax2.text(az_rad, r, f'{prn}', fontsize=5, color=color,
                 ha='center', va='bottom', alpha=0.8)

    ax2.scatter(0, 0, c='black', s=80, marker='o', label='RX')
    ax2.set_ylim(0, 90)
    ax2.set_yticks([0, 30, 60, 90])
    ax2.set_yticklabels(['Zenith', '60°', '30°', 'Horizon'])
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.legend(loc='upper right', fontsize=8)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Plot saved: {save_path}")
    plt.show()


# 星座颜色 (仅用于图例)
SYS_COLORS = {
    'GPS':      'green',
    'GLONASS':  'green',
    'BEIDOU':   'green',
    'QZSS':     'green',
    'SBAS':     'green',
    'GALILEO':  'green',
}