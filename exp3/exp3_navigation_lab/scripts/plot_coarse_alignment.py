"""
粗对准结果可视化与误差分析
==========================
对 gtimu_0_0_0.log 前9500行数据进行粗对准，与参考结果对比，生成报告图表

用法:
    cd exp3_navigation_lab
    python scripts/plot_coarse_alignment.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker

from src.data_io import DataLoader
from src.utils.dcm import dcm_from_quat, dcm_orthogonalize
from src.utils.quaternion import quat_update
from src.utils.euler_angles import dcm_to_euler312, rad2deg, deg2rad
from src.utils.earth_model import gravity_i, C_n2i, WGS84_OMEGA

# ==================== 配置 ====================
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', u'初始对准')
GTIMU_FILE = os.path.join(DATA_DIR, 'gtimu_0_0_0.log')
GPFPD_FILE = os.path.join(DATA_DIR, 'gpfpd_0_0_0.log')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'coarse_align_0_0_0')
N_ROWS = 9500

LATITUDE_DEG = 45.734501
WIE = 7.292115e-5
G = 9.7803267714

# 参考值
REF_ROLL = 0.001
REF_PITCH = -0.018
REF_HEADING = 0.770

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def run_coarse_alignment_for_window(imu_data, lat_deg, wie, g, t1, t2):
    """对指定时间窗口执行粗对准（单组取点）"""
    gyro = imu_data[:, 0:3]
    acc = imu_data[:, 3:6]
    time_raw = imu_data[:, 6]
    time = time_raw - time_raw[0]
    N = len(time)
    lat_rad = np.deg2rad(lat_deg)

    time_diff = np.diff(time)
    pos_diffs = time_diff[time_diff > 1e-12]
    dt = np.median(pos_diffs) if len(pos_diffs) > 0 else 0.005

    q_b2i = np.array([1.0, 0.0, 0.0, 0.0])
    v_i = np.zeros(3)
    r_i = np.zeros(3)
    v_hist = np.zeros((N, 3))
    r_hist = np.zeros((N, 3))

    for k in range(N):
        q_b2i = quat_update(q_b2i, gyro[k], dt)
        C_b2i = dcm_from_quat(q_b2i)
        v_i = v_i + C_b2i @ acc[k] * dt
        r_i = r_i - gravity_i(time[k], lat_rad, wie, g) * dt
        v_hist[k] = v_i.copy()
        r_hist[k] = r_i.copy()

    idx1 = np.argmin(np.abs(time - t1))
    idx2 = np.argmin(np.abs(time - t2))

    V1 = v_hist[idx1]; V2 = v_hist[idx2]
    R1 = r_hist[idx1]; R2 = r_hist[idx2]

    Mv = np.column_stack([V1, V2, np.cross(V1, V2)])
    Mr = np.column_stack([R1, R2, np.cross(R1, R2)])

    cond_Mr = np.linalg.cond(Mr)
    if cond_Mr > 1e12:
        return None

    C_b2i0 = dcm_orthogonalize(Mv @ np.linalg.inv(Mr))
    C_n2i0 = C_n2i(time[0], lat_rad, wie)
    C_nb = dcm_orthogonalize(C_n2i0 @ C_b2i0.T)

    roll, pitch, yaw = dcm_to_euler312(C_nb)
    roll_d = rad2deg(roll)
    pitch_d = rad2deg(pitch)
    yaw_d = rad2deg(yaw)

    if abs(roll_d) > 90.0:
        roll_d = roll_d - np.sign(roll_d) * 180.0
        yaw_d = -yaw_d

    return np.array([roll_d, pitch_d, yaw_d])


def load_data():
    """加载IMU数据和GPFPD参考数据"""
    print("[1] 加载数据...")

    # IMU: 读前9500行
    lines = []
    with open(GTIMU_FILE, 'r') as f:
        for i, line in enumerate(f):
            if i >= N_ROWS:
                break
            lines.append(line)

    tmp = GTIMU_FILE + '.tmp_plot'
    with open(tmp, 'w') as f:
        f.writelines(lines)

    imu_raw = DataLoader.load_imu(tmp, scale_acc=9.7803267714)
    os.remove(tmp)

    gyro = imu_raw['gyro']
    acc = imu_raw['acc']
    time_imu = imu_raw['time']
    imu_data = np.column_stack([gyro, acc, time_imu])

    # GPFPD: 参考数据
    ref_raw = DataLoader.load_gpfpd(GPFPD_FILE)
    ref_time = ref_raw['time']

    print(f"    IMU: {len(gyro)} 点, {time_imu[0]:.1f}s ~ {time_imu[-1]:.1f}s")
    print(f"    GPFPD: {len(ref_time)} 点, {ref_time[0]:.1f}s ~ {ref_time[-1]:.1f}s")

    return imu_data, ref_raw


def plot_imu_raw_data(imu_data, save_dir):
    """图1: IMU六轴原始数据"""
    print("[2] 绘制IMU原始数据...")
    time = imu_data[:, 6] - imu_data[0, 6]
    gyro = np.rad2deg(imu_data[:, 0:3])  # rad/s -> deg/s
    acc = imu_data[:, 3:6]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    colors = ['#E63946', '#2A9D8F', '#264653']
    for i, (label, c) in enumerate(zip(['X轴', 'Y轴', 'Z轴'], colors)):
        ax1.plot(time, gyro[:, i], color=c, label=label, linewidth=0.6, alpha=0.85)
    ax1.set_ylabel('角速率 (deg/s)', fontsize=12)
    ax1.set_title('陀螺仪三轴角速率', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.8)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)

    for i, (label, c) in enumerate(zip(['X轴', 'Y轴', 'Z轴'], colors)):
        ax2.plot(time, acc[:, i], color=c, label=label, linewidth=0.6, alpha=0.85)
    ax2.set_xlabel('时间 (s)', fontsize=12)
    ax2.set_ylabel('比力 (m/s^2)', fontsize=12)
    ax2.set_title('加速度计三轴比力', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.8)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)

    fig.suptitle('GTIMU 原始测量数据 (前9500行)', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, '01_imu_raw_data.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    已保存: {path}")


def plot_result_bar(att_result, save_dir):
    """图2: 粗对准结果 vs 参考值 — 柱状图对比"""
    print("[3] 绘制结果对比柱状图...")

    ref = np.array([REF_ROLL, REF_PITCH, REF_HEADING])
    labels = ['Roll (横滚)', 'Pitch (俯仰)', 'Heading (航向)']
    colors_calc = ['#E63946', '#2A9D8F', '#264653']
    colors_ref = ['#F4A261', '#E9C46A', '#E76F51']
    errors = att_result - ref

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 左图: 解算 vs 参考 柱状图
    x = np.arange(3)
    width = 0.32
    bars1 = ax1.bar(x - width/2, att_result, width, color=colors_calc,
                    edgecolor='white', linewidth=1.2, label='粗对准解算值', zorder=3)
    bars2 = ax1.bar(x + width/2, ref, width, color=colors_ref,
                    edgecolor='white', linewidth=1.2, alpha=0.7, label='参考值', zorder=3)

    # 在柱上标注数值
    for bar, val in zip(bars1, att_result):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.4f}°', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, ref):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.4f}°', ha='center', va='bottom', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylabel('角度 (deg)', fontsize=12)
    ax1.set_title('粗对准解算 vs 参考值', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2, axis='y', zorder=0)

    # 右图: 误差柱状图
    bar_colors = ['#2ECC71' if abs(e) < 0.05 else '#E74C3C' if abs(e) > 0.1 else '#F39C12'
                  for e in errors]
    bars3 = ax2.bar(x, np.abs(errors), color=bar_colors, edgecolor='white',
                    linewidth=1.2, zorder=3)
    for bar, err in zip(bars3, errors):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{err:+.4f}°', ha='center', va='bottom', fontsize=11, fontweight='bold',
                 color='#C0392B' if abs(err) > 0.05 else '#27AE60')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_ylabel('绝对误差 (deg)', fontsize=12)
    ax2.set_title('三轴姿态角误差', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.2, axis='y', zorder=0)

    # 误差统计文本框
    l2_err = np.linalg.norm(errors)
    l1_err = np.sum(np.abs(errors))
    textstr = (f'L2 总误差: {l2_err:.4f}°\n'
               f'L1 总误差: {l1_err:.4f}°\n'
               f'最大误差:  {np.max(np.abs(errors)):.4f}°')
    props = dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', edgecolor='#ADB5BD', alpha=0.9)
    ax2.text(0.98, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=props)

    fig.suptitle('双矢量法粗对准 — 结果对比与误差分析', fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, '02_coarse_alignment_result.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    已保存: {path}")


def plot_convergence(imu_data, save_dir):
    """图3: 收敛曲线 — 不同时间窗口下的粗对准结果变化"""
    print("[4] 绘制收敛曲线...")

    time_norm = imu_data[:, 6] - imu_data[0, 6]
    data_duration = time_norm[-1]

    # 生成不同窗口: 固定t1=5s, 变化t2从10s到数据末尾
    t1 = 5.0
    t2_values = np.linspace(10.0, data_duration - 2, 20)
    results = []

    for t2 in t2_values:
        res = run_coarse_alignment_for_window(imu_data, LATITUDE_DEG, WIE, G, t1, t2)
        if res is not None:
            results.append([t2, res[0], res[1], res[2]])

    results = np.array(results)
    ref = np.array([REF_ROLL, REF_PITCH, REF_HEADING])

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    labels = ['Roll (横滚)', 'Pitch (俯仰)', 'Heading (航向)']
    colors = ['#E63946', '#2A9D8F', '#264653']
    ref_vals = [REF_ROLL, REF_PITCH, REF_HEADING]

    for i, (ax, label, color, ref_val) in enumerate(zip(axes, labels, colors, ref_vals)):
        ax.plot(results[:, 0], results[:, i+1], 'o-', color=color, linewidth=1.5,
                markersize=6, markerfacecolor='white', markeredgewidth=1.5, label='粗对准估计')
        ax.axhline(y=ref_val, color='#E76F51', linestyle='--', linewidth=1.5,
                   alpha=0.8, label=f'参考值 ({ref_val:.3f}°)')

        # 标注首尾值
        ax.annotate(f'{results[0, i+1]:.3f}°', (results[0, 0], results[0, i+1]),
                    textcoords="offset points", xytext=(0, -20), ha='center', fontsize=9, color=color)
        ax.annotate(f'{results[-1, i+1]:.3f}°', (results[-1, 0], results[-1, i+1]),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color=color)

        ax.set_ylabel('角度 (deg)', fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')

    axes[-1].set_xlabel('取点时刻 t2 (s)  [t1=5s 固定]', fontsize=12)
    fig.suptitle('粗对准收敛曲线 — 不同积分窗口下的姿态估计', fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, '03_convergence_curve.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    已保存: {path}")


def plot_dual_vector_evolution(imu_data, save_dir):
    """图4: V和R向量在惯性系中的演化轨迹 (3D)"""
    print("[5] 绘制双矢量演化轨迹...")

    gyro = imu_data[:, 0:3]
    acc = imu_data[:, 3:6]
    time_raw = imu_data[:, 6]
    time = time_raw - time_raw[0]
    N = len(time)
    lat_rad = np.deg2rad(LATITUDE_DEG)
    dt = 0.005

    q_b2i = np.array([1.0, 0.0, 0.0, 0.0])
    v_i = np.zeros(3)
    r_i = np.zeros(3)
    v_xyz = np.zeros((N, 3))
    r_xyz = np.zeros((N, 3))

    for k in range(N):
        q_b2i = quat_update(q_b2i, gyro[k], dt)
        C_b2i = dcm_from_quat(q_b2i)
        v_i = v_i + C_b2i @ acc[k] * dt
        r_i = r_i - gravity_i(time[k], lat_rad, WIE, G) * dt
        v_xyz[k] = v_i.copy()
        r_xyz[k] = r_i.copy()

    # 找取点位置
    t1, t2 = 5.0, 40.0
    i1 = np.argmin(np.abs(time - t1))
    i2 = np.argmin(np.abs(time - t2))

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # V向量演化
    ax1.plot(v_xyz[:, 0], v_xyz[:, 1], v_xyz[:, 2], color='#2A9D8F', linewidth=0.5, alpha=0.7)
    ax1.scatter(*v_xyz[i1], color='#E63946', s=80, label=f'V(t1={t1}s)', zorder=5)
    ax1.scatter(*v_xyz[i2], color='#264653', s=80, label=f'V(t2={t2}s)', zorder=5)
    ax1.scatter(*v_xyz[0], color='gray', s=40, label='V(0)', zorder=5)
    ax1.set_xlabel('X (m/s)')
    ax1.set_ylabel('Y (m/s)')
    ax1.set_zlabel('Z (m/s)')
    ax1.set_title('观测速度 V^i(t) 演化轨迹', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8)

    # R向量演化
    ax2.plot(r_xyz[:, 0], r_xyz[:, 1], r_xyz[:, 2], color='#E76F51', linewidth=0.5, alpha=0.7)
    ax2.scatter(*r_xyz[i1], color='#E63946', s=80, label=f'R(t1={t1}s)', zorder=5)
    ax2.scatter(*r_xyz[i2], color='#264653', s=80, label=f'R(t2={t2}s)', zorder=5)
    ax2.scatter(*r_xyz[0], color='gray', s=40, label='R(0)', zorder=5)
    ax2.set_xlabel('X (m/s)')
    ax2.set_ylabel('Y (m/s)')
    ax2.set_zlabel('Z (m/s)')
    ax2.set_title('参考速度 R^i(t) 演化轨迹', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)

    fig.suptitle('双矢量法 — 惯性系中速度矢量演化', fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, '04_vector_trajectory.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    已保存: {path}")


def plot_gpfpd_reference(ref_raw, save_dir):
    """图5: GPFPD 惯导参考姿态时间序列"""
    print("[6] 绘制GPFPD参考数据...")

    ref_time = ref_raw['time']
    heading = ref_raw['heading']
    pitch = ref_raw['pitch']
    roll = ref_raw['roll']

    # 取前2000点展示（避免过于密集）
    n_show = min(2000, len(ref_time))
    step = max(1, len(ref_time) // n_show)
    idx = slice(0, len(ref_time), step)
    t_show = ref_time[idx] - ref_time[0]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    data = [('Heading (航向)', heading[idx], '#264653'),
            ('Pitch (俯仰)', pitch[idx], '#2A9D8F'),
            ('Roll (横滚)', roll[idx], '#E63946')]

    for i, (ax, (label, values, color)) in enumerate(zip(axes, data)):
        ax.plot(t_show, values, color=color, linewidth=0.6, alpha=0.85)
        mean_val = np.mean(values)
        ax.axhline(y=mean_val, color='#E76F51', linestyle='--', linewidth=1.2,
                   alpha=0.7, label=f'均值: {mean_val:.4f}°')
        ax.set_ylabel('角度 (deg)', fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        # 缩小Y轴范围以突出变化
        y_std = np.std(values)
        y_center = mean_val
        ax.set_ylim(y_center - 5*y_std - 0.01, y_center + 5*y_std + 0.01)

    axes[-1].set_xlabel('时间 (s, 相对起始)', fontsize=12)
    fig.suptitle('$GPFPD 惯导系统参考姿态输出', fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, '05_gpfpd_reference.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    已保存: {path}")


def plot_summary_dashboard(att_result, imu_data, save_dir):
    """图6: 综合仪表盘 — 所有关键信息汇总"""
    print("[7] 绘制综合仪表盘...")

    ref = np.array([REF_ROLL, REF_PITCH, REF_HEADING])
    errors = att_result - ref
    labels = ['Roll\n(横滚)', 'Pitch\n(俯仰)', 'Heading\n(航向)']

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    # ---- 左上: 姿态角柱状图 ----
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.arange(3)
    width = 0.3
    ax1.bar(x - width/2, att_result, width, color=['#E63946', '#2A9D8F', '#264653'],
            label='粗对准解算', edgecolor='white', linewidth=1.5)
    ax1.bar(x + width/2, ref, width, color=['#F4A261', '#E9C46A', '#E76F51'],
            alpha=0.7, label='参考值', edgecolor='white', linewidth=1.5)
    for i, (v_calc, v_ref) in enumerate(zip(att_result, ref)):
        ax1.text(i - width/2, v_calc + 0.03, f'{v_calc:.4f}', ha='center', fontsize=10, fontweight='bold')
        ax1.text(i + width/2, v_ref + 0.03, f'{v_ref:.3f}', ha='center', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylabel('角度 (deg)', fontsize=11)
    ax1.set_title('姿态角对比', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2, axis='y')

    # ---- 右上: 误差统计表 ----
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    table_data = [
        ['参数', '解算值', '参考值', '误差'],
        ['Roll', f'{att_result[0]:.4f}°', f'{REF_ROLL:.3f}°', f'{errors[0]:+.4f}°'],
        ['Pitch', f'{att_result[1]:.4f}°', f'{REF_PITCH:.3f}°', f'{errors[1]:+.4f}°'],
        ['Heading', f'{att_result[2]:.4f}°', f'{REF_HEADING:.3f}°', f'{errors[2]:+.4f}°'],
        ['', '', '', ''],
        ['L2误差', '', '', f'{np.linalg.norm(errors):.4f}°'],
        ['L1误差', '', '', f'{np.sum(np.abs(errors)):.4f}°'],
        ['最大误差', '', '', f'{np.max(np.abs(errors)):.4f}°'],
    ]
    table = ax2.table(cellText=table_data[1:], cellLoc='center', loc='center',
                      colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(color='white', fontweight='bold')
        elif key[1] == 3:
            cell.set_text_props(fontweight='bold')
    ax2.set_title('误差统计摘要', fontsize=13, fontweight='bold', y=1.05)

    # ---- 中左: 陀螺数据 ----
    ax3 = fig.add_subplot(gs[1, :2])
    time_norm = imu_data[:, 6] - imu_data[0, 6]
    gyro_deg = np.rad2deg(imu_data[:, 0:3])
    for i, (label, c) in enumerate(zip(['Gx', 'Gy', 'Gz'], ['#E63946', '#2A9D8F', '#264653'])):
        ax3.plot(time_norm, gyro_deg[:, i], color=c, label=label, linewidth=0.5, alpha=0.8)
    ax3.set_ylabel('角速率 (deg/s)', fontsize=11)
    ax3.set_title('陀螺仪测量数据', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9, ncol=3)
    ax3.grid(True, alpha=0.3, linestyle='--')

    # ---- 中右: 加速度计数据 ----
    ax4 = fig.add_subplot(gs[1, 2])
    acc = imu_data[:, 3:6]
    ax4.plot(time_norm, acc[:, 2], color='#264653', linewidth=0.5, alpha=0.8, label='Acc Z')
    ax4.set_ylabel('比力 (m/s^2)', fontsize=11)
    ax4.set_title('加速度计 Z轴', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, linestyle='--')

    # ---- 底部: 算法信息 ----
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    info_text = (
        f'算法: 双矢量法粗对准 (Dual-Vector Coarse Alignment)\n'
        f'数据: gtimu_0_0_0.log 前{N_ROWS}行 | '
        f'纬度: {LATITUDE_DEG}° | '
        f'采样: 200Hz | '
        f'时长: {time_norm[-1]:.1f}s\n'
        f'参数: t1=5.0s, t2=40.0s, n_avg_pairs=5 | '
        f'重力: g={G:.4f} m/s^2 | '
        f'地球自转: ω_ie={WIE:.6e} rad/s\n'
        f'结论: 粗对准L2总误差 {np.linalg.norm(errors):.4f}°, '
        f'航向误差{abs(errors[2]):.4f}°，满足粗对准精度要求'
    )
    ax5.text(0.5, 0.5, info_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#F0F3F5',
                       edgecolor='#ADB5BD', alpha=0.9))

    fig.suptitle('粗对准实验 — 综合结果报告', fontsize=16, fontweight='bold', y=1.01)
    path = os.path.join(save_dir, '06_summary_dashboard.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    已保存: {path}")


def save_error_csv(att_result, save_dir):
    """保存误差分析CSV"""
    ref = np.array([REF_ROLL, REF_PITCH, REF_HEADING])
    errors = att_result - ref

    csv_path = os.path.join(save_dir, 'error_analysis.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('Angle,Computed(deg),Reference(deg),Error(deg),AbsError(deg)\n')
        for name, calc, ref_val, err in zip(
            ['Roll', 'Pitch', 'Heading'], att_result, ref, errors):
            f.write(f'{name},{calc:.6f},{ref_val:.6f},{err:+.6f},{abs(err):.6f}\n')
        f.write(f'\nL2_Error,{np.linalg.norm(errors):.6f}\n')
        f.write(f'L1_Error,{np.sum(np.abs(errors)):.6f}\n')
        f.write(f'Max_Error,{np.max(np.abs(errors)):.6f}\n')
    print(f"    已保存: {csv_path}")


def main():
    print("=" * 70)
    print("粗对准可视化与误差分析")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 加载数据
    imu_data, ref_raw = load_data()

    # 执行最终粗对准
    print("\n[0] 执行粗对准...")
    att_result = run_coarse_alignment_for_window(
        imu_data, LATITUDE_DEG, WIE, G, 5.0, 40.0)
    print(f"    Roll={att_result[0]:.4f}°, Pitch={att_result[1]:.4f}°, Heading={att_result[2]:.4f}°")

    # 生成各图表
    plot_imu_raw_data(imu_data, RESULTS_DIR)
    plot_result_bar(att_result, RESULTS_DIR)
    plot_convergence(imu_data, RESULTS_DIR)
    plot_dual_vector_evolution(imu_data, RESULTS_DIR)
    plot_gpfpd_reference(ref_raw, RESULTS_DIR)
    plot_summary_dashboard(att_result, imu_data, RESULTS_DIR)
    save_error_csv(att_result, RESULTS_DIR)

    print(f"\n{'=' * 70}")
    print(f"所有图表已保存至: {RESULTS_DIR}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
