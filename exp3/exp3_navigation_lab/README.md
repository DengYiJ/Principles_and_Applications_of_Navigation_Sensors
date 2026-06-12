# 实验(3) 惯性导航实验 — 捷联惯导解算系统

基于双矢量粗对准 + 8维卡尔曼滤波精对准 + 四元数姿态更新的完整SINS解算流水线。

## 快速开始

```bash
# 1. 安装依赖
pip install numpy matplotlib pyyaml

# 2. 粗对准 (双矢量法, 单点取点)
python scripts/coarse_align_9500.py          # 0_0_0 数据集
python scripts/coarse_align_30_0_0.py        # 30_0_0 数据集

# 3. 精对准 (8维KF, 全量数据) + 全套图表
python scripts/fine_align_final.py           # 两组一起跑

# 4. 姿态更新 (3组多位置静态) + 对比作图
python scripts/attitude_update_v2.py
```

## 实验结果速览

### 粗对准 (双矢量法, 单点取点)

| 姿态 | Roll (°) | Pitch (°) | Heading (°) | L2误差 | 参考Heading |
|------|----------|-----------|-------------|--------|-------------|
| 0_0_0 | 0.039 | -0.018 | 0.798 | 0.047° | 0.770° |
| 30_0_0 | 0.063 | 0.017 | 330.717 | 0.102° | 330.642° |

### 精对准 (8维KF, 全量数据, 航向锁定于粗对准)

| 姿态 | Roll (°) | Pitch (°) | Heading (°) | L2误差 |
|------|----------|-----------|-------------|--------|
| 0_0_0 | -0.003 | -0.019 | 0.798 | 0.028° |
| 30_0_0 | 0.005 | -0.016 | 330.717 | 0.075° |

### 姿态更新 (纯惯性递推, 统一3-2-1转序, 与GPFPD原生格式一致)

| 数据集 | 时长 | 文件名含义 | Heading RMSE | Pitch RMSE | Roll RMSE |
|--------|------|-----------|:---:|:---:|:---:|
| 0_20_0 | 341s | 俯仰20° | 0.41° | 0.28° | 0.62° |
| 0_0_90 | 319s | 横滚90° | 0.53° | 0.09° | 0.19° |
| -30_-20_180 | 356s | 偏航-30°俯仰-20°横滚180° | 0.04° | 1.31° | 0.65° |

## 项目结构

```
exp3_navigation_lab/
├── configs/exp3_config.yaml     # 全局配置
├── data/
│   ├── 初始对准/                 # 静态对准数据
│   └── 姿态更新/                 # 多位置静态测试数据
├── src/
│   ├── alignment/               # coarse_aligner + fine_aligner (8-state KF)
│   ├── attitude/                # attitude_updater
│   ├── data_io/                 # data_loader (GTIMU/GPFPD解析)
│   ├── utils/                   # quaternion, dcm, euler_angles, earth_model, kalman_filter
│   ├── analysis/                # comparison_analyzer, plotter
│   └── preprocessing/           # preprocessor, outlier_detector
├── scripts/                     # 运行脚本
│   ├── coarse_align_9500.py     # 0_0_0 粗对准
│   ├── coarse_align_30_0_0.py   # 30_0_0 粗对准
│   ├── fine_align_final.py      # 两组精对准 (单点粗对准 + 8维KF)
│   └── attitude_update_v2.py    # 三组姿态更新 (统一3-2-1转序)
├── results/                     # 输出图表和CSV
└── README.md
```

## 算法概述

### 1. 粗对准 — 双矢量法

利用重力矢量在惯性系中随地���自转改变方向，在两个时刻(t1, t2)构造观测/参考矢量对，解算初始姿态矩阵。

- 核心: `C_b^i(0) = [V1,V2,V1×V2] · [R1,R2,R1×R2]⁻¹`
- 最终: `C_n^b = C_n^i(0) @ C_b^i(0)^T` — nav-to-body DCM
- 数据量: ~40-70s (单点取点), 不依赖全量数据
- 精度: ~0.1° L2

### 2. 精对准 — 8维卡尔曼滤波

```
状态 X = [φ_E, φ_N, δv_E, δv_N, ε_x, ε_y, ∇_x, ∇_y]
量测 z = [δv_E, δv_N]  (静基座真速=0)
```

- **可观测**: φ_E, φ_N, δv_E, δv_N, ε_x, ε_y, ∇_x, ∇_y
- **不可观测 (移除)**: φ_U (航向误差), δv_U, ε_z, ∇_z
- **航向锁定**: φ_U = 0, 航向维持粗对准结果
- **量测选择**: dim_z=2, 仅水平速度。垂直通道 fⁿ+gⁿ≈2g 会淹没失准角信号
- **精度**: L2 0.028°~0.075°, Roll/Pitch 比粗对准提升 10-50x

### 3. 姿态更新 — 四元数递推

- 陀螺补偿: `ω_nb^b = ω_ib^b - ε - C_n^b·ω_in^n` (零偏+地球自转)
- 积分: 毕卡二阶法, 200Hz
- 转序约定: GPFPD用3-2-1(航向→俯仰→横滚), 算法内部用3-1-2
- 比较方式: 统一转3-2-1与GPFPD原生格式对比

## 数据格式

### $GTIMU (IMU原始数据)
```
$GTIMU,GPSWeek,GPSTime,GyroX,GyroY,GyroZ,AccX,AccY,AccZ,Tpr*cs
```
- Gyro: 原始值 °/s → ×0.0174533 → rad/s
- Acc: 原始值 g → ×9.78033 → m/s²
- 系统内部已完成圆锥/划桨误差补偿，可直接使用

### $GPFPD (惯导参考输出)
```
$GPFPD,GPSWeek,GPSTime,Heading,Pitch,Roll,Lat,Lon,Alt,Ve,Vn,Vu,...*cs
```
- **转序: 3-2-1** (航向→俯仰→横滚) — 与算法内部3-1-2不同
- 同一物理姿态在两种转序下欧拉角数值完全不同
- 比较时需统一到同一转序（DCM空间等价，欧拉角数值不同）

## 已修复的关键Bug

| # | 问题 | 文件 | 影响 |
|---|------|------|------|
| 1 | DataLoader: `$GTIMU,`前缀剥离偏移1字符 | data_loader.py | 所有字段索引错位1列 |
| 2 | scale_acc: 0.00978→9.78 (g→m/s²) | data_loader.py, config | 比力值偏小1000倍 |
| 3 | C_n^b公式: C_n^i^T@C_b^i → C_n^i@C_b^i^T | coarse_aligner.py | 计算body→nav而非nav→body |
| 4 | euler321_to_dcm: 矩阵6处符号错误 | euler_angles.py | 3-2-1往返不一致 |
| 5 | dcm_to_euler321: 提取公式索引错误 | euler_angles.py | 同上 |
| 6 | FineAligner: 12维→8维, dim_z=3→2 | fine_aligner.py | 不可观测状态导致KF发散 |
| 7 | GPFPD二进制头: 编码错误 | data_loader.py | 姿态更新GPFPD无法加载 |
