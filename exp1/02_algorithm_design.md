# [Algorithm Design] 惯性导航传感器标定算法方案设计

> **依据规则**：`clinerules/10_algorithm_design.md` — 系统工程设计管道全链路
> **输入资产**：`01_experiment_guide_analysis.md` — 实验指导书深度剖析
> **目标**：从文献/需求到高内聚低耦合工程设计的全链路闭航蓝图

---

## 🧭 Stage 0：需求剖析与多维约束 (Requirement Analysis)

### 项目目标审计

| 维度 | 内容 |
|:---|:---|
| **本质痛点** | 捷联惯导系统中加速度计与陀螺仪因制造工艺存在标度因数误差、安装误差、零偏及随机噪声，若不标定直接使用，导航位置误差将快速发散 |
| **最终交付结果** | 加速度计综合误差矩阵 **K_a**(3×3) + 零偏 **D_a**(3×1)；陀螺仪综合误差矩阵 **K_g**(3×3) + 零偏 **D_g**(3×1)；Allan方差曲线与ARW/BI指标 |
| **核心量化评价指标** | ① 标定残差（最小二乘拟合残差RMS）② 正反转积分差值一致性 ③ 八位置零偏标准差 ④ Allan方差双对数曲线拟合优度 |

### I/O 边界梳理

| 方向 | 数据类型 | 内容说明 | 数据速率 |
|:---|:---|:---|:---|
| **输入** | `GTIMU` NMEA语句 | `GPSWeek, GPSTime, GyroX, GyroY, GyroZ, AccX, AccY, AccZ, Tpr` | 200Hz |
| **输入** | 转台设置参数（人工/文件） | 姿态角度序列、转速序列(±10~±50°/s)、旋转方向 | 离线配置 |
| **输出** | `CalibResult` 结构体 | **K_a**(3×3), **D_a**(3×1), **K_g**(3×3), **D_g**(3×1) | 一次性 |
| **输出** | `AllanResult` 结构体 | `tau`向量, `sigma`向量, ARW, BiasInstability | 一次性 |

### 硬核边界约束

| 约束维度 | 具体约束 | 设计影响 |
|:---------|:---------|:---------|
| **实时性频率** | 离线标定（非实时），无严格实时性要求 | 允许Python/numpy实现，无需C++优化 |
| **部署平台** | Windows/Linux 台式机 | 无需嵌入式交叉编译 |
| **数据处理** | 200Hz × 2min × (6+8)位置 + 2h静态 ≈ 150万行数据 | 需设计流式/分段处理，避免内存溢出 |
| **数值精度** | 最小二乘矩阵条件数需监控，防止病态解 | 需实现矩阵条件数计算与警示机制 |
| **物理约束** | 重力加速度g ≈ 9.8 m/s²；地球自转速率 ≈ 15 °/h | 参数物理范围断言（Stage 6） |

---

## 🗺 Stage 1：总体架构与逻辑链设计 (Global Architecture)

### 系统简介

本系统基于**多位置/速率实验编排 + 最小二乘/积分参数辨识 + Allan方差时域分析**框架，实现捷联惯导系统加速度计与陀螺仪综合误差参数（标度因数-安装误差耦合矩阵、零偏向量、随机噪声指标）的全链路离线标定。

### 总体流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    IMU 标定算法系统顶层架构                                 │
└─────────────────────────────────────────────────────────────────────────┘

  原始GTIMU数据 ───────────────────────────────────────────────────────────┐
        │                                                                  │
        ▼                                                                  │
┌───────────────────┐                                                     │
│   数据加载与解析    │  ← 解析GTIMU语句，按位置/速率/方向分组标记                │
│  (DataLoader)     │                                                     │
└────────┬──────────┘                                                     │
         │                                                                 │
         ▼                                                                 │
┌───────────────────┐                                                     │
│   数据预处理模块    │  ← 均值滤波、异常值剔除、角度增量积分                      │
│  (Preprocessor)    │                                                     │
└────────┬──────────┘                                                     │
         │                                                                 │
         ├──────────────────────────────────────────────────────────────────┤
         │                          │                      │               │
         ▼                          ▼                      ▼               │
┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐       │
│ 加速度计六位置标定  │  │ 陀螺仪速率标定     │  │  陀螺仪八位置零偏   │       │
│  (AccelCalibrator) │─▶│  (GyroRateCalib)   │─▶│  (GyroBiasCalib)  │       │
│  最小二乘参数辨识   │  │  正反转积分参数提取  │  │  扣除K_g后均值法    │       │
└────────┬──────────┘  └────────┬──────────┘  └────────┬──────────┘       │
         │                      │                      │                   │
         ▼                      ▼                      ▼                   │
┌──────────────────────────────────────────────────────┐                  │
│                 结果组装与输出模块                      │                  │
│               (ResultAssembler)                      │◀─────────────────┘
│         输出 K_a, D_a, K_g, D_g                      │
└──────────────────────────────────────────────────────┘

         ▼ (可选独立流程)
┌──────────────────────────────────────────────────────┐
│              Allan方差分析模块                         │
│            (AllanVarianceAnalyzer)                    │
│          ≥2h静态数据 → 双对数曲线 → ARW & BI          │
└──────────────────────────────────────────────────────┘
```

### 模块关系论证

| 依赖链 | 数据流向 | 架构理由 |
|:-------|:---------|:---------|
| `DataLoader` → `Preprocessor` | 原始GTIMU字串流 → 数值数组 | 将低层文本解析与高层算法解耦，便于支持不同数据格式 |
| `Preprocessor` → `AccelCalibrator` | 6位置均值 → 6×4矩阵 | 均值滤波必须在参数辨识前完成，先降噪再求解 |
| `Preprocessor` → `GyroRateCalib` | 正反转角度增量 → 4π除法 | 积分和差运算独立封装，便于单元测试积分算法正确性 |
| `GyroRateCalib` → `GyroBiasCalib` | **K_g**(3×3) → 式(22)输入 | **严格顺序依赖**：八位置零偏解算必须提前已知**K_g** |
| `DataLoader` → `AllanVarianceAnalyzer` | 原始静态数据 → 分组方差 | Allan方差需原始高频数据（不均值），单独数据路径 |

---

## 🧩 Stage 2：高内聚低耦合模块划分 (Module Decomposition)

### 模块1：`DataLoader` — 数据加载与解析

| 属性 | 内容 |
|:---|:---|
| **模块名称** | `DataLoader` |
| **模块职责** | 读取GTIMU/NMEA格式原始数据文件，按实验场景（位置编组、速率编组、静态编组）解析为结构化`DataFrame`或`DataChunk`集合 |
| **明确 I/O** | **入参**：`file_paths: List[str]`, `file_type: str = "GTIMU"`；**出参**：`raw_data: RawDataBundle`（含`gyro_arr: ndarray[N×3]`, `acc_arr: ndarray[N×3]`, `timestamps: ndarray[N]`, `scenario_tags: List[str]`） |
| **依赖树** | 无（基础输入层） |
| **设计合理性** | 数据格式解析与后续算法逻辑完全无关，单独模块允许轻松扩展不同IMU数据格式（如KVH、Xsens、ADIS等），仅需替换此模块即可适配 |

### 模块2：`Preprocessor` — 数据预处理

| 属性 | 内容 |
|:---|:---|
| **模块名称** | `Preprocessor` |
| **模块职责** | 执行均值滤波（200Hz→每个位置1个均值）、异常值剔除（3σ准则）、角度增量积分（梯形法/辛普森法）、旋转方向标记 |
| **明确 I/O** | **入参**：`raw_data: RawDataBundle`；**出参**：`processed_data: ProcessedData`（含`acc_means: dict[pose_id→ndarray[3]]`, `gyro_integrals: dict[rate_setting→ndarray[3]]`, `static_gyro: ndarray[M×3]`） |
| **依赖树** | 依赖 `DataLoader` |
| **设计合理性** | 预处理是时间序列数据处理的通用步骤，与具体标定算法的参数解算逻辑解耦。若需替换滤波算法（如低通→带通），仅改此模块 |

### 模块3：`AccelCalibrator` — 加速度计六位置标定

| 属性 | 内容 |
|:---|:---|
| **模块名称** | `AccelCalibrator` |
| **模块职责** | 基于六位置加速度均值数据，按式(9)构建输入矩阵**A**(6×4)，对X/Y/Z轴分别执行最小二乘解算(式8)，组装**K_a**(3×3)和**D_a**(3×1)，并计算拟合残差评估标准 |
| **明确 I/O** | **入参**：`acc_means: dict[pose_id→ndarray[3]]`, `pose_table: List[Tuple[float,float,float]]`（6个位置的理论重力分量）；**出参**：`accel_calib: AccelCalibResult`（含`K_a: ndarray[3×3]`, `D_a: ndarray[3]`, `residuals: ndarray[6]`, `condition_number: float`） |
| **依赖树** | 依赖 `Preprocessor` |
| **设计合理性** | 加速度计标定是独立的线性最小二乘问题，与陀螺仪标定无数据依赖关系，可独立测试。矩阵条件数监控自带防御机制 |

### 模块4：`GyroRateCalibrator` — 陀螺仪速率标定（综合误差矩阵）

| 属性 | 内容 |
|:---|:---|
| **模块名称** | `GyroRateCalibrator` |
| **模块职责** | 基于三位置×10转速×正反转的陀螺角增量数据，按式(18-20)计算正反转积分差值/和值，提取**K_g**(3×3)的9个元素，对多转速结果取均值 |
| **明确 I/O** | **入参**：`gyro_integrals: dict[rate_setting→tuple(J_pos, J_neg)]`，`rate_config: RateConfig`（转速序列、方向标志）；**出参**：`gyro_rate_calib: GyroRateCalibResult`（含`K_g: ndarray[3×3]`, `K_g_std: ndarray[3×3]`各转速间标准差） |
| **依赖树** | 依赖 `Preprocessor` |
| **设计合理性** | 速率标定的"积分-差值-取均值"逻辑自成一体，与陀螺零偏标定**顺序依赖但数据隔离**，若速率标定失败（如转台数据异常），零偏标定需降级 |

### 模块5：`GyroBiasCalibrator` — 陀螺仪八位置零偏标定

| 属性 | 内容 |
|:---|:---|
| **模块名称** | `GyroBiasCalibrator` |
| **模块职责** | 基于八位置陀螺均值数据，利用已知的**K_g**按式(22)计算每个位置的零偏估计，再按式(23)取8位置均值得到最终**D_g** |
| **明确 I/O** | **入参**：`gyro_means: dict[pose_id→ndarray[3]]`，`K_g: ndarray[3×3]`（来自GyroRateCalibrator），`pose_table: List[Tuple[float,float,float]]`（8个姿态的地球自转理论投影）；**出参**：`gyro_bias_calib: GyroBiasCalibResult`（含`D_g: ndarray[3]`, `D_g_std: ndarray[3]`），`bias_per_pose: ndarray[8×3]`） |
| **依赖树** | 依赖 `Preprocessor`，`GyroRateCalibrator`（提供**K_g**） |
| **设计合理性** | 该模块必须与`GyroRateCalibrator`分离设计，原因：(1) 数据来源不同（速率标定+八位置）；(2) 计算方法不同（积分差值 vs 均值法）；(3) 允许在有**K_g**先验时独立复算零偏 |

### 模块6：`AllanVarianceAnalyzer` — Allan方差静态噪声分析

| 属性 | 内容 |
|:---|:---|
| **模块名称** | `AllanVarianceAnalyzer` |
| **模块职责** | 对≥2小时静态陀螺数据实施Allan方差计算：逐步增大相关时间τ→重叠分组统计方差→双对数坐标拟合斜率→提取ARW(°/√h)与Bias Instability(°/h) |
| **明确 I/O** | **入参**：`static_gyro: ndarray[M×3]`（M≥200Hz×7200s=1,440,000），`fs: float=200.0`；**出参**：`allan_result: AllanResult`（含`tau: ndarray[P]`, `sigma: ndarray[P×3]`, `ARW: ndarray[3]`, `BI: ndarray[3]`, `fitted_curves: dict`） |
| **依赖树** | 依赖 `DataLoader`（直接取原始静态数据，不经均值预处理） |
| **设计合理性** | Allan方差分析是独立的后处理模块，需原始高频数据（不可均值），与确定性参数标定完全并行，可作为独立模块或脚本运行 |

### 模块7：`ResultAssembler` — 结果组装与输出

| 属性 | 内容 |
|:---|:---|
| **模块名称** | `ResultAssembler` |
| **模块职责** | 将 `AccelCalibrator`、`GyroRateCalibrator`、`GyroBiasCalibrator` 的输出组装为统一标定报告结构体，生成可视化图表（标定前后对比、残差分布图），序列化为JSON/YAML格式 |
| **明确 I/O** | **入参**：`accel_result: AccelCalibResult`, `gyro_rate_result: GyroRateCalibResult`, `gyro_bias_result: GyroBiasCalibResult`；**出参**：`calibration_report: CalibrationReport`（含全部参数、质量指标、路径存盘） |
| **依赖树** | 依赖 `AccelCalibrator`, `GyroRateCalibrator`, `GyroBiasCalibrator` |
| **设计合理性** | 结果组装与可视化是独立的展示层，与核心标定算法完全解耦，允许灵活切换输出格式（控制台/文件/数据库） |

---

## 🔄 Stage 3：数据生命周期与维度流设计 (Dataflow Design)

### 全生命周期追踪

```
[采集阶段]                               [解析阶段]                           [预处理阶段]
GTIMU文本流(200Hz)                    原始数值数组                          分组统计量
┌─────────────┐       parse_gtimu()    ┌─────────────┐       group_by_scenario()   ┌──────────────┐
│ *.log / *.txt│ ─────────────────────▶│ gyro[N×3]   │ ──────────────────────────▶│ acc_means     │
│ NMEA语句     │                      │ acc [N×3]   │                          │ [pose_id→Vec3]│
│ ≈150万行     │                      │ time[N]     │                          │ gyro_means    │
└─────────────┘                       └─────────────┘                          │ [pose_id→Vec3]│
       │                                    │                                   │ gyro_integrals│
       │                                    │                                   │ [rate→(J+,J-)]│
       │                                    │                                   └──────────────┘
       │                                    │                                          │
       ▼                                    ▼                                          ▼
[长期静态数据]                        ┌─────────────────────┐               ┌──────────────────────┐
┌─────────────┐                      │ GTIMU静态片段        │              │                   │
│ static.log  │                      │ gyro_static[1.44M×3] │              │                   │
│ ≥2h静止     │                      │ (200Hz×7200s)       │              │                   │
└─────────────┘                      └─────────────────────┘              │                   │
                                                                          │                   │
                                                                          ▼                   ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│ [算法阶段]                                                                                     │
│                                                                                               │
│  acc_means[6]                                                                                 │
│  + pose_table[6×(ax,ay,az)]  ──▶  AccelCalibrator  ──▶  K_a[3×3], D_a[3], residuals[6]      │
│                                                                                               │
│  gyro_integrals[3×10×2]                                                                       │
│  + rate_config              ──▶  GyroRateCalibrator ──▶  K_g[3×3], K_g_std[3×3]              │
│                                                                                               │
│  gyro_means[8] + K_g[3×3]                                                                     │
│  + pose_table[8×(ωx,ωy,ωz)] ──▶  GyroBiasCalibrator ──▶  D_g[3], D_g_std[3], bias_p8[8×3]  │
│                                                                                               │
│  gyro_static[1.44M×3] + fs                                                                   │
│                             ──▶  AllanVarianceAnalyzer ──▶  tau[P], sigma[P×3], ARW[3], BI[3]│
└───────────────────────────────────────────────────────────────────────────────────────────────┘
                                                                          │
                                                                          ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ [组装阶段]                                                                 │
│  CalibrationReport                                                        │
│  ├── accel:   {K_a[3×3], D_a[3], cond_num, residuals_rms}                │
│  ├── gyro:    {K_g[3×3], K_g_std[3×3], D_g[3], D_g_std[3]}               │
│  ├── allan:   {ARW[3], BI[3], tau_arr, sigma_arr}                        │
│  └── quality: {reprojection_error, bias_consistency, allan_fit_goodness}  │
└───────────────────────────────────────────────────────────────────────────┘
```

### 流转指标约束（维度与格式）

| 流程步骤 | 数据类型 | 矩阵维度 | 物理含义 | 变换操作 |
|:---------|:---------|:---------|:---------|:---------|
| **原始数据** | `float64` | `N×7` (gx,gy,gz,ax,ay,az,Tpr) | 传感器原始数字量(加工程变换后) | — |
| **位置均值** | `float64` | `6×3` (acc) / `8×3` (gyro) | 每个位置12000个样本的算术平均 → 压制白噪声 | Mean(12000)→1 |
| **角度增量** | `float64` | `3×10×2` (3轴×10转速×正反) | 旋转一周陀螺输出的梯形积分 = 实际转过的角度 + 误差项 | Trapz(每秒)→累计 |
| **理论输入矩阵** | `float64` | `6×4` (acc X轴) | 重力分量在6个位置的投影矩阵，含`[±g, 0,0,1]`行 | 式(9)构造 |
| **综合误差矩阵** | `float64` | `3×3` | 标度因数+安装误差的乘性耦合整体 | LSQ解6×4→4→组装9 |
| **零偏向量** | `float64` | `3` | 常值偏置(°/h或m/s²) | 式(22)→8位置均值 |
| **Allan方差** | `float64` | `P×3` (P~100个τ) | 不同相关时间下的陀螺噪声方差 | 重叠分组方差 |
| **ARW/BI** | `float64` | `3` | 角度随机游走(°/√h) / 零偏不稳定性(°/h) | 双对数曲线斜率拟合 |

---

## 📐 Stage 4：高层算法逻辑与伪代码 (Algorithmic Logic)

### 算法主循环（高层运行时序）

```
Step 1: 数据加载与分组
    └── 遍历所有GTIMU数据文件 → 按scenario_tag（六位置/速率标定/八位置/静态）分组标记

Step 2: 加速度计六位置标定 ← 独立链路
    ├── Step 2.1: 对每个位置的加速度数据取算术均值
    ├── Step 2.2: 按式(9)构建6×4输入矩阵 A (每轴独立)
    ├── Step 2.3: 最小二乘法求解 X = (A^T A)^{-1} A^T · b  (b为6个位置的该轴均值)
    ├── Step 2.4: 从X提取K_a行向量和D_a标量，组装3×3+3×1
    └── Step 2.5: 计算拟合残差与矩阵条件数 → 输出质量控制标志

Step 3: 陀螺仪速率标定 ← 独立链路
    ├── Step 3.1: 对每个转速配置（轴方向×转速值×旋转方向），计算顺时针和逆时针旋转一周的角度增量J
    ├── Step 3.2: 正反转角度增量相减: ΔJ = J_pos - J_neg → 消去零偏与地球自转
    ├── Step 3.3: K_g(i,i) = ΔJ / (4π)  → 提取对角元素
    ├── Step 3.4: K_g(i,j) 同理提取交轴元素
    ├── Step 3.5: 对10个转速的结果取均值，并计算标准差
    └── Step 3.6: 输出K_g(3×3)及置信度

Step 4: 陀螺仪八位置零偏标定 ← 依赖Step 3
    ├── Step 4.1: 对每个位置的陀螺数据取均值
    ├── Step 4.2: 利用K_g按式(22): D_g^(i) = mean_i - K_g · ω_earth^(i)
    ├── Step 4.3: D_g = mean(D_g^(1), ..., D_g^(8))
    └── Step 4.4: 计算D_g_std = std(D_g^(1), ..., D_g^(8)) → 零偏估计一致性指标

Step 5: Allan方差分析 ← 独立链路
    ├── Step 5.1: 对≥2h静态陀螺数据，逐步增大相关时间τ (τ = τ₀·2^k, k=0,1,...)
    ├── Step 5.2: 对每个τ，将数据划分为重叠/非重叠子段，计算子段均值的方差
    ├── Step 5.3: 绘制log(σ) vs log(τ) 双对数曲线
    ├── Step 5.4: 在τ^-1/2斜率区拟合ARW: σ(τ) = ARW / √τ
    └── Step 5.5: 在曲线最低点提取Bias Instability

Step 6: 结果组装与输出
    ├── Step 6.1: 汇总所有标定参数
    ├── Step 6.2: 计算综合质量指标（重投影误差、一致性、拟合优度）
    └── Step 6.3: 序列化为JSON/YAML → 保存文件 + 控制台摘要输出
```

### 泛型伪代码 (Language-Agnostic Pseudocode)

```
// ============================================================
// Pseudo 1: 加速度计六位置标定核心
// ============================================================
FUNCTION calibrate_accel(acc_means[6][3], pose_table[6][3]) -> (K_a[3][3], D_a[3], residual[6])
    // 理论重力投影: 在位置i，加速度计感受的比力为 -g_enu 在载体系的投影
    // 简化: 6位置对应 ±g_X, ±g_Y, ±g_Z 方向

    FOR EACH axis IN {X, Y, Z}:
        // 构建输入矩阵 A (6×4):
        // 每行 = [a_x^(i), a_y^(i), a_z^(i), 1]
        // 其中 a_x^(i), a_y^(i), a_z^(i) 为位置i的理论重力分量
        A = matrix(6, 4)
        FOR i IN 0..5:
            A[i][0] = pose_table[i].ax
            A[i][1] = pose_table[i].ay
            A[i][2] = pose_table[i].az
            A[i][3] = 1.0

        // 构建观测向量 b (6×1): 6个位置的axis轴加速度均值
        b = vector(6)
        FOR i IN 0..5:
            b[i] = acc_means[i][axis]

        // 最小二乘求解: X = (A^T A)^{-1} A^T b
        X = solve_least_squares(A, b)

        // X = [K_Xx, K_Xy, K_Xz, D_X]^T
        K_a[axis][0] = X[0]    // K_Xx
        K_a[axis][1] = X[1]    // K_Xy
        K_a[axis][2] = X[2]    // K_Xz
        D_a[axis]    = X[3]    // D_X

        // 计算拟合残差: residual = A·X - b
        residual[axis] = compute_residual(A, X, b)

        // 计算条件数 cond(A^T A) → 病态警示
        cond_num = compute_condition_number(A)
        IF cond_num > COND_THRESHOLD:
            FLAG_WARNING("Accel calibrator: ill-conditioned matrix A")

    RETURN (K_a, D_a, residual)
END FUNCTION


// ============================================================
// Pseudo 2: 陀螺仪速率标定核心（正反转积分法）
// ============================================================
FUNCTION calibrate_gyro_rate(gyro_data[axis_idx][rate_idx][direction][time_series],
                              rate_config) -> (K_g[3][3], K_g_std[3][3])

    // 每个axis_idx = 0,1,2 对应X,Y,Z轴转台旋转
    // 每个rate_idx 对应 ±10, ±20, ±30, ±40, ±50 °/s
    // direction = 0 (逆时针/正向), 1 (顺时针/负向)
    // time_series 为旋转一周内的陀螺三轴输出

    K_g = zeros(3, 3)
    K_g_accum = zeros(3, 3)  // 用于多转速取均值
    RATE_COUNT = number_of_rates

    FOR EACH rate_idx:
        FOR EACH axis_idx:
            // 提取正反转数据
            gyro_pos = gyro_data[axis_idx][rate_idx][POSITIVE]
            gyro_neg = gyro_data[axis_idx][rate_idx][NEGATIVE]

            // 计算角度增量: 梯形积分
            // J_pos = ∫(gyro_pos(t)) dt  积分整周
            // J_neg = ∫(gyro_neg(t)) dt  积分整周
            J_pos = trapezoidal_integrate(gyro_pos, dt)
            J_neg = trapezoidal_integrate(gyro_neg, dt)

            // 正反转相减 → 消去零偏和地球自转 → ΔJ = 2 * 2π * K_g_row
            // ΔJ = J_pos - J_neg  = [4π*K_gXx, 4π*K_gYx, 4π*K_gZx]^T
            delta_J = J_pos - J_neg

            // 提取K_g对应列 (axis_idx列)
            // K_g[0][axis_idx] = delta_J[0] / (4π)
            // K_g[1][axis_idx] = delta_J[1] / (4π)
            // K_g[2][axis_idx] = delta_J[2] / (4π)
            FOR row IN 0..2:
                K_g_accum[row][axis_idx] += delta_J[row] / (4 * PI)

        // 可选: 对当前转速的K_g进行地球自转修正
        // 正反转和 + 扣除地球自转投影

    // 多转速取均值
    K_g = K_g_accum / RATE_COUNT
    // K_g_std = 各转速K_g的标准差（置信度）

    RETURN (K_g, K_g_std)
END FUNCTION


// ============================================================
// Pseudo 3: 陀螺仪八位置零偏标定
// ============================================================
FUNCTION calibrate_gyro_bias(gyro_means[8][3], K_g[3][3],
                              omega_earth_poses[8][3]) -> (D_g[3], D_g_std[3])

    // omega_earth_poses[i] = 位置i下地球自转在载体系的理论投影
    // gyro_means[i] = 位置i的陀螺输出均值(含K_g·omega + D_g + noise)

    bias_estimates = zeros(8, 3)

    FOR i IN 0..7:
        // 式(22): D_g^(i) = mean_i - K_g · ω_earth^(i)
        predicted_omega = matrix_vector_multiply(K_g, omega_earth_poses[i])
        bias_estimates[i] = gyro_means[i] - predicted_omega

    // 式(23): D_g = (1/8) Σ D_g^(i)
    D_g = mean(bias_estimates, axis=0)

    // 标准差: 评估8个位置零偏估计的一致性
    D_g_std = std(bias_estimates, axis=0)

    RETURN (D_g, D_g_std)
END FUNCTION


// ============================================================
// Pseudo 4: Allan方差分析
// ============================================================
FUNCTION analyze_allan_variance(gyro_static[N][3], fs) -> (tau[P], sigma[P][3],
                                                            ARW[3], BI[3])
    dt = 1.0 / fs

    // 构建τ序列: 对数均匀分布
    // τ_k = τ₀ · 2^k, k=0,1,...,K
    // 直到 τ_max < N·dt / 9  (保证足够子段数)
    tau = generate_logspace_tau(dt, N)

    FOR EACH axis IN 0..2:
        data = gyro_static[:, axis]
        sigma_axis = vector(length(tau))

        FOR EACH k, tau_k IN tau:
            m = round(tau_k / dt)    // 子段长度(样本数)
            L = floor(N / m)          // 子段数(非重叠)

            // 非重叠Allan方差(简化)
            // 亦可实现重叠Allan方差以提高统计量
            segments = reshape(data[0:L*m], (L, m))
            segment_means = mean(segments, axis=1)

            // Allan方差 = (1/2)·E[(y_{k+1} - y_k)²]
            diff = segment_means[1:L] - segment_means[0:L-1]
            sigma_axis[k] = sqrt(0.5 * mean(diff^2))

        sigma[:, axis] = sigma_axis

        // 拟合ARW: 在τ取较小值时, σ(τ)*√τ ≈ ARW
        // 寻找log-log斜率≈-1/2的区间
        low_tau_idx = find_slope_region(log(tau), log(sigma_axis), target_slope=-0.5)
        ARW[axis] = median(sigma_axis[low_tau_idx] * sqrt(tau[low_tau_idx]))

        // 提取Bias Instability: Allan方差曲线最低点
        min_idx = argmin(sigma_axis)
        BI[axis] = sigma_axis[min_idx]

    RETURN (tau, sigma, ARW, BI)
END FUNCTION
```

---

## 🎯 Stage 5：文献到工程双向映射 (Paper-to-Code Mapping)

### 公式-代码追溯链

| 论文公式 | 工程实现模块/函数 | 核心变量/参数 | 说明 |
|:---------|:------------------|:--------------|:-----|
| 式(1): $\hat{\mathbf{a}} = \mathbf{K}_a \mathbf{a} + \mathbf{D}_a + \boldsymbol{\varepsilon}_a$ | `AccelCalibrator.calibrate_accel()` | $\mathbf{K}_a$→`K_a[3×3]`; $\mathbf{D}_a$→`D_a[3]` | 加速度计误差模型→最小二乘参数辨识 |
| 式(5): $\hat{a}_X^{(i)} = D_{aX} + \mathbf{K}_{aX} \cdot \mathbf{a}^{(i)T}$ | `AccelCalibrator._build_input_matrix()` | $\mathbf{A}$(6×4) = `A`; $\mathbf{a}_X$(6×1) = `b` | 单轴观测方程→矩阵构造 |
| 式(8): $\hat{\mathbf{X}} = (\mathbf{A}^T \mathbf{A})^{-1} \mathbf{A}^T \mathbf{a}_X$ | `AccelCalibrator._solve_lsq()` | $\hat{\mathbf{X}}$(4×1)=`X` | 最小二乘闭式解→`numpy.linalg.lstsq` |
| 式(10): $\hat{\boldsymbol{\omega}}_g = \mathbf{K}_g \boldsymbol{\omega}_g + \mathbf{D}_g + \boldsymbol{\varepsilon}_g$ | `GyroRateCalibrator`, `GyroBiasCalibrator` | $\mathbf{K}_g$→`K_g[3×3]`; $\mathbf{D}_g$→`D_g[3]` | 陀螺误差模型→两个独立模块分别求K_g与D_g |
| 式(18): $J^{\text{正}} = 2\pi \mathbf{K}_g + \mathbf{D}_g t_x$ | `GyroRateCalibrator._integrate_one_revolution()` | $J^{\text{正}}$→`J_pos[3]`; $t_x$→`rev_time` | 正转一周角度积分 |
| 式(19): $K_{gXx} = (J_x^{(+)} - J_x^{(-)}) / (4\pi)$ | `GyroRateCalibrator._extract_Kg_element()` | $J^{(+)}-J^{(-)}$→`delta_J[3]` | 正反转差值提取综合误差 |
| 式(22): $\hat{D}_{gX}^{(i)} = \hat{\omega}_{gX}^{(i)} - \mathbf{K}_{gX} \boldsymbol{\omega}_e^{(i)}$ | `GyroBiasCalibrator._compute_bias_per_pose()` | $\hat{D}_{gX}^{(i)}$→`bias_per_pose[i,0]` | 单位置零偏估计 |
| 式(23): $\hat{D}_{gX} = \frac{1}{8}\sum_{i=1}^8\hat{D}_{gX}^{(i)}$ | `GyroBiasCalibrator._average_bias()` | $\hat{D}_{gX}$→`D_g[0]` | 多位置零偏取均值 |
| Allan方差定义: $\sigma^2(\tau) = \frac{1}{2}\langle(\bar{y}_{k+1} - \bar{y}_k)^2\rangle$ | `AllanVarianceAnalyzer._compute_allan_variance()` | $\sigma^2(\tau)$→`sigma_axis[k]` | 重叠/非重叠Allan方差核心计算 |

### 原理图-软件映射

| 实验指导书原理图/表 | 对应软件子系统 | 实现函数 |
|:-------------------|:--------------|:---------|
| 表1：加速度计六位置姿态表 | `common.pose_tables` | `build_accel_pose_table()` |
| 表2：陀螺仪八位置姿态表 | `common.pose_tables` | `build_gyro_pose_table()` |
| 式(9)输入矩阵构造 | `AccelCalibrator` | `_build_input_matrix()` |
| 三位置旋转轴×正反转编排 | `GyroRateCalibrator` | `_parse_rate_scenario()` |
| Allan方差双对数曲线 | `AllanVarianceAnalyzer` | `plot_allan_curve()` |

---

## Stage 5.5：复现风险分析 (Reproduction Risk Analysis)

| 风险类别 | 已知细节 | 缺失细节 | 所需假设 | 潜在影响 | 工程近似策略 |
|:---------|:---------|:---------|:---------|:---------|:-------------|
| **转台参考系对齐** | 载体坐标系与转台坐标系的关系未明确定义 | 转台零位与载体系之间的转换矩阵 | 假设转台零位与IMU载体系对准 | 安装误差角耦合进综合误差矩阵**K**，导致标定参数偏移 | 在报告中标记此假设；若已知转台安装偏差角，可前置扣除 |
| **均值滤波窗口** | "2分钟数据取均值" | 是否预滤除异常值（如转台启动瞬态）；均值前是否低通滤波 | 假设直接算数平均即可，不额外滤波 | 瞬态干扰（转台启动过冲）污染均值 | 实现3σ异常值剔除：剔除距均值>3σ的样本后再均值 |
| **角度积分方法** | "正反转一周的角度增量" | 积分方法（梯形法/辛普森法）；整周判断标准（过零点/编码器触发） | 假设梯形法即可，整周由转台设置保证 | 积分截断误差影响**K_g**精度 | 实现梯形+辛普森两种积分，对比差异；建议使用转台编码器脉冲标记整周 |
| **地球自转模型** | 需扣除地球自转影响 | 当地纬度值；地球自转速率精确值(7.2921150×10⁻⁵ rad/s) | 假设know当地纬度作为配置输入 | 纬度错误导致正反转和值的修正误差 | 将纬度作为配置参数暴露；提供默认海平面纬度45°N |
| **多转速取均值** | "10个转速取平均" | 是否加权平均；各转速标度因数线性度 | 假设标度因数与转速无关 | 若传感器存在转速非线性，均值掩盖了转速相关偏差 | 输出每个转速的**K_g**估计值（`K_g_per_rate`），供用户检查线性度 |
| **Allan方差置信度** | "双对数曲线拟合" | 子段最小样本数约束；重叠Allan方差的实现细节 | 非重叠Allan方差 | 大τ区间的方差估计点数少，置信度低 | 实现重叠Allan方差（提高统计量）并标注置信区间（±1σ误差棒） |
| **零偏稳定性** | 假设零偏在标定全程不变 | 实验顺序的时间戳；温升曲线 | 假设陀螺仪已充分预热(开机>30min) | 若预热不充分，六位置/八位置间零偏漂移导致解偏 | 在结果中输出每个位置的零偏估计序列（`bias_per_pose`），供用户检查漂移趋势 |
| **GTIMU数据格式** | RANGEA/SATXYZ2A格式 | 各字段的字节偏移量与解析规则 | 假设字段顺序固定、空格分隔即可安全解析 | 解析错误导致整批数据无效 | 实现健壮的字段Index映射表与Block Size校验（Stage 6） |

---

## 🔌 Stage 6：接口规范与异常防御 (Interface & Exception Design)

### 契约式接口定义

#### 核心数据结构

```python
# ============================================================
# 数据结构定义 (DataType / NamedTuple / Dataclass)
# ============================================================

@dataclass
class RawDataBundle:
    """原始数据捆绑包——DataLoader产出"""
    gyro: np.ndarray          # shape=(N, 3), dtype=float64, 陀螺三轴输出(°/s)
    accel: np.ndarray         # shape=(N, 3), dtype=float64, 加速度三轴输出(m/s²)
    timestamps: np.ndarray    # shape=(N,), dtype=float64, GPS时间(s)
    scenario_tags: List[str]  # shape=(N,), 每条数据的场景标签: "pos1","pos2",...,"rate_1_pos","rate_1_neg",...,"static"
    metadata: Dict            # 文件头部元数据(采样率、日期、实验配置等)

@dataclass
class ProcessedData:
    """预处理数据——Preprocessor产出"""
    acc_means: Dict[str, np.ndarray]      # key=pose_id, value=ndarray[3], 单位m/s²
    gyro_means: Dict[str, np.ndarray]     # key=pose_id, value=ndarray[3], 单位°/s
    gyro_integrals: Dict[str, Tuple[np.ndarray, np.ndarray]]  # key=rate_setting, value=(J_pos[3], J_neg[3]), 单位°
    static_gyro: np.ndarray               # shape=(M, 3), dtype=float64, 原始静态陀螺数据(不均值)
    static_timestamps: np.ndarray         # shape=(M,), dtype=float64

@dataclass
class AccelCalibResult:
    """加速度计标定结果"""
    K_a: np.ndarray           # shape=(3, 3), dtype=float64
    D_a: np.ndarray           # shape=(3,), dtype=float64, 单位m/s²
    residuals: np.ndarray     # shape=(3, 6), dtype=float64, 每轴每位置的拟合残差
    condition_number: float   # 输入矩阵条件数
    reprojection_error: float # 重投影RMS误差

@dataclass
class GyroRateCalibResult:
    """陀螺速率标定结果"""
    K_g: np.ndarray           # shape=(3, 3), dtype=float64
    K_g_std: np.ndarray       # shape=(3, 3), dtype=float64, 多转速标准差
    K_g_per_rate: np.ndarray  # shape=(10, 3, 3), dtype=float64, 每个转速的K_g(供非线性检查)
    earth_rate_corrected: bool  # 是否已做地球自转修正

@dataclass
class GyroBiasCalibResult:
    """陀螺零偏标定结果"""
    D_g: np.ndarray           # shape=(3,), dtype=float64, 单位°/h 或 °/s
    D_g_std: np.ndarray       # shape=(3,), dtype=float64, 8位置标准差
    bias_per_pose: np.ndarray # shape=(8, 3), dtype=float64, 每个位置的零偏估计(供温漂检查)

@dataclass
class AllanResult:
    """Allan方差分析结果"""
    tau: np.ndarray           # shape=(P,), dtype=float64, 相关时间(s)
    sigma: np.ndarray         # shape=(P, 3), dtype=float64, Allan标准差(°/h)
    ARW: np.ndarray           # shape=(3,), dtype=float64, 角度随机游走(°/√h)
    BI: np.ndarray            # shape=(3,), dtype=float64, 零偏不稳定性(°/h)
    fitted_log_slopes: np.ndarray  # shape=(3,), ARW拟合区间的log-log斜率(应≈-0.5)

@dataclass
class CalibrationReport:
    """完整标定报告——ResultAssembler产出"""
    accel: AccelCalibResult
    gyro_rate: GyroRateCalibResult
    gyro_bias: GyroBiasCalibResult
    allan: Optional[AllanResult]   # Allan方差可选(若未提供静态数据则为None)
    metadata: Dict                  # 标定元数据(时间、配置、文件来源)
    quality_flags: Dict[str, bool]  # 质量控制标志
```

#### 模块API边界

| 模块 | 主入口方法 | 入参 | 出参 | 异常场景 |
|:-----|:-----------|:-----|:-----|:---------|
| `DataLoader` | `load(file_paths, file_type)` | `file_paths: List[Path]`, `file_type: str` | `RawDataBundle` | 文件不存在→`FileNotFoundError`；格式解析错误→`ParseError`(含行号) |
| `Preprocessor` | `process(raw_data, config)` | `raw_data: RawDataBundle`, `config: PreprocConfig` | `ProcessedData` | 数据不足→`InsufficientDataError`；采样率异常→`SampleRateWarning` |
| `AccelCalibrator` | `calibrate(acc_means, pose_table)` | `acc_means: Dict[str, ndarray]`, `pose_table: List[Tuple]` | `AccelCalibResult` | 条件数>阈值→`IllConditionedWarning`；数据NaN→`NumericalError` |
| `GyroRateCalibrator` | `calibrate(gyro_integrals, config)` | `gyro_integrals: Dict`, `config: RateConfig` | `GyroRateCalibResult` | 积分值异常(相减≈0)→`DivisionByZeroError` |
| `GyroBiasCalibrator` | `calibrate(gyro_means, K_g, pose_table)` | `gyro_means: Dict`, `K_g: ndarray`, `pose_table: List[Tuple]` | `GyroBiasCalibResult` | 维度不匹配→`DimensionMismatchError` |
| `AllanVarianceAnalyzer` | `analyze(static_gyro, fs)` | `static_gyro: ndarray[N×3]`, `fs: float` | `AllanResult` | 数据时长不足→`InsufficientDurationError`(需≥1h) |
| `ResultAssembler` | `assemble(results, output_path)` | `results: Dict[str, object]`, `output_path: Path` | `CalibrationReport` | 写入权限→`PermissionError` |

### 防断裂异常处理 (Error Handling)

| 异常场景 | 检测条件 | 降级策略 | 日志输出 |
|:---------|:---------|:---------|:---------|
| **数据为空** | `RawDataBundle.gyro.shape[0] == 0` | 抛出`InsufficientDataError`，不继续运算 | ERROR: "No data loaded from {file_path}" |
| **维度不匹配** | K_g.shape != (3,3) 传入GyroBiasCalibrator | 抛出`DimensionMismatchError`，提示预期形状 | ERROR: "K_g expected (3,3) got {shape}" |
| **传感器异常丢失** | 某位置数据全为0或NaN | 跳过该位置，降级为少位置标定；若跳过位置≥2则报错 | WARNING: "Pose {pose_id} data invalid, skipping" |
| **GNSS掉线(GPS时间戳不连续)** | 时间戳间隔>2×采样间隔 | 分段标记，在分段内分别计算均值/积分 | WARNING: "Timestamp gap detected at index {idx}" |
| **最小二乘病态** | cond(A^T A) > 1e8 | 输出condition_number警告，但仍继续计算，供用户判断 | WARNING: "Ill-conditioned matrix (cond={cond})" |
| **正反转积分异常** | J_pos ≈ J_neg (差值接近0) | 检查转台方向标志；若确认方向标志正确，则抛出`GyroRateCalibError` | ERROR: "Pos/Neg integrals nearly equal, check rotation direction" |
| **Allan方差数据不足** | M < 200Hz × 3600s (1h) | 降低最低τ上限，标注置信度降低 | WARNING: "Static data <1h, Allan variance uncertainty elevated" |

### ⚡ 物理边界断言与多维校验设计

#### 1️⃣ 解析层格式硬规约（GTIMU语句解析）

```python
# GTIMU语句字段Index映射表 & Block Size校验
GTIMU_FIELD_MAP = {
    "GPSWeek":   {"index": 0, "dtype": int},
    "GPSTime":   {"index": 1, "dtype": float},
    "GyroX":     {"index": 2, "dtype": float, "unit": "deg/s"},
    "GyroY":     {"index": 3, "dtype": float},
    "GyroZ":     {"index": 4, "dtype": float},
    "AccX":      {"index": 5, "dtype": float, "unit": "m/s²"},
    "AccY":      {"index": 6, "dtype": float},
    "AccZ":      {"index": 7, "dtype": float},
    "Tpr":       {"index": 8, "dtype": float, "unit": "°C"},
}

MIN_FIELD_COUNT = 9     # 最少字段数
MAX_FIELD_COUNT = 9     # 严格9字段，超出报warn

def parse_gtimu_line(line: str) -> Optional[List[str]]:
    tokens = line.strip().split(",")  # 或split()取决于实际分隔符
    assert MIN_FIELD_COUNT <= len(tokens) <= MAX_FIELD_COUNT, \
        f"GTIMU field count mismatch: expected {MIN_FIELD_COUNT}, got {len(tokens)}, line={line[:80]}"
    # 若有非数字token，标记该行为无效
    # ...
```

#### 2️⃣ 级联量级动态监控

```python
# 所有数据转换核心逻辑处预留shape/dtype/value_range实时监测槽位
MONITOR_SLOTS = {
    "raw_data":         {"shape_check": lambda x: x.ndim == 2 and x.shape[1] >= 7},
    "acc_means":        {"value_range": (-30, 30), "unit": "m/s²"},     # 加速度计输出应在±30m/s²内
    "gyro_means":       {"value_range": (-500, 500), "unit": "°/s"},     # 陀螺输出应在±500°/s内
    "angle_increment":  {"value_range": (-720, 720), "unit": "°"},      # 一周积分应在±720°内
    "K_a_diag":         {"value_range": (0.5, 1.5), "unit": "scalefactor"},  # 标度因数应在0.5~1.5
    "K_g_diag":         {"value_range": (0.5, 1.5), "unit": "scalefactor"},
    "D_a":              {"value_range": (-10, 10), "unit": "m/s²"},
    "D_g":              {"value_range": (-360, 360), "unit": "°/h"},     # 零偏单位转换后
}
```

#### 3️⃣ 严格物理区间断言

```python
# 在状态更新/参数解算前置条件中实施强制断言

# —— 卫星空间位置（如有）：约束在 10^7 m 数量级 ——
# （本实验无卫星数据，保留插槽）

# —— 加速度数值物理合理性 ——
assert np.all(np.abs(acc_means_flat) < 30), \
    f"Accel magnitude exceeds physical bounds: max={np.max(np.abs(acc_means_flat))} m/s²"

# —— 角速率数值物理合理性 ——
assert np.all(np.abs(gyro_means_flat) < 500), \
    f"Gyro rate exceeds physical bounds: max={np.max(np.abs(gyro_means_flat))} °/s"

# —— 综合误差矩阵对角线应在合理标度因数范围内 ——
K_a_diag = np.diag(K_a)
assert np.all((K_a_diag > 0.5) & (K_a_diag < 1.5)), \
    f"K_a diagonal (scalefactors) out of range [0.5,1.5]: {K_a_diag}"

# —— 零偏应在合理范围内 ——
assert np.all(np.abs(D_a) < 10), f"D_a magnitude > 10 m/s²: {D_a}"
# D_g单位若为°/s, 则地球自转约0.0042°/s, 零偏一般<1°/s
# D_g单位若为°/h, 则零偏一般<360°/h
if bias_unit == "deg/s":
    assert np.all(np.abs(D_g) < 1.0), f"D_g magnitude > 1.0 °/s: {D_g}"
elif bias_unit == "deg/h":
    assert np.all(np.abs(D_g) < 360), f"D_g magnitude > 360 °/h: {D_g}"

# —— 地球自转投影一致性 ——
# 在纬度45°处, ω_earth ≈ 10.5 °/h (≈0.0029 °/s)
# 各位置的陀螺均值应与地球自转投影+零偏量级吻合
expected_omega_magnitude = 0.0042  # °/s 最大(赤道)
assert np.all(np.abs(gyro_means) < 1.0), \
    f"Gyro means unreasonably large (>1 °/s), check units or raw data"
```

---

## ⚙ Stage 7：超参数与系统配置设计 (Hyperparameter Taxonomy)

### 参数矩阵分类

#### 数据参数

| 参数名 | 默认值 | 物理作用 | 边界影响 |
|:-------|:-------|:---------|:---------|
| `sampling_rate` | 200 Hz | 原始数据采样频率 | 过低(<100Hz)导致混叠；过高(>1000Hz)增大数据处理量 |
| `accel_static_duration` | 120 s | 每个位置的加速度静态采集时长 | 过短(<30s)均值不收敛；过长(>300s)受零偏漂移影响 |
| `gyro_static_duration` | 120 s | 陀螺八位置采集时长 | 同上 |
| `allan_min_duration` | 7200 s | Allan方差所需最小静态数据时长 | <3600s(1h)大τ区间置信度不足 |

#### 算法参数

| 参数名 | 默认值 | 物理作用 | 边界影响 |
|:-------|:-------|:---------|:---------|
| `accel_pose_count` | 6 | 加速度计标定位置数 | 必须≥4(每轴4参数所需最小值)，<6则超定性不足 |
| `gyro_bias_pose_count` | 8 | 陀螺零偏标定位置数 | 必须≥1，越多零偏估计精度越高(σ/√n) |
| `rate_values_degps` | [10,20,30,40,50] | 速率标定转速序列(°/s) | 过低(<2°/s)信噪比低；过高可能超出传感器量程 |
| `rate_direction_count` | 2 | 正反旋转方向数 | 恒为2(正+反)，不可更改 |
| `rate_position_count` | 3 | 速率标定的转台位置数(绕X,Y,Z轴) | 必须为3(覆盖三个正交轴) |
| `allan_tau_base` | 2 | Allan方差τ序列的倍增基数 | 2^(k)为常用选择；基数过大会降低τ分辨率 |
| `allan_min_cluster_count` | 9 | Allan方差最小子段数 | 过小(<4)大τ估计方差不可靠 |

#### 质量控制参数

| 参数名 | 默认值 | 物理作用 | 边界影响 |
|:-------|:-------|:---------|:---------|
| `cond_threshold` | 1e8 | 最小二乘矩阵条件数告警阈值 | 超过阈值表示输入矩阵病态，标定结果不可信 |
| `outlier_sigma_threshold` | 3.0 | 异常值剔除的σ倍数 | 过小(<2)剔除过多有效数据；过大(>5)异常值残留 |
| `residual_rms_threshold` | 0.5 m/s² | 加速度计拟合残差RMS告警阈值 | 超过阈值表示模型拟合不佳或数据质量差 |
| `bias_consistency_threshold` | 0.5 °/h | 八位置零偏标准差告警阈值 | 超过阈值表示零偏不稳定或地球自转修正有误 |

#### 系统/工程参数

| 参数名 | 默认值 | 物理作用 | 边界影响 |
|:-------|:-------|:---------|:---------|
| `earth_rotation_rate` | 7.2921150e-5 rad/s | 地球自转角速率常数 | 该值来自IERS标准，不应改动 |
| `local_latitude` | 45.0 deg | 实验当地纬度 | 影响地球自转在各个姿态的投影分量；若未知则假设45°N |
| `g_magnitude` | 9.7803267714 m/s² | 重力加速度模量(赤道海平面) | 若需更高精度应代入当地重力公式或实测值 |
| `output_format` | "yaml" | 标定结果输出格式 | 可选yaml/json/txt |
| `visualization_enabled` | true | 是否生成标定结果图表 | 关闭则仅输出数值参数 |

### 超参数配置文件结构 (YAML)

```yaml
# configs/calibration_config.yaml

data:
  sampling_rate: 200          # Hz
  accel_static_duration: 120  # seconds
  gyro_static_duration: 120   # seconds
  allan_min_duration: 7200    # seconds
  allan_tau_base: 2
  allan_min_cluster_count: 9

algorithm:
  accel_pose_count: 6
  gyro_bias_pose_count: 8
  rate_values_degps: [10, 20, 30, 40, 50]
  rate_direction_count: 2
  rate_position_count: 3

quality:
  cond_threshold: 1.0e8
  outlier_sigma_threshold: 3.0
  residual_rms_threshold: 0.5   # m/s²
  bias_consistency_threshold: 0.5  # deg/h

physical_constants:
  earth_rotation_rate: 7.2921150e-5  # rad/s
  local_latitude: 45.0               # deg
  g_magnitude: 9.7803267714           # m/s²

system:
  output_format: "yaml"
  visualization_enabled: true
  output_dir: "./calibration_results"
```

---

## 📂 Stage 8：工业级工程目录结构设计 (Project Directory Tree)

```text
imu_calibration/
│
├── configs/                          # 统一配置管理
│   ├── calibration_config.yaml       # 标定算法超参数(见Stage 7)
│   └── pose_tables.yaml              # 六位置/八位置姿态表
│
├── data/                             # 数据文件
│   ├── raw/                          # 原始GTIMU数据文件
│   │   ├── accel_six_pose/           # 加速度计六位置数据
│   │   ├── gyro_rate_calib/          # 陀螺仪速率标定数据
│   │   ├── gyro_eight_pose/          # 陀螺仪八位置零偏数据
│   │   └── allan_static/             # Allan方差静态数据(≥2h)
│   └── processed/                    # 预处理后的中间数据(可选cache)
│
├── imu_calibration/                  # 核心算法包(包名)
│   │
│   ├── __init__.py                   # 包入口, 暴露CalibrationPipeline
│   │
│   ├── io/                           # 数据IO层 → 对应Stage2模块1
│   │   ├── __init__.py
│   │   ├── data_loader.py            # DataLoader: 加载&解析GTIMU
│   │   ├── gtimu_parser.py           # GTIMU语句解析器(字段Index映射表+Block Size校验)
│   │   └── result_serializer.py      # 标定结果序列化(JSON/YAML)
│   │
│   ├── preprocessing/                # 数据预处理 → 对应Stage2模块2
│   │   ├── __init__.py
│   │   ├── preprocessor.py           # Preprocessor: 均值滤波/积分
│   │   ├── outlier_filter.py         # 3σ异常值剔除
│   │   └── integrator.py             # 角度增量积分器(梯形/辛普森)
│   │
│   ├── calibration/                  # 核心标定算法 → 对应Stage2模块3-5
│   │   ├── __init__.py
│   │   ├── accel_calibrator.py       # AccelCalibrator: 加速度计六位置标定
│   │   ├── gyro_rate_calibrator.py   # GyroRateCalibrator: 速率标定(综合误差矩阵)
│   │   └── gyro_bias_calibrator.py   # GyroBiasCalibrator: 八位置零偏标定
│   │
│   ├── analysis/                     # 噪声分析 → 对应Stage2模块6
│   │   ├── __init__.py
│   │   └── allan_variance_analyzer.py  # AllanVarianceAnalyzer
│   │
│   ├── assembly/                     # 结果组装 → 对应Stage2模块7
│   │   ├── __init__.py
│   │   ├── result_assembler.py       # ResultAssembler: 标定报告组装
│   │   └── report_generator.py       # 报告生成(包含质量控制标志评估)
│   │
│   ├── common/                       # 公共组件
│   │   ├── __init__.py
│   │   ├── types.py                  # 数据结构定义(@dataclass见Stage6)
│   │   ├── constants.py              # 物理常数(ω_e, g, π等)
│   │   ├── pose_tables.py            # 六位置/八位置理论姿态表
│   │   ├── assertions.py             # 物理边界断言函数(Stage6防断裂)
│   │   └── monitors.py               # shape/dtype/value_range监测槽
│   │
│   └── pipeline.py                   # 标定流水线编排器(组合所有模块)
│
├── visualization/                    # 可视化脚本
│   ├── __init__.py
│   ├── plot_calibration.py           # 标定结果可视化(残差分布、K矩阵热力图)
│   └── plot_allan.py                 # Allan方差双对数曲线绘制
│
├── scripts/                          # 自动化运行脚本
│   ├── run_calibration.py            # 一键标定脚本(Data→Report全流程)
│   ├── run_allan_analysis.py         # 单独运行Allan方差分析
│   └── verify_calibration.py         # 标定验证脚本(用验证数据对标定前后对比)
│
├── tests/                            # 单元测试与集成测试
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_gtimu_parser.py
│   ├── test_preprocessor.py
│   ├── test_integrator.py
│   ├── test_accel_calibrator.py
│   ├── test_gyro_rate_calibrator.py
│   ├── test_gyro_bias_calibrator.py
│   ├── test_allan_variance_analyzer.py
│   ├── test_result_assembler.py
│   ├── test_pipeline.py              # 端到端集成测试
│   └── fixtures/                     # 测试用数据文件
│       ├── sample_gtimu_6pose.log
│       ├── sample_gtimu_rate.log
│       └── sample_gtimu_static.log
│
├── docs/                             # 架构维护文档
│   ├── algorithm_design.md           # 本文档(Stage0-8完整设计)
│   ├── api_reference.md              # API接口参考文档(Stage6详细接口)
│   ├── formula_to_code_mapping.md    # 公式-代码追溯链(Stage5详细扩展)
│   └── reproduction_risks.md         # 复现风险与工程近似(Stage5.5详细)
│
├── calibration_results/              # 标定结果输出目录
│   └── .gitkeep
│
├── requirements.txt                  # Python依赖(numpy, scipy, matplotlib, pyyaml)
├── setup.py                          # 包安装配置
├── README.md                         # 项目自述
└── .gitignore
```

### 目录结构设计理由

| 目录 | 存在原因 | 如果不独立存在会怎样 |
|:-----|:---------|:---------------------|
| `configs/` | 将所有可变参数与代码分离，实现"不改代码调整算法" | 硬编码参数遍布各模块，修改实验配置需改多处代码 |
| `io/` | 数据解析与后续算法逻辑完全正交；便于替换不同数据格式 | Parser逻辑混杂在算法模块中，无法独立测试 |
| `preprocessing/` | 预处理是独立信号处理逻辑，可独立验证滤波/积分算法正确性 | AccelCalibrator混入积分逻辑，违反单一职责 |
| `calibration/` | 三个子模块有严格顺序依赖但数据路径不同，独立设计便于调试 | 所有的标定逻辑堆在一个文件中，代码量爆炸且难以复用 |
| `analysis/` | Allan方差是独立的后处理模块，可单独复用到其他噪声分析任务 | 与非相关的标定代码耦合，不利于单独使用 |
| `tests/fixtures/` | 测试需要固定已知结果的数据文件，不可依赖真实转台数据 | 测试无法离线运行，每次测试需连接转台 |
| `docs/` | 架构设计、公式-代码追溯链、复现风险等文档需随代码维护 | 半年后无人记得设计决策依据 |

---

## 附录：模块间数据依赖图谱

```
DataLoader
  │
  ├──▶ Preprocessor
  │      │
  │      ├──▶ AccelCalibrator ──────────────────────────▶ ResultAssembler
  │      │                                                     │
  │      ├──▶ GyroRateCalibrator ──▶ GyroBiasCalibrator ──▶    │
  │      │                                                     │
  │      └─────────────────────────────────────────────────────┘
  │
  └──▶ AllanVarianceAnalyzer (独立链路，不依赖标定结果)
```

> **依赖方向**：从左到右为数据依赖方向，不可逆。
> **关键依赖**：`GyroBiasCalibrator` → `GyroRateCalibrator`（需 **K_g** 作为输入）
> **并行可能性**：`AccelCalibrator` 与 `GyroRateCalibrator` 可并行执行（无数据依赖）
> **独立链路**：`AllanVarianceAnalyzer` 与确定性标定完全解耦，可单独独立运行