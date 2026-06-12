# 卫星导航单点定位算法设计文档

> **框架**：`skills/10_algorithm_design.md`（算法架构设计技能）  
> **设计依据**：`02_experiment_guide_analysis.md`（实验指导书深度分析）  
> **目标**：从"裸数据（日志文本）"到"接收机经纬高 + 精度评估"的全链路单点定位算法  
> **输入资产**：`exp2/data.txt`（NovAtel 接收机原始日志）、`satellite_navigation_solution.py`（参考实现）

---

## 📊 Stage 0：需求剖析与多维约束

### 项目目标审计

| 维度 | 内容 |
| :--- | :--- |
| **本质痛点** | 学生从 GNSS 接收机拿到的是"已经解算好的经纬高"黑盒结果，不理解从原始星历/伪距到定位坐标的底层数学物理链路。 |
| **最终交付** | 一个完整的单点定位解算程序，输入为接收机原始日志（ASCII 文本），输出为接收机 ECEF 坐标/经纬高 + 解算精度评估报告。 |
| **量化评价指标** | ① 卫星位置解算结果与 SATXYZ2A 固件输出偏差 ≤ 10 cm；② 自定位结果与 BESTPOSA 商用固件结果偏差 ≤ 20 m（3D）；③ 最小二乘迭代在 10 次内收敛；④ 支持任意 N≥4 颗 GPS 卫星的定位解算。 |

### I/O 边界梳理

| 方向 | 数据形态 | 说明 |
| :--- | :--- | :--- |
| **输入** | `data.txt`（纯文本，UTF-8） | NovAtel 接收机 ASCII 日志，包含 `#RANGEA`（伪距）、`#GPSEPHEMA`（星历）、`#SATXYZ2A`（固件卫星坐标）、`#BESTPOSA`（固件定位结果）四种日志条目 |
| **输入** | `eph` 矩阵（N×23，float64） | 从日志解析出的结构化星历参数，每行对应一颗 GPS 卫星 |
| **输入** | `pr` 矩阵（N×2，float64） | 从日志解析出的伪距值，第一列 PRN，第二列伪距（m） |
| **输出** | 接收机 ECEF 坐标 (X, Y, Z) | 地心地固直角坐标系，单位：m |
| **输出** | 接收机经纬高 (lon, lat, h) | WGS-84 大地坐标系，单位：deg, deg, m |
| **输出** | 精度评估报告 | 与 BESTPOSA 的偏差：ΔE/ΔN/ΔU + 2D RMS + 3D Error |
| **输出** | 可视化 | 3D 地球 + 卫星分布图 + 星空图 |

### 硬核边界约束

| 约束维度 | 具体限制 |
| :--- | :--- |
| **实时性** | 批处理模式（非实时），无严格延迟要求，但单次解算应 < 1s（Python） |
| **平台** | Python 3.9+ / NumPy / matplotlib（教学环境标准配置） |
| **数据量** | 单次采集 ~100-500 KB 原始日志，约 8-12 颗 GPS 卫星 |
| **精度要求** | 广播星历单点定位理论精度：水平 ≤ 10m，高程 ≤ 15m（95%） |
| **鲁棒性** | 需处理：星历过期（TOE 超 7200s）、伪距信号类型异常、矩阵奇异/病态、卫星数不足 4 颗 |

---

## 🗺️ Stage 1：总体架构与逻辑链设计

### 系统简介

本系统基于 **ICD-GPS-200 标准广播星历参数**，实现从 NovAtel 接收机 ASCII 日志到 WGS-84 经纬高的完整单点定位解算链路，核心算法包括 **9 步卫星位置解算（含摄动改正）→ 误差修正（钟差/地球自转）→ 加权最小二乘迭代**，并提供与商用固件结果的双轨对比验证。

### 总体流程图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        SPP Single Point Positioning                      │
│                          (Algorithm Pipeline)                            │
└──────────────────────────────────────────────────────────────────────────┘

data.txt (raw log) 
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  [Stage A] 数据提取层 (Data Extractor)              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ A1: 观测  │  │ A2: 星历  │  │ A3: 伪距  │         │
│  │ 时刻提取  │  │ 矩阵提取  │  │ 矩阵提取  │         │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
│       │             │             │                  │
│       ▼             ▼             ▼                  │
│  t_obs(浮点)   eph(N×23)    pr_list(N×1)            │
│  +week(int)    矩阵(浮点)    伪距(浮点)              │
└─────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  [Stage B] 卫星位置解算层 (SV Position Solver)      │
│                                                     │
│  eph[i] + t_obs ──► 9步标准流程 ──► (Xʲ,Yʲ,Zʲ)    │
│                                                     │
│  1. tk = t_obs - toe  (周内秒归化)                  │
│  2. n = √(GM/A³) + Δn  (平均角速度)                 │
│  3. Mk = M₀ + n·tk    (平近点角)                    │
│  4. Ek = Mk + e·sin(Ek)  (开普勒方程迭代)           │
│  5. νk = atan2(√(1-e²)sinEk, cosEk-e)  (真近点角)  │
│  6. Φk = νk + ω       (升交角距)                    │
│  7. δu/δr/δi = C·sin(2Φ)+C·cos(2Φ)  (摄动改正)     │
│  8. Ωk = Ω₀+(Ω̇-ωe)tk-ωe·toe  (升交点经度)          │
│  9. xp→ECEF 坐标旋转                                │
│                                                     │
│  输出: sat_pos(N×3)  ECEF卫星坐标矩阵               │
└─────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  [Stage C] 误差修正层 (Error Correction)            │
│                                                     │
│  ┌──────────────────┐  ┌──────────────────┐         │
│  │ C1: 卫星钟差修正  │  │ C2: 地球自转修正  │         │
│  │ dt = af0+af1·dt   │  │ X' = X+ωe·t·Y   │         │
│  │     +af2·dt²      │  │ Y' = Y-ωe·t·X   │         │
│  │     -2√(GM)       │  │ Z' = Z          │         │
│  │     ·e√A·sin(Ek)  │  │ t = ρ/c         │         │
│  │     /c²           │  └────────┬────────┘         │
│  └────────┬──────────┘           │                  │
│           │                       │                  │
│           ▼                       ▼                  │
│      rho_corr = ρ - c·dt      sat_pos_corr(N×3)     │
└─────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  [Stage D] 定位解算引擎 (Positioning Engine)        │
│                                                     │
│  X = [0,0,0,0]ᵀ  (初始: 地心 + 钟差=0)            │
│  for iter ≤ max_iter:                               │
│    for j in 1..N:                                   │
│      ρ̂ⱼ = ||SVⱼ - X[:3]|| + X[3]   (预测伪距)     │
│      Hⱼ = [-(X-SVⱼ)/ρ̂, 1]        (方向余弦+钟差)   │
│      yⱼ = ρ_corrⱼ - ρ̂ⱼ          (残差)             │
│    ΔX = (HᵀH)⁻¹Hᵀy                                │
│    X = X + ΔX                                       │
│    if ||ΔX[:3]|| < ε: break                         │
│                                                     │
│  输出: Xecef(X,Y,Z) + clock_bias                    │
└─────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  [Stage E] 坐标变换与精度评估层 (Post-processor)    │
│                                                     │
│  ┌────────────────────┐  ┌────────────────────┐     │
│  │ E1: ECEF→Geodetic  │  │ E2: 精度验证       │     │
│  │ 迭代: φ,h          │  │ vs BESTPOSA        │     │
│  │ lon = atan2(Y,X)   │  │ ΔE/ΔN/ΔU + RMS     │     │
│  │ 直到 Δφ<1e-10      │  │ 可视化输出         │     │
│  └────────┬───────────┘  └────────┬───────────┘     │
│           │                       │                  │
│           ▼                       ▼                  │
│  (lon,lat,h) + 精度报告 + 可视化图                  │
└─────────────────────────────────────────────────────┘
```

### 模块关系论证

| 关系 | 论证 |
| :--- | :--- |
| **A → B** | 数据提取层是卫星位置解算的前置条件。没有正确的星历矩阵，卫星位置解算就没有输入参数。 |
| **A → C** | 伪距矩阵是误差修正的输入源，观测时刻 t_obs 是钟差修正的时间基准。 |
| **B + C → D** | 卫星位置（B 的输出）提供几何观测方向，修正后的伪距（C 的输出）提供距离量测。两者在最小二乘迭代中联合使用，缺一不可。 |
| **D → E** | ECEF 坐标在导航应用中不可直接使用，必须转换为大地坐标系。精度验证需要商用固件结果作为基准。 |
| **解耦理由** | A→B→C→D→E 是**严格单向数据流**，各阶段之间只有明确的数据传递，没有反向依赖或循环依赖。这种结构使得每一步都可以独立调试验证。 |

---

## 🧩 Stage 2：高内聚低耦合模块划分

### 模块细化矩阵

#### Module A1：`ExtractObsTime`

| 属性 | 内容 |
| :--- | :--- |
| **模块名称** | `ExtractObsTime` |
| **模块职责** | 从 `#RANGEA` 日志头段中解析观测时刻 t_obs（周内秒）及 GPS 周计数。 |
| **明确 I/O** | 输入：原始日志文本（str）；输出：`(week: int, t_obs: float)` |
| **依赖树** | 无（独立模块） |
| **设计合理性** | 时刻提取是时间维度唯一入口，所有后续流程的时间参考均基于此值，必须独立为单一模块以明确时间基准。不与其他提取逻辑合并可减少"一个模块多职责"的耦合风险。 |

#### Module A2：`ExtractEphemerisMatrix`

| 属性 | 内容 |
| :--- | :--- |
| **模块名称** | `ExtractEphemerisMatrix` |
| **模块职责** | 从 `#GPSEPHEMA` 日志行中解析星历参数，按 ICD-GPS-200 标准筛选有效星历（PRN 1-37、周匹配、TOE 差值 ≤ 7200s），构造 N×23 星历矩阵。 |
| **明确 I/O** | 输入：`(raw_text, obs_week, t_obs)`；输出：`eph: np.ndarray(N×23) + valid_prns: List[int]` |
| **依赖树** | 依赖 `ExtractObsTime` 的输出（obs_week, t_obs）作为筛选条件 |
| **设计合理性** | 星历解析是数据提取中最复杂的部分（28 个字段的字段映射、多条星历的合并去重），独立为模块便于维护 ICD-GPS-200 标准映射关系和筛选逻辑的修改。 |

#### Module A3：`ExtractPseudorangeMatrix`

| 属性 | 内容 |
| :--- | :--- |
| **模块名称** | `ExtractPseudorangeMatrix` |
| **模块职责** | 从 `#RANGEA` 日志行中提取伪距测量值，按信号类型掩码（L1 C/A 码）筛选有效伪距，构造 N×2 伪距矩阵。 |
| **明确 I/O** | 输入：`(raw_text, obs_week, valid_prns)`；输出：`pr: np.ndarray(N×2)` |
| **依赖树** | 依赖 `ExtractEphemerisMatrix` 输出的 `valid_prns` 以取交集 |
| **设计合理性** | 伪距提取涉及复杂的数据块解析（每 11 字段为一个观测块，需跳过非 L1 信号），独立模块确保信号类型筛选规则（掩码 04/0b/0c）的准确实现。 |

#### Module B：`SatellitePositionSolver`

| 属性 | 内容 |
| :--- | :--- |
| **模块名称** | `SatellitePositionSolver` |
| **模块职责** | 基于广播星历参数，按 ICD-GPS-200 9 步标准算法计算每颗卫星的 ECEF 坐标。 |
| **明确 I/O** | 输入：`(eph_row: np.ndarray(23,), t_obs: float)`；输出：`(X, Y, Z, Ek, A, e): tuple` — 其中 Ek/A/e 供钟差修正使用 |
| **依赖树** | 依赖 A2 输出的 eph 矩阵 |
| **设计合理性** | 卫星位置解算是整个定位链路中最核心的数学模块。开普勒方程迭代求解、摄动改正、坐标旋转变换共涉及约 20 步浮点运算，独立封装便于单元测试和与 SATXYZ2A 固件输出逐星对比验证。 |

#### Module C1：`ClockCorrection`

| 属性 | 内容 |
| :--- | :--- |
| **模块名称** | `ClockCorrection` |
| **模块职责** | 计算卫星钟差修正量（含相对论效应），从伪距中扣除。 |
| **明确 I/O** | 输入：`(eph_row, t_obs, Ek, A, e)`；输出：`dt: float`（钟差修正量，单位 s） |
| **依赖树** | 依赖 B 输出的 Ek, A, e |
| **设计合理性** | 卫星钟差涉及多项式（af0+af1·Δt+af2·Δt²）和相对论修正两项公式，且是伪距修正的第一步。独立模块使得"开关测试"（含/不含钟差修正的对比实验）易于实现。 |

#### Module C2：`EarthRotationCorrection`

| 属性 | 内容 |
| :--- | :--- |
| **模块名称** | `EarthRotationCorrection` |
| **模块职责** | 修正信号传输期间地球自转引起的 Sagnac 效应，对卫星 ECEF 坐标施加旋转改正。 |
| **明确 I/O** | 输入：`(X, Y, Z, rho)`；输出：`(X', Y', Z')` |
| **依赖树** | 依赖 B 输出的卫星坐标 + A3 输出的伪距 rho |
| **设计合理性** | 地球自转校正是对卫星坐标的几何修正，需根据伪距计算出信号传播延时（~67ms，产生最大 ~30m 赤道处偏差）。独立封装便于评估该修正项的实际影响量级。 |

#### Module D：`LeastSquaresSolver`

| 属性 | 内容 |
| :--- | :--- |
| **模块名称** | `LeastSquaresSolver` |
| **模块职责** | 通过迭代最小二乘估计接收机位置 (X,Y,Z) 和钟差 c·δt。初始值为地心 (0,0,0)，迭代直到 ‖ΔX[:3]‖ < tol（默认 1e-4 m）或达到最大迭代次数。 |
| **明确 I/O** | 输入：`(sat_pos: np.ndarray(N×3), corrected_pseudorange: np.ndarray(N,), max_iter=20, tol=1e-4)`；输出：`state: np.ndarray(4,)` — [X, Y, Z, c·δt] |
| **依赖树** | 依赖 B + C1 + C2 的全部输出 |
| **设计合理性** | 最小二乘迭代是整个定位的"发动机"，该模块实现了从观测方程到状态估计的核心数学变换。独立设计使得（加权）最小二乘、RAIM 异常检测等算法变体可以方便地替换。 |

#### Module E1：`EcefToGeodetic`

| 属性 | 内容 |
| :--- | :--- |
| **模块名称** | `EcefToGeodetic` |
| **模块职责** | 将 ECEF 笛卡尔坐标转换为 WGS-84 大地坐标系下的经度、纬度、高度。 |
| **明确 I/O** | 输入：`(X, Y, Z)`；输出：`(lon_deg, lat_deg, h_m)` |
| **依赖树** | 依赖 D 输出的接收机 ECEF 坐标 |
| **设计合理性** | ECEF→大地坐标变换是独立于定位解算的数学问题，使用迭代法求解，与定位算法无关。独立模块便于替换为直接法（如 Bowring 算法）进行比较。 |

#### Module E2：`AccuracyValidator`

| 属性 | 内容 |
| :--- | :--- |
| **模块名称** | `AccuracyValidator` |
| **模块职责** | 将自解算结果与 BESTPOSA 参考结果对比，计算 ΔE/ΔN/ΔU、2D RMS、3D Error，输出精度评估报告。 |
| **明确 I/O** | 输入：`(our_lon, our_lat, our_h, ref_lon, ref_lat, ref_h)`；输出：`(dlon, dlat, dh, r2d, r3d)` — 打印格式化报告 |
| **依赖树** | 依赖 E1（自解算结果）+ BESTPOSA 日志解析结果 |
| **设计合理性** | 精度验证是实验对比的"闭环"关键。不独立成模块，学生就难以在编码过程中迭代式地评估和改进自己的实现。 |

---

## 🔄 Stage 3：数据生命周期与维度流设计

### 生命周期追踪

```
┌─────────────┐    ┌───────────────┐    ┌─────────────────┐    ┌──────────────┐    ┌───────────────┐
│  原始数据    │    │  结构化数据    │    │  卫星位置 +     │    │  状态估计     │    │  最终结果      │
│  (文本)     │───►│  (矩阵/向量)  │───►│  修正伪距      │───►│  (ECEF矢量)   │───►│  (经纬高+偏差)  │
└─────────────┘    └───────────────┘    └─────────────────┘    └──────────────┘    └───────────────┘
  data.txt          eph(N×23)           sat_pos(N×3)            X(4,) =            lon_s(标量)
  (ASCII,          pr(N×2)             rho_corr(N,)            [X,Y,Z,cδt]        lat_s(标量)
   100-500KB)      t_obs(标量)                                                      h_s(标量)
                    week(整型)                                                      ΔE/ΔN/ΔU(标量)
                                                                                    2D/3D误差(标量)
```

### 流转指标约束

| 处理阶段 | 数据格式 | 维度/形状 | 核心物理含义 | 变换说明 |
| :--- | :--- | :--- | :--- | :--- |
| **日志解析** | str | ~100-500 KB | 接收机原始输出的 ASCII 日志文本 | 语法解析：文本 → 结构化字段 |
| **观测时刻提取** | float | 标量 | 数据采集的 GPS 周内秒时刻 [s] | 所有时间相关计算的 T₀ 基准 |
| **星历矩阵提取** | np.float64 | N×23 | 每行: PRN, week, A, Δn, M₀, e, ω, C_uc, C_us, C_rc, C_rs, C_ic, C_is, i₀, di/dt, Ω₀, Ω̇, t_oe, t_oc, T_GD, a_f0, a_f1, a_f2 | ICD-GPS-200 参数映射 |
| **伪距矩阵提取** | np.float64 | N×2 | 每行: PRN, raw_pseudorange [m] | 原始测量值，含所有误差 |
| **卫星位置解算** | np.float64 | N×3 | 每颗卫星在 ECEF 下的 (X,Y,Z) 坐标 [m] | 9 步非线性映射：参数→坐标 |
| **卫星钟差修正** | np.float64 | N, | 每颗卫星的钟差修正量 [s] | 应用于伪距：ρ_c = ρ - c·Δt_sat |
| **地球自转修正** | np.float64 | N×3 | 修正 Sagnac 效应后的卫星坐标 [m] | X' = X + ω_e·τ·Y, 类似 |
| **最小二乘迭代** | np.float64 | 4, | [X_rx, Y_rx, Z_rx, c·δt_rx] [m] + [m] | 牛顿迭代：H(ΔX) = y |
| **ECEF→大地坐标** | float | 3 个标量 | 经度[°], 纬度[°], 高度[m] | 迭代法：φ←atan(Z/p/(1-e²N/(N+h))) |
| **精度评估** | float | 5 个标量 | ΔE[m], ΔN[m], ΔU[m], 2DRMS[m], 3DError[m] | 投影到东北天(ENU)坐标系 |

---

## 📐 Stage 4：高层算法逻辑与伪代码

### 主循环流程

```
Step 1:  读取原始日志文件 data.txt 到内存字符串
Step 2:  字符串正则扫描 #RANGEA 头段 → 解析出 GPS_week, t_obs
Step 3:  字符串扫描所有 #GPSEPHEMA 行 → 按 ICD-GPS-200 筛选标准
         (PRN∈[1,37], week匹配, |TOE-t_obs|≤7200s) → eph_matrix(N×23)
Step 4:  字符串扫描所有 #RANGEA 行 → 解析伪距观测块（每块 10 字段），
         保留信号掩码属 L1 C/A 码的伪距 → pr_matrix(N×2)
Step 5:  PRN 交互验证：取 eph 侧 PRN 与 pr 侧 PRN 的交集作为有效卫星集。
         按 PRN 升序重新排列 eph_matrix 和 pr_matrix，行序对齐。
Step 6:  for i = 1 to N:
            调用卫星位置解算(eph[i], t_obs) → (X_s, Y_s, Z_s, Ek, A, e)
Step 7:  for i = 1 to N:
            调用钟差修正(eph[i], t_obs, Ek, A, e) → dt_s
            rho_corr[i] = pr[i] - c * dt_s
            调用地球自转修正(X_s, Y_s, Z_s, rho_corr[i]) → (X'_s, Y'_s, Z'_s)
Step 8:  调用最小二乘迭代(sat_pos_corr(N×3), rho_corr(N,)) → X_rx(4,)
Step 9:  调用 ECEF→大地坐标(X_rx[0], X_rx[1], X_rx[2]) → (lon, lat, h)
Step 10: 调用精度验证(lon, lat, h, BESTPOSA_lon, BESTPOSA_lat, BESTPOSA_h)
         → 输出偏差报告 + 可视化
```

### 泛型伪代码

```
ALGORITHM: GPS_SinglePointPositioning
INPUT:    raw_log (string) — NovAtel ASCII log file content
OUTPUT:   (lon_deg, lat_deg, h_m) — receiver position in WGS-84
          accuracy_report — comparison vs BESTPOSA

// ====== PHASE 1: DATA EXTRACTION ======

FUNCTION ExtractObsTime(raw_log):
    FOR EACH line IN raw_log:
        IF line CONTAINS "#RANGEA" AND ";" IN line:
            header ← extract_before_semicolon(line)
            fields ← SPLIT(header, ",")
            week ← PARSE_INT(fields[5])
            t_obs ← PARSE_FLOAT(fields[6])  // GPS time-of-week in seconds
            RETURN (week, t_obs)
    RAISE ERROR "RANGEA header not found"

FUNCTION ExtractEphemerisMatrix(raw_log, obs_week, t_obs):
    eph_dict ← EMPTY_MAP(PRN → (TOE, data_array))
    FOR EACH line IN raw_log:
        IF line CONTAINS "#GPSEPHEMA" AND ";" IN line:
            header, data ← SPLIT(line, ";")
            hdr_fields ← SPLIT(header, ",") 
            IF PARSE_INT(hdr_fields[5]) ≠ obs_week: CONTINUE
            data_fields ← SPLIT(data, ",") 
            prn ← PARSE_INT(data_fields[0])
            IF prn < 1 OR prn > 37: CONTINUE
            toe ← PARSE_FLOAT(data_fields[7])  // TOE field
            IF ABS(toe - t_obs) > 7200: CONTINUE  // ephemeris too old
            // Keep latest ephemeris per PRN (by TOE)
            IF prn NOT IN eph_dict OR toe > eph_dict[prn].toe:
                eph_dict[prn] ← (toe, data_fields)
    
    valid_prns ← SORTED_KEYS(eph_dict)
    N ← LENGTH(valid_prns)
    eph ← ALLOCATE_MATRIX(N, 23)
    FOR i = 1 TO N:
        prn ← valid_prns[i]
        d ← eph_dict[prn].data
        eph[i, 0] ← prn
        eph[i, 1] ← obs_week
        // Map ICD-GPS-200 parameters to columns 2-22
        COLUMN_MAP = {2:8, 3:9, 4:10, 5:11, 6:12, 7:13, 8:14, 9:15,
                      10:16, 11:17, 12:18, 13:19, 14:20, 15:21, 16:22,
                      17:7, 18:24, 19:25, 20:26, 21:27, 22:28}
        FOR EACH (col, idx) IN COLUMN_MAP:
            eph[i, col] ← PARSE_FLOAT(d[idx])
    RETURN (eph, valid_prns)

FUNCTION ExtractPseudorangeMatrix(raw_log, obs_week, valid_prns):
    prn_set ← SET(valid_prns)
    pr_data ← EMPTY_MAP(PRN → pseudorange)
    FOR EACH line IN raw_log:
        IF line CONTAINS "#RANGEA" AND ";" IN line:
            header, data ← SPLIT(line, ";")
            hdr_fields ← SPLIT(header, ",")
            IF PARSE_INT(hdr_fields[5]) ≠ obs_week: CONTINUE
            obs_blocks ← SPLIT(data, ",") 
            // Each 10 fields = one satellite observation block
            FOR j = 0 TO LENGTH(obs_blocks)-10 STEP 10:
                block ← obs_blocks[j:j+10]
                prn ← PARSE_INT(block[0])
                IF prn NOT IN prn_set: CONTINUE
                sig_mask ← PARSE_HEX_INT(block[9]) & 0x0F
                IF sig_mask NOT IN {0x04, 0x08, 0x0C}: CONTINUE  // L1 only
                pseudorange ← PARSE_FLOAT(block[2])
                IF prn NOT IN pr_data:
                    pr_data[prn] ← pseudorange  // keep first L1
    
    N ← LENGTH(valid_prns)
    pr ← ALLOCATE_MATRIX(N, 2)
    FOR i = 1 TO N:
        prn ← valid_prns[i]
        pr[i, 0] ← prn
        IF prn IN pr_data:
            pr[i, 1] ← pr_data[prn]
        ELSE:
            pr[i, 1] ← 0.0  // missing pseudorange, flag
            RAISE WARNING "No L1 pseudorange for PRN " + prn
    RETURN pr

// ====== PHASE 2: SATELLITE POSITION COMPUTATION ======

FUNCTION ComputeSatellitePosition(eph_row, t_obs):
    // Unpack ephemeris parameters
    A       ← eph_row[2]    // sqrt(A) in ICD, but stored as A directly
    dn      ← eph_row[3]    // Δn
    M0      ← eph_row[4]    // M₀
    e       ← eph_row[5]    // eccentricity
    om      ← eph_row[6]    // ω (argument of perigee)
    Cuc     ← eph_row[7]    // C_uc
    Cus     ← eph_row[8]    // C_us
    Crc     ← eph_row[9]    // C_rc
    Crs     ← eph_row[10]   // C_rs
    Cic     ← eph_row[11]   // C_ic
    Cis     ← eph_row[12]   // C_is
    i0      ← eph_row[13]   // i₀
    didt    ← eph_row[14]   // di/dt
    O0      ← eph_row[15]   // Ω₀
    Odot    ← eph_row[16]   // Ω̇
    toe     ← eph_row[17]   // t_oe
    
    // Step 1: Time from ephemeris reference epoch
    tk ← t_obs - toe
    // Handle week crossovers
    IF tk > 302400:  tk ← tk - 604800
    IF tk < -302400: tk ← tk + 604800
    
    // Step 2: Mean angular velocity
    n ← SQRT(GM / A³) + dn        // GM = 3.986005e14
    
    // Step 3: Mean anomaly
    Mk ← M0 + n * tk
    
    // Step 4: Eccentric anomaly (Kepler equation, Newton iteration)
    Ek ← Mk
    FOR iter = 1 TO 100:
        En ← Mk + e * SIN(Ek)
        IF ABS(En - Ek) < 1e-12:   // convergence threshold
            Ek ← En
            BREAK
        Ek ← En
    
    // Step 5: True anomaly
    νk ← ATAN2(SQRT(1 - e²) * SIN(Ek), COS(Ek) - e)
    
    // Step 6: Argument of latitude
    Φk ← νk + om
    
    // Step 7: Harmonic perturbations
    δu ← Cus * SIN(2Φk) + Cuc * COS(2Φk)
    δr ← Crs * SIN(2Φk) + Crc * COS(2Φk)
    δi ← Cis * SIN(2Φk) + Cic * COS(2Φk)
    
    // Corrected argument, radius, inclination
    uk ← Φk + δu
    rk ← A * (1 - e * COS(Ek)) + δr
    ik ← i0 + δi + didt * tk
    
    // Step 8: Longitude of ascending node
    Ωk ← O0 + (Odot - ωe) * tk - ωe * toe   // ωe = 7.2921151467e-5
    
    // Step 9: Position in orbital frame
    xp ← rk * COS(uk)
    yp ← rk * SIN(uk)
    
    // Transform to ECEF
    cO ← COS(Ωk);  sO ← SIN(Ωk)
    ci ← COS(ik);  si ← SIN(ik)
    
    Xs ← xp * cO - yp * ci * sO
    Ys ← xp * sO + yp * ci * cO
    Zs ← yp * si
    
    RETURN (Xs, Ys, Zs, Ek, A, e)

// ====== PHASE 3: ERROR CORRECTIONS ======

FUNCTION ClockCorrection(eph_row, t_obs, Ek, A, e):
    toc ← eph_row[18]       // t_oc
    af0 ← eph_row[20]       // a_f0
    af1 ← eph_row[21]       // a_f1
    af2 ← eph_row[22]       // a_f2
    dt ← t_obs - toc
    // Polynomial clock correction
    dts ← af0 + af1 * dt + af2 * dt²
    // Relativistic correction
    dtr ← -2 * SQRT(GM) * e * SQRT(A) * SIN(Ek) / C²
    RETURN dts + dtr        // total satellite clock bias [s]

FUNCTION EarthRotationCorrection(X, Y, Z, rho):
    τ ← rho / C              // signal travel time [s]
    X' ← X + ωe * τ * Y     // ωe = 7.2921151467e-5 rad/s
    Y' ← Y - ωe * τ * X
    Z' ← Z
    RETURN (X', Y', Z')

// ====== PHASE 4: LEAST SQUARES POSITIONING ======

FUNCTION LeastSquaresSolution(sat_pos_Nx3, rho_corr_N, max_iter=20, tol=1e-4):
    // Initial state: Earth center + zero clock bias
    X ← [0.0, 0.0, 0.0, 0.0]ᵀ
    
    FOR iter = 1 TO max_iter:
        // Compute predicted pseudorange and observation matrix
        FOR j = 1 TO N:
            dx ← sat_pos[j,0] - X[0]
            dy ← sat_pos[j,1] - X[1]
            dz ← sat_pos[j,2] - X[2]
            rho_hat ← SQRT(dx² + dy² + dz²)
            
            // Line-of-sight unit vector (negated) + clock column
            H[j, :] ← [-dx/rho_hat, -dy/rho_hat, -dz/rho_hat, 1.0]
            
            // Measurement residual
            y[j] ← rho_corr[j] - (rho_hat + X[3])
        
        // Least squares solution: ΔX = (HᵀH)⁻¹Hᵀy
        ΔX ← SOLVE_NORMAL_EQUATION(H, y)
        X ← X + ΔX
        
        // Convergence check (position only, exclude clock bias)
        IF NORM(ΔX[0:3]) < tol:
            OUTPUT "Converged after " + iter + " iterations"
            BREAK
    
    RETURN X    // [X_rx, Y_rx, Z_rx, cδt]

// ====== PHASE 5: COORDINATE TRANSFORM & VALIDATION ======

FUNCTION EcefToGeodetic(X, Y, Z):
    lon ← ATAN2(Y, X)
    p ← SQRT(X² + Y²)
    
    // Handle near-polar case
    IF p < 1e-6:
        lat ← SIGN(Z) * 90°
        h ← ABS(Z) - A_WGS84 * SQRT(1 - e²)
        RETURN (lon, lat, h)
    
    // Iterative latitude and height computation
    phi ← ATAN(Z / p / (1 - e²))      // initial guess
    FOR iter = 1 TO 100:
        N ← A_WGS84 / SQRT(1 - e² * SIN²(phi))
        h ← p / COS(phi) - N
        phi_new ← ATAN(Z / p / (1 - e² * N/(N+h)))
        IF ABS(phi_new - phi) < 1e-10:
            phi ← phi_new
            BREAK
        phi ← phi_new
    
    N ← A_WGS84 / SQRT(1 - e² * SIN²(phi))
    h ← p / COS(phi) - N
    
    RETURN (lon * 180/π, phi * 180/π, h)

FUNCTION AccuracyValidator(our_lon, our_lat, our_h, ref_lon, ref_lat, ref_h):
    dLon ← our_lon - ref_lon    [deg]
    dLat ← our_lat - ref_lat    [deg]
    dH   ← our_h - ref_h        [m]
    
    // Convert angular differences to meters (local ENU)
    RN ← A_WGS84 / SQRT(1 - e² * SIN²(ref_lat * π/180))
    dN ← dLat * (π/180) * RN                    [m]
    dE ← dLon * (π/180) * RN * COS(ref_lat * π/180) [m]
    
    // 2D horizontal RMS
    r2 ← SQRT(dN² + dE²)
    // 3D error
    r3 ← SQRT(dE² + dN² + dH²)
    
    OUTPUT "========== ACCURACY REPORT =========="
    OUTPUT "Solved:    (lon, lat, h) = (our_lon, our_lat, our_h)"
    OUTPUT "Reference: (lon, lat, h) = (ref_lon, ref_lat, ref_h)"
    OUTPUT "dE = dE m, dN = dN m, dU = dH m"
    OUTPUT "2D RMS  = r2 m"
    OUTPUT "3D Error = r3 m"
    OUTPUT "====================================="
    
    RETURN (dE, dN, dH, r2, r3)

// ====== MAIN ======

PROCEDURE MAIN():
    raw_log ← READ_FILE("data.txt")
    
    (week, t_obs) ← ExtractObsTime(raw_log)
    (eph, vprn) ← ExtractEphemerisMatrix(raw_log, week, t_obs)
    pr ← ExtractPseudorangeMatrix(raw_log, week, vprn)
    
    N ← ROWS(pr)
    ASSERT ROWS(eph) == N, "Satellite count mismatch"
    
    // Satellite position computation
    sat_pos ← ALLOCATE(N, 3)
    Ek_list ← ALLOCATE(N)
    e_list  ← ALLOCATE(N)
    A_list  ← ALLOCATE(N)
    FOR i = 1 TO N:
        (Xs, Ys, Zs, Ek, A, e) ← ComputeSatellitePosition(eph[i], t_obs)
        sat_pos[i] ← (Xs, Ys, Zs)
        Ek_list[i] ← Ek;  e_list[i] ← e;  A_list[i] ← A
    
    // Error corrections
    rho_corr ← ALLOCATE(N)
    sat_pos_corr ← ALLOCATE(N, 3)
    FOR i = 1 TO N:
        dt_s ← ClockCorrection(eph[i], t_obs, Ek_list[i], A_list[i], e_list[i])
        rho_corr[i] ← pr[i, 1] - C * dt_s
        (Xc, Yc, Zc) ← EarthRotationCorrection(sat_pos[i], rho_corr[i])
        sat_pos_corr[i] ← (Xc, Yc, Zc)
    
    // Positioning
    X_rx ← LeastSquaresSolution(sat_pos_corr, rho_corr)
    
    // Coordinate transform
    (lon, lat, h) ← EcefToGeodetic(X_rx[0], X_rx[1], X_rx[2])
    
    // Reference from BESTPOSA
    (ref_lon, ref_lat, ref_h) ← ParseBESTPOSA(raw_log)
    
    // Accuracy validation
    AccuracyValidator(lon, lat, h, ref_lon, ref_lat, ref_h)
    
    // Visualization
    PlotEarthAndSatellites(all_sats, (lon, lat, h))
    
    OUTPUT "DONE"
```

---

## 🎯 Stage 5：文献到工程双向映射

### 核心公式 - 工程映射表

| 论文/ICD 公式 | 工程实现函数 | 说明 |
| :--- | :--- | :--- |
| $\rho^{(j)} = \sqrt{(X^{(j)}-X)^2 + (Y^{(j)}-Y)^2 + (Z^{(j)}-Z)^2} + c\cdot\delta t + \varepsilon$ | `LeastSquaresSolution()` → 预测伪距 $\hat{\rho}$ | 观测方程是定位算法的基础数学模型，残差 $y = \rho_{corr} - \hat{\rho}$ 驱动最小二乘迭代 |
| $E = M + e\sin E$ | `ComputeSatellitePosition()` Step 4 | 开普勒方程迭代求解（Newton-Raphson，收敛阈值 1e-12 rad） |
| $\nu = \arctan2(\sqrt{1-e^2}\sin E, \cos E - e)$ | `ComputeSatellitePosition()` Step 5 | 真近点角计算，使用 arctan2 确保象限正确 |
| $\delta_u = C_{us}\sin(2\Phi) + C_{uc}\cos(2\Phi)$ | `ComputeSatellitePosition()` Step 7 | 升交角距摄动改正，$\delta_r$ 和 $\delta_i$ 类似 |
| $\Omega_k = \Omega_0 + (\dot{\Omega} - \omega_e)t_k - \omega_e t_{oe}$ | `ComputeSatellitePosition()` Step 8 | 升交点经度需考虑地球自转速率 $\omega_e$ |
| $\Delta t_{sv} = a_{f0} + a_{f1}\Delta t + a_{f2}\Delta t^2 + \Delta t_r$ | `ClockCorrection()` | 卫星钟差 = 多项式项 + 相对论修正项 |
| $\boldsymbol{X}' = \boldsymbol{X} + \frac{\omega_e\rho}{c} \begin{bmatrix}Y\\-X\\0\end{bmatrix}$ | `EarthRotationCorrection()` | Sagnac 效应修正（~30m max at equator） |
| $\Delta\boldsymbol{X} = (\boldsymbol{H}^T\boldsymbol{H})^{-1}\boldsymbol{H}^T\boldsymbol{y}$ | `LeastSquaresSolution()` → 核心求解 | 最小二乘正规方程解 |
| $\begin{cases}\phi = \arctan\left(\frac{Z}{p(1-e^2N/(N+h))}\right)\\ h = \frac{p}{\cos\phi} - N\end{cases}$ | `EcefToGeodetic()` | WGS-84 大地坐标迭代变换 |

### 算法流程 - 工程映射

| ICD/论文算法步骤 | 工程模块 | 说明 |
| :--- | :--- | :--- |
| 卫星位置解算 9 步标准流程（ICD-GPS-200 20.3.3.4.3） | `ComputeSatellitePosition()` | 完整实现从 t_k → M_k → E_k → ν_k → Φ_k → δu/δr/δi → u_k/r_k/i_k → Ω_k → ECEF |
| 最小二乘迭代定位（Paper Sec 3.4） | `LeastSquaresSolution()` | 第 2-5 步循环：初始化为 (0,0,0,0)，逐次逼近 |
| 误差修正模型（Paper Sec 3.3） | `ClockCorrection()` + `EarthRotationCorrection()` | 钟差 + 相对论 + 地球自转（电离层/对流层修正为可扩展预留接口） |

### Reproduction Risk Analysis

| 风险类别 | 已知信息 | 缺失信息 | 所需假设 | 复现风险 | 工程近似策略 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **星历字段映射** | 指导书给出了 23 列的字段-数据段位置映射表 | 某些字段索引可能存在 ±1 偏移（依赖于日志格式版本） | 假设 NovAtel 日志数据段的字段编号从 1 开始 | 🟡 中等 | 与 SATXYZ2A 固件输出的卫星坐标逐颗对比以验证映射正确性 |
| **开普勒方程迭代** | 使用牛顿迭代法求解 $E = M + e\sin E$ | 未指定收敛阈值的具体数值 | 默认 1e-12 或 1e-15（视浮点精度而定） | 🟢 低 | 迭代在 4-8 次内始终收敛，阈值宽松到 1e-10 也无影响 |
| **信号类型掩码** | 掩码末尾两位为 04/0b/0c 且标识 L1 频点 | 未说明掩码的位定义细节（8 位 vs 16 位） | 假设 0x0F 掩码取低 4 位 | 🟡 中等 | 实际 RANGEA 日志解析中需仔细核对 `blk[9]` 的十六进制解析 |
| **电离层/对流层修正** | 指导书列出概念但未给出模型参数 | α_n, β_n（Klobuchar 模型的电离层参数）未从星历中提取；Saastamoinen 模型未实现 | 为了教学简化，可以忽略大气修正 | 🔴 **高** | 本设计中暂不实现大气修正，预期偏差增大但代码复杂度显著降低 |
| **最小二乘收敛判据** | 使用 $\|\Delta X\| < \varepsilon$ | 未指定 $\varepsilon$ 值 | 默认 1e-4 m（位置分量） | 🟢 低 | 1e-4 在双精度浮点下合理，且与实际定位精度（数米）充分解耦 |

---

## 🔌 Stage 6：接口规范与异常防御

### 契约式接口定义

#### 核心数据结构

```python
class EphemerisRow(NamedTuple):
    """单颗卫星广播星历，23 列"""
    prn: int           # 卫星 PRN 编号
    week: int          # GPS 周计数
    A: float           # 轨道半长轴 [m]
    delta_n: float     # 平均角速度修正 [rad/s]
    M0: float          # 参考时刻平近点角 [rad]
    e: float           # 轨道偏心率
    omega: float       # 近地点角距 [rad]
    Cuc: float         # 纬度幅角余弦调和修正振幅 [rad]
    Cus: float         # 纬度幅角正弦调和修正振幅 [rad]
    Crc: float         # 轨道半径余弦调和修正振幅 [m]
    Crs: float         # 轨道半径正弦调和修正振幅 [m]
    Cic: float         # 轨道倾角余弦调和修正振幅 [rad]
    Cis: float         # 轨道倾角正弦调和修正振幅 [rad]
    i0: float          # 参考时刻轨道倾角 [rad]
    idot: float        # 轨道倾角变化率 [rad/s]
    Omega0: float      # 参考时刻升交点赤经 [rad]
    Omegadot: float    # 升交点赤经变化率 [rad/s]
    toe: float         # 星历参考时刻 [s] (GPS week seconds)
    toc: float         # 星钟参考时刻 [s]
    Tgd: float         # L1-L2 群延迟差 [s]
    af0: float         # 卫星钟差多项式系数 0 [s]
    af1: float         # 卫星钟差多项式系数 1 [s/s]
    af2: float         # 卫星钟差多项式系数 2 [s/s²]
```

```python
class ReceiverState(NamedTuple):
    """接收机状态估计向量"""
    X: float           # ECEF X [m]
    Y: float           # ECEF Y [m]
    Z: float           # ECEF Z [m]
    clock_bias: float  # 接收机钟差 c·δt [m]
```

```python
class GeodeticPosition(NamedTuple):
    """WGS-84 大地坐标系位置"""
    lon_deg: float     # 经度 [°]
    lat_deg: float     # 纬度 [°]
    height_m: float    # 椭球高度 [m]
```

#### 主要函数签名

```python
def extract_obs_time(data_text: str) -> Tuple[int, float]:
    """解析观测时刻
    Args:
        data_text: 原始日志文本
    Returns:
        (week, t_obs) — GPS 周计数和本周秒数
    Raises:
        ValueError: 未找到 RANGEA 日志头
    """

def extract_ephemeris_matrix(data_text: str, obs_week: int, t_obs: float) -> Tuple[np.ndarray, List[int]]:
    """提取星历矩阵
    Returns:
        (eph_matrix N×23, valid_prns)
    Raises:
        ValueError: 没有有效 GPS 星历
    """

def extract_pseudorange_matrix(data_text: str, obs_week: int, valid_prns: List[int]) -> np.ndarray:
    """提取伪距矩阵
    Returns:
        pr_matrix N×2 (PRN, pseudorange_m)
    """

def compute_satellite_position(eph_row: np.ndarray, t_obs: float) -> Tuple[float, float, float, float, float, float]:
    """计算卫星 ECEF 位置
    Returns:
        (X, Y, Z, Ek, A, e) — ECEF 坐标 + 开普勒参数（供钟差修正用）
    """

def apply_clock_correction(eph_row: np.ndarray, t_obs: float, Ek: float, A: float, e: float) -> float:
    """卫星钟差修正
    Returns:
        dts: 卫星钟差 [s]（包含相对论效应）
    """

def apply_earth_rotation_correction(X: float, Y: float, Z: float, pseudorange: float) -> Tuple[float, float, float]:
    """地球自转 Sagnac 修正
    Returns:
        (X', Y', Z') 修正后卫星坐标
    """

def least_squares_solution(sat_pos: np.ndarray, rho_corr: np.ndarray, 
                          max_iter: int = 20, tol: float = 1e-4) -> np.ndarray:
    """最小二乘迭代定位
    Args:
        sat_pos: N×3 卫星 ECEF 坐标
        rho_corr: N 个修正后伪距
    Returns:
        X: [X_rx, Y_rx, Z_rx, cδt]
    Raises:
        np.linalg.LinalgError: 观测矩阵 H 奇异（卫星几何分布病态）
    """

def ecef_to_geodetic(X: float, Y: float, Z: float) -> Tuple[float, float, float]:
    """ECEF → WGS-84 大地坐标转换
    Returns:
        (lon_deg, lat_deg, h_m)
    """

def validate_accuracy(our_lon: float, our_lat: float, our_h: float,
                     ref_lon: float, ref_lat: float, ref_h: float) -> Tuple[float, float, float, float, float]:
    """精度验证
    Returns:
        (dE, dN, dU, r2d, r3d) — 东北天偏差 + 2D/3D误差
    """
```

### 防断裂异常处理

| 异常场景 | 检测条件 | 降级/恢复策略 |
| :--- | :--- | :--- |
| **RANGEA 日志头缺失** | `extract_obs_time` 中无匹配行 | 抛出 `ValueError("RANGEA header not found")`，禁止继续 |
| **星历为空** | `extract_ephemeris_matrix` 中 N == 0 | 抛出 `ValueError("No valid GPS ephemeris")`，需检查日志是否包含 GPSEPHEMA |
| **卫星数 < 4** | 有效 PRN 交集数量 < 4 | 抛出 `ValueError("Insufficient satellites: N < 4")`，无法定位 |
| **星历/伪距数量不匹配** | eph.shape[0] != pr.shape[0] | `AssertionError` 终止，调试阶段快速暴露 bug |
| **伪距缺失** | 某个有效 PRN 没有对应 L1 伪距 | 打印 `WARNING: PRN {prn} no L1 pseudorange`，该行置 0，仍继续定位（降级为 N-1 星） |
| **观测矩阵奇异** | `np.linalg.inv(H.T@H)` 抛出 `LinAlgError` | 回退到 `np.linalg.pinv(H.T@H)`（伪逆），打印警告但继续迭代 |
| **经纬高迭代不收敛** | `ecef_to_geodetic` 中 100 次迭代未收敛（Δφ > 1e-10） | 以最后一次迭代结果作为近似值，打印警告 |
| **BESTPOSA 缺失** | 日志中无 `SOL_COMPUTED` 的 BESTPOSA 行 | 跳过精度对比，将自解算结果同时作为最终结果和参考值 |

---

## ⚙️ Stage 7：超参数与系统配置设计

### 参数矩阵分类

#### 物理常数参数

| 参数 | 符号 | 值 | 物理作用 | 边界影响 |
| :--- | :--- | :--- | :--- | :--- |
| 光速 | C | 299792458 m/s | 钟差←→距离转换；信号传播延时计算 | 常数值是定死的，不会改动 |
| 地球引力常数 | GM | 3.986005e14 m³/s² | 卫星平均角速度计算 | 影响卫星位置精度 ~1m 量级 |
| 地球自转角速度 | ωe | 7.2921151467e-5 rad/s | 升交点赤经变化 + Sagnac 修正 | 赤道处最大影响 ~30m |
| WGS-84 长半轴 | A_WGS84 | 6378137.0 m | 大地坐标转换基准 | 影响高度计算 ~1m |
| WGS-84 扁率 | F_WGS84 | 1/298.257223563 | 大地坐标转换基准 | 影响纬度/高度精度 |

#### 算法超参数

| 参数 | 默认值 | 物理作用 | 范围 | 性能影响 |
| :--- | :--- | :--- | :--- | :--- |
| 开普勒方程收敛阈值 | 1e-12 rad | 控制偏近点角 E 的迭代精度 | 1e-8 ~ 1e-15 | 过高=<2m 卫星位置误差；过低=浪费迭代 |
| 最小二乘最大迭代次数 | 20 | 防止死循环 | 5 ~ 50 | 过低=未收敛；过高=无意义（4-8 次已足够） |
| 最小二乘收敛阈值 | 1e-4 m | 位置增量 ‖ΔX[:3]‖ 的收敛判据 | 1e-2 ~ 1e-6 | 过松=提前终止影响精度；过紧=迭代次数增加 |
| 星历最大时差 | 7200 s | |TOE - t_obs| 的最大允许值 | ≤3600（保守）~14400（宽松） | 过小=卫星数量不足；过大=使用过期星历 |
| 有效卫星最小数量 | 4 | 定位所需的最小卫星数 | 4 ~ 6 | 4 星是数学可解的最小要求，<4 则无法定位 |

#### 数据相关参数

| 参数 | 说明 | 典型值 |
| :--- | :--- | :--- |
| 日志文件路径 | 输入原始日志的文件路径 | `exp2/data.txt` |
| GPS 星座卫星 PRN 范围 | GPS 卫星 PRN 编号有效范围 | 1 ~ 37 |
| L1 信号类型掩码 | 有效 L1 C/A 码的十六进制掩码值 | 0x04, 0x08, 0x0C |
| 观测块字段数 | RANGEA 中每个卫星观测块的字段数量 | 10 或 11（取决于日志版本） |

---

## 📂 Stage 8：工业级工程目录结构设计

```text
exp2/
├── data/
│   ├── data.txt                    # 原始接收机日志（输入）
│   └── satellite_visualization.png # 卫星分布可视化（输出）
│
├── algorithms/
│   ├── __init__.py
│   ├── extractor.py                # 数据提取层
│   │   ├── extract_obs_time()      #   A1: 观测时刻提取
│   │   ├── extract_ephemeris_matrix()  # A2: 星历矩阵提取
│   │   └── extract_pseudorange_matrix() # A3: 伪距矩阵提取
│   ├── satellite.py                # 卫星位置解算层
│   │   ├── compute_satellite_position() # B: 9 步卫星位置解算
│   │   └── compute_all_satellite_positions() # B 的批量版本
│   ├── corrections.py              # 误差修正层
│   │   ├── apply_clock_correction()    # C1: 卫星钟差修正
│   │   └── apply_earth_rotation_correction() # C2: 地球自转修正
│   ├── solver.py                   # 定位解算引擎
│   │   ├── least_squares_solution()    # D: 最小二乘迭代
│   │   └── weighted_least_squares()    # D扩展: 加权最小二乘
│   └── transform.py                # 坐标变换层
│       ├── ecef_to_geodetic()      #   E1: ECEF→大地坐标
│       ├── ecef_to_enu()           #   E1扩展: ECEF→ENU
│       └── geodetic_to_ecef()      #   E1逆变换
│
├── evaluation/
│   ├── __init__.py
│   └── accuracy.py                 # 精度评估层
│       ├── validate_accuracy()     #   E2: 精度验证
│       └── plot_skyplot()          #   星空图可视化
│
├── visualization/
│   ├── __init__.py
│   ├── plot_earth.py               # 3D 地球+卫星分布
│   └── plot_skyplot.py             # 卫星星空图（极坐标）
│
├── configs/
│   └── constants.py                # 所有物理常数 + 算法超参数
│
├── tests/
│   ├── __init__.py
│   ├── test_extractor.py           # 数据提取单元测试
│   ├── test_satellite.py           # 卫星位置解算单元测试
│   ├── test_corrections.py         # 误差修正单元测试
│   ├── test_solver.py              # 最小二乘单元测试
│   └── test_transform.py           # 坐标变换单元测试
│
├── scripts/
│   ├── run_spp.sh                  # 一键运行脚本
│   └── run_comparison.sh           # 多组数据对比运行脚本
│
├── docs/
│   ├── algorithm_design.md         # 本文件：算法设计文档
│   └── experiment_analysis.md      # 实验指导书分析报告（02）
│
├── 02_experiment_guide_analysis.md # 实验指导书深度分析
├── 03_algorithm_design.md          # 本文件
├── satellite_navigation_solution.py # 参考实现（Monolithic）
├── satellite_visualization.png     # 可视化输出
├── debug_test.py                   # 调试脚本
├── data.txt                        # 原始数据
├── raw_data.txt                    # 备用原始数据
├── readme.md                       # 实验说明
└── task_tree.md                    # 任务树
```

### 目录设计说明

| 目录 | 设计理由 |
| :--- | :--- |
| `algorithms/` | 核心算法逻辑与数据/IO 完全解耦，不依赖文件路径。每个模块文件职责单一（提取/卫星/修正/求解器/变换），便于独立测试。 |
| `evaluation/` | 精度评估单独抽取，因为实验需要"自解算 vs 参考"的双轨对比，该模块可能包含多种评估指标（RMS/CEP/2DRMS 等）。 |
| `visualization/` | 可视化逻辑与算法逻辑完全分离。3D 地球图 + 极坐标星空图是独立于定位解算的输出模块。 |
| `configs/constants.py` | 所有常数集中在单一文件，避免散落在各模块中难以维护。WGS-84 参数、GM、ωe 等在整个系统内必须绝对一致。 |
| `tests/` | 每个算法模块都有对应单元测试。卫星位置解算（9 步流程）是最需要单元测试的部分，因为涉及 20+ 步浮点运算，极易因参数映射错误产生微小但累积的偏差。 |
| `scripts/` | 一键运行脚本，方便学生/教师快速执行完整流程。 |

---

## 附录：实验报告模板对应关系

| 实验报告要求（模板第 5 节） | 对应算法模块/输出 | 说明 |
| :--- | :--- | :--- |
| (1) 接收机配套软件界面截图/照片 | N/A（外部操作） | 记录 NovAtel Connect 的定位界面 |
| (2) 可见卫星编号/定位授时结果 | `ExtractObsTime` → `(week, t_obs)` + `valid_prns` | 输出 GPS 周、周内秒、有效 PRN 列表 |
| (3) 原始星历数据 | `ExtractEphemerisMatrix` → `eph(N×23)` | 打印星历矩阵（或记录到文件） |
| (4) 原始伪距测量数据 | `ExtractPseudorangeMatrix` → `pr(N×2)` | 打印各卫星 PRN+伪距值 |
| (5) 原始单点定位结果 | 解析 BESTPOSA → `(ref_lon, ref_lat, ref_h)` | 商用固件定位结果 |
| (6) 原始卫星位置信息 | `ComputeSatellitePosition` → `sat_pos(N×3)` | 各卫星 ECEF 坐标，可与 SATXYZ2A 对比 |
| (7) 程序定位数据 | `LeastSquaresSolver` + `EcefToGeodetic` → 自解算经纬高 + 精度偏差 | 参与定位卫星的 PRN/高度角/方位角/坐标、自解算结果、与参考值的偏差 |
| (8) 完整单点定位程序 | 整个算法管线 | 提交可运行的程序代码 |

---

*设计日期：2026-06-08*  
*设计框架：`skills/10_algorithm_design.md`*  
*设计依据：`02_experiment_guide_analysis.md`*