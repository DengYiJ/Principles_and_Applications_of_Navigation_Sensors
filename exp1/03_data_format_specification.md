# 数据格式规范 (Data Format Specification)

> **用途**：定义 `$GTIMU` 与 `$GPFPD` 两种 NMEA 格式消息的字段映射、单位、校验规则与解析边界。
> **输入来源**：实验采集的实际数据文件（八位置/加速度计/零偏/速率标定目录）
> **关联设计**：`02_algorithm_design.md` → Stage 6 → `GTIMU_FIELD_MAP`

---

## 1. `$GTIMU` 消息 — IMU原始数据

### 协议格式

```
[$]GTIMU, GPSWeek, GPSTime, GyroX, GyroY, GyroZ, AccX, AccY, AccZ, Tpr*CS\r\n
```

- 前缀 `$` 可选（部分行有，部分行无）
- 字段分隔符：`,`（逗号）
- 校验和后缀：`*CS`（2字符十六进制校验和），可选
- 行尾：`\r\n`

### 字段索引映射表

| Index | 字段名 | 数据类型 | 单位 | 物理范围 | 说明 |
|:-----:|:-------|:---------|:-----|:---------|:-----|
| 0 | `GPSWeek` | int | 周 | [0, 4096] | GPS周数，实验数据中恒为0 |
| 1 | `GPSTime` | float | s | [0, 604800] | GPS周内秒，分辨率0.005s(=200Hz) |
| 2 | `GyroX` | float | °/s | [-500, 500] | X轴角速率 |
| 3 | `GyroY` | float | °/s | [-500, 500] | Y轴角速率 |
| 4 | `GyroZ` | float | °/s | [-500, 500] | Z轴角速率 |
| 5 | `AccX` | float | m/s² | [-30, 30] | X轴加速度 |
| 6 | `AccY` | float | m/s² | [-30, 30] | Y轴加速度 |
| 7 | `AccZ` | float | m/s² | [-30, 30] | Z轴加速度 |
| 8 | `Tpr` | float | °C | [-40, 85] | 传感器温度 |

### 校验规则

```python
MIN_FIELDS = 9          # 至少9个逗号分隔字段
MAX_FIELDS = 9          # 严格9字段
FIELD_SEPARATOR = ","
SAMPLE_RATE = 200.0     # Hz
EXPECTED_DT = 0.005     # s = 1/200
```

### 实际数据样例

```
$GTIMU,0,2299.16500,-0.0014,0.0150,-0.0006,-0.0043,1.0001,-0.0077,30.8*57
```

---

## 2. `$GPFPD` 消息 — 转台基准/组合导航结果

### 协议格式

```
[$]GPFPD, GPSWeek, GPSTime, Roll, Pitch, Heading, Lat, Lon, Height, Ve, Vn, Vu, Status, NSV, Age*CS\r\n
```

### 字段索引映射表

| Index | 字段名 | 数据类型 | 单位 | 物理范围 | 说明 |
|:-----:|:-------|:---------|:-----|:---------|:-----|
| 0 | `GPSWeek` | int | 周 | [0, 4096] | GPS周数 |
| 1 | `GPSTime` | float | s | [0, 604800] | GPS周内秒 |
| 2 | `Roll` | float | ° | [-180, 180] | 横滚角 |
| 3 | `Pitch` | float | ° | [-90, 90] | 俯仰角 |
| 4 | `Heading` | float | ° | [0, 360] | 航向角 |
| 5 | `Lat` | float | ° | [-90, 90] | 纬度(GPS) |
| 6 | `Lon` | float | ° | [-180, 180] | 经度(GPS) |
| 7 | `Height` | float | m | [-500, 10000] | 高程 |
| 8 | `Ve` | float | m/s | [-500, 500] | 东向速度 |
| 9 | `Vn` | float | m/s | [-500, 500] | 北向速度 |
| 10 | `Vu` | float | m/s | [-500, 500] | 天向速度 |
| 11 | `Status` | int | — | [0, 6] | 导航解状态 |
| 12 | `NSV` | int | — | [0, 32] | 可见卫星数 |
| 13 | `Age` | float | s | [0, 999] | 差分龄期 |

### 实际数据样例

```
$GPFPD,0,18857.93000,0.803,-0.017,0.004,45.73450012,126.63420014,145.996,0.001,0.000,-0.000,0.000,0,0,08*4C
```

---

## 3. 数据文件命名规约与实验场景映射

### 加速度计六位置标定

| 文件名 | 实验场景 | 预期姿态 |
|:-------|:---------|:---------|
| `gtimu_1.log` | 位置1 | X轴朝上(天向) |
| `gtimu_2.log` | 位置2 | X轴朝下(地向) |
| `gtimu_3.log` | 位置3 | Y轴朝上(天向) |
| `gtimu_4.log` | 位置4 | Y轴朝下(地向) |
| `gtimu_5.log` | 位置5 | Z轴朝上(天向) |
| `gtimu_6.log` | 位置6 | Z轴朝下(地向) |

### 陀螺仪八位置零偏标定

| 文件名 | 实验场景 |
|:-------|:---------|
| `gtimu_1.log` ~ `gtimu_8.log` | 8个不同姿态（零偏多位置目录） |

### 陀螺仪速率标定

| 文件模式 | 说明 |
|:---------|:-----|
| `gtimu_+/-RR.log` | IMU原始数据，RR=转速(10/20/30/40/50) |
| `gpfpd_+/-RR.log` | 转台基准数据，RR=转速 |
| 目录：`位置一/位置二/位置三` | 分别绕X/Y/Z轴旋转 |

### Allan方差静态分析

| 文件名 | 说明 |
|:-------|:-----|
| `gtimu_3.5h.log` | 约3.5小时静态IMU数据(200Hz) |

---

## 4. 解析器伪代码契约

```python
def parse_gtimu_line(line: str) -> Optional[dict]:
    """
    解析单行$GTIMU语句。
    
    契约规则：
    1. 去除首尾空白
    2. 去除可选前缀'$'
    3. 以逗号分割
    4. 校验字段数 == 9 (Assertion)
    5. 可选去除校验和后缀 '*CS'
    6. 按GTIMU_FIELD_MAP类型转换
    7. 返回dict或None(解析失败)
    
    Assertions:
    - len(fields) == 9
    - float(GyroX/Y/Z) in [-500, 500]
    - float(AccX/Y/Z) in [-30, 30]
    - float(Tpr) in [-40, 85]
    """

def parse_gpfpd_line(line: str) -> Optional[dict]:
    """
    解析单行$GPFPD语句。
    
    契约规则：同GTIMU，但字段数==14。
    
    Assertions:
    - float(Roll) in [-180, 180]
    - float(Pitch) in [-90, 90]
    - float(Heading) in [0, 360]
    - float(Lat) in [-90, 90]
    - float(Lon) in [-180, 180]
    - int(NSV) in [0, 32]
    """
```

---

## 5. 边界情况与异常处理

| 场景 | 处理策略 |
|:-----|:---------|
| 空行 | 跳过，不报错 |
| 前缀 `$` 缺失 | 自动补全逻辑不必须，分割时忽略即可 |
| 字段数不等于9/14 | 标记无效行，记录WARNING + 行号 |
| 字段无法转换为数字 | 标记无效行，记录WARNING + 原始内容 |
| 时间戳不连续(>0.01s间隔) | 分段标记，记录WARNING |
| 校验和错误(可选) | 记录DEBUG级别日志 |
| 数据全零持续>1s | 标记"传感器异常丢失" |