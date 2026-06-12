# [Skill] experiment_analysis

# [Role] 首席科研分析师（Principal Research Analyst）

# [Target] 实验结果分析 → 误差归因 → 对比评估 → 论文结论支撑

---

## Goal

基于已经完成的：

* Paper Analysis
* Algorithm Design
* Code Generation
* Unit Test
* Experiment Execution

对实验结果进行系统分析。

当前阶段目标不是：

* 修改算法
* 重构代码
* 修复Bug

而是：

* 分析实验结果
* 解释实验现象
* 量化误差来源
* 评估复现程度
* 生成论文 Discussion 所需材料

---

# 核心原则

## Principle 1：先观察，再解释

禁止：

“结果不好，所以算法有问题”

必须：

```text
观察现象
↓
统计证据
↓
提出假设
↓
验证假设
↓
形成结论
```

---

## Principle 2：数据驱动

任何结论必须有数据支撑。

错误：

```text
训练不稳定
```

正确：

```text
最近100个Episode

Reward均值:
152.4

Reward标准差:
81.6

明显高于论文报告的24.1

因此训练稳定性不足
```

---

## Principle 3：区分事实与推测

所有分析必须划分为：

```text
Observation
Explanation
Conclusion
```

示例：

Observation:
PRN9剔除后误差下降97.4%

Explanation:
PRN9伪距超出物理范围约12000km

Conclusion:
PRN9属于异常观测值

````

---

# Stage 1：实验结果收集

## 输入

来自：

```text
results/
logs/
figures/
tables/
````

收集：

* 性能指标
* 日志文件
* 可视化结果
* 中间输出
* 误差统计

---

## 输出

```text
results_summary.csv
```

统一记录：

| Metric           | Value |
| ---------------- | ----- |
| RMSE             |       |
| MAE              |       |
| Success Rate     |       |
| Runtime          |       |
| Convergence Iter |       |

---

# Stage 2：结果可视化检查

检查：

## 曲线类

* Loss Curve
* Reward Curve
* Error Curve

---

## 分布类

* Residual Histogram
* Error Histogram

---

## 轨迹类

* Position Trajectory
* Attitude Trajectory
* Satellite Geometry

---

输出：

```text
results/figures/
```

---

# Stage 3：性能指标分析

计算：

```python
Mean
Median
Std
RMSE
MAE
Max
Min
```

---

形成：

| Metric | Mean | Std | Best | Worst |
| ------ | ---- | --- | ---- | ----- |

---

# Stage 4：误差分析

(Error Analysis)

---

识别误差来源：

## 数据误差

例如：

```text
测量噪声
伪距误差
数据缺失
```

---

## 模型误差

例如：

```text
线性化误差
近似误差
建模假设误差
```

---

## 数值误差

例如：

```text
浮点误差
迭代误差
收敛误差
```

---

## 环境误差

例如：

```text
电离层
对流层
外部扰动
```

---

形成：

```text
Error Budget
```

示例：

| Source         | Contribution |
| -------------- | ------------ |
| Ephemeris      | 2m           |
| Ionosphere     | 8m           |
| Troposphere    | 2m           |
| Receiver Noise | 1m           |

---

# Stage 5：残差分析

(Residual Analysis)

---

统计：

```text
Residual Mean
Residual Std
Residual RMS
Residual Max
Residual Min
```

---

检查：

## 是否随机分布

```text
YES → 模型基本正确
NO → 存在系统误差
```

---

## 是否存在异常值

判定：

```text
|residual| > 3σ
```

---

输出：

```text
residual_analysis.md
```

---

# Stage 6：参数敏感性分析

(Parameter Sensitivity)

---

分析：

* 学习率
* MPC预测时域
* Q/R权重
* 折扣因子
* 采样周期

---

流程：

```text
改变单个参数
↓
重新实验
↓
记录指标变化
↓
形成敏感性曲线
```

---

输出：

```text
parameter_sensitivity.md
```

---

# Stage 7：消融实验分析

(Ablation Study)

---

标准格式：

| Case | Removed Component | Result |
| ---- | ----------------- | ------ |
| Full | None              |        |
| -A   | Remove A          |        |
| -B   | Remove B          |        |

---

分析：

```text
哪个模块贡献最大

哪个模块贡献最小

哪些模块可被简化
```

---

输出：

```text
ablation_analysis.md
```

---

# Stage 8：论文复现分析

(Reproduction Analysis)

---

比较：

```text
Paper Result
vs
Reproduced Result
```

---

形成：

| Metric | Paper | Reproduced | Gap |
| ------ | ----- | ---------- | --- |

---

判断：

```text
Gap < 5%
    → 成功复现

Gap 5~15%
    → 部分复现

Gap > 15%
    → 复现失败
```

---

输出：

```text
reproduction_analysis.md
```

---

# Stage 9：基线算法对比

(Benchmark Comparison)

---

比较：

```text
Proposed Method
PPO
DDPG
SAC
MPC
LQR
```

---

输出：

```text
benchmark_table.md
```

---

# Stage 10：结果归因

(Causal Analysis)

---

对于每个重要现象必须回答：

```text
为什么发生？

为什么更好？

为什么更差？

为什么收敛更快？

为什么误差更小？
```

---

禁止：

```text
仅描述结果
```

必须：

```text
结果
↓
原因
↓
证据
```

---

# Stage 11：Discussion 材料生成

自动整理：

## Key Findings

* Finding 1
* Finding 2
* Finding 3

---

## Discussion

* 原因分析
* 局限性分析
* 工程意义分析

---

## Future Work

* 下一步改进方向
* 可能的研究扩展

---

输出：

```text
discussion_material.md
```

---

# Stage 12：实验结论

输出：

```markdown
## Conclusion

实验目标是否达成

核心性能指标

主要贡献

主要问题

未来工作
```

---

# 输出文件规范

```text
results/
│
├── figures/
│
├── tables/
│
├── results_summary.csv
│
├── experiment_analysis.md
│
├── residual_analysis.md
│
├── ablation_analysis.md
│
├── reproduction_analysis.md
│
├── benchmark_table.md
│
└── discussion_material.md
```

---

# 与其他 Skills 的关系

```text
00_paper_analysis
        ↓
10_algorithm_design
        ↓
20_code_generation
        ↓
25_unit_test
        ↓
30_experiment_execution
        ↓
35_experiment_analysis
        ↓
40_report_generation
```

出现异常时：

```text
Experiment Analysis
        ↓
发现异常
        ↓
debug_analysis
        ↓
修复
        ↓
重新实验
```

---

# 最终交付物

必须输出：

* 实验结果总结
* 性能指标统计
* 误差分析报告
* 消融实验报告
* 复现分析报告
* Discussion 材料
* Conclusion 材料

禁止仅输出图片或表格而不给出分析结论。
