# -exp2/
├── main.py                          # ✅ 主入口 (保持不变)
├── data/                            # 📁 原始数据
│   ├── data.txt                     # ← 从根目录移入
│   └── raw_data.txt                 # ← 从根目录移入
│
├── configs/                         # ✅ 已有
│   └── constants.py
│
├── algorithms/                      # ✅ 已有
│   ├── __init__.py
│   ├── extractor.py
│   ├── satellite.py
│   ├── corrections.py
│   ├── solver.py
│   └── transform.py
│
├── evaluation/                      # ✅ 已有
│   └── accuracy.py
│
├── visualization/                   # ✅ 已有
│   └── plot_earth.py
│
├── tests/                           # ✅ 已有
│   └── test_satellite_position.py
│
├── scripts/                         # 📁 调试脚本
│   ├── debug_time_chain.py          # ← 从根目录移入
│   ├── debug_rangea_fields.py       # ← 从根目录移入
│   ├── debug_lsq_trace.py           # ← 从根目录移入
│   ├── debug_clock_sign.py          # ← 从根目录移入
│   ├── debug_clock_sign_final.py    # ← 从根目录移入
│   ├── debug_exclude_prn9.py        # ← 从根目录移入
│   ├── debug_three_cases.py         # ← 从根目录移入
│   ├── debug_residual_analysis.py   # ← 从根目录移入
│   └── debug_measurement_validation.py  # ← 从根目录移入
│
├── docs/                            # 📁 文档
│   ├── analysis/                    #   算法/实验分析
│   │   ├── 02_experiment_guide_analysis.md  # ← 从根目录移入
│   │   ├── 03_algorithm_design.md          # ← 从根目录移入
│   │   └── 算法与数据字段映射验证报告.md    # ← 从根目录移入
│   │
│   ├── debug_reports/               #   调试报告
│   │   ├── 04_调试复盘与根因分析.md         # ← 从根目录移入
│   │   ├── 05_时间偏移根因分析报告.md       # ← 从根目录移入
│   │   ├── 06_LSQ追踪与7000km误差根因分析.md # ← 从根目录移入
│   │   ├── 07_剔除PRN9后剩余186km误差排查.md # ← 从根目录移入
│   │   └── 08_钟差符号根因确认与修复报告.md # ← 从根目录移入
│   │
│   └── 数据字段含义解析.docx        # ← 从根目录移入
│
├── results/                         # ✅ 已有 (8项输出)
│   ├── 01_software_screenshot.txt
│   ├── ...
│   ├── 08_program_info.txt
│   └── visualization.png
│
└── readme.md                        # ✅ 保留

#运行main.py即可运行代码
