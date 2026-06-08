# 运行说明
    python main.py
    
# 项目目录结构

```text
exp2/
├── main.py                               # ✅ 主函数入口 (保持不变)
├── data/                                 # 📁 原始数据
│   ├── data.txt                          # 实验数据
│   └── raw_data.txt                      
├── configs/                              # ✅ 参数设置
│   └── constants.py
├── algorithms/                           # ✅ 算法
│   ├── __init__.py
│   ├── extractor.py
│   ├── satellite.py
│   ├── corrections.py
│   ├── solver.py
│   └── transform.py
├── evaluation/                           # ✅ 精度评估
│   └── accuracy.py
├── visualization/                        # ✅ 可视化skyplot
│   └── plot_earth.py
├── tests/                                # 
│   └── test_satellite_position.py
├── scripts/                              # 📁 调试脚本
│   ├── debug_time_chain.py               
│   ├── debug_rangea_fields.py            
│   ├── debug_lsq_trace.py                 
│   ├── debug_clock_sign.py               
│   ├── debug_clock_sign_final.py          
│   ├── debug_exclude_prn9.py            
│   ├── debug_three_cases.py              
│   ├── debug_residual_analysis.py       
│   └── debug_measurement_validation.py   
├── docs/                                 # 📁 文档
│   ├── analysis/                         #    算法/实验分析
│   │   ├── 02_experiment_guide_analysis.md # 实验分析
│   │   ├── 03_algorithm_design.md        # 算法设计
│   │   └── 算法与数据字段映射验证报告.md   # 
│   ├── debug_reports/                    #    调试报告
│   │   ├── 04_调试复盘与根因分析.md        
│   │   ├── 05_时间偏移根因分析报告.md      
│   │   ├── 06_LSQ追踪与7000km误差根因分析.md 
│   │   ├── 07_剔除PRN9后剩余186km误差排查.md 
│   │   └── 08_钟差符号根因确认与修复报告.md 
│   └── 数据字段含义解析.docx               
├── results/                              # ✅ 8项输出
│   ├── 01_software_screenshot.txt
│   ├── ...
│   ├── 08_program_info.txt
│   └── visualization.png
└── readme.md                             # ✅ 
