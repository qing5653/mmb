# MathorCup C题工作空间目录规划

## 1. 目录结构

```text
mmb/
├── 2026年MathorCup_C题.md
├── 附件1_样例数据.csv
├── 建模分析报告.md
├── README.md
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── pipeline/
│   ├── q1/
│   ├── q2/
│   ├── q3/
│   └── common/
├── outputs/
│   ├── q1/
│   ├── q2/
│   └── q3/
└── docs/
    └── 工作空间目录规划.md
```

## 2. 各目录职责

- data/raw：原始输入数据，只读不改。
- data/processed：预处理后的训练集、验证集、测试集，以及样本1/2/3抽取文件。
- src/pipeline：总流程脚本（从数据读取到结果汇总的一键执行入口）。
- src/q1：问题1相关脚本（特征筛选、OR值估计、统计检验）。
- src/q2：问题2相关脚本（模型训练、阈值确定、可解释性分析）。
- src/q3：问题3相关脚本（约束建模、枚举求解、个体方案输出）。
- src/common：公共模块（配置、评估函数、可视化工具、I/O工具）。
- outputs/q1：问题1结果表与图。
- outputs/q2：问题2结果表与图。
- outputs/q3：问题3结果表与图。
- docs：流程文档、实验记录和提交前说明。

## 3. 协作规范

- 变量口径统一：问题2默认继承问题1筛选出的关键指标，再补充临床必要变量。
- 防数据泄漏：标准化、特征筛选、阈值搜索均只在训练/验证阶段完成；测试集仅封板评估。
- 结果可追溯：报告中的每个结论都要能在outputs目录找到对应表图。

## 4. 建议的脚本顺序

1. src/pipeline/01_data_check.py
2. src/pipeline/02_split_data.py
3. src/q1/11_feature_selection.py
4. src/q1/12_or_analysis.py
5. src/q2/21_train_models.py
6. src/q2/22_threshold_tuning.py
7. src/q2/23_interpretability.py
8. src/q3/31_build_constraints.py
9. src/q3/32_optimize_plan.py
10. src/q3/33_generate_samples_123.py
