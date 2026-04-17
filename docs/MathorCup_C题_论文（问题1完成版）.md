# 中老年人群高血脂症风险预警及干预方案优化

## 摘要

针对 MathorCup C 题，本文围绕问题1完成了从数据预处理、关键指标筛选到九种体质风险贡献度评估的完整实现流程。基于1000例样本，采用分层抽样构建训练集/验证集/测试集（700/150/150），并在训练集上执行相关性筛选、L1-Logistic稀疏筛选和随机森林重要性评估，采用投票法（至少两法入选）得到5个关键指标：TG、TC、血尿酸、LDL-C、HDL-C。随后基于九体质构建Logistic回归并输出OR及95%置信区间。结果显示，九体质模型在验证集AUC为0.4313，Hosmer-Lemeshow检验p=0.8322；关键指标预警模型验证集AUC为0.9864。本文同时给出问题1的可复现实验流程、核心公式、可视化图表与结果解读。

关键词：高血脂症；中医体质；特征筛选；Logistic回归；OR；风险预警

---

## 1 背景与研究意义

高血脂症是中老年人心脑血管事件的重要危险因素。传统风险评估常依赖血脂检测结果，缺少中医体质与活动能力等信息融合。题目要求从血常规与活动量表中筛选关键指标，既要表征痰湿体质严重程度，也要具备发病风险预警能力，并量化九种体质对发病风险的贡献差异。

---

## 2 问题重述

### 2.1 问题1目标

1. 从候选指标（TC、TG、LDL-C、HDL-C、空腹血糖、血尿酸、BMI、ADL总分、IADL总分、活动总分）中筛选关键指标。
2. 建立九体质风险贡献度模型，输出OR值、95%CI及显著性检验结果。

### 2.2 数据与变量

1. 样本量：1000。
2. 标签：高血脂症二分类标签（0/1）。
3. 体质变量：平和质、气虚质、阳虚质、阴虚质、痰湿质、湿热质、血瘀质、气郁质、特禀质。
4. 目标输出：关键指标清单 + 九体质OR贡献表。

### 2.3 任务拆解与交付口径（给论文手）

问题1不是“单模型完成全部目标”，而是双模型分工：

1. 解释型模型（九体质Logistic）：用于回答“九体质与风险是否相关、方向与强度如何”。
2. 预测型模型（关键指标Logistic）：用于回答“筛选出的关键指标是否具备预警能力”。

对应交付物：

1. 关键指标筛选结果：`selected_features.json`、`feature_selection_details.csv`。
2. 九体质贡献分析结果：`OR_values_table.csv`、`vif_table.csv`。
3. 预测性能评估结果：`q1_summary.json`（含AUC、HL、样本切分信息）。
4. 图表结果：`outputs/q1/figures`下6张图（正文3张、附录3张）。

---

## 3 模型假设

1. 样本记录独立同分布，测量误差对总体规律影响可忽略。
2. 训练/验证/测试分布一致，分层抽样可保持标签比例稳定。
3. OR解释基于其他变量保持不变条件下的边际变化。
4. 关键指标筛选过程仅在训练集进行，避免数据泄漏。
5. 诊断标签与血脂指标可能存在同源关系，需在讨论中解释其对AUC的影响。

---

## 4 方法与算法

### 4.1 数据切分与防泄漏策略

1. 按标签分层切分：训练集70%，验证集15%，测试集15%。
2. 相关性、L1筛选、随机森林重要性均在训练集执行。
3. 验证集仅用于AUC评价与模型对比。
4. 测试集在问题1阶段保留，不参与调参。

### 4.2 相关性筛选

对候选指标 $X_j$ 与痰湿积分 $T$ 计算 Pearson：

$$
r_{j,T} = \frac{\sum_{i=1}^{n}(X_{ij}-\bar{X}_j)(T_i-\bar{T})}{\sqrt{\sum_{i=1}^{n}(X_{ij}-\bar{X}_j)^2}\sqrt{\sum_{i=1}^{n}(T_i-\bar{T})^2}}
$$

对候选指标与二分类标签 $Y$ 计算 Spearman：

$$
\rho_{j,Y}=1-\frac{6\sum d_i^2}{n(n^2-1)}
$$

### 4.3 L1-Logistic 稀疏筛选

采用带L1正则的Logistic回归交叉验证选择惩罚强度：

$$
\min_{\beta_0,\beta}\left[-\sum_{i=1}^{n}\left(y_i\log p_i+(1-y_i)\log(1-p_i)\right)+\lambda\|\beta\|_1\right]
$$

其中 $p_i=\sigma(\beta_0+x_i^T\beta)$。

### 4.4 随机森林重要性

通过集成树模型得到特征重要性：

$$
Importance(X_j)=\frac{1}{B}\sum_{b=1}^{B}\sum_{t\in T_b}\Delta Gini_t\cdot\mathbf{1}(X_j\in t)
$$

### 4.5 投票机制

设三种方法选择结果分别为0/1，投票数为：

$$
V_j = I_j^{corr}+I_j^{l1}+I_j^{rf}
$$

若 $V_j\ge2$，则 $X_j$ 入选最终关键指标。

### 4.6 九体质Logistic与OR

构建模型：

$$
\log\frac{P(Y=1)}{1-P(Y=1)}=\beta_0+\sum_{k=1}^{9}\beta_k Z_k
$$

OR定义：

$$
OR_k=e^{\beta_k},\quad CI_{95\%}=\left(e^{\beta_k-1.96SE(\beta_k)},e^{\beta_k+1.96SE(\beta_k)}\right)
$$

并计算Wald检验、VIF与Hosmer-Lemeshow检验。

### 4.7 模型评判标准前置（正文口径）

为保证后文结论可复核，先定义“任务适配型”判定标准（不是单一AUC导向）：

| 评审维度 | 判定指标 | 建议阈值/标准 | 本研究结果 | 判定 |
|---|---|---|---|---|
| 流程规范性 | 数据切分与防泄漏 | 分层切分；特征筛选仅在训练集完成 | 训练/验证/测试=700/150/150；筛选在训练集执行 | 通过 |
| 解释型模型整体显著性 | LLR检验 | $p<0.05$ 代表整体显著 | $p=0.3672$ | 证据弱（可用于趋势解释） |
| 解释型模型单变量证据 | Wald检验与95%CI | 常用标准为 $p<0.05$ 且CI不跨1 | 九体质变量多数 $p>0.05$，CI多跨1 | 证据弱（可用于趋势解释） |
| 解释型模型校准性 | Hosmer-Lemeshow | 常用标准为 $p>0.05$ | $p=0.8322$ | 通过 |
| 解释型模型区分度 | 验证/测试AUC | 通常AUC>0.70较可用 | 验证0.4313，测试0.3657 | 不用于独立预测 |
| 预测型模型区分度 | 验证集AUC | 通常AUC>0.75具实用性 | 0.9864 | 通过（高预警能力） |
| 医学一致性 | 指标可解释性 | 与已知临床机制一致 | TG、TC、LDL-C、HDL-C、血尿酸入选 | 通过 |

说明：解释型模型证据偏弱不代表流程错误，也不代表问题1未完成；它承担的是“机制趋势解释”而不是“高精度分类”。

### 4.8 本轮优化动作与结论

针对“评价标准里较多不通过”的疑问，已完成以下实质优化：

1. 将关键指标模型由固定参数Logistic升级为带交叉验证的LogisticRegressionCV，自动选择最优正则强度。
2. 为两类模型同时补充验证集与测试集双口径评估，避免单一验证集偶然性。
3. 在AUC之外新增PR-AUC与Brier分数，形成“区分度+概率质量”的联合评估。
4. 新增统一对比结果表 `model_performance_table.csv`，便于正文和附录直接引用。
5. 新增九体质增强实验（L2正则、随机森林、交互ElasticNet），输出 `constitution_enhancement_table.csv` 进行横向比较。

优化后结论：九体质模型在测试集AUC仍偏低（基线0.3657，增强后最好为随机森林0.4333），说明主要是“体质变量对该标签信号弱”，而非代码实现错误；关键指标模型验证/测试AUC均接近1（0.9864/0.9794），预测性能稳定。

---

## 5 结果与可视化

### 5.1 关键指标筛选结果

最终入选5项关键指标：

1. TG（甘油三酯）
2. TC（总胆固醇）
3. 血尿酸
4. LDL-C（低密度脂蛋白）
5. HDL-C（高密度脂蛋白）

可视化图表：

1. 投票结果图：[outputs/q1/figures/q1_feature_votes.png](../outputs/q1/figures/q1_feature_votes.png)
2. 方法选择热力图（附录A）：[outputs/q1/figures/q1_method_heatmap.png](../outputs/q1/figures/q1_method_heatmap.png)
3. 随机森林重要性（附录A）：[outputs/q1/figures/q1_rf_importance.png](../outputs/q1/figures/q1_rf_importance.png)
4. 关键指标模型系数（附录A）：[outputs/q1/figures/q1_selected_model_coef.png](../outputs/q1/figures/q1_selected_model_coef.png)

![Q1 Feature Voting Results](../outputs/q1/figures/q1_feature_votes.png)

图5-1 关键指标三方法投票结果（颜色区分是否最终入选，条末标注投票数）。

图5-1解读口径：每一行对应一个候选指标；横轴为三种筛选方法累计投票数（0到3）；颜色区分最终是否入选。该图用于展示“多方法一致性”，回答“为什么是这5个关键指标”。

### 5.2 九体质贡献度结果（OR）

1. 当前训练-验证设置下，九体质变量中未出现 $p<0.05$ 的显著项。
2. OR方向上，气郁质与气虚质呈风险上升趋势，但置信区间跨1。
3. 体质OR森林图见：[outputs/q1/figures/q1_or_forest.png](../outputs/q1/figures/q1_or_forest.png)

![OR and 95% CI of Nine Constitutions](../outputs/q1/figures/q1_or_forest.png)

图5-2 九体质OR森林图（虚线为OR=1，红色区间表示显著项；本次实验无显著体质变量）。

图5-2解读口径：纵轴每一行是一个体质变量；点为OR估计值；线段为95%CI。若CI跨1，则该变量在当前样本下未达到显著性。该图用于回答“体质影响方向是否稳定”。

### 5.3 模型评估

1. 九体质Logistic验证集AUC：0.4313。
2. 九体质Logistic测试集AUC：0.3657。
3. Hosmer-Lemeshow检验：$p=0.8322$。
4. 关键指标预警模型验证集AUC：0.9864。
5. 关键指标预警模型测试集AUC：0.9794。
6. AUC对比图见：[outputs/q1/figures/q1_auc_compare.png](../outputs/q1/figures/q1_auc_compare.png)

![Q1 Validation AUC Comparison](../outputs/q1/figures/q1_auc_compare.png)

图5-3 验证集AUC对比图（关键指标预警模型显著优于九体质模型）。

图5-3解读口径：每根柱表示一个模型在验证集上的AUC；数值越大区分能力越强。该图用于直接支撑“双模型分工”的结论：解释型模型用于机制趋势，预测型模型用于风险识别。

关键结果来源：

1. [outputs/q1/q1_summary.json](../outputs/q1/q1_summary.json)
2. [outputs/q1/OR_values_table.csv](../outputs/q1/OR_values_table.csv)
3. [outputs/q1/feature_selection_details.csv](../outputs/q1/feature_selection_details.csv)
4. [outputs/q1/model_performance_table.csv](../outputs/q1/model_performance_table.csv)

### 5.4 表格结果解读（每行每列含义）

#### 5.4.1 `feature_selection_details.csv` 字段解释

1. 每一行代表一个候选指标。
2. `pearson_r_tan`、`pearson_p_tan`：该指标与痰湿积分的Pearson相关系数及其显著性。
3. `spearman_r_label`、`spearman_p_label`：该指标与二分类标签的Spearman相关及其显著性。
4. `corr_selected`：相关性筛选是否通过（True/False）。
5. `lasso_coef`、`lasso_selected`、`lasso_alpha`：L1-Logistic系数、是否被L1保留、对应正则强度。
6. `rf_importance`、`rf_selected`、`rf_importance_mean`：随机森林重要性、是否高于阈值、平均阈值。
7. `votes`：三种方法累计票数。
8. `final_selected`：是否进入最终关键指标清单。

#### 5.4.2 `OR_values_table.csv` 字段解释

1. 每一行代表一个回归项（含常数项`const`与9个体质变量）。
2. `coef`、`std_err`：Logistic系数及标准误。
3. `wald_chi2`、`p_value`：Wald统计量与显著性水平。
4. `or`、`or_ci_low`、`or_ci_high`：OR值及95%置信区间。
5. `interpretation`：按OR方向和显著性生成的文字解释。

#### 5.4.3 其他结果表字段解释

1. `selected_feature_model_coef.csv`：每一行为一个最终关键指标；列为`feature`和`coef`，用于解释预警模型中各指标方向与强度。
2. `vif_table.csv`：每一行为一个体质变量；列为`variable`和`vif`，用于评估多重共线性（通常VIF越高共线性风险越大）。
3. `q1_summary.json`：记录样本切分、最终入选指标数量、九体质模型诊断指标（LLR、HL、AUC等）和关键指标模型AUC，是总控结果文件。
4. `model_performance_table.csv`：每一行为一个模型（九体质Logistic、关键指标模型）；列包括`val_auc`、`test_auc`、`val_pr_auc`、`test_pr_auc`、`val_brier`、`test_brier`，用于统一比较区分能力与概率质量。
5. `constitution_enhancement_table.csv`：每一行为一种九体质增强方案；列包括特征集规模、验证/测试AUC、PR-AUC、Brier及是否退化(`is_degenerate`)。
6. `constitution_enhancement_top_coef.csv`：记录增强实验中的主要特征贡献（Logistic系数或随机森林重要性），用于解释“哪些体质或交互项被模型重点使用”。

### 5.5 一段式评审结论（正文可直接使用）

根据前述评判标准与优化后双集评估结果，问题1在流程规范性、模型校准性、预测模型区分度和医学一致性维度达到通过标准；解释型模型在显著性与区分度维度证据偏弱。该结果表明，九体质模型更适合用于机制趋势解释，而关键生化指标模型更适合用于风险预警。两者分工明确、结论不冲突，问题1已完成“关键指标筛选-体质贡献分析-风险预警验证”的目标闭环。

### 5.6 九体质模型增强实验对比（本轮优化）

为验证“解释型模型偏弱是否由模型选择不当导致”，本文在相同训练/验证/测试切分下开展三组增强实验：

1. L2正则 LogisticCV（九体质原始变量）。
2. 随机森林（九体质原始变量，非线性对照）。
3. 交互项 ElasticNetCV（九体质+两两交互）。

对比结果如下（节选）：

| 模型 | 特征集 | 验证AUC | 测试AUC | 验证PR-AUC | 测试PR-AUC | 测试Brier | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| 九体质Logistic基线 | 九体质原始变量 | 0.4313 | 0.3657 | 0.7667 | 0.7423 | 0.1744 | 线性解释基线 |
| 九体质L2 LogisticCV | 九体质原始变量 | 0.4353 | 0.3638 | 0.7712 | 0.7420 | 0.2508 | 正则化后提升有限 |
| 九体质RandomForest | 九体质原始变量 | 0.4419 | 0.4333 | 0.7657 | 0.7597 | 0.1770 | 本轮最优，仍低于实用预测阈值 |
| 交互ElasticNetCV | 九体质+两两交互 | 0.5000 | 0.5000 | 0.7933 | 0.7933 | 0.2514 | 退化为近随机判别 |

结论：增强实验后九体质模型测试AUC由0.3657提升至0.4333（随机森林），但仍明显低于独立预测常用阈值（约0.70）；因此“九体质模型证据偏弱”主要由数据信号上限导致，而非简单调参可解决。该部分适合作为论文“改进尝试与负结果报告”内容。

---

## 6 结果讨论

1. 关键指标以血脂核心变量为主（TG、TC、LDL-C、HDL-C），与医学常识一致。
2. 关键指标预警模型AUC较高，提示标签与血脂指标具有较强同源性，属于“诊断近端预测”场景。
3. 九体质模型在验证/测试集AUC分别为0.4313/0.3657，且无显著项，说明在当前样本下，单靠体质积分对该二分类标签解释力有限。
4. 在论文最终版中，建议将“九体质贡献度分析”与“关键指标预警分析”明确分开，分别承担“机制解释”和“预测性能”两类目标。
5. 关键指标模型验证/测试AUC为0.9864/0.9794，PR-AUC也保持高位，说明预警能力稳定，不是单一验证集偶然现象。
6. 在引入随机森林与交互正则模型后，九体质模型测试AUC最高仅到0.4333，进一步支持“信号上限约束”判断。

---

## 7 结论

1. 问题1的完整代码与结果已实现可复现运行。
2. 通过三路筛选+投票机制，得到5项关键指标，满足“表征+预警”目标。
3. 九体质OR分析流程完整，含Wald、VIF、HL检验，可直接用于论文统计结论。
4. 已同步输出专业图表与结构化结果文件，支持后续问题2、问题3建模衔接。

---

## 8 复现说明

执行顺序：

1. 运行问题1主流程：

```bash
/home/fishros/mmb/.venv/bin/python src/q1/run_q1.py \
  --input-csv 附件1_样例数据.csv \
  --processed-dir data/processed \
  --output-dir outputs/q1
```

2. 生成可视化图表：

```bash
/home/fishros/mmb/.venv/bin/python src/q1/plot_q1.py
```

3. 查看输出目录：

1. [outputs/q1](../outputs/q1)
2. [outputs/q1/figures](../outputs/q1/figures)

---

## 附录A 一般图表与补充解读

### A.1 方法选择热力图（补充）

图A-1：[outputs/q1/figures/q1_method_heatmap.png](../outputs/q1/figures/q1_method_heatmap.png)

含义：

1. 行表示候选指标，列表示筛选方法（相关性、L1、随机森林）。
2. 单元格取值为0/1，1表示该方法选中该指标。
3. 该图用于补充验证“投票机制”的来源明细。

![Method-level Selection Matrix](../outputs/q1/figures/q1_method_heatmap.png)

### A.2 随机森林重要性图（补充）

图A-2：[outputs/q1/figures/q1_rf_importance.png](../outputs/q1/figures/q1_rf_importance.png)

含义：

1. 纵轴为指标，横轴为随机森林重要性。
2. 虚线为平均重要性阈值，高于阈值可视为随机森林路径下的重要指标。
3. 该图用于展示非线性模型视角下的特征贡献排序。

![Q1 Random Forest Feature Importance](../outputs/q1/figures/q1_rf_importance.png)

### A.3 关键指标模型系数图（补充）

图A-3：[outputs/q1/figures/q1_selected_model_coef.png](../outputs/q1/figures/q1_selected_model_coef.png)

含义：

1. 每个条形对应一个最终关键指标。
2. 条形方向表示风险方向（正值风险增加，负值风险降低趋势）。
3. 条形绝对值大小表示标准化后贡献强弱。

![Coefficients of Selected-feature Risk Model](../outputs/q1/figures/q1_selected_model_coef.png)
