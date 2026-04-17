# 中老年人群高血脂症风险预警及干预方案优化（问题2完成版）

## 摘要

针对问题2，本文构建了融合血脂指标、痰湿体质积分、活动能力评分及基础信息的三级风险预警模型，输出低/中/高风险分层结果，并给出阈值选取依据与高风险核心特征组合。方法上采用随机森林获得个体风险分值，并与临床可解释规则融合形成复合风险指数，再按分位阈值与规则门控联合划分三级风险。结果显示：样本被划分为低风险38例、中风险676例、高风险286例；各层高血脂阳性率分别为0.0000、0.7604、0.9755，呈显著递增。高风险核心组合主要包括“TG异常+TC异常”“痰湿高分+TG异常”“活动能力低+TC异常”等，表明血脂异常、痰湿偏颇与活动能力下降的叠加效应是高风险识别的关键。

关键词：三级风险分层；痰湿体质；复合风险指数；特征组合挖掘；高血脂预警

---

## 1 问题2目标与交付

### 1.1 目标

1. 构建可输出低/中/高三级风险的预警模型。
2. 明确三级风险阈值选取依据（概率阈值、复合指数阈值、临床规则阈值）。
3. 识别痰湿体质高风险人群核心特征组合，并给出解释。

### 1.2 交付文件

1. 风险预测明细：[outputs/q2/q2_risk_predictions.csv](../outputs/q2/q2_risk_predictions.csv)
2. 阈值与规则依据：[outputs/q2/q2_thresholds.json](../outputs/q2/q2_thresholds.json)
3. 风险层汇总：[outputs/q2/q2_risk_tier_summary.csv](../outputs/q2/q2_risk_tier_summary.csv)
4. 验证集分层汇总：[outputs/q2/q2_risk_tier_summary_val.csv](../outputs/q2/q2_risk_tier_summary_val.csv)
5. 测试集分层汇总：[outputs/q2/q2_risk_tier_summary_test.csv](../outputs/q2/q2_risk_tier_summary_test.csv)
6. 特征重要性：[outputs/q2/q2_feature_importance.csv](../outputs/q2/q2_feature_importance.csv)
7. 高风险核心组合：[outputs/q2/q2_high_risk_core_combos.csv](../outputs/q2/q2_high_risk_core_combos.csv)
8. 汇总结果：[outputs/q2/q2_summary.json](../outputs/q2/q2_summary.json)
9. 图表目录：[outputs/q2/figures](../outputs/q2/figures)
10. 多随机种子稳健性：[outputs/q2/q2_robustness_seed_repeat.csv](../outputs/q2/q2_robustness_seed_repeat.csv)
11. 特征消融结果：[outputs/q2/q2_ablation_results.csv](../outputs/q2/q2_ablation_results.csv)
12. 稳健性汇总：[outputs/q2/q2_robustness_summary.json](../outputs/q2/q2_robustness_summary.json)
13. 概率校准分箱表：[outputs/q2/q2_calibration_table.csv](../outputs/q2/q2_calibration_table.csv)
14. 概率校准汇总：[outputs/q2/q2_calibration_summary.json](../outputs/q2/q2_calibration_summary.json)
15. 分层阳性率Bootstrap区间：[outputs/q2/q2_tier_bootstrap_ci.csv](../outputs/q2/q2_tier_bootstrap_ci.csv)

---

## 2 方法与阈值设计

### 2.1 多维融合特征

模型输入包含四类信息：

1. 核心血脂及代谢：TC、TG、LDL-C、HDL-C、空腹血糖、血尿酸、BMI。
2. 体质信息：九体质积分（含痰湿质）。
3. 活动能力：活动量表总分（ADL+IADL）。
4. 基础特征：年龄组、性别、吸烟史、饮酒史。

并构造血脂异常计数变量 `abnormal_lipid_count` 作为可解释增强特征。

### 2.2 预警模型

采用随机森林得到个体风险分值 $p_i$，并在验证集评估AUC、PR-AUC、Brier分数。

### 2.3 三级风险分层：复合风险指数

为避免单一模型分值主导，定义复合风险指数：

$$
R_i=0.45\cdot\frac{L_i}{4}+0.25\cdot\frac{T_i}{100}+0.20\cdot\left(1-\frac{A_i}{100}\right)+0.10\cdot p_i
$$

其中：

1. $L_i$：血脂异常计数（0-4）。
2. $T_i$：痰湿质积分。
3. $A_i$：活动量表总分。
4. $p_i$：模型风险分值。

### 2.4 阈值选取依据

#### 2.4.1 概率阈值（模型层）

1. 在验证集上用Youden指数得到最优二分类阈值 $t^*=0.9632$。
2. 设 $t_{low}=t^*-0.15=0.8132$，$t_{high}=\min(0.95,t^*+0.15)=0.95$。

#### 2.4.2 复合风险指数阈值（分层层）

1. 以训练集复合指数分位数设阈：
2. `index_low` = 35%分位数 = 0.3830。
3. `index_high` = 75%分位数 = 0.5100。

#### 2.4.3 临床规则阈值（门控层）

1. 痰湿高分阈值 `tan_high=56`，重度阈值 `tan_very_high=60`。
2. 低活动阈值 `activity_low=42`，较好活动阈值 `activity_good=55`。
3. 血脂异常判据：TC>6.2、TG>1.7、LDL-C>3.1、HDL-C<1.04。

### 2.5 三级风险规则

1. 高风险：满足 `R_i >= index_high`，或满足任一高风险临床门控规则。
2. 低风险：满足 `R_i < index_low` 且同时满足低风险门控规则。
3. 中风险：其余样本。

### 2.6 模型好坏验证设计（稳健性）

为验证模型是否“过拟合或偶然高分”，增加两类稳健性实验：

1. 多随机种子重复：在10个随机种子（42-51）下重复训练与评估，观察AUC、PR-AUC、Brier波动。
2. 特征消融实验：分别去除核心血脂、去除体质特征、去除活动特征、去除血脂异常计数，比较性能变化。
3. 概率校准评估：按分箱统计预测概率与真实发生率，计算ECE/MCE。
4. 分层区间估计：对各风险层阳性率进行Bootstrap（1000次）估计95%置信区间。

对应脚本：`src/q2/validate_q2.py`。

---

## 3 结果

### 3.1 模型性能

1. 验证集AUC：1.0000。
2. 测试集AUC：1.0000。
3. 验证集PR-AUC：1.0000。
4. 测试集PR-AUC：1.0000。

说明：该数据中标签与血脂指标存在强同源关系，因此区分性能极高，属于诊断近端预警场景。

### 3.2 三级风险分层结果

风险层汇总表见 [outputs/q2/q2_risk_tier_summary.csv](../outputs/q2/q2_risk_tier_summary.csv)，并补充验证集与测试集分层汇总分别见 [outputs/q2/q2_risk_tier_summary_val.csv](../outputs/q2/q2_risk_tier_summary_val.csv)、[outputs/q2/q2_risk_tier_summary_test.csv](../outputs/q2/q2_risk_tier_summary_test.csv)。

1. 低风险：38例，阳性率0.0000。
2. 中风险：676例，阳性率0.7604。
3. 高风险：286例，阳性率0.9755。

### 3.3 高风险核心特征组合

核心组合表见 [outputs/q2/q2_high_risk_core_combos.csv](../outputs/q2/q2_high_risk_core_combos.csv)。代表性组合如下：

1. TG异常 + TC异常（高风险支持度47.04%，lift=2.39）。
2. TG异常 + LDL异常（支持度31.85%，lift=2.55）。
3. 痰湿高分 + TG异常（支持度31.85%，lift=2.41）。
4. 活动能力低 + TC异常（支持度26.67%，lift=2.08）。
5. TG异常 + TC异常 + 饮酒史（三元组合，支持度24.44%，lift=2.40）。

结论：高风险并非单一特征触发，而是“血脂异常叠加 + 痰湿偏颇 + 低活动/行为因素”共同驱动。

### 3.4 稳健性验证结果

#### 3.4.1 多随机种子重复结果

结果文件见 [outputs/q2/q2_robustness_seed_repeat.csv](../outputs/q2/q2_robustness_seed_repeat.csv) 与 [outputs/q2/q2_robustness_summary.json](../outputs/q2/q2_robustness_summary.json)。

1. 测试集AUC均值=1.0000，标准差=0.0000。
2. 测试集PR-AUC均值=1.0000，标准差约0。
3. 测试集Brier均值=0.00169，标准差=0.00038。

说明：模型在不同随机切分下表现极其稳定。

#### 3.4.2 特征消融结果

结果文件见 [outputs/q2/q2_ablation_results.csv](../outputs/q2/q2_ablation_results.csv)。

1. 全模型测试AUC=1.0000。
2. 去除核心血脂后测试AUC仍为1.0000，但Brier由0.00231上升至0.00756。
3. 去除血脂异常计数后测试AUC仍为1.0000，但Brier上升至0.01154。

解释：该数据集中可替代特征强，AUC出现“顶格效应”；因此判断模型优劣不能只看AUC，还应结合Brier和分层稳定性。

### 3.5 概率校准结果（ECE/MCE）

结果文件见 [outputs/q2/q2_calibration_summary.json](../outputs/q2/q2_calibration_summary.json) 与 [outputs/q2/q2_calibration_table.csv](../outputs/q2/q2_calibration_table.csv)。

1. 验证集ECE=0.0172，MCE=0.0894。
2. 测试集ECE=0.0192，MCE=0.1086。
3. 分箱结果显示高概率区（接近1）预测与真实事件率差距较小，低概率区存在可接受的偏差。

说明：ECE低于0.02，表明模型概率输出具备较好校准性，适合用于风险分层阈值解释。

### 3.6 分层阳性率Bootstrap区间

结果文件见 [outputs/q2/q2_tier_bootstrap_ci.csv](../outputs/q2/q2_tier_bootstrap_ci.csv)。

1. 训练集中风险阳性率95%CI约为[0.7324, 0.8071]，高风险约为[0.9521, 0.9947]。
2. 验证集中风险阳性率95%CI约为[0.6346, 0.8077]，高风险样本全阳性（区间收敛至1）。
3. 测试集中风险阳性率95%CI约为[0.6556, 0.8222]，高风险约为[0.9074, 1.0000]。

说明：尽管验证/测试样本量较小导致区间较宽，但“中风险 < 高风险”关系在各数据切分下保持稳定。

---

## 4 图表与解读

### 4.1 正文重点图

1. 风险层样本分布图：[outputs/q2/figures/q2_risk_tier_distribution.png](../outputs/q2/figures/q2_risk_tier_distribution.png)
2. 风险分值箱线图（含阈值线）：[outputs/q2/figures/q2_risk_score_boxplot.png](../outputs/q2/figures/q2_risk_score_boxplot.png)
3. 高风险核心组合图：[outputs/q2/figures/q2_high_risk_core_combos.png](../outputs/q2/figures/q2_high_risk_core_combos.png)

![Q2 Risk Tier Distribution](../outputs/q2/figures/q2_risk_tier_distribution.png)

图4-1 低中高风险样本量分布。横轴为风险层级，纵轴为样本数；用于展示分层规模结构。

![Q2 Risk Score by Tier](../outputs/q2/figures/q2_risk_score_boxplot.png)

图4-2 各风险层模型分值分布。横轴为风险层级，纵轴为风险分值；虚线表示概率参考阈值，体现分层与风险分值的一致性。

![Q2 Core Feature Combinations in High-risk Group](../outputs/q2/figures/q2_high_risk_core_combos.png)

图4-3 高风险核心组合支持度。横轴为组合在高风险样本内支持度，纵轴为特征组合；用于回答“高风险由哪些特征叠加形成”。

### 4.2 附录图

1. 特征重要性图：[outputs/q2/figures/q2_feature_importance_top12.png](../outputs/q2/figures/q2_feature_importance_top12.png)

---

## 5 表格字段释义（每行每列含义）

### 5.1 `q2_risk_predictions.csv`

1. 每一行：一个样本。
2. `样本ID`：个体唯一编号。
3. `高血脂症二分类标签`：真实标签0/1。
4. `risk_score`：模型预测风险分值。
5. `risk_index`：复合风险指数。
6. `risk_level`：三级风险输出。
7. `rule_hit_high_*`：高风险临床规则命中标记（0/1）。

### 5.2 `q2_risk_tier_summary.csv`

1. 每一行：一个风险层（低/中/高）。
2. `sample_count`：该层样本数。
3. `positive_count`：该层标签为1的数量。
4. `positive_rate`：该层阳性率。
5. `mean_score`：该层平均模型分值。

补充：`q2_risk_tier_summary_val.csv` 与 `q2_risk_tier_summary_test.csv` 结构相同，分别用于验证集与测试集分层有效性评估。

### 5.3 `q2_thresholds.json`

1. `probability_threshold`：模型概率阈值及依据。
2. `risk_index_threshold`：复合指数阈值、公式与依据。
3. `clinical_threshold`：痰湿、活动、血脂异常阈值。
4. `tier_rules`：低/中/高风险分层规则文本。

### 5.4 `q2_feature_importance.csv`

1. 每一行：一个模型输入特征。
2. `importance`：随机森林特征重要性。

### 5.5 `q2_high_risk_core_combos.csv`

1. 每一行：一个高风险核心组合（2元或3元）。
2. `combo`：组合名称。
3. `combo_size`：组合规模（2或3）。
4. `support_high`：在高风险组内支持度。
5. `support_all`：在全样本支持度。
6. `lift`：相对提升度，越高说明越偏向高风险。

### 5.6 `q2_robustness_seed_repeat.csv`

1. 每一行：一个随机种子下的重复实验结果。
2. `seed`：随机种子编号。
3. `val_auc`、`test_auc`：验证/测试AUC。
4. `val_pr_auc`、`test_pr_auc`：验证/测试PR-AUC。
5. `val_brier`、`test_brier`：验证/测试Brier分数。

### 5.7 `q2_ablation_results.csv`

1. 每一行：一个消融实验。
2. `experiment`：实验名称（full_model、remove_core_lipids等）。
3. `n_features`：该实验使用特征数。
4. 其余性能列与主实验一致，用于比较性能下降幅度。

### 5.8 `q2_robustness_summary.json`

1. `seed_repeat`：多随机种子结果的均值/标准差/极值。
2. `ablation`：关键消融结论（如去除核心血脂后性能变化）。
3. `conclusion_hint`：稳健性解释提示语。

### 5.9 `q2_calibration_table.csv`

1. 每一行：一个概率分箱（按验证/测试分别统计）。
2. `bin`：概率区间。
3. `n`：分箱样本数。
4. `mean_pred`：分箱平均预测概率。
5. `event_rate`：分箱真实发生率。
6. `abs_gap`：校准误差绝对值 $|mean\_pred-event\_rate|$。
7. `weight`：分箱权重（样本占比）。
8. `split`：数据切分（val/test）。

### 5.10 `q2_calibration_summary.json`

1. `val.ece`、`test.ece`：期望校准误差（越低越好）。
2. `val.mce`、`test.mce`：最大校准误差（反映最差分箱偏差）。
3. `n_bins_effective`：有效分箱数。

### 5.11 `q2_tier_bootstrap_ci.csv`

1. 每一行：某个数据切分(train/val/test)下某个风险层的统计结果。
2. `positive_rate`：阳性率点估计。
3. `ci95_low`、`ci95_high`：95%置信区间上下界。
4. `ci95_width`：区间宽度，用于反映估计稳定性。
5. `bootstrap_n`：Bootstrap重采样次数。

---

## 6 论文可直接引用的结论段

问题2采用“模型风险分值+临床可解释规则”的双层分层策略，构建了低、中、高三级风险预警体系。阈值设置上，概率阈值由验证集Youden指数确定，复合风险指数阈值由训练集分位数确定，临床门控规则由痰湿积分、活动能力和血脂异常标准共同约束。结果显示，三层风险阳性率呈阶梯式上升（低0.0000、中0.7604、高0.9755），并识别出“TG异常+TC异常”“痰湿高分+TG异常”“活动能力低+TC异常”等高风险核心组合，说明高风险形成机制具有明显的多因素叠加特征。

稳健性验证进一步表明：在10个随机种子下模型指标几乎无波动；消融实验中AUC因数据顶格效应保持1.0，但Brier对特征删减敏感，提示模型校准质量会下降。因此本研究采用“区分能力+AUC、概率质量+Brier、分层有效性”三维联合评价模型好坏。

进一步地，校准评估显示验证/测试ECE分别为0.0172/0.0192，Bootstrap区间结果显示各切分下高风险层阳性率始终高于中风险层，支持分层结论的稳定性与概率解释性。

---

## 7 复现命令

```bash
/home/fishros/mmb/.venv/bin/python src/q2/run_q2.py \
  --input-csv 附件1_样例数据.csv \
  --output-dir outputs/q2

/home/fishros/mmb/.venv/bin/python src/q2/plot_q2.py

/home/fishros/mmb/.venv/bin/python src/q2/validate_q2.py \
  --input-csv 附件1_样例数据.csv \
  --output-dir outputs/q2 \
  --seed-start 42 \
  --seed-count 10
```
