%% MathorCup C题 - MATLAB专业论文图批量生成脚本
% 兼容: MATLAB R2024a
% 用法: 在项目根目录执行: run('src/matlab/plot_professional_all.m')

clear; clc;

projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
outDir = fullfile(projectRoot, 'outputs', 'matlab_figures');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

fontName = chooseChineseFont();
set(groot, 'defaultAxesFontName', fontName, ...
    'defaultTextFontName', fontName, ...
    'defaultAxesFontSize', 11, ...
    'defaultLineLineWidth', 1.6, ...
    'defaultAxesLineWidth', 1.0, ...
    'defaultFigureColor', 'w');

fprintf('使用字体: %s\n', fontName);

%% ========== Q1 专业图 ==========
q1Dir = fullfile(projectRoot, 'outputs', 'q1');

tFS = readtable(fullfile(q1Dir, 'feature_selection_details.csv'), ...
    'VariableNamingRule', 'preserve', 'TextType', 'string');
tOR = readtable(fullfile(q1Dir, 'OR_values_table.csv'), ...
    'VariableNamingRule', 'preserve', 'TextType', 'string');
tCoef = readtable(fullfile(q1Dir, 'selected_feature_model_coef.csv'), ...
    'VariableNamingRule', 'preserve', 'TextType', 'string');

plotQ1VoteLollipop(tFS, outDir);
plotQ1ORForest(tOR, outDir);
plotQ1CoefWaterfall(tCoef, outDir);

%% ========== Q2 专业图 ==========
q2Dir = fullfile(projectRoot, 'outputs', 'q2');

tTier = readtable(fullfile(q2Dir, 'q2_risk_tier_summary.csv'), ...
    'VariableNamingRule', 'preserve', 'TextType', 'string');
tPred = readtable(fullfile(q2Dir, 'q2_risk_predictions.csv'), ...
    'VariableNamingRule', 'preserve', 'TextType', 'string');
tImp = readtable(fullfile(q2Dir, 'q2_feature_importance.csv'), ...
    'VariableNamingRule', 'preserve', 'TextType', 'string');
tCal = readtable(fullfile(q2Dir, 'q2_calibration_table.csv'), ...
    'VariableNamingRule', 'preserve', 'TextType', 'string');
tBoot = readtable(fullfile(q2Dir, 'q2_tier_bootstrap_ci.csv'), ...
    'VariableNamingRule', 'preserve', 'TextType', 'string');

plotQ2TierAndPositiveRate(tTier, outDir);
plotQ2ScoreViolin(tPred, outDir);
plotQ2ImportanceHorizontal(tImp, outDir);
plotQ2CalibrationCurve(tCal, outDir);
plotQ2BootstrapCI(tBoot, outDir);

%% ========== Q3 专业图 ==========
q3Dir = fullfile(projectRoot, 'outputs', 'q3');

tPlan = readtable(fullfile(q3Dir, 'q3_patient_optimal_plans.csv'), ...
    'VariableNamingRule', 'preserve', 'TextType', 'string');
tSample = readtable(fullfile(q3Dir, 'q3_sample_1_2_3_optimal_plan.csv'), ...
    'VariableNamingRule', 'preserve', 'TextType', 'string');
tSens = readtable(fullfile(q3Dir, 'q3_sensitivity_summary.csv'), ...
    'VariableNamingRule', 'preserve', 'TextType', 'string');

plotQ3Pareto(tPlan, outDir);
plotQ3TrajectoryRibbon(tSample, outDir);
plotQ3SensitivityBars(tSens, outDir);

fprintf('已完成: MATLAB专业图输出目录 -> %s\n', outDir);

%% ======================== 局部函数区 ========================

function fontName = chooseChineseFont()
fonts = string(listfonts);
candidates = ["Microsoft YaHei", "SimHei", "PingFang SC", "Heiti SC", ...
    "Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", ...
    "Arial Unicode MS", "DejaVu Sans"];
idx = find(ismember(candidates, fonts), 1, 'first');
if isempty(idx)
    fontName = 'DejaVu Sans';
else
    fontName = candidates(idx);
end
end

function exportPubFig(fig, outPath)
set(fig, 'Renderer', 'painters');
exportgraphics(fig, outPath, 'Resolution', 600);
end

function ax = beautifyAxes(ax)
ax.Box = 'off';
ax.TickDir = 'out';
ax.GridAlpha = 0.22;
ax.MinorGridAlpha = 0.12;
ax.XGrid = 'on';
ax.YGrid = 'on';
end

function labels = mapFeatureLabels(labels)
old = ["TG（甘油三酯）", "TC（总胆固醇）", "LDL-C（低密度脂蛋白）", ...
    "HDL-C（高密度脂蛋白）", "活动量表总分（ADL总分+IADL总分）"];
new = ["TG", "TC", "LDL-C", "HDL-C", "活动总分"];
for i = 1:numel(old)
    labels = replace(labels, old(i), new(i));
end
end

function plotQ1VoteLollipop(tFS, outDir)
[~, idx] = sortrows([double(tFS.votes), double(tFS.rf_importance)], [-1 -1]);
t = tFS(idx, :);
feat = mapFeatureLabels(string(t.feature));
v = double(t.votes);

fig = figure('Position', [80, 80, 980, 560]);
ax = axes(fig); hold(ax, 'on');
for i = 1:numel(v)
    x = [0, v(i)];
    y = [i, i];
    plot(ax, x, y, '-', 'Color', [0.72 0.76 0.82], 'LineWidth', 2.2);
end
isSel = logical(t.final_selected);
scatter(ax, v(~isSel), find(~isSel), 85, 'MarkerFaceColor', [0.67 0.72 0.78], ...
    'MarkerEdgeColor', [0.3 0.35 0.4], 'LineWidth', 1.2);
scatter(ax, v(isSel), find(isSel), 95, 'MarkerFaceColor', [0.08 0.49 0.45], ...
    'MarkerEdgeColor', [0.02 0.27 0.25], 'LineWidth', 1.2);

ax.YTick = 1:numel(feat);
ax.YTickLabel = feat;
ax.XLim = [0 3.3];
ax.XTick = 0:1:3;
ax.YDir = 'reverse';
xlabel(ax, '投票数（三方法集成）');
ylabel(ax, '候选指标');
title(ax, 'Q1 关键指标投票结果（棒棒糖图）', 'FontWeight', 'bold');
legend(ax, {'投票连线', '未入选', '入选'}, 'Location', 'southeast');
beautifyAxes(ax);
exportPubFig(fig, fullfile(outDir, 'q1_vote_lollipop_matlab.png'));
close(fig);
end

function plotQ1ORForest(tOR, outDir)
t = tOR(~strcmp(string(tOR.variable), "const"), :);
var = string(t.variable);
orv = double(t.or);
lo = double(t.or_ci_low);
hi = double(t.or_ci_high);
pv = double(t.p_value);

[~, idx] = sort(orv, 'descend');
var = var(idx); orv = orv(idx); lo = lo(idx); hi = hi(idx); pv = pv(idx);

fig = figure('Position', [80, 80, 980, 580]);
ax = axes(fig); hold(ax, 'on');

for i = 1:numel(orv)
    colorLine = [0.36 0.43 0.51];
    if pv(i) < 0.05
        colorLine = [0.85 0.23 0.2];
    end
    line(ax, [lo(i), hi(i)], [i, i], 'Color', colorLine, 'LineWidth', 2.0);
    scatter(ax, orv(i), i, 65, 'filled', 'MarkerFaceColor', colorLine, ...
        'MarkerEdgeColor', [0.15 0.15 0.15]);
end
xline(ax, 1.0, '--k', 'OR=1', 'LabelVerticalAlignment', 'middle', ...
    'LabelHorizontalAlignment', 'left');
set(ax, 'XScale', 'log');
ax.YTick = 1:numel(var);
ax.YTickLabel = var;
ax.YDir = 'reverse';

xlabel(ax, 'OR（对数尺度）');
ylabel(ax, '体质变量');
title(ax, 'Q1 九体质 OR 森林图（95%CI）', 'FontWeight', 'bold');
beautifyAxes(ax);
exportPubFig(fig, fullfile(outDir, 'q1_or_forest_matlab.png'));
close(fig);
end

function plotQ1CoefWaterfall(tCoef, outDir)
feat = mapFeatureLabels(string(tCoef.feature));
coef = double(tCoef.coef);
[~, idx] = sort(abs(coef), 'descend');
feat = feat(idx); coef = coef(idx);

fig = figure('Position', [80, 80, 920, 530]);
ax = axes(fig); hold(ax, 'on');
colors = repmat([0.12 0.57 0.85], numel(coef), 1);
colors(coef < 0, :) = repmat([0.87 0.31 0.25], sum(coef < 0), 1);

b = barh(ax, coef, 'FaceColor', 'flat', 'EdgeColor', [0.18 0.2 0.22]);
b.CData = colors;
yline(ax, 0, 'w');
xline(ax, 0, '--k', 'LineWidth', 1.0);
ax.YTick = 1:numel(feat);
ax.YTickLabel = feat;
ax.YDir = 'reverse';
xlabel(ax, '标准化系数');
ylabel(ax, '关键指标');
title(ax, 'Q1 关键指标模型系数贡献图', 'FontWeight', 'bold');
beautifyAxes(ax);
exportPubFig(fig, fullfile(outDir, 'q1_coef_contrib_matlab.png'));
close(fig);
end

function plotQ2TierAndPositiveRate(tTier, outDir)
order = ["低风险", "中风险", "高风险"];
[~, idx] = ismember(string(tTier.risk_level), order);
t = tTier(idx > 0, :);
[~, j] = sort(idx(idx > 0));
t = t(j, :);

levels = string(t.risk_level);
n = double(t.sample_count);
pr = double(t.positive_rate);

fig = figure('Position', [80, 80, 980, 560]);
ax1 = axes(fig); hold(ax1, 'on');
bar(ax1, categorical(levels, order), n, 0.58, 'FaceColor', [0.24 0.56 0.83], ...
    'EdgeColor', [0.16 0.2 0.27]);
ylabel(ax1, '样本数');

ax2 = axes('Position', ax1.Position, 'Color', 'none');
h2 = plot(ax2, 1:numel(levels), pr * 100, '-o', 'LineWidth', 2.2, ...
    'Color', [0.85 0.33 0.10], 'MarkerFaceColor', [0.85 0.33 0.10]);
ylabel(ax2, '阳性率（%）');
ax2.XLim = ax1.XLim;
ax2.XTick = 1:numel(levels);
ax2.XTickLabel = {};
ax2.YAxisLocation = 'right';
ax2.Box = 'off';

for i = 1:numel(levels)
    text(ax2, i, pr(i) * 100 + 1.8, sprintf('%.1f%%', pr(i) * 100), ...
        'HorizontalAlignment', 'center', 'Color', [0.85 0.33 0.10], 'FontSize', 10);
end

title(ax1, 'Q2 风险层规模与阳性率双轴图', 'FontWeight', 'bold');
grid(ax1, 'on'); ax1.GridAlpha = 0.2; ax1.Box = 'off';
legend(ax2, h2, {'阳性率'}, 'Location', 'northwest');
exportPubFig(fig, fullfile(outDir, 'q2_tier_positive_dualaxis_matlab.png'));
close(fig);
end

function plotQ2ScoreViolin(tPred, outDir)
levels = ["低风险", "中风险", "高风险"];
L = string(tPred.risk_level);
S = double(tPred.risk_score);

fig = figure('Position', [80, 80, 980, 560]);
ax = axes(fig); hold(ax, 'on');
for i = 1:numel(levels)
    si = S(L == levels(i));
    if isempty(si)
        continue;
    end
    [f, xi] = ksdensity(si, 'NumPoints', 120, 'Support', [0 1]);
    f = f / max(f) * 0.28;
    patch(ax, [i - f, fliplr(i + f)], [xi, fliplr(xi)], [0.32 0.70 0.52], ...
        'FaceAlpha', 0.28, 'EdgeColor', [0.15 0.34 0.24], 'LineWidth', 1.1);
    med = median(si, 'omitnan');
    q1 = quantile(si, 0.25);
    q3 = quantile(si, 0.75);
    plot(ax, [i - 0.15, i + 0.15], [med, med], '-', 'Color', [0.10 0.10 0.10], 'LineWidth', 2.2);
    plot(ax, [i, i], [q1, q3], '-', 'Color', [0.10 0.10 0.10], 'LineWidth', 2.2);
end
xlim(ax, [0.5 3.5]);
xticks(ax, 1:3);
xticklabels(ax, cellstr(levels));
ylim(ax, [0 1]);
ylabel(ax, '模型风险分值');
xlabel(ax, '风险层级');
title(ax, 'Q2 风险分值分布（小提琴 + 四分位）', 'FontWeight', 'bold');
beautifyAxes(ax);
exportPubFig(fig, fullfile(outDir, 'q2_score_violin_matlab.png'));
close(fig);
end

function plotQ2ImportanceHorizontal(tImp, outDir)
t = sortrows(tImp, 'importance', 'descend');
t = t(1:min(12, height(t)), :);
feat = mapFeatureLabels(string(t.feature));
imp = double(t.importance);

[imp, idx] = sort(imp, 'ascend');
feat = feat(idx);

fig = figure('Position', [80, 80, 980, 600]);
ax = axes(fig);
b = barh(ax, imp, 0.65, 'FaceColor', [0.30 0.46 0.88], ...
    'EdgeColor', [0.12 0.18 0.35]);
for i = 1:numel(imp)
    text(ax, imp(i) + 0.005, i, sprintf('%.3f', imp(i)), ...
        'VerticalAlignment', 'middle', 'FontSize', 9);
end
ax.YTick = 1:numel(feat);
ax.YTickLabel = feat;
xlabel(ax, '重要性');
ylabel(ax, '特征');
title(ax, 'Q2 特征重要性 Top12（论文版）', 'FontWeight', 'bold');
beautifyAxes(ax);
exportPubFig(fig, fullfile(outDir, 'q2_importance_top12_matlab.png'));
close(fig);
end

function plotQ2CalibrationCurve(tCal, outDir)
% 取测试集优先，其次验证集
splits = unique(string(tCal.split));
if any(splits == "test")
    t = tCal(string(tCal.split) == "test", :);
else
    t = tCal(string(tCal.split) == splits(1), :);
end
x = double(t.mean_pred);
y = double(t.event_rate);
w = double(t.n);

fig = figure('Position', [80, 80, 720, 620]);
ax = axes(fig); hold(ax, 'on');
plot(ax, [0 1], [0 1], '--', 'Color', [0.25 0.25 0.25], 'LineWidth', 1.3);
scatter(ax, x, y, 24 + 2.5 * w, 'filled', 'MarkerFaceColor', [0.17 0.50 0.83], ...
    'MarkerFaceAlpha', 0.75, 'MarkerEdgeColor', [0.08 0.22 0.40]);
plot(ax, x, y, '-', 'Color', [0.17 0.50 0.83], 'LineWidth', 1.8);

xlabel(ax, '预测概率（分箱均值）');
ylabel(ax, '观测阳性率');
title(ax, sprintf('Q2 校准曲线（%s集）', char(string(t.split(1)))), 'FontWeight', 'bold');
axis(ax, [0 1 0 1]); axis(ax, 'square');
beautifyAxes(ax);
exportPubFig(fig, fullfile(outDir, 'q2_calibration_curve_matlab.png'));
close(fig);
end

function plotQ2BootstrapCI(tBoot, outDir)
% 仅展示 test 集，其次 val 集
sp = unique(string(tBoot.split));
if any(sp == "test")
    t = tBoot(string(tBoot.split) == "test", :);
elseif any(sp == "val")
    t = tBoot(string(tBoot.split) == "val", :);
else
    t = tBoot(string(tBoot.split) == sp(1), :);
end

order = ["低风险", "中风险", "高风险"];
[~, idx] = ismember(string(t.risk_level), order);
t = t(idx > 0, :);
[~, j] = sort(idx(idx > 0));
t = t(j, :);

lev = string(t.risk_level);
rate = double(t.positive_rate);
lo = double(t.ci95_low);
hi = double(t.ci95_high);

fig = figure('Position', [80, 80, 760, 480]);
ax = axes(fig); hold(ax, 'on');
for i = 1:numel(rate)
    line(ax, [lo(i), hi(i)] * 100, [i, i], 'Color', [0.27 0.33 0.38], 'LineWidth', 2.2);
    scatter(ax, rate(i) * 100, i, 70, 'filled', 'MarkerFaceColor', [0.82 0.30 0.25], ...
        'MarkerEdgeColor', [0.2 0.2 0.2]);
end
ax.YTick = 1:numel(lev);
ax.YTickLabel = lev;
ax.YDir = 'reverse';
xlabel(ax, '阳性率（%）');
ylabel(ax, '风险层级');
title(ax, 'Q2 分层阳性率 Bootstrap 95%CI', 'FontWeight', 'bold');
beautifyAxes(ax);
exportPubFig(fig, fullfile(outDir, 'q2_bootstrap_ci_matlab.png'));
close(fig);
end

function plotQ3Pareto(tPlan, outDir)
x = double(tPlan.total_cost_6m);
y = double(tPlan.tan_reduction_rate) * 100;
s = double(tPlan.frequency_per_week);
c = double(tPlan.activity_intensity);

fig = figure('Position', [80, 80, 980, 560]);
ax = axes(fig); hold(ax, 'on');
sc = scatter(ax, x, y, 30 + 11 * s, c, 'filled', ...
    'MarkerFaceAlpha', 0.75, 'MarkerEdgeColor', [0.18 0.2 0.22]);
colormap(ax, turbo(3));
cb = colorbar(ax);
cb.Label.String = '运动强度';

xline(ax, 2000, '--r', '预算上限 2000', 'LineWidth', 1.3, ...
    'LabelVerticalAlignment', 'middle', 'LabelHorizontalAlignment', 'left');

xlabel(ax, '6个月总成本');
ylabel(ax, '痰湿降幅率（%）');
title(ax, 'Q3 成本-疗效帕累托散点图', 'FontWeight', 'bold');
beautifyAxes(ax);
exportPubFig(fig, fullfile(outDir, 'q3_pareto_scatter_matlab.png'));
close(fig);
end

function plotQ3TrajectoryRibbon(tSample, outDir)
fig = figure('Position', [80, 80, 920, 560]);
ax = axes(fig); hold(ax, 'on');

sampleIDs = double(tSample.sample_id);
cmap = lines(numel(sampleIDs));
for i = 1:numel(sampleIDs)
    traj = string(tSample.trajectory(i));
    parts = split(traj, ';');
    vals = str2double(parts);
    m = 0:(numel(vals)-1);
    plot(ax, m, vals, '-o', 'Color', cmap(i, :), 'MarkerFaceColor', cmap(i, :), ...
        'LineWidth', 2.2, 'DisplayName', sprintf('样本 %d', sampleIDs(i)));
end
xlabel(ax, '月份');
ylabel(ax, '预测痰湿积分');
title(ax, 'Q3 样本1/2/3 干预轨迹图', 'FontWeight', 'bold');
legend(ax, 'Location', 'northeast');
beautifyAxes(ax);
exportPubFig(fig, fullfile(outDir, 'q3_sample_trajectory_matlab.png'));
close(fig);
end

function plotQ3SensitivityBars(tSens, outDir)
scenario = string(tSens.scenario);
drop = double(tSens.mean_tan_reduction_rate) * 100;
cost = double(tSens.mean_total_cost_6m);

fig = figure('Position', [80, 80, 980, 560]);
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

ax1 = nexttile(tl, 1);
bar(ax1, categorical(scenario), drop, 0.62, 'FaceColor', [0.18 0.64 0.52], ...
    'EdgeColor', [0.10 0.25 0.21]);
title(ax1, '平均痰湿降幅率对比');
ylabel(ax1, '降幅率（%）');
beautifyAxes(ax1);

ax2 = nexttile(tl, 2);
bar(ax2, categorical(scenario), cost, 0.62, 'FaceColor', [0.30 0.50 0.83], ...
    'EdgeColor', [0.12 0.20 0.36]);
title(ax2, '平均6个月成本对比');
ylabel(ax2, '成本');
beautifyAxes(ax2);

title(tl, 'Q3 敏感性情景结果对比（双图）', 'FontWeight', 'bold');
exportPubFig(fig, fullfile(outDir, 'q3_sensitivity_dualbar_matlab.png'));
close(fig);
end
