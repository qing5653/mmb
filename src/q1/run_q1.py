#!/usr/bin/env python3
"""问题1完整可执行脚本。

功能覆盖：
1. 读取原始数据并完成分层切分（train/val/test）。
2. 在训练集上执行三类特征筛选：相关性、Lasso、随机森林。
3. 采用投票机制输出最终关键指标（>=2票）。
4. 建立九种体质 Logistic 回归，输出 OR、95%CI、Wald、VIF、HL检验等。
5. 在验证集评估 Logistic AUC。

运行示例：
python src/q1/run_q1.py \
  --input-csv "附件1_样例数据.csv" \
  --processed-dir "data/processed" \
  --output-dir "outputs/q1"
"""

from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


@dataclass
class ColumnMap:
    sample_id: str
    label: str
    tan_score: str
    constitution_scores: List[str]
    candidate_features: List[str]


def find_column(df: pd.DataFrame, pattern: str) -> str:
    regex = re.compile(pattern)
    for col in df.columns:
        if regex.search(col):
            return col
    raise KeyError(f"未找到匹配列：{pattern}")


def build_column_map(df: pd.DataFrame) -> ColumnMap:
    sample_id = find_column(df, r"样本ID")
    label = find_column(df, r"高血脂症二分类标签")

    constitution_scores = [
        find_column(df, r"^平和质$"),
        find_column(df, r"^气虚质$"),
        find_column(df, r"^阳虚质$"),
        find_column(df, r"^阴虚质$"),
        find_column(df, r"^痰湿质$"),
        find_column(df, r"^湿热质$"),
        find_column(df, r"^血瘀质$"),
        find_column(df, r"^气郁质$"),
        find_column(df, r"^特禀质$"),
    ]
    tan_score = find_column(df, r"^痰湿质$")

    # ADL、IADL与活动总分存在构成关系，候选特征中强制互斥：
    # 优先使用活动总分；若总分字段缺失则退回ADL与IADL。
    adl_col = find_column(df, r"ADL总分")
    iadl_col = find_column(df, r"IADL总分")
    try:
        activity_total_col = find_column(df, r"活动量表总分")
        activity_features = [activity_total_col]
    except KeyError:
        activity_features = [adl_col, iadl_col]

    candidate_features = [
        find_column(df, r"HDL-C"),
        find_column(df, r"LDL-C"),
        find_column(df, r"TG"),
        find_column(df, r"TC"),
        find_column(df, r"空腹血糖"),
        find_column(df, r"血尿酸"),
        find_column(df, r"BMI"),
    ] + activity_features

    return ColumnMap(
        sample_id=sample_id,
        label=label,
        tan_score=tan_score,
        constitution_scores=constitution_scores,
        candidate_features=candidate_features,
    )


def to_numeric_frame(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def split_data(df: pd.DataFrame, label_col: str, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df[label_col],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df[label_col],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def run_correlation_selection(
    train_df: pd.DataFrame,
    features: List[str],
    tan_col: str,
    label_col: str,
) -> pd.DataFrame:
    rows = []
    for f in features:
        sub = train_df[[f, tan_col, label_col]].dropna()
        if len(sub) < 10:
            rows.append(
                {
                    "feature": f,
                    "pearson_r_tan": np.nan,
                    "pearson_p_tan": np.nan,
                    "spearman_r_label": np.nan,
                    "spearman_p_label": np.nan,
                    "corr_selected": False,
                }
            )
            continue

        pearson_r, pearson_p = stats.pearsonr(sub[f], sub[tan_col])
        spear_r, spear_p = stats.spearmanr(sub[f], sub[label_col])

        # 二分类标签与连续变量的相关系数通常偏小，标签侧阈值适当放宽
        pearson_ok = (abs(pearson_r) > 0.20) and (pearson_p < 0.05)
        spearman_ok = (abs(spear_r) > 0.10) and (spear_p < 0.05)
        corr_selected = bool(pearson_ok or spearman_ok)

        rows.append(
            {
                "feature": f,
                "pearson_r_tan": float(pearson_r),
                "pearson_p_tan": float(pearson_p),
                "spearman_r_label": float(spear_r),
                "spearman_p_label": float(spear_p),
                "corr_selected": corr_selected,
            }
        )
    return pd.DataFrame(rows)


def run_l1_logistic_selection(train_df: pd.DataFrame, features: List[str], label_col: str, seed: int) -> pd.DataFrame:
    sub = train_df[features + [label_col]].dropna().copy()
    x = sub[features].values
    y = sub[label_col].astype(int).values

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    clf = LogisticRegressionCV(
        penalty="l1",
        solver="saga",
        Cs=np.logspace(-2, 2, 20),
        cv=10,
        scoring="roc_auc",
        class_weight="balanced",
        random_state=seed,
        max_iter=5000,
        n_jobs=-1,
        refit=True,
    )
    clf.fit(x_scaled, y)
    coef = clf.coef_.ravel()
    c_best = float(clf.C_[0])

    rows = []
    for f, c in zip(features, coef):
        rows.append(
            {
                "feature": f,
                "lasso_coef": float(c),
                "lasso_selected": bool(abs(c) > 1e-6),
                "lasso_alpha": float(1.0 / c_best),
            }
        )
    return pd.DataFrame(rows)


def run_selected_feature_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_features: List[str],
    label_col: str,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if not selected_features:
        return pd.DataFrame(columns=["feature", "coef"]), {
            "val_auc": np.nan,
            "test_auc": np.nan,
            "val_pr_auc": np.nan,
            "test_pr_auc": np.nan,
            "val_brier": np.nan,
            "test_brier": np.nan,
            "n_train": 0,
            "n_val": 0,
            "n_test": 0,
        }

    train_sub = train_df[selected_features + [label_col]].dropna().copy()
    val_sub = val_df[selected_features + [label_col]].dropna().copy()
    test_sub = test_df[selected_features + [label_col]].dropna().copy()

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_sub[selected_features].values)
    y_train = train_sub[label_col].astype(int).values
    x_val = scaler.transform(val_sub[selected_features].values)
    y_val = val_sub[label_col].astype(int).values
    x_test = scaler.transform(test_sub[selected_features].values)
    y_test = test_sub[label_col].astype(int).values

    model = LogisticRegressionCV(
        Cs=np.logspace(-3, 2, 15),
        cv=10,
        scoring="roc_auc",
        penalty="l2",
        solver="lbfgs",
        class_weight="balanced",
        random_state=seed,
        max_iter=3000,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    val_prob = model.predict_proba(x_val)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]

    val_auc = float(roc_auc_score(y_val, val_prob))
    test_auc = float(roc_auc_score(y_test, test_prob))
    val_pr_auc = float(average_precision_score(y_val, val_prob))
    test_pr_auc = float(average_precision_score(y_test, test_prob))
    val_brier = float(brier_score_loss(y_val, val_prob))
    test_brier = float(brier_score_loss(y_test, test_prob))

    coef_df = pd.DataFrame({"feature": selected_features, "coef": model.coef_.ravel()}).sort_values(
        "coef", ascending=False
    )
    metrics = {
        "val_auc": val_auc,
        "test_auc": test_auc,
        "val_pr_auc": val_pr_auc,
        "test_pr_auc": test_pr_auc,
        "val_brier": val_brier,
        "test_brier": test_brier,
        "best_C": float(model.C_[0]),
        "n_train": int(len(train_sub)),
        "n_val": int(len(val_sub)),
        "n_test": int(len(test_sub)),
    }
    return coef_df, metrics


def run_rf_selection(train_df: pd.DataFrame, features: List[str], label_col: str, seed: int) -> pd.DataFrame:
    sub = train_df[features + [label_col]].dropna().copy()
    x = sub[features].values
    y = sub[label_col].astype(int).values

    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf.fit(x, y)
    importances = rf.feature_importances_
    mean_imp = float(np.mean(importances))

    rows = []
    for f, imp in zip(features, importances):
        rows.append(
            {
                "feature": f,
                "rf_importance": float(imp),
                "rf_selected": bool(imp > mean_imp),
                "rf_importance_mean": mean_imp,
            }
        )
    return pd.DataFrame(rows)


def hosmer_lemeshow_test(y_true: np.ndarray, y_prob: np.ndarray, groups: int = 10) -> Tuple[float, float, int]:
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    df = df.dropna().copy()
    if df.empty:
        return np.nan, np.nan, 0

    # qcut 可能因重复值减少分组数
    df["bin"] = pd.qcut(df["p"], q=groups, duplicates="drop")
    agg = df.groupby("bin", observed=True).agg(obs=("y", "sum"), n=("y", "count"), exp=("p", "sum"))
    bins = len(agg)
    if bins < 3:
        return np.nan, np.nan, bins

    eps = 1e-12
    term1 = (agg["obs"] - agg["exp"]) ** 2 / (agg["exp"] + eps)
    term2 = ((agg["n"] - agg["obs"]) - (agg["n"] - agg["exp"])) ** 2 / ((agg["n"] - agg["exp"]) + eps)
    hl_stat = float((term1 + term2).sum())
    df_hl = bins - 2
    p_value = float(stats.chi2.sf(hl_stat, df_hl))
    return hl_stat, p_value, bins


def run_logistic_or_analysis(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    constitution_cols: List[str],
    label_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    train_sub = train_df[constitution_cols + [label_col]].dropna().copy()
    val_sub = val_df[constitution_cols + [label_col]].dropna().copy()
    test_sub = test_df[constitution_cols + [label_col]].dropna().copy()

    x_train = sm.add_constant(train_sub[constitution_cols], has_constant="add")
    y_train = train_sub[label_col].astype(int)

    model = sm.Logit(y_train, x_train)
    result = model.fit(disp=False)

    params = result.params
    bse = result.bse
    pvals = result.pvalues
    conf = result.conf_int()
    conf.columns = ["ci_low_log", "ci_high_log"]

    or_table = pd.DataFrame(
        {
            "variable": params.index,
            "coef": params.values,
            "std_err": bse.values,
            "wald_chi2": (params.values / bse.values) ** 2,
            "p_value": pvals.values,
            "or": np.exp(params.values),
            "or_ci_low": np.exp(conf["ci_low_log"].values),
            "or_ci_high": np.exp(conf["ci_high_log"].values),
        }
    )

    def interpret(row: pd.Series) -> str:
        low, high, or_v = row["or_ci_low"], row["or_ci_high"], row["or"]
        if low > 1:
            return "显著风险因子"
        if high < 1:
            return "显著保护因子"
        if or_v > 1:
            return "风险趋势（未显著）"
        if or_v < 1:
            return "保护趋势（未显著）"
        return "无明显影响"

    or_table["interpretation"] = or_table.apply(interpret, axis=1)

    # VIF
    x_vif = train_sub[constitution_cols].copy()
    vif_rows = []
    for i, col in enumerate(x_vif.columns):
        vif_rows.append({"variable": col, "vif": float(variance_inflation_factor(x_vif.values, i))})
    vif_table = pd.DataFrame(vif_rows)

    # 验证集 AUC
    x_val = sm.add_constant(val_sub[constitution_cols], has_constant="add")
    y_val = val_sub[label_col].astype(int).values
    val_prob = result.predict(x_val)
    val_auc = float(roc_auc_score(y_val, val_prob))

    x_test = sm.add_constant(test_sub[constitution_cols], has_constant="add")
    y_test = test_sub[label_col].astype(int).values
    test_prob = result.predict(x_test)
    test_auc = float(roc_auc_score(y_test, test_prob))
    val_pr_auc = float(average_precision_score(y_val, val_prob))
    test_pr_auc = float(average_precision_score(y_test, test_prob))
    val_brier = float(brier_score_loss(y_val, val_prob))
    test_brier = float(brier_score_loss(y_test, test_prob))

    # Hosmer-Lemeshow
    train_prob = result.predict(x_train)
    hl_stat, hl_p, hl_bins = hosmer_lemeshow_test(y_train.values, train_prob.values, groups=10)

    diagnostics = {
        "n_train": int(len(train_sub)),
        "n_val": int(len(val_sub)),
        "n_test": int(len(test_sub)),
        "log_likelihood": float(result.llf),
        "llr_p_value": float(result.llr_pvalue),
        "pseudo_r2_mcfadden": float(result.prsquared),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "hosmer_lemeshow_stat": hl_stat,
        "hosmer_lemeshow_p": hl_p,
        "hosmer_lemeshow_bins": int(hl_bins),
        "val_auc": val_auc,
        "test_auc": test_auc,
        "val_pr_auc": val_pr_auc,
        "test_pr_auc": test_pr_auc,
        "val_brier": val_brier,
        "test_brier": test_brier,
    }

    return or_table, vif_table, diagnostics


def _compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def _build_constitution_interactions(df: pd.DataFrame, constitution_cols: List[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in constitution_cols:
        out[c] = df[c]
    for a, b in combinations(constitution_cols, 2):
        out[f"{a}*{b}"] = df[a] * df[b]
    return out


def run_constitution_enhancement_experiments(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    constitution_cols: List[str],
    label_col: str,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_sub = train_df[constitution_cols + [label_col]].dropna().copy()
    val_sub = val_df[constitution_cols + [label_col]].dropna().copy()
    test_sub = test_df[constitution_cols + [label_col]].dropna().copy()

    y_train = train_sub[label_col].astype(int).values
    y_val = val_sub[label_col].astype(int).values
    y_test = test_sub[label_col].astype(int).values

    rows: List[Dict[str, float]] = []
    coef_rows: List[Dict[str, float]] = []

    # 实验1：九体质原始变量 + L2正则LogisticCV
    x_train_raw = train_sub[constitution_cols].values
    x_val_raw = val_sub[constitution_cols].values
    x_test_raw = test_sub[constitution_cols].values
    scaler_raw = StandardScaler()
    x_train_raw = scaler_raw.fit_transform(x_train_raw)
    x_val_raw = scaler_raw.transform(x_val_raw)
    x_test_raw = scaler_raw.transform(x_test_raw)

    l2_model = LogisticRegressionCV(
        Cs=np.logspace(-3, 2, 15),
        cv=5,
        scoring="roc_auc",
        penalty="l2",
        solver="lbfgs",
        class_weight="balanced",
        random_state=seed,
        max_iter=4000,
        n_jobs=-1,
        refit=True,
    )
    l2_model.fit(x_train_raw, y_train)
    val_prob_l2 = l2_model.predict_proba(x_val_raw)[:, 1]
    test_prob_l2 = l2_model.predict_proba(x_test_raw)[:, 1]
    val_m_l2 = _compute_binary_metrics(y_val, val_prob_l2)
    test_m_l2 = _compute_binary_metrics(y_test, test_prob_l2)
    rows.append(
        {
            "model": "Constitution L2 LogisticCV",
            "feature_set": "constitution_raw",
            "n_features": int(len(constitution_cols)),
            "best_C": float(l2_model.C_[0]),
            "best_l1_ratio": np.nan,
            "val_auc": val_m_l2["auc"],
            "test_auc": test_m_l2["auc"],
            "val_pr_auc": val_m_l2["pr_auc"],
            "test_pr_auc": test_m_l2["pr_auc"],
            "val_brier": val_m_l2["brier"],
            "test_brier": test_m_l2["brier"],
            "is_degenerate": False,
        }
    )

    for name, coef in zip(constitution_cols, l2_model.coef_.ravel()):
        coef_rows.append(
            {
                "experiment": "Constitution L2 LogisticCV",
                "feature": name,
                "coef": float(coef),
                "abs_coef": float(abs(coef)),
            }
        )

    # 实验1.5：九体质原始变量 + 随机森林（非线性对照）
    rf_model = RandomForestClassifier(
        n_estimators=500,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf_model.fit(train_sub[constitution_cols].values, y_train)
    val_prob_rf = rf_model.predict_proba(val_sub[constitution_cols].values)[:, 1]
    test_prob_rf = rf_model.predict_proba(test_sub[constitution_cols].values)[:, 1]
    val_m_rf = _compute_binary_metrics(y_val, val_prob_rf)
    test_m_rf = _compute_binary_metrics(y_test, test_prob_rf)
    rows.append(
        {
            "model": "Constitution RandomForest",
            "feature_set": "constitution_raw",
            "n_features": int(len(constitution_cols)),
            "best_C": np.nan,
            "best_l1_ratio": np.nan,
            "val_auc": val_m_rf["auc"],
            "test_auc": test_m_rf["auc"],
            "val_pr_auc": val_m_rf["pr_auc"],
            "test_pr_auc": test_m_rf["pr_auc"],
            "val_brier": val_m_rf["brier"],
            "test_brier": test_m_rf["brier"],
            "is_degenerate": False,
        }
    )

    for name, imp in zip(constitution_cols, rf_model.feature_importances_):
        coef_rows.append(
            {
                "experiment": "Constitution RandomForest",
                "feature": name,
                "coef": float(imp),
                "abs_coef": float(abs(imp)),
            }
        )

    # 实验2：九体质 + 两两交互项 + ElasticNet LogisticCV
    x_train_int_df = _build_constitution_interactions(train_sub, constitution_cols)
    x_val_int_df = _build_constitution_interactions(val_sub, constitution_cols)
    x_test_int_df = _build_constitution_interactions(test_sub, constitution_cols)
    feature_names_int = x_train_int_df.columns.tolist()

    scaler_int = StandardScaler()
    x_train_int = scaler_int.fit_transform(x_train_int_df.values)
    x_val_int = scaler_int.transform(x_val_int_df.values)
    x_test_int = scaler_int.transform(x_test_int_df.values)

    enet_model = LogisticRegressionCV(
        Cs=np.logspace(-3, 2, 15),
        cv=5,
        scoring="roc_auc",
        penalty="elasticnet",
        l1_ratios=[0.2, 0.5, 0.8],
        solver="saga",
        class_weight="balanced",
        random_state=seed,
        max_iter=6000,
        n_jobs=-1,
        refit=True,
    )
    enet_model.fit(x_train_int, y_train)
    val_prob_enet = enet_model.predict_proba(x_val_int)[:, 1]
    test_prob_enet = enet_model.predict_proba(x_test_int)[:, 1]
    enet_coef = enet_model.coef_.ravel()
    degenerate_enet = bool((np.std(test_prob_enet) < 1e-6) or np.all(np.abs(enet_coef) < 1e-8))
    val_m_enet = _compute_binary_metrics(y_val, val_prob_enet)
    test_m_enet = _compute_binary_metrics(y_test, test_prob_enet)
    rows.append(
        {
            "model": "Constitution Interaction ElasticNetCV",
            "feature_set": "constitution_pairwise_interaction",
            "n_features": int(len(feature_names_int)),
            "best_C": float(enet_model.C_[0]),
            "best_l1_ratio": float(enet_model.l1_ratio_[0]) if enet_model.l1_ratio_ is not None else np.nan,
            "val_auc": val_m_enet["auc"],
            "test_auc": test_m_enet["auc"],
            "val_pr_auc": val_m_enet["pr_auc"],
            "test_pr_auc": test_m_enet["pr_auc"],
            "val_brier": val_m_enet["brier"],
            "test_brier": test_m_enet["brier"],
            "is_degenerate": degenerate_enet,
        }
    )

    coef_int = enet_coef
    top_idx = np.argsort(np.abs(coef_int))[::-1][:15]
    for idx in top_idx:
        coef_rows.append(
            {
                "experiment": "Constitution Interaction ElasticNetCV",
                "feature": feature_names_int[idx],
                "coef": float(coef_int[idx]),
                "abs_coef": float(abs(coef_int[idx])),
            }
        )

    enhancement_df = pd.DataFrame(rows).sort_values(["is_degenerate", "test_auc"], ascending=[True, False])
    coef_df = pd.DataFrame(coef_rows).sort_values(["experiment", "abs_coef"], ascending=[True, False])
    return enhancement_df, coef_df


def main() -> None:
    parser = argparse.ArgumentParser(description="MathorCup C题-问题1完整可执行代码")
    parser.add_argument("--input-csv", type=str, default="附件1_样例数据.csv", help="原始CSV路径")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="切分数据输出目录")
    parser.add_argument("--output-dir", type=str, default="outputs/q1", help="问题1结果输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)

    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    col_map = build_column_map(df)

    numeric_cols = list(
        set(
            [col_map.label, col_map.tan_score]
            + col_map.constitution_scores
            + col_map.candidate_features
        )
    )
    df = to_numeric_frame(df, numeric_cols)

    # 至少保证标签非空
    df = df[df[col_map.label].notna()].copy().reset_index(drop=True)
    df[col_map.label] = df[col_map.label].astype(int)

    train_df, val_df, test_df = split_data(df, label_col=col_map.label, seed=args.seed)
    train_df.to_csv(processed_dir / "train_set.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(processed_dir / "val_set.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(processed_dir / "test_set.csv", index=False, encoding="utf-8-sig")

    # 特征筛选三路
    corr_df = run_correlation_selection(
        train_df,
        features=col_map.candidate_features,
        tan_col=col_map.tan_score,
        label_col=col_map.label,
    )
    lasso_df = run_l1_logistic_selection(
        train_df,
        features=col_map.candidate_features,
        label_col=col_map.label,
        seed=args.seed,
    )
    rf_df = run_rf_selection(
        train_df,
        features=col_map.candidate_features,
        label_col=col_map.label,
        seed=args.seed,
    )

    merged = corr_df.merge(lasso_df, on="feature", how="left").merge(rf_df, on="feature", how="left")
    merged.columns = [str(c).strip() for c in merged.columns]
    merged["votes"] = (
        merged[["corr_selected", "lasso_selected", "rf_selected"]]
        .astype(int)
        .sum(axis=1)
    )
    merged["final_selected"] = merged["votes"] >= 2
    merged = merged.sort_values(["final_selected", "votes", "rf_importance"], ascending=[False, False, False])

    selected_features = merged.loc[merged["final_selected"], "feature"].tolist()
    if not selected_features:
        selected_features = merged.sort_values("rf_importance", ascending=False)["feature"].head(3).tolist()

    merged.to_csv(output_dir / "feature_selection_details.csv", index=False, encoding="utf-8-sig")
    with open(output_dir / "selected_features.json", "w", encoding="utf-8") as f:
        json.dump({"selected_features": selected_features}, f, ensure_ascii=False, indent=2)

    # 用筛选后的关键指标训练预警模型，并在验证集评估AUC
    selected_coef_df, selected_metrics = run_selected_feature_model(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        selected_features=selected_features,
        label_col=col_map.label,
        seed=args.seed,
    )
    selected_coef_df.to_csv(output_dir / "selected_feature_model_coef.csv", index=False, encoding="utf-8-sig")

    # 九体质 OR 分析
    or_table, vif_table, diagnostics = run_logistic_or_analysis(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        constitution_cols=col_map.constitution_scores,
        label_col=col_map.label,
    )
    or_table.to_csv(output_dir / "OR_values_table.csv", index=False, encoding="utf-8-sig")
    vif_table.to_csv(output_dir / "vif_table.csv", index=False, encoding="utf-8-sig")

    enhancement_df, enhancement_coef_df = run_constitution_enhancement_experiments(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        constitution_cols=col_map.constitution_scores,
        label_col=col_map.label,
        seed=args.seed,
    )
    enhancement_df.to_csv(output_dir / "constitution_enhancement_table.csv", index=False, encoding="utf-8-sig")
    enhancement_coef_df.to_csv(output_dir / "constitution_enhancement_top_coef.csv", index=False, encoding="utf-8-sig")

    perf_df = pd.DataFrame(
        [
            {
                "model": "Constitution Logistic",
                "val_auc": diagnostics.get("val_auc"),
                "test_auc": diagnostics.get("test_auc"),
                "val_pr_auc": diagnostics.get("val_pr_auc"),
                "test_pr_auc": diagnostics.get("test_pr_auc"),
                "val_brier": diagnostics.get("val_brier"),
                "test_brier": diagnostics.get("test_brier"),
            },
            {
                "model": "Selected-feature Risk Model",
                "val_auc": selected_metrics.get("val_auc"),
                "test_auc": selected_metrics.get("test_auc"),
                "val_pr_auc": selected_metrics.get("val_pr_auc"),
                "test_pr_auc": selected_metrics.get("test_pr_auc"),
                "val_brier": selected_metrics.get("val_brier"),
                "test_brier": selected_metrics.get("test_brier"),
            },
        ]
    )
    perf_df.to_csv(output_dir / "model_performance_table.csv", index=False, encoding="utf-8-sig")

    run_summary = {
        "input_csv": str(input_csv),
        "processed_dir": str(processed_dir),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "sample_size": int(len(df)),
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "candidate_feature_count": int(len(col_map.candidate_features)),
        "selected_feature_count": int(len(selected_features)),
        "selected_features": selected_features,
        "diagnostics": diagnostics,
        "selected_feature_model": selected_metrics,
        "constitution_enhancement_best": enhancement_df.head(1).to_dict(orient="records"),
    }
    with open(output_dir / "q1_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    print("Q1运行完成")
    print(f"- 训练/验证/测试: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    print(f"- 最终关键指标数: {len(selected_features)}")
    print(f"- 验证集AUC(Logistic九体质): {diagnostics['val_auc']:.4f}")
    print(f"- 测试集AUC(Logistic九体质): {diagnostics['test_auc']:.4f}")
    print(f"- 验证集AUC(关键指标预警模型): {selected_metrics['val_auc']:.4f}")
    print(f"- 测试集AUC(关键指标预警模型): {selected_metrics['test_auc']:.4f}")
    if not enhancement_df.empty:
        best_row = enhancement_df.iloc[0]
        print(
            "- 九体质增强实验最佳模型: "
            f"{best_row['model']} (测试AUC={best_row['test_auc']:.4f})"
        )
    print(f"- 输出目录: {output_dir}")


if __name__ == "__main__":
    main()
