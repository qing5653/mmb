#!/usr/bin/env python3
"""问题2：多维度融合三级风险预警模型。

输出：
1) q2_risk_predictions.csv：每个样本的风险等级与模型概率。
2) q2_thresholds.json：三级风险阈值与临床规则依据。
3) q2_feature_importance.csv：模型特征重要性。
4) q2_high_risk_core_combos.csv：高风险人群核心特征组合。
5) q2_summary.json：核心评估指标与分层结果汇总。
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split


@dataclass
class ColumnMap:
    sample_id: str
    label: str
    tan_score: str
    activity_total: str
    age_group: str
    sex: str
    smoking: str
    drinking: str
    constitution_scores: List[str]
    core_lipids: List[str]
    candidate_features: List[str]


def find_column(df: pd.DataFrame, pattern: str) -> str:
    regex = re.compile(pattern)
    for col in df.columns:
        if regex.search(col):
            return col
    raise KeyError(f"未找到匹配列: {pattern}")


def build_column_map(df: pd.DataFrame) -> ColumnMap:
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

    hdl = find_column(df, r"HDL-C")
    ldl = find_column(df, r"LDL-C")
    tg = find_column(df, r"TG")
    tc = find_column(df, r"TC")
    activity_total = find_column(df, r"活动量表总分")

    candidate_features = [
        hdl,
        ldl,
        tg,
        tc,
        find_column(df, r"空腹血糖"),
        find_column(df, r"血尿酸"),
        find_column(df, r"BMI"),
        activity_total,
        find_column(df, r"年龄组"),
        find_column(df, r"性别"),
        find_column(df, r"吸烟史"),
        find_column(df, r"饮酒史"),
    ] + constitution_scores

    return ColumnMap(
        sample_id=find_column(df, r"样本ID"),
        label=find_column(df, r"高血脂症二分类标签"),
        tan_score=find_column(df, r"^痰湿质$"),
        activity_total=activity_total,
        age_group=find_column(df, r"年龄组"),
        sex=find_column(df, r"性别"),
        smoking=find_column(df, r"吸烟史"),
        drinking=find_column(df, r"饮酒史"),
        constitution_scores=constitution_scores,
        core_lipids=[tc, tg, ldl, hdl],
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


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def get_optimal_cutoff_by_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    idx = int(np.argmax(youden))
    return float(thresholds[idx])


def add_lipid_abnormal_flags(df: pd.DataFrame, col_map: ColumnMap) -> pd.DataFrame:
    out = df.copy()
    tc, tg, ldl, hdl = col_map.core_lipids
    out["abn_tc"] = (out[tc] > 6.2).astype(int)
    out["abn_tg"] = (out[tg] > 1.7).astype(int)
    out["abn_ldl"] = (out[ldl] > 3.1).astype(int)
    out["abn_hdl"] = (out[hdl] < 1.04).astype(int)
    out["abnormal_lipid_count"] = out[["abn_tc", "abn_tg", "abn_ldl", "abn_hdl"]].sum(axis=1)
    return out


def assign_risk_level(
    df: pd.DataFrame,
    prob_col: str,
    index_col: str,
    tan_col: str,
    activity_col: str,
    t_low: float,
    index_low: float,
    index_high: float,
    tan_high: float,
    tan_very_high: float,
    activity_low: float,
    activity_good: float,
) -> pd.Series:
    clinical_high = (
        ((df["abnormal_lipid_count"] >= 3) & (df[tan_col] >= tan_high))
        | ((df["abnormal_lipid_count"] >= 2) & (df[tan_col] >= tan_very_high))
        | ((df[tan_col] >= tan_very_high) & (df[activity_col] <= activity_low))
    )

    clinical_low = (
        (df["abnormal_lipid_count"] <= 1)
        & (df[tan_col] < 40)
        & (df[activity_col] >= activity_good)
        & (df[prob_col] < t_low)
    )

    pred_high = df[index_col] >= index_high
    pred_low = df[index_col] < index_low

    risk = np.where(pred_high | clinical_high, "高风险", np.where(pred_low & clinical_low, "低风险", "中风险"))
    return pd.Series(risk, index=df.index)


def summarize_risk_tiers(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    agg = (
        df.groupby("risk_level", observed=True)
        .agg(
            sample_count=(label_col, "size"),
            positive_count=(label_col, "sum"),
            positive_rate=(label_col, "mean"),
            mean_score=("risk_score", "mean"),
        )
        .reset_index()
    )
    order = {"低风险": 0, "中风险": 1, "高风险": 2}
    agg["_order"] = agg["risk_level"].map(order)
    agg = agg.sort_values("_order").drop(columns=["_order"])
    return agg


def compute_risk_index(
    df: pd.DataFrame,
    prob_col: str,
    tan_col: str,
    activity_col: str,
) -> pd.Series:
    return (
        0.45 * (df["abnormal_lipid_count"] / 4.0)
        + 0.25 * (df[tan_col] / 100.0)
        + 0.20 * (1.0 - df[activity_col] / 100.0)
        + 0.10 * df[prob_col]
    )


def build_high_risk_combos(
    df: pd.DataFrame,
    col_map: ColumnMap,
    tan_high: float,
    activity_low: float,
) -> pd.DataFrame:
    out = df.copy()

    # 离散化规则（用于特征组合挖掘）
    out["痰湿高分"] = out[col_map.tan_score] >= tan_high
    out["活动能力低"] = out[col_map.activity_total] <= activity_low
    out["TG异常"] = out["abn_tg"] == 1
    out["TC异常"] = out["abn_tc"] == 1
    out["LDL异常"] = out["abn_ldl"] == 1
    out["HDL偏低"] = out["abn_hdl"] == 1
    out["BMI偏高"] = out[find_column(out, r"BMI")] >= 24
    uric_col = find_column(out, r"血尿酸")
    out["血尿酸偏高"] = out[uric_col] >= out[uric_col].quantile(0.75)
    out["年龄偏高"] = out[col_map.age_group] >= 4
    out["吸烟史"] = out[col_map.smoking] == 1
    out["饮酒史"] = out[col_map.drinking] == 1

    flag_cols = [
        "痰湿高分",
        "活动能力低",
        "TG异常",
        "TC异常",
        "LDL异常",
        "HDL偏低",
        "BMI偏高",
        "血尿酸偏高",
        "年龄偏高",
        "吸烟史",
        "饮酒史",
    ]

    high_df = out[out["risk_level"] == "高风险"].copy()
    if high_df.empty:
        return pd.DataFrame(columns=["combo", "combo_size", "support_high", "support_all", "lift", "count_high"])

    rows = []
    n_high = len(high_df)
    n_all = len(out)

    for k in [2, 3]:
        for cols in combinations(flag_cols, k):
            mask_high = high_df[list(cols)].all(axis=1)
            mask_all = out[list(cols)].all(axis=1)
            support_high = float(mask_high.mean())
            support_all = float(mask_all.mean())
            count_high = int(mask_high.sum())

            if support_high < 0.12:
                continue

            lift = float(support_high / (support_all + 1e-12))
            rows.append(
                {
                    "combo": " + ".join(cols),
                    "combo_size": k,
                    "support_high": support_high,
                    "support_all": support_all,
                    "lift": lift,
                    "count_high": count_high,
                    "n_high": n_high,
                    "n_all": n_all,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["combo", "combo_size", "support_high", "support_all", "lift", "count_high"])

    combo_df = pd.DataFrame(rows).sort_values(["support_high", "lift", "combo_size"], ascending=[False, False, True])
    return combo_df.head(20).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="MathorCup C题-问题2三级风险预警模型")
    parser.add_argument("--input-csv", type=str, default="附件1_样例数据.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/q2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    col_map = build_column_map(df)

    numeric_cols = list(
        set(
            [col_map.label, col_map.sample_id, col_map.tan_score, col_map.activity_total]
            + col_map.candidate_features
            + col_map.constitution_scores
        )
    )
    df = to_numeric_frame(df, numeric_cols)

    required_cols = [col_map.sample_id, col_map.label] + col_map.candidate_features
    df = df.dropna(subset=required_cols).copy().reset_index(drop=True)
    df[col_map.label] = df[col_map.label].astype(int)

    df = add_lipid_abnormal_flags(df, col_map)

    train_df, val_df, test_df = split_data(df, col_map.label, args.seed)

    model_features = col_map.candidate_features + ["abnormal_lipid_count"]
    x_train = train_df[model_features].values
    y_train = train_df[col_map.label].values
    x_val = val_df[model_features].values
    y_val = val_df[col_map.label].values
    x_test = test_df[model_features].values
    y_test = test_df[col_map.label].values

    model = RandomForestClassifier(
        n_estimators=700,
        random_state=args.seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=3,
    )
    model.fit(x_train, y_train)

    val_prob = model.predict_proba(x_val)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]

    val_metrics = evaluate_binary(y_val, val_prob)
    test_metrics = evaluate_binary(y_test, test_prob)

    t_star = get_optimal_cutoff_by_youden(y_val, val_prob)
    t_low = float(max(0.05, t_star - 0.15))
    t_high = float(min(0.95, t_star + 0.15))

    tan_q75 = float(train_df[col_map.tan_score].quantile(0.75))
    activity_q25 = float(train_df[col_map.activity_total].quantile(0.25))
    activity_q60 = float(train_df[col_map.activity_total].quantile(0.60))

    tan_high = float(max(55.0, tan_q75))
    tan_very_high = float(max(60.0, tan_high + 2.0))
    activity_low = float(min(45.0, activity_q25))
    activity_good = float(max(55.0, activity_q60))

    train_prob = model.predict_proba(train_df[model_features].values)[:, 1]
    train_eval_df = train_df[
        [
            col_map.sample_id,
            col_map.label,
            col_map.tan_score,
            col_map.activity_total,
            "abnormal_lipid_count",
            "abn_tc",
            "abn_tg",
            "abn_ldl",
            "abn_hdl",
        ]
    ].copy()
    train_eval_df["risk_score"] = train_prob
    train_eval_df["risk_index"] = compute_risk_index(
        train_eval_df,
        prob_col="risk_score",
        tan_col=col_map.tan_score,
        activity_col=col_map.activity_total,
    )

    # 仅使用训练集风险指数确定分层阈值，避免评估泄漏
    index_low = float(train_eval_df["risk_index"].quantile(0.35))
    index_high = float(train_eval_df["risk_index"].quantile(0.75))

    all_prob = model.predict_proba(df[model_features].values)[:, 1]
    pred_df = df[
        [
            col_map.sample_id,
            col_map.label,
            col_map.tan_score,
            col_map.activity_total,
            "abnormal_lipid_count",
            "abn_tc",
            "abn_tg",
            "abn_ldl",
            "abn_hdl",
        ]
    ].copy()
    pred_df["risk_score"] = all_prob
    pred_df["risk_index"] = compute_risk_index(
        pred_df,
        prob_col="risk_score",
        tan_col=col_map.tan_score,
        activity_col=col_map.activity_total,
    )

    train_ids = set(train_df[col_map.sample_id].tolist())
    val_ids = set(val_df[col_map.sample_id].tolist())
    pred_df["data_split"] = pred_df[col_map.sample_id].map(
        lambda x: "train" if x in train_ids else ("val" if x in val_ids else "test")
    )

    pred_df["risk_level"] = assign_risk_level(
        pred_df,
        prob_col="risk_score",
        index_col="risk_index",
        tan_col=col_map.tan_score,
        activity_col=col_map.activity_total,
        t_low=t_low,
        index_low=index_low,
        index_high=index_high,
        tan_high=tan_high,
        tan_very_high=tan_very_high,
        activity_low=activity_low,
        activity_good=activity_good,
    )

    # 规则命中标记，便于解释
    pred_df["rule_hit_high_1"] = (
        (pred_df["abnormal_lipid_count"] >= 3) & (pred_df[col_map.tan_score] >= tan_high)
    ).astype(int)
    pred_df["rule_hit_high_2"] = (
        (pred_df["abnormal_lipid_count"] >= 2) & (pred_df[col_map.tan_score] >= tan_very_high)
    ).astype(int)
    pred_df["rule_hit_high_3"] = (
        (pred_df[col_map.tan_score] >= tan_very_high) & (pred_df[col_map.activity_total] <= activity_low)
    ).astype(int)

    pred_df.to_csv(output_dir / "q2_risk_predictions.csv", index=False, encoding="utf-8-sig")

    importance_df = pd.DataFrame({"feature": model_features, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    importance_df.to_csv(output_dir / "q2_feature_importance.csv", index=False, encoding="utf-8-sig")

    tier_df = summarize_risk_tiers(pred_df, col_map.label)
    tier_val_df = summarize_risk_tiers(pred_df[pred_df["data_split"] == "val"], col_map.label)
    tier_test_df = summarize_risk_tiers(pred_df[pred_df["data_split"] == "test"], col_map.label)
    tier_df.to_csv(output_dir / "q2_risk_tier_summary.csv", index=False, encoding="utf-8-sig")
    tier_val_df.to_csv(output_dir / "q2_risk_tier_summary_val.csv", index=False, encoding="utf-8-sig")
    tier_test_df.to_csv(output_dir / "q2_risk_tier_summary_test.csv", index=False, encoding="utf-8-sig")

    combo_df = build_high_risk_combos(
        pred_df.join(df[col_map.candidate_features], rsuffix="_full"),
        col_map=col_map,
        tan_high=tan_high,
        activity_low=activity_low,
    )
    combo_df.to_csv(output_dir / "q2_high_risk_core_combos.csv", index=False, encoding="utf-8-sig")

    thresholds = {
        "probability_threshold": {
            "youden_optimal_t_star": t_star,
            "t_low": t_low,
            "t_high": t_high,
            "basis": "在验证集上按Youden指数得到二分类最优阈值t_star，再采用t_star±0.15构建低/高风险概率阈值。",
        },
        "risk_index_threshold": {
            "index_low": index_low,
            "index_high": index_high,
            "formula": "0.45*(血脂异常计数/4)+0.25*(痰湿质/100)+0.20*(1-活动总分/100)+0.10*(模型风险分)",
            "basis": "按训练集复合风险指数分位数（35%/75%）形成低/高风险分界，中间区间为中风险。",
        },
        "clinical_threshold": {
            "tan_high": tan_high,
            "tan_very_high": tan_very_high,
            "activity_low": activity_low,
            "activity_good": activity_good,
            "lipid_abnormal_definition": {
                "TC": ">6.2 mmol/L",
                "TG": ">1.7 mmol/L",
                "LDL-C": ">3.1 mmol/L",
                "HDL-C": "<1.04 mmol/L",
            },
            "basis": "痰湿阈值结合训练集分位数与题目建议区间；活动阈值结合训练集分位数与量表分段。",
        },
        "tier_rules": {
            "high_risk": [
                "risk_index >= index_high",
                "(abnormal_lipid_count >= 3) and (痰湿质 >= tan_high)",
                "(abnormal_lipid_count >= 2) and (痰湿质 >= tan_very_high)",
                "(痰湿质 >= tan_very_high) and (活动量表总分 <= activity_low)",
            ],
            "low_risk": [
                "risk_index < index_low",
                "risk_score < t_low",
                "abnormal_lipid_count <= 1",
                "痰湿质 < 40",
                "活动量表总分 >= activity_good",
            ],
            "middle_risk": ["其余样本归为中风险"],
        },
    }

    with open(output_dir / "q2_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)

    summary = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "sample_size": int(len(df)),
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "model": "RandomForestClassifier(n_estimators=700, min_samples_leaf=3)",
        "model_features": model_features,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "risk_index_threshold": {"index_low": index_low, "index_high": index_high},
        "risk_level_counts": pred_df["risk_level"].value_counts().to_dict(),
        "risk_level_positive_rate": tier_df.set_index("risk_level")["positive_rate"].to_dict(),
        "risk_level_positive_rate_val": tier_val_df.set_index("risk_level")["positive_rate"].to_dict(),
        "risk_level_positive_rate_test": tier_test_df.set_index("risk_level")["positive_rate"].to_dict(),
        "top_high_risk_combos": combo_df.head(8).to_dict(orient="records"),
        "thresholds_file": "q2_thresholds.json",
    }

    with open(output_dir / "q2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Q2运行完成")
    print(f"- 验证集AUC: {val_metrics['auc']:.4f}")
    print(f"- 测试集AUC: {test_metrics['auc']:.4f}")
    print(f"- 低/中/高风险样本数: {pred_df['risk_level'].value_counts().to_dict()}")
    print(f"- 输出目录: {output_dir}")


if __name__ == "__main__":
    main()
