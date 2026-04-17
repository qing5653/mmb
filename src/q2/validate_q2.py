#!/usr/bin/env python3
"""问题2稳健性验证：特征消融 + 多随机种子重复。

输出：
1) q2_robustness_seed_repeat.csv
2) q2_ablation_results.csv
3) q2_robustness_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass
class ColumnMap:
    label: str
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
        label=find_column(df, r"高血脂症二分类标签"),
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


def add_lipid_abnormal_flags(df: pd.DataFrame, col_map: ColumnMap) -> pd.DataFrame:
    out = df.copy()
    tc, tg, ldl, hdl = col_map.core_lipids
    out["abn_tc"] = (out[tc] > 6.2).astype(int)
    out["abn_tg"] = (out[tg] > 1.7).astype(int)
    out["abn_ldl"] = (out[ldl] > 3.1).astype(int)
    out["abn_hdl"] = (out[hdl] < 1.04).astype(int)
    out["abnormal_lipid_count"] = out[["abn_tc", "abn_tg", "abn_ldl", "abn_hdl"]].sum(axis=1)
    return out


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def train_and_eval(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    label_col: str,
    seed: int,
) -> Dict[str, float]:
    x_train = train_df[features].values
    y_train = train_df[label_col].values
    x_val = val_df[features].values
    y_val = val_df[label_col].values
    x_test = test_df[features].values
    y_test = test_df[label_col].values

    model = RandomForestClassifier(
        n_estimators=700,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=3,
    )
    model.fit(x_train, y_train)

    val_prob = model.predict_proba(x_val)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]

    val_m = evaluate(y_val, val_prob)
    test_m = evaluate(y_test, test_prob)

    return {
        "val_auc": val_m["auc"],
        "test_auc": test_m["auc"],
        "val_pr_auc": val_m["pr_auc"],
        "test_pr_auc": test_m["pr_auc"],
        "val_brier": val_m["brier"],
        "test_brier": test_m["brier"],
    }


def summarize_metric(df: pd.DataFrame, prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_mean": float(df[prefix].mean()),
        f"{prefix}_std": float(df[prefix].std(ddof=0)),
        f"{prefix}_min": float(df[prefix].min()),
        f"{prefix}_max": float(df[prefix].max()),
    }


def fit_model(
    train_df: pd.DataFrame,
    features: List[str],
    label_col: str,
    seed: int,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=700,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=3,
    )
    model.fit(train_df[features].values, train_df[label_col].values)
    return model


def build_calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[pd.DataFrame, Dict[str, float]]:
    cal = pd.DataFrame({"y": y_true, "p": y_prob}).dropna().copy()
    cal["bin"] = pd.qcut(cal["p"], q=n_bins, duplicates="drop")
    grouped = (
        cal.groupby("bin", observed=True)
        .agg(
            n=("y", "size"),
            mean_pred=("p", "mean"),
            event_rate=("y", "mean"),
        )
        .reset_index()
    )
    grouped["abs_gap"] = (grouped["event_rate"] - grouped["mean_pred"]).abs()
    n_all = float(grouped["n"].sum())
    grouped["weight"] = grouped["n"] / n_all if n_all > 0 else 0.0
    ece = float((grouped["weight"] * grouped["abs_gap"]).sum()) if n_all > 0 else np.nan
    mce = float(grouped["abs_gap"].max()) if not grouped.empty else np.nan
    out = grouped.copy()
    out["bin"] = out["bin"].astype(str)
    return out, {"ece": ece, "mce": mce, "n_bins_effective": int(len(out))}


def bootstrap_tier_positive_rate_ci(
    pred_df: pd.DataFrame,
    label_col: str,
    group_col: str = "risk_level",
    split_col: str = "data_split",
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    if split_col in pred_df.columns:
        split_values = pred_df[split_col].dropna().unique().tolist()
    else:
        split_values = ["all"]

    for split in split_values:
        sub = pred_df if split == "all" else pred_df[pred_df[split_col] == split]
        if sub.empty:
            continue

        groups = sub[group_col].dropna().unique().tolist()
        for g in groups:
            g_sub = sub[sub[group_col] == g]
            n = len(g_sub)
            if n == 0:
                continue

            boot_vals = []
            arr = g_sub[label_col].values
            for _ in range(n_boot):
                idx = rng.integers(0, n, size=n)
                boot_vals.append(float(arr[idx].mean()))

            low = float(np.quantile(boot_vals, 0.025))
            high = float(np.quantile(boot_vals, 0.975))
            rows.append(
                {
                    "split": split,
                    "risk_level": g,
                    "sample_count": int(n),
                    "positive_rate": float(arr.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "ci95_width": float(high - low),
                    "bootstrap_n": int(n_boot),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        split_order = {"train": 0, "val": 1, "test": 2, "all": 3}
        risk_order = {"低风险": 0, "中风险": 1, "高风险": 2}
        out["_s"] = out["split"].map(split_order).fillna(99)
        out["_r"] = out["risk_level"].map(risk_order).fillna(99)
        out = out.sort_values(["_s", "_r"]).drop(columns=["_s", "_r"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Q2稳健性验证")
    parser.add_argument("--input-csv", type=str, default="附件1_样例数据.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/q2")
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv, encoding="utf-8-sig")
    col_map = build_column_map(df)

    numeric_cols = list(set([col_map.label] + col_map.candidate_features + col_map.core_lipids))
    df = to_numeric_frame(df, numeric_cols)
    df = df.dropna(subset=[col_map.label] + col_map.candidate_features).copy().reset_index(drop=True)
    df[col_map.label] = df[col_map.label].astype(int)
    df = add_lipid_abnormal_flags(df, col_map)

    full_features = col_map.candidate_features + ["abnormal_lipid_count"]

    # 1) 多随机种子重复稳定性
    seed_rows = []
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    for seed in seeds:
        train_df, val_df, test_df = split_data(df, col_map.label, seed)
        m = train_and_eval(train_df, val_df, test_df, full_features, col_map.label, seed)
        seed_rows.append({"seed": seed, **m})

    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(output_dir / "q2_robustness_seed_repeat.csv", index=False, encoding="utf-8-sig")

    # 2) 特征消融
    base_seed = args.seed_start
    train_df, val_df, test_df = split_data(df, col_map.label, base_seed)

    ablations = {
        "full_model": full_features,
        "remove_core_lipids": [f for f in full_features if f not in col_map.core_lipids],
        "remove_constitution": [f for f in full_features if f not in col_map.constitution_scores],
        "remove_activity": [f for f in full_features if "活动量表总分" not in f],
        "remove_lipid_count": [f for f in full_features if f != "abnormal_lipid_count"],
    }

    ablation_rows = []
    for name, feats in ablations.items():
        m = train_and_eval(train_df, val_df, test_df, feats, col_map.label, base_seed)
        ablation_rows.append(
            {
                "experiment": name,
                "n_features": len(feats),
                **m,
            }
        )

    ablation_df = pd.DataFrame(ablation_rows).sort_values("test_auc", ascending=False)
    ablation_df.to_csv(output_dir / "q2_ablation_results.csv", index=False, encoding="utf-8-sig")

    full_test_auc = float(ablation_df.loc[ablation_df["experiment"] == "full_model", "test_auc"].iloc[0])
    remove_core_auc = float(ablation_df.loc[ablation_df["experiment"] == "remove_core_lipids", "test_auc"].iloc[0])

    # 3) 概率校准（基线模型）
    base_model = fit_model(train_df, full_features, col_map.label, base_seed)
    val_prob = base_model.predict_proba(val_df[full_features].values)[:, 1]
    test_prob = base_model.predict_proba(test_df[full_features].values)[:, 1]
    cal_val_df, cal_val_stats = build_calibration_table(val_df[col_map.label].values, val_prob, n_bins=10)
    cal_test_df, cal_test_stats = build_calibration_table(test_df[col_map.label].values, test_prob, n_bins=10)
    cal_val_df["split"] = "val"
    cal_test_df["split"] = "test"
    cal_df = pd.concat([cal_val_df, cal_test_df], ignore_index=True)
    cal_df.to_csv(output_dir / "q2_calibration_table.csv", index=False, encoding="utf-8-sig")

    cal_summary = {
        "val": cal_val_stats,
        "test": cal_test_stats,
        "note": "ECE越低表示概率校准越好；MCE反映最差分箱偏差。",
    }
    with open(output_dir / "q2_calibration_summary.json", "w", encoding="utf-8") as f:
        json.dump(cal_summary, f, ensure_ascii=False, indent=2)

    # 4) 风险层阳性率Bootstrap置信区间（依赖run_q2输出）
    pred_path = output_dir / "q2_risk_predictions.csv"
    if pred_path.exists():
        pred_df = pd.read_csv(pred_path, encoding="utf-8-sig")
        tier_ci_df = bootstrap_tier_positive_rate_ci(
            pred_df=pred_df,
            label_col=col_map.label,
            group_col="risk_level",
            split_col="data_split",
            n_boot=args.bootstrap_iterations,
            seed=args.seed_start,
        )
        tier_ci_df.to_csv(output_dir / "q2_tier_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    else:
        tier_ci_df = pd.DataFrame()

    summary = {
        "seed_repeat": {
            "seed_start": args.seed_start,
            "seed_count": args.seed_count,
            **summarize_metric(seed_df, "val_auc"),
            **summarize_metric(seed_df, "test_auc"),
            **summarize_metric(seed_df, "val_pr_auc"),
            **summarize_metric(seed_df, "test_pr_auc"),
            **summarize_metric(seed_df, "val_brier"),
            **summarize_metric(seed_df, "test_brier"),
        },
        "ablation": {
            "full_model_test_auc": full_test_auc,
            "remove_core_lipids_test_auc": remove_core_auc,
            "test_auc_drop_without_core_lipids": float(full_test_auc - remove_core_auc),
            "table_file": "q2_ablation_results.csv",
        },
        "calibration": {
            "table_file": "q2_calibration_table.csv",
            "summary_file": "q2_calibration_summary.json",
            "val_ece": cal_val_stats["ece"],
            "test_ece": cal_test_stats["ece"],
            "val_mce": cal_val_stats["mce"],
            "test_mce": cal_test_stats["mce"],
        },
        "tier_bootstrap_ci": {
            "table_file": "q2_tier_bootstrap_ci.csv" if not tier_ci_df.empty else "",
            "bootstrap_n": int(args.bootstrap_iterations),
            "available": bool(not tier_ci_df.empty),
        },
        "conclusion_hint": "当AUC顶格时，应联合Brier、ECE与分层CI评估模型好坏。",
    }

    with open(output_dir / "q2_robustness_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Q2稳健性验证完成")
    print(f"- Seeds: {seeds[0]}..{seeds[-1]} (count={len(seeds)})")
    print(f"- Full model test AUC: {full_test_auc:.4f}")
    print(f"- Remove core lipids test AUC: {remove_core_auc:.4f}")
    print(f"- Drop: {full_test_auc - remove_core_auc:.4f}")
    print(f"- Test ECE: {cal_test_stats['ece']:.6f}")
    print(f"- 输出目录: {output_dir}")


if __name__ == "__main__":
    main()
