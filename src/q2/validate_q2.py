#!/usr/bin/env python3
"""问题2稳健性验证（网文式重算版本）。

输出：
1) q2_robustness_seed_repeat.csv
2) q2_ablation_results.csv
3) q2_robustness_summary.json
4) q2_calibration_table.csv
5) q2_calibration_summary.json
6) q2_tier_bootstrap_ci.csv
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass
class ColumnMap:
    sample_id: str
    label: str
    tan_score: str
    activity_total: str
    core_lipids: List[str]
    constitution_scores: List[str]
    candidate_features: List[str]


def find_column(df: pd.DataFrame, pattern: str) -> str:
    regex = re.compile(pattern)
    for col in df.columns:
        if regex.search(col):
            return col
    raise KeyError(f"未找到匹配列: {pattern}")


def find_column_optional(df: pd.DataFrame, pattern: str) -> Optional[str]:
    regex = re.compile(pattern)
    for col in df.columns:
        if regex.search(col):
            return col
    return None


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

    candidate_features = [
        hdl,
        ldl,
        tg,
        tc,
        find_column(df, r"空腹血糖"),
        find_column(df, r"血尿酸"),
        find_column(df, r"BMI"),
        find_column(df, r"活动量表总分"),
        find_column(df, r"年龄组"),
        find_column(df, r"性别"),
        find_column(df, r"吸烟史"),
        find_column(df, r"饮酒史"),
    ] + constitution_scores

    return ColumnMap(
        sample_id=find_column(df, r"样本ID"),
        label=find_column(df, r"高血脂症二分类标签"),
        tan_score=find_column(df, r"^痰湿质$"),
        activity_total=find_column(df, r"活动量表总分"),
        core_lipids=[tc, tg, ldl, hdl],
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


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def fit_predict(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    features: List[str],
    label_col: str,
    seed: int,
) -> np.ndarray:
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=seed,
    )
    model.fit(train_df[features].values, train_df[label_col].values)
    return model.predict_proba(eval_df[features].values)[:, 1]


def train_and_eval(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    label_col: str,
    seed: int,
) -> Dict[str, float]:
    val_prob = fit_predict(train_df, val_df, features, label_col, seed)
    test_prob = fit_predict(train_df, test_df, features, label_col, seed)

    val_m = evaluate(val_df[label_col].values, val_prob)
    test_m = evaluate(test_df[label_col].values, test_prob)

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


def build_calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[pd.DataFrame, Dict[str, float]]:
    cal = pd.DataFrame({"y": y_true, "p": y_prob}).dropna().copy()
    cal["bin"] = pd.qcut(cal["p"], q=n_bins, duplicates="drop")
    grouped = (
        cal.groupby("bin", observed=True)
        .agg(n=("y", "size"), mean_pred=("p", "mean"), event_rate=("y", "mean"))
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

    split_values = pred_df[split_col].dropna().unique().tolist() if split_col in pred_df.columns else ["all"]

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

            arr = g_sub[label_col].values
            boot_vals = []
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
    parser = argparse.ArgumentParser(description="Q2稳健性验证（网文式）")
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

    numeric_cols = list(set([col_map.label, col_map.sample_id] + col_map.candidate_features))
    df = to_numeric_frame(df, numeric_cols)
    df = df.dropna(subset=[col_map.label, col_map.sample_id] + col_map.candidate_features).copy().reset_index(drop=True)
    df[col_map.label] = df[col_map.label].astype(int)

    with_lipid_features = col_map.candidate_features
    no_lipid_features = [f for f in with_lipid_features if f not in col_map.core_lipids]

    # 1) 多随机种子重复（使用网文主轨A：去血脂）
    seed_rows = []
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    for seed in seeds:
        train_df, val_df, test_df = split_data(df, col_map.label, seed)
        m = train_and_eval(train_df, val_df, test_df, no_lipid_features, col_map.label, seed)
        seed_rows.append({"seed": seed, **m})

    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(output_dir / "q2_robustness_seed_repeat.csv", index=False, encoding="utf-8-sig")

    # 2) 特征消融（围绕网文口径）
    base_seed = args.seed_start
    train_df, val_df, test_df = split_data(df, col_map.label, base_seed)

    ablations = {
        "track_A_no_lipid": no_lipid_features,
        "track_B_with_lipid": with_lipid_features,
        "remove_constitution": [f for f in no_lipid_features if f not in col_map.constitution_scores],
        "remove_activity": [f for f in no_lipid_features if f != col_map.activity_total],
    }

    ablation_rows = []
    for name, feats in ablations.items():
        m = train_and_eval(train_df, val_df, test_df, feats, col_map.label, base_seed)
        ablation_rows.append({"experiment": name, "n_features": len(feats), **m})

    ablation_df = pd.DataFrame(ablation_rows).sort_values("test_auc", ascending=False)
    ablation_df.to_csv(output_dir / "q2_ablation_results.csv", index=False, encoding="utf-8-sig")

    # 3) 概率校准（主轨A）
    model_val_prob = fit_predict(train_df, val_df, no_lipid_features, col_map.label, base_seed)
    model_test_prob = fit_predict(train_df, test_df, no_lipid_features, col_map.label, base_seed)

    cal_val_df, cal_val_stats = build_calibration_table(val_df[col_map.label].values, model_val_prob, n_bins=10)
    cal_test_df, cal_test_stats = build_calibration_table(test_df[col_map.label].values, model_test_prob, n_bins=10)
    cal_val_df["split"] = "val"
    cal_test_df["split"] = "test"
    cal_df = pd.concat([cal_val_df, cal_test_df], ignore_index=True)
    cal_df.to_csv(output_dir / "q2_calibration_table.csv", index=False, encoding="utf-8-sig")

    cal_summary = {
        "val": cal_val_stats,
        "test": cal_test_stats,
        "note": "基于去血脂主轨模型A计算，ECE越低表示概率校准越好。",
    }
    with open(output_dir / "q2_calibration_summary.json", "w", encoding="utf-8") as f:
        json.dump(cal_summary, f, ensure_ascii=False, indent=2)

    # 4) 分层Bootstrap区间（依赖run_q2输出）
    pred_path = output_dir / "q2_risk_predictions.csv"
    if pred_path.exists():
        pred_df = pd.read_csv(pred_path, encoding="utf-8-sig")
        tier_ci_df = bootstrap_tier_positive_rate_ci(
            pred_df=pred_df,
            label_col=col_map.label,
            group_col="risk_level",
            split_col="data_split",
            n_boot=args.bootstrap_iterations,
            seed=base_seed,
        )
        tier_ci_df.to_csv(output_dir / "q2_tier_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(output_dir / "q2_tier_bootstrap_ci.csv", index=False, encoding="utf-8-sig")

    # 5) 汇总
    summary = {
        "seed_range": [args.seed_start, args.seed_start + args.seed_count - 1],
        "seed_repeat_rows": int(len(seed_df)),
        "ablation_rows": int(len(ablation_df)),
        "main_track": "track_A_no_lipid",
        "track_A_features": len(no_lipid_features),
        "track_B_features": len(with_lipid_features),
        "seed_repeat_auc": summarize_metric(seed_df, "test_auc"),
        "seed_repeat_pr_auc": summarize_metric(seed_df, "test_pr_auc"),
        "seed_repeat_brier": summarize_metric(seed_df, "test_brier"),
        "best_ablation": ablation_df.iloc[0].to_dict() if not ablation_df.empty else {},
        "calibration": cal_summary,
    }

    with open(output_dir / "q2_robustness_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Q2验证完成（网文式）")
    print(f"- 种子重复条数: {len(seed_df)}")
    print(f"- 消融实验条数: {len(ablation_df)}")
    print(f"- 输出目录: {output_dir}")


if __name__ == "__main__":
    main()
