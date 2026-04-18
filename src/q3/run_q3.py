#!/usr/bin/env python3
"""问题3：痰湿体质患者6个月干预优化模型。"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class ColumnMap:
    sample_id: str
    constitution_label: str
    tan_score: str
    activity_total: str
    age_group: str


def find_column(df: pd.DataFrame, pattern: str) -> str:
    regex = re.compile(pattern)
    for col in df.columns:
        if regex.search(col):
            return col
    raise KeyError(f"未找到匹配列: {pattern}")


def build_column_map(df: pd.DataFrame) -> ColumnMap:
    return ColumnMap(
        sample_id=find_column(df, r"样本ID"),
        constitution_label=find_column(df, r"体质标签"),
        tan_score=find_column(df, r"^痰湿质$"),
        activity_total=find_column(df, r"活动量表总分"),
        age_group=find_column(df, r"年龄组"),
    )


def to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def regulation_level_by_tan(tan_score: float) -> int:
    # 附表2：调理分级适用区间
    if tan_score <= 58:
        return 1
    if tan_score <= 61:
        return 2
    return 3


def max_intensity_by_age(age_group: int) -> int:
    # 附表3：年龄约束
    if age_group <= 2:
        return 3
    if age_group <= 4:
        return 2
    return 1


def max_intensity_by_activity(activity_total: float) -> int:
    # 附表3：活动量表总分约束
    if activity_total < 40:
        return 1
    if activity_total < 60:
        return 2
    return 3


def monthly_drop_rate(reg_level: int, intensity: int, freq_per_week: int) -> float:
    # 假设1：中医调理分级对应每月痰湿积分基础下降率
    reg_drop_map = {1: 0.01, 2: 0.03, 3: 0.05}
    reg_drop = reg_drop_map[reg_level]

    # 假设2：活动干预效果按题目给定经验规则
    if freq_per_week < 5:
        activity_drop = 0.0
    else:
        activity_drop = 0.03 * (intensity - 1) + 0.01 * (freq_per_week - 5)

    # 假设3：月下降率上限，避免不合理过大下降
    return float(min(0.30, reg_drop + activity_drop))


def simulate_6_month(tan_init: float, monthly_rate: float) -> Tuple[float, List[float]]:
    scores = [float(tan_init)]
    current = float(tan_init)
    for _ in range(6):
        current = current * (1.0 - monthly_rate)
        scores.append(float(current))
    return float(current), scores


def _select_best_candidate(candidates: List[Dict[str, object]], mode: str) -> Dict[str, object]:
    """Select best candidate under the specified objective mode.

    mode='upper': efficacy-first upper-bound solution (original objective)
    mode='practical': add adherence penalty to avoid unrealistic high frequency
    """
    if mode == "upper":
        return sorted(
            candidates,
            key=lambda x: (
                x["tan_final_6m"],
                x["total_cost_6m"],
                x["frequency_per_week"],
                x["activity_intensity"],
            ),
        )[0]

    # Practical objective: keep efficacy primary but penalize very high frequency/intensity.
    # The soft cap is 7 times/week; above this, adherence burden rises quickly.
    if mode == "practical":
        def practical_score(x: Dict[str, object]) -> float:
            freq = float(x["frequency_per_week"])
            inten = float(x["activity_intensity"])
            # Prefer 5-7 times/week in practical mode unless efficacy gain is substantial.
            freq_penalty = 1.20 * max(0.0, freq - 6.0) ** 2
            inten_penalty = 0.50 * max(0.0, inten - 1.0)
            return float(x["tan_final_6m"]) + freq_penalty + inten_penalty

        return sorted(
            candidates,
            key=lambda x: (
                practical_score(x),
                x["tan_final_6m"],
                x["total_cost_6m"],
                x["frequency_per_week"],
                x["activity_intensity"],
            ),
        )[0]

    # Balanced objective: jointly optimize efficacy, cost and adherence burden.
    # This mode is intended as the default recommendation for real-world execution.
    if mode == "balanced":
        tan_init = max(float(candidates[0]["tan_init"]), 1e-6)

        def balanced_score(x: Dict[str, object]) -> float:
            tan_final = float(x["tan_final_6m"])
            cost = float(x["total_cost_6m"])
            freq = float(x["frequency_per_week"])
            inten = float(x["activity_intensity"])

            efficacy_term = tan_final / tan_init
            cost_term = cost / 2000.0
            # Prefer middle frequency (about 6-7/week) instead of edge solutions.
            freq_term = ((freq - 6.5) / 3.5) ** 2
            # Prefer moderate intensity unless efficacy gain is significant.
            inten_term = ((inten - 2.0) / 1.5) ** 2

            return 0.62 * efficacy_term + 0.23 * cost_term + 0.10 * freq_term + 0.05 * inten_term

        return sorted(
            candidates,
            key=lambda x: (
                balanced_score(x),
                x["tan_final_6m"],
                x["total_cost_6m"],
            ),
        )[0]

    raise ValueError(f"unknown mode: {mode}")


def optimize_single_patient(sample_id: int, age_group: int, activity_total: float, tan_init: float, mode: str = "upper") -> Dict[str, object]:
    reg_level = regulation_level_by_tan(tan_init)
    intensity_max = min(max_intensity_by_age(age_group), max_intensity_by_activity(activity_total))

    reg_monthly_cost = {1: 30, 2: 80, 3: 130}
    train_unit_cost = {1: 3, 2: 5, 3: 8}

    candidates = []
    for intensity in range(1, intensity_max + 1):
        for freq in range(1, 11):
            reg_cost_6m = reg_monthly_cost[reg_level] * 6
            train_cost_6m = train_unit_cost[intensity] * freq * 24
            total_cost = reg_cost_6m + train_cost_6m
            if total_cost > 2000:
                continue

            m_rate = monthly_drop_rate(reg_level, intensity, freq)
            tan_final, traj = simulate_6_month(tan_init, m_rate)
            reduction = tan_init - tan_final
            reduction_rate = reduction / tan_init if tan_init > 0 else 0.0

            candidates.append(
                {
                    "sample_id": int(sample_id),
                    "regulation_level": int(reg_level),
                    "activity_intensity": int(intensity),
                    "frequency_per_week": int(freq),
                    "monthly_drop_rate": float(m_rate),
                    "tan_init": float(tan_init),
                    "tan_final_6m": float(tan_final),
                    "tan_reduction": float(reduction),
                    "tan_reduction_rate": float(reduction_rate),
                    "regulation_cost_6m": float(reg_cost_6m),
                    "activity_cost_6m": float(train_cost_6m),
                    "total_cost_6m": float(total_cost),
                    "trajectory": traj,
                }
            )

    if not candidates:
        # 极端情况下给一个保底方案
        m_rate = monthly_drop_rate(reg_level, 1, 1)
        tan_final, traj = simulate_6_month(tan_init, m_rate)
        reg_cost_6m = reg_monthly_cost[reg_level] * 6
        train_cost_6m = train_unit_cost[1] * 1 * 24
        total_cost = reg_cost_6m + train_cost_6m
        return {
            "sample_id": int(sample_id),
            "regulation_level": int(reg_level),
            "activity_intensity": 1,
            "frequency_per_week": 1,
            "monthly_drop_rate": float(m_rate),
            "tan_init": float(tan_init),
            "tan_final_6m": float(tan_final),
            "tan_reduction": float(tan_init - tan_final),
            "tan_reduction_rate": float((tan_init - tan_final) / tan_init if tan_init > 0 else 0.0),
            "regulation_cost_6m": float(reg_cost_6m),
            "activity_cost_6m": float(train_cost_6m),
            "total_cost_6m": float(total_cost),
            "trajectory": traj,
            "note": "no-feasible-under-budget",
        }

    best = _select_best_candidate(candidates, mode=mode)
    best["objective_mode"] = mode
    return best


def build_matching_rules(opt_df: pd.DataFrame) -> pd.DataFrame:
    if opt_df.empty:
        return pd.DataFrame()

    work = opt_df.copy()
    work["tan_bin"] = pd.cut(
        work["tan_init"],
        bins=[-np.inf, 58, 61, np.inf],
        labels=["tan<=58", "59<=tan<=61", "tan>=62"],
    )
    work["activity_bin"] = pd.cut(
        work["activity_total"],
        bins=[-np.inf, 40, 60, np.inf],
        labels=["act<40", "40<=act<60", "act>=60"],
        right=False,
    )

    grp = (
        work.groupby(["age_group", "tan_bin", "activity_bin"], observed=True)
        .agg(
            n=("sample_id", "size"),
            reg_level_mode=("regulation_level", lambda s: int(s.mode().iloc[0])),
            intensity_mode=("activity_intensity", lambda s: int(s.mode().iloc[0])),
            freq_mode=("frequency_per_week", lambda s: int(s.mode().iloc[0])),
            mean_reduction_rate=("tan_reduction_rate", "mean"),
            mean_cost=("total_cost_6m", "mean"),
        )
        .reset_index()
    )
    return grp.sort_values(["age_group", "tan_bin", "activity_bin"])


def main() -> None:
    parser = argparse.ArgumentParser(description="MathorCup C题-问题3 6个月干预方案优化")
    parser.add_argument("--input-csv", type=str, default="附件1_样例数据.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/q3")
    parser.add_argument("--target-constitution", type=int, default=5, help="目标体质标签，默认痰湿质=5")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv, encoding="utf-8-sig")
    col_map = build_column_map(df)

    numeric_cols = [
        col_map.sample_id,
        col_map.constitution_label,
        col_map.tan_score,
        col_map.activity_total,
        col_map.age_group,
    ]
    df = to_numeric(df, numeric_cols)
    df = df.dropna(subset=numeric_cols).copy().reset_index(drop=True)

    target_df = df[df[col_map.constitution_label].astype(int) == int(args.target_constitution)].copy()

    results_upper = []
    results_practical = []
    results_balanced = []
    for _, row in target_df.iterrows():
        plan_upper = optimize_single_patient(
            sample_id=int(row[col_map.sample_id]),
            age_group=int(row[col_map.age_group]),
            activity_total=float(row[col_map.activity_total]),
            tan_init=float(row[col_map.tan_score]),
            mode="upper",
        )
        plan_practical = optimize_single_patient(
            sample_id=int(row[col_map.sample_id]),
            age_group=int(row[col_map.age_group]),
            activity_total=float(row[col_map.activity_total]),
            tan_init=float(row[col_map.tan_score]),
            mode="practical",
        )
        plan_balanced = optimize_single_patient(
            sample_id=int(row[col_map.sample_id]),
            age_group=int(row[col_map.age_group]),
            activity_total=float(row[col_map.activity_total]),
            tan_init=float(row[col_map.tan_score]),
            mode="balanced",
        )

        for p in (plan_upper, plan_practical, plan_balanced):
            p["age_group"] = int(row[col_map.age_group])
            p["activity_total"] = float(row[col_map.activity_total])

        results_upper.append(plan_upper)
        results_practical.append(plan_practical)
        results_balanced.append(plan_balanced)

    upper_df = pd.DataFrame(results_upper).sort_values("sample_id").reset_index(drop=True)
    practical_df = pd.DataFrame(results_practical).sort_values("sample_id").reset_index(drop=True)
    opt_df = pd.DataFrame(results_balanced).sort_values("sample_id").reset_index(drop=True)

    # 导出上限解对照
    upper_out = upper_df.copy()
    upper_out["trajectory"] = upper_out["trajectory"].apply(lambda x: ";".join([f"{v:.3f}" for v in x]))
    upper_out.to_csv(output_dir / "q3_patient_optimal_plans_upper.csv", index=False, encoding="utf-8-sig")

    # 导出上限解（兼容旧文件名）
    out_df = opt_df.copy()
    out_df["trajectory"] = out_df["trajectory"].apply(lambda x: ";".join([f"{v:.3f}" for v in x]))
    out_df.to_csv(output_dir / "q3_patient_optimal_plans.csv", index=False, encoding="utf-8-sig")

    # 导出可执行折中解（旧版practical）
    practical_out = practical_df.copy()
    practical_out["trajectory"] = practical_out["trajectory"].apply(lambda x: ";".join([f"{v:.3f}" for v in x]))
    practical_out.to_csv(output_dir / "q3_patient_optimal_plans_practical.csv", index=False, encoding="utf-8-sig")

    sample_df = opt_df[opt_df["sample_id"].isin([1, 2, 3])].copy()
    sample_df_out = sample_df.copy()
    sample_df_out["trajectory"] = sample_df_out["trajectory"].apply(lambda x: ";".join([f"{v:.3f}" for v in x]))
    sample_df_out.to_csv(output_dir / "q3_sample_1_2_3_optimal_plan.csv", index=False, encoding="utf-8-sig")

    sample_df_upper = upper_df[upper_df["sample_id"].isin([1, 2, 3])].copy()
    sample_df_upper_out = sample_df_upper.copy()
    sample_df_upper_out["trajectory"] = sample_df_upper_out["trajectory"].apply(
        lambda x: ";".join([f"{v:.3f}" for v in x])
    )
    sample_df_upper_out.to_csv(
        output_dir / "q3_sample_1_2_3_optimal_plan_upper.csv", index=False, encoding="utf-8-sig"
    )

    sample_df_practical = practical_df[practical_df["sample_id"].isin([1, 2, 3])].copy()
    sample_df_practical_out = sample_df_practical.copy()
    sample_df_practical_out["trajectory"] = sample_df_practical_out["trajectory"].apply(
        lambda x: ";".join([f"{v:.3f}" for v in x])
    )
    sample_df_practical_out.to_csv(
        output_dir / "q3_sample_1_2_3_optimal_plan_practical.csv", index=False, encoding="utf-8-sig"
    )

    rules_df = build_matching_rules(opt_df)
    rules_df.to_csv(output_dir / "q3_matching_rules.csv", index=False, encoding="utf-8-sig")

    summary = {
        "input_csv": str(args.input_csv),
        "output_dir": str(output_dir),
        "target_constitution_label": int(args.target_constitution),
        "n_target_patients": int(len(opt_df)),
        "solution_upper": {
            "mean_tan_reduction_rate": float(upper_df["tan_reduction_rate"].mean()) if not upper_df.empty else np.nan,
            "mean_total_cost_6m": float(upper_df["total_cost_6m"].mean()) if not upper_df.empty else np.nan,
            "median_total_cost_6m": float(upper_df["total_cost_6m"].median()) if not upper_df.empty else np.nan,
            "freq_distribution": upper_df["frequency_per_week"].value_counts().sort_index().to_dict(),
        },
        "solution_practical": {
            "mean_tan_reduction_rate": float(practical_df["tan_reduction_rate"].mean()) if not practical_df.empty else np.nan,
            "mean_total_cost_6m": float(practical_df["total_cost_6m"].mean()) if not practical_df.empty else np.nan,
            "median_total_cost_6m": float(practical_df["total_cost_6m"].median()) if not practical_df.empty else np.nan,
            "freq_distribution": practical_df["frequency_per_week"].value_counts().sort_index().to_dict(),
        },
        "solution_balanced": {
            "mean_tan_reduction_rate": float(opt_df["tan_reduction_rate"].mean()) if not opt_df.empty else np.nan,
            "mean_total_cost_6m": float(opt_df["total_cost_6m"].mean()) if not opt_df.empty else np.nan,
            "median_total_cost_6m": float(opt_df["total_cost_6m"].median()) if not opt_df.empty else np.nan,
            "freq_distribution": opt_df["frequency_per_week"].value_counts().sort_index().to_dict(),
        },
        # backward-compatible fields now use balanced solution as default/main
        "mean_tan_reduction_rate": float(opt_df["tan_reduction_rate"].mean()) if not opt_df.empty else np.nan,
        "mean_total_cost_6m": float(opt_df["total_cost_6m"].mean()) if not opt_df.empty else np.nan,
        "median_total_cost_6m": float(opt_df["total_cost_6m"].median()) if not opt_df.empty else np.nan,
        "sample_1_2_3_plans": sample_df[[
            "sample_id",
            "regulation_level",
            "activity_intensity",
            "frequency_per_week",
            "tan_init",
            "tan_final_6m",
            "tan_reduction_rate",
            "total_cost_6m",
        ]].to_dict(orient="records"),
        "sample_1_2_3_plans_upper": sample_df_upper[[
            "sample_id",
            "regulation_level",
            "activity_intensity",
            "frequency_per_week",
            "tan_init",
            "tan_final_6m",
            "tan_reduction_rate",
            "total_cost_6m",
        ]].to_dict(orient="records"),
        "sample_1_2_3_plans_practical": sample_df_practical[[
            "sample_id",
            "regulation_level",
            "activity_intensity",
            "frequency_per_week",
            "tan_init",
            "tan_final_6m",
            "tan_reduction_rate",
            "total_cost_6m",
        ]].to_dict(orient="records"),
    }

    with open(output_dir / "q3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Q3运行完成")
    print(f"- 目标体质标签: {args.target_constitution}")
    print(f"- 覆盖样本数: {len(opt_df)}")
    if not opt_df.empty:
        print(f"- 平均6个月痰湿降幅: {opt_df['tan_reduction_rate'].mean():.4f}")
        print(f"- 平均6个月总成本: {opt_df['total_cost_6m'].mean():.2f}")
    print(f"- 输出目录: {output_dir}")


if __name__ == "__main__":
    main()
