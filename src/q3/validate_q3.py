#!/usr/bin/env python3
"""问题3稳健性与敏感性验证。"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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
    if tan_score <= 58:
        return 1
    if tan_score <= 61:
        return 2
    return 3


def max_intensity_by_age(age_group: int) -> int:
    if age_group <= 2:
        return 3
    if age_group <= 4:
        return 2
    return 1


def max_intensity_by_activity(activity_total: float) -> int:
    if activity_total < 40:
        return 1
    if activity_total < 60:
        return 2
    return 3


def monthly_drop_rate(reg_level: int, intensity: int, freq_per_week: int, drop_scale: float) -> float:
    reg_drop_map = {1: 0.01, 2: 0.03, 3: 0.05}
    reg_drop = reg_drop_map[reg_level]

    if freq_per_week < 5:
        activity_drop = 0.0
    else:
        activity_drop = 0.03 * (intensity - 1) + 0.01 * (freq_per_week - 5)

    base = reg_drop + activity_drop
    return float(min(0.30, max(0.0, base * drop_scale)))


def simulate_6_month(tan_init: float, monthly_rate: float) -> float:
    current = float(tan_init)
    for _ in range(6):
        current = current * (1.0 - monthly_rate)
    return float(current)


def optimize_patient_scenario(
    age_group: int,
    activity_total: float,
    tan_init: float,
    budget_cap: float,
    drop_scale: float,
    objective_mode: str = "balanced",
) -> Dict[str, float]:
    reg_level = regulation_level_by_tan(tan_init)
    intensity_max = min(max_intensity_by_age(age_group), max_intensity_by_activity(activity_total))

    reg_monthly_cost = {1: 30, 2: 80, 3: 130}
    train_unit_cost = {1: 3, 2: 5, 3: 8}

    candidates: List[Dict[str, float]] = []
    for intensity in range(1, intensity_max + 1):
        for freq in range(1, 11):
            reg_cost_6m = reg_monthly_cost[reg_level] * 6
            train_cost_6m = train_unit_cost[intensity] * freq * 24
            total_cost = reg_cost_6m + train_cost_6m
            if total_cost > budget_cap:
                continue

            m_rate = monthly_drop_rate(reg_level, intensity, freq, drop_scale=drop_scale)
            tan_final = simulate_6_month(tan_init, m_rate)
            candidates.append(
                {
                    "regulation_level": float(reg_level),
                    "activity_intensity": float(intensity),
                    "frequency_per_week": float(freq),
                    "tan_final_6m": tan_final,
                    "total_cost_6m": float(total_cost),
                }
            )

    if not candidates:
        return {
            "regulation_level": float(reg_level),
            "activity_intensity": 1.0,
            "frequency_per_week": 1.0,
            "tan_final_6m": float(tan_init),
            "total_cost_6m": np.nan,
            "is_feasible": 0.0,
        }

    if objective_mode == "upper":
        best = sorted(candidates, key=lambda x: (x["tan_final_6m"], x["total_cost_6m"]))[0]
    else:
        tan_init_safe = max(float(tan_init), 1e-6)

        def balanced_score(x: Dict[str, float]) -> float:
            efficacy_term = float(x["tan_final_6m"]) / tan_init_safe
            cost_term = float(x["total_cost_6m"]) / max(float(budget_cap), 1.0)
            freq_term = ((float(x["frequency_per_week"]) - 6.5) / 3.5) ** 2
            inten_term = ((float(x["activity_intensity"]) - 2.0) / 1.5) ** 2
            return 0.62 * efficacy_term + 0.23 * cost_term + 0.10 * freq_term + 0.05 * inten_term

        best = sorted(
            candidates,
            key=lambda x: (
                balanced_score(x),
                x["tan_final_6m"],
                x["total_cost_6m"],
            ),
        )[0]
    best["is_feasible"] = 1.0
    return best


def evaluate_scenario(
    target_df: pd.DataFrame,
    col_map: ColumnMap,
    budget_cap: float,
    drop_scale: float,
    scenario_name: str,
    objective_mode: str,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    for _, r in target_df.iterrows():
        tan_init = float(r[col_map.tan_score])
        out = optimize_patient_scenario(
            age_group=int(r[col_map.age_group]),
            activity_total=float(r[col_map.activity_total]),
            tan_init=tan_init,
            budget_cap=budget_cap,
            drop_scale=drop_scale,
            objective_mode=objective_mode,
        )
        tan_final = float(out["tan_final_6m"])
        reduction_rate = (tan_init - tan_final) / tan_init if tan_init > 0 else 0.0
        rows.append(
            {
                "scenario": scenario_name,
                "sample_id": int(r[col_map.sample_id]),
                "tan_init": tan_init,
                "tan_final_6m": tan_final,
                "tan_reduction_rate": float(reduction_rate),
                "total_cost_6m": float(out["total_cost_6m"]) if not np.isnan(out["total_cost_6m"]) else np.nan,
                "is_feasible": int(out["is_feasible"]),
            }
        )

    res = pd.DataFrame(rows)
    summary = {
        "scenario": scenario_name,
        "budget_cap": float(budget_cap),
        "drop_scale": float(drop_scale),
        "n_patients": int(len(res)),
        "feasible_rate": float(res["is_feasible"].mean()),
        "mean_tan_reduction_rate": float(res["tan_reduction_rate"].mean()),
        "mean_total_cost_6m": float(res["total_cost_6m"].mean(skipna=True)),
        "median_total_cost_6m": float(res["total_cost_6m"].median(skipna=True)),
    }
    return res, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Q3敏感性验证")
    parser.add_argument("--input-csv", type=str, default="附件1_样例数据.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/q3")
    parser.add_argument("--target-constitution", type=int, default=5)
    parser.add_argument(
        "--objective-mode",
        type=str,
        default="balanced",
        choices=["balanced", "upper"],
        help="敏感性分析采用的目标模式：balanced(默认) 或 upper(疗效上界)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv, encoding="utf-8-sig")
    col_map = build_column_map(df)
    cols = [
        col_map.sample_id,
        col_map.constitution_label,
        col_map.tan_score,
        col_map.activity_total,
        col_map.age_group,
    ]
    df = to_numeric(df, cols)
    df = df.dropna(subset=cols).copy()
    target_df = df[df[col_map.constitution_label].astype(int) == int(args.target_constitution)].copy()

    scenarios = [
        ("baseline", 2000.0, 1.00),
        ("budget_1600", 1600.0, 1.00),
        ("budget_1800", 1800.0, 1.00),
        ("budget_2200", 2200.0, 1.00),
        ("drop_x0.90", 2000.0, 0.90),
        ("drop_x1.10", 2000.0, 1.10),
    ]

    all_rows = []
    summaries = []
    for name, budget_cap, drop_scale in scenarios:
        res, summary = evaluate_scenario(
            target_df=target_df,
            col_map=col_map,
            budget_cap=budget_cap,
            drop_scale=drop_scale,
            scenario_name=name,
            objective_mode=args.objective_mode,
        )
        all_rows.append(res)
        summaries.append(summary)

    detail_df = pd.concat(all_rows, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    detail_df.to_csv(out_dir / "q3_sensitivity_detail.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "q3_sensitivity_summary.csv", index=False, encoding="utf-8-sig")

    sample_view = detail_df[detail_df["sample_id"].isin([1, 2, 3])].copy()
    sample_view.to_csv(out_dir / "q3_sensitivity_sample_1_2_3.csv", index=False, encoding="utf-8-sig")

    baseline = summary_df[summary_df["scenario"] == "baseline"].iloc[0]
    summary_json = {
        "n_target_patients": int(len(target_df)),
        "objective_mode": args.objective_mode,
        "baseline": {
            "mean_tan_reduction_rate": float(baseline["mean_tan_reduction_rate"]),
            "mean_total_cost_6m": float(baseline["mean_total_cost_6m"]),
        },
        "scenarios": summaries,
    }
    with open(out_dir / "q3_sensitivity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    print("Q3敏感性验证完成")
    print(f"- 目标体质样本数: {len(target_df)}")
    print(f"- 输出文件: {out_dir / 'q3_sensitivity_summary.csv'}")


if __name__ == "__main__":
    main()
