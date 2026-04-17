#!/usr/bin/env python3
"""问题3可视化脚本。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _setup() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=320, bbox_inches="tight")
    plt.close()


def plot_plan_distribution(df: pd.DataFrame, out: Path) -> None:
    view = df.groupby(["regulation_level", "activity_intensity"], as_index=False).size()

    plt.figure(figsize=(8.6, 6.0))
    ax = sns.barplot(
        data=view,
        x="regulation_level",
        y="size",
        hue="activity_intensity",
        palette="viridis",
    )
    plt.title("Q3 Plan Distribution", weight="bold")
    plt.xlabel("Regulation Level")
    plt.ylabel("Patient Count")
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.text(p.get_x() + p.get_width() / 2, h + 0.5, f"{int(h)}", ha="center", fontsize=10)
    _save(out / "q3_plan_distribution.png")


def plot_cost_reduction(df: pd.DataFrame, out: Path) -> None:
    plt.figure(figsize=(8.8, 6.2))
    sns.scatterplot(
        data=df,
        x="total_cost_6m",
        y="tan_reduction_rate",
        hue="activity_intensity",
        size="frequency_per_week",
        palette="magma",
        alpha=0.8,
    )
    plt.title("Q3 Cost vs Reduction Rate", weight="bold")
    plt.xlabel("Total Cost (6 months)")
    plt.ylabel("Tan-score Reduction Rate")
    _save(out / "q3_cost_vs_reduction.png")


def plot_sample_trajectory(sample_df: pd.DataFrame, out: Path) -> None:
    if sample_df.empty:
        return

    rows = []
    for _, r in sample_df.iterrows():
        vals = [float(x) for x in str(r["trajectory"]).split(";")]
        for month, v in enumerate(vals):
            rows.append({"sample_id": int(r["sample_id"]), "month": month, "tan_score": v})

    traj = pd.DataFrame(rows)
    plt.figure(figsize=(8.8, 6.0))
    sns.lineplot(data=traj, x="month", y="tan_score", hue="sample_id", marker="o", linewidth=2.2)
    plt.title("Q3 Sample 1/2/3 Tan-score Trajectory", weight="bold")
    plt.xlabel("Month")
    plt.ylabel("Predicted Tan-score")
    _save(out / "q3_sample_1_2_3_trajectory.png")


def main() -> None:
    _setup()

    q3_dir = Path("outputs/q3")
    fig_dir = q3_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(q3_dir / "q3_patient_optimal_plans.csv", encoding="utf-8-sig")
    sample_df = pd.read_csv(q3_dir / "q3_sample_1_2_3_optimal_plan.csv", encoding="utf-8-sig")

    plot_plan_distribution(df, fig_dir)
    plot_cost_reduction(df, fig_dir)
    plot_sample_trajectory(sample_df, fig_dir)

    print("Q3可视化完成")
    print(f"输出目录: {fig_dir}")


if __name__ == "__main__":
    main()
