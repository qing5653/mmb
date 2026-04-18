#!/usr/bin/env python3
"""问题3可视化脚本。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager


def _setup() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    preferred = [
        "AR PL UMing CN",
        "Droid Sans Fallback",
        "Noto Sans CJK JP",
        "Noto Serif CJK JP",
        "AR PL UMing CN",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((f for f in preferred if f in installed), "DejaVu Sans")
    plt.rcParams["font.family"] = [chosen]
    plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
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
    plt.title("问题三最优方案分布", weight="bold")
    plt.xlabel("调理等级")
    plt.ylabel("患者数")
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
    plt.title("问题三成本与降幅关系", weight="bold")
    plt.xlabel("6个月总成本")
    plt.ylabel("痰湿降幅率")
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
    plt.title("问题三样本1/2/3痰湿积分轨迹", weight="bold")
    plt.xlabel("月份")
    plt.ylabel("预测痰湿积分")
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
