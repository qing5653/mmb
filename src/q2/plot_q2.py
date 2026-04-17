#!/usr/bin/env python3
"""问题2可视化脚本。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FEATURE_MAP = {
    "痰湿质": "Tan-score",
    "活动量表总分（ADL总分+IADL总分）": "Activity total",
    "TG（甘油三酯）": "TG",
    "TC（总胆固醇）": "TC",
    "LDL-C（低密度脂蛋白）": "LDL-C",
    "HDL-C（高密度脂蛋白）": "HDL-C",
    "空腹血糖": "Fasting glucose",
    "血尿酸": "Uric acid",
    "平和质": "Balanced",
    "气虚质": "Qi-deficient",
    "阳虚质": "Yang-deficient",
    "阴虚质": "Yin-deficient",
    "湿热质": "Damp-heat",
    "血瘀质": "Blood-stasis",
    "气郁质": "Qi-depressed",
    "特禀质": "Special diathesis",
}

RISK_MAP = {"低风险": "Low", "中风险": "Medium", "高风险": "High"}
COMBO_TOKEN_MAP = {
    "痰湿高分": "High tan-score",
    "活动能力低": "Low activity",
    "TG异常": "TG abnormal",
    "TC异常": "TC abnormal",
    "LDL异常": "LDL abnormal",
    "HDL偏低": "Low HDL",
    "BMI偏高": "High BMI",
    "血尿酸偏高": "High uric acid",
    "年龄偏高": "Older age",
    "吸烟史": "Smoking",
    "饮酒史": "Drinking",
}


def _map_feature_name(s: str) -> str:
    return FEATURE_MAP.get(s, s)


def _map_combo(s: str) -> str:
    out = s
    for k, v in COMBO_TOKEN_MAP.items():
        out = out.replace(k, v)
    return out


def _setup() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "#fcfcfd"


def _save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=320, bbox_inches="tight")
    plt.close()


def plot_risk_distribution(pred: pd.DataFrame, out: Path) -> None:
    order = ["低风险", "中风险", "高风险"]
    pred = pred.copy()
    pred["risk_level"] = pred["risk_level"].map(RISK_MAP)
    order = ["Low", "Medium", "High"]
    cnt = pred["risk_level"].value_counts().reindex(order).fillna(0).reset_index()
    cnt.columns = ["risk_level", "count"]

    plt.figure(figsize=(8.5, 5.8))
    ax = sns.barplot(data=cnt, x="risk_level", y="count", hue="risk_level", palette=["#2a9d8f", "#e9c46a", "#e76f51"], legend=False)
    plt.title("Q2 Risk Tier Distribution", weight="bold")
    plt.xlabel("Risk Tier")
    plt.ylabel("Sample Count")
    for p in ax.patches:
        h = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2, h + max(cnt["count"]) * 0.01, f"{int(h)}", ha="center", va="bottom", fontsize=11)
    _save(out / "q2_risk_tier_distribution.png")


def plot_score_box(pred: pd.DataFrame, thresholds: dict, out: Path) -> None:
    pred = pred.copy()
    pred["risk_level"] = pred["risk_level"].map(RISK_MAP)
    order = ["Low", "Medium", "High"]
    t_low = thresholds["probability_threshold"]["t_low"]
    t_high = thresholds["probability_threshold"]["t_high"]

    plt.figure(figsize=(9.4, 6.0))
    sns.boxplot(
        data=pred,
        x="risk_level",
        y="risk_score",
        order=order,
        hue="risk_level",
        palette=["#2a9d8f", "#e9c46a", "#e76f51"],
        dodge=False,
        linewidth=1.2,
        legend=False,
    )
    plt.axhline(t_low, color="#2a9d8f", linestyle="--", linewidth=1.3, label=f"t_low={t_low:.3f}")
    plt.axhline(t_high, color="#e76f51", linestyle="--", linewidth=1.3, label=f"t_high={t_high:.3f}")
    plt.title("Q2 Risk Score by Tier", weight="bold")
    plt.xlabel("Risk Tier")
    plt.ylabel("Model Risk Score")
    plt.legend(frameon=False, loc="upper left")
    _save(out / "q2_risk_score_boxplot.png")


def plot_feature_importance(imp: pd.DataFrame, out: Path) -> None:
    imp = imp.copy()
    imp["feature"] = imp["feature"].map(_map_feature_name)
    top = imp.head(12).copy().sort_values("importance", ascending=True)
    plt.figure(figsize=(10, 6.8))
    ax = sns.barplot(data=top, x="importance", y="feature", hue="feature", palette="viridis", legend=False)
    plt.title("Q2 Model Top Feature Importance", weight="bold")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    for p in ax.patches:
        w = p.get_width()
        y = p.get_y() + p.get_height() / 2
        ax.text(w + top["importance"].max() * 0.01, y, f"{w:.3f}", va="center", fontsize=10)
    _save(out / "q2_feature_importance_top12.png")


def plot_core_combos(combo: pd.DataFrame, out: Path) -> None:
    if combo.empty:
        return

    top = combo.head(10).copy().sort_values("support_high", ascending=True)
    top["combo"] = top["combo"].map(_map_combo)
    plt.figure(figsize=(11, 7.2))
    ax = sns.barplot(data=top, x="support_high", y="combo", hue="combo", palette="magma", legend=False)
    plt.title("Q2 Core Feature Combinations in High-risk Group", weight="bold")
    plt.xlabel("Support in High-risk Group")
    plt.ylabel("Feature Combination")
    for p in ax.patches:
        w = p.get_width()
        y = p.get_y() + p.get_height() / 2
        ax.text(w + 0.005, y, f"{w:.2%}", va="center", fontsize=9)
    _save(out / "q2_high_risk_core_combos.png")


def main() -> None:
    _setup()

    q2_dir = Path("outputs/q2")
    fig_dir = q2_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(q2_dir / "q2_risk_predictions.csv", encoding="utf-8-sig")
    imp = pd.read_csv(q2_dir / "q2_feature_importance.csv", encoding="utf-8-sig").sort_values("importance", ascending=False)
    combo = pd.read_csv(q2_dir / "q2_high_risk_core_combos.csv", encoding="utf-8-sig")

    with open(q2_dir / "q2_thresholds.json", "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    plot_risk_distribution(pred, fig_dir)
    plot_score_box(pred, thresholds, fig_dir)
    plot_feature_importance(imp, fig_dir)
    plot_core_combos(combo, fig_dir)

    print("Q2可视化完成")
    print(f"输出目录: {fig_dir}")


if __name__ == "__main__":
    main()
