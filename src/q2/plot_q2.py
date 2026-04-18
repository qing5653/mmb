#!/usr/bin/env python3
"""问题2可视化脚本。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager


FEATURE_MAP = {
    "痰湿质": "痰湿积分",
    "活动量表总分（ADL总分+IADL总分）": "活动总分",
    "TG（甘油三酯）": "TG",
    "TC（总胆固醇）": "TC",
    "LDL-C（低密度脂蛋白）": "LDL-C",
    "HDL-C（高密度脂蛋白）": "HDL-C",
    "空腹血糖": "空腹血糖",
    "血尿酸": "血尿酸",
    "平和质": "平和质",
    "气虚质": "气虚质",
    "阳虚质": "阳虚质",
    "阴虚质": "阴虚质",
    "湿热质": "湿热质",
    "血瘀质": "血瘀质",
    "气郁质": "气郁质",
    "特禀质": "特禀质",
}

RISK_MAP = {"低风险": "低风险", "中风险": "中风险", "高风险": "高风险"}
COMBO_TOKEN_MAP = {
    "痰湿高分": "痰湿高分",
    "活动能力低": "活动能力低",
    "TG异常": "TG异常",
    "TC异常": "TC异常",
    "LDL异常": "LDL异常",
    "HDL偏低": "HDL偏低",
    "BMI偏高": "BMI偏高",
    "血尿酸偏高": "血尿酸偏高",
    "年龄偏高": "年龄偏高",
    "吸烟史": "吸烟史",
    "饮酒史": "饮酒史",
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
    order = ["低风险", "中风险", "高风险"]
    cnt = pred["risk_level"].value_counts().reindex(order).fillna(0).reset_index()
    cnt.columns = ["risk_level", "count"]

    plt.figure(figsize=(8.5, 5.8))
    ax = sns.barplot(data=cnt, x="risk_level", y="count", hue="risk_level", palette=["#2a9d8f", "#e9c46a", "#e76f51"], legend=False)
    plt.title("问题二风险层样本分布", weight="bold")
    plt.xlabel("风险层级")
    plt.ylabel("样本数")
    for p in ax.patches:
        h = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2, h + max(cnt["count"]) * 0.01, f"{int(h)}", ha="center", va="bottom", fontsize=11)
    _save(out / "q2_risk_tier_distribution.png")


def plot_score_box(pred: pd.DataFrame, thresholds: dict, out: Path) -> None:
    pred = pred.copy()
    pred["risk_level"] = pred["risk_level"].map(RISK_MAP)
    order = ["低风险", "中风险", "高风险"]
    prob_cfg = thresholds.get("probability_threshold", {})
    t_low = prob_cfg.get("t_low")
    t_high = prob_cfg.get("t_high")

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
    if t_low is not None:
        plt.axhline(t_low, color="#2a9d8f", linestyle="--", linewidth=1.3, label=f"t_low={t_low:.3f}")
    if t_high is not None:
        plt.axhline(t_high, color="#e76f51", linestyle="--", linewidth=1.3, label=f"t_high={t_high:.3f}")
    plt.title("问题二各风险层风险分值分布", weight="bold")
    plt.xlabel("风险层级")
    plt.ylabel("模型风险分值")
    if t_low is not None or t_high is not None:
        plt.legend(frameon=False, loc="upper left")
    _save(out / "q2_risk_score_boxplot.png")


def plot_feature_importance(imp: pd.DataFrame, out: Path) -> None:
    imp = imp.copy()
    imp["feature"] = imp["feature"].map(_map_feature_name)
    top = imp.head(12).copy().sort_values("importance", ascending=True)
    plt.figure(figsize=(10, 6.8))
    ax = sns.barplot(data=top, x="importance", y="feature", hue="feature", palette="viridis", legend=False)
    plt.title("问题二模型特征重要性（前12）", weight="bold")
    plt.xlabel("重要性")
    plt.ylabel("特征")
    for p in ax.patches:
        w = p.get_width()
        y = p.get_y() + p.get_height() / 2
        ax.text(w + top["importance"].max() * 0.01, y, f"{w:.3f}", va="center", fontsize=10)
    _save(out / "q2_feature_importance_top12.png")


def plot_core_combos(combo: pd.DataFrame, out: Path) -> None:
    if combo.empty:
        return

    combo = combo.copy()
    if "support_high" not in combo.columns and "positive_rate" in combo.columns:
        combo["support_high"] = combo["positive_rate"]

    top = combo.head(10).copy().sort_values("support_high", ascending=True)
    top["combo"] = top["combo"].map(_map_combo)
    plt.figure(figsize=(11, 7.2))
    ax = sns.barplot(data=top, x="support_high", y="combo", hue="combo", palette="magma", legend=False)
    plt.title("问题二高风险核心特征组合", weight="bold")
    plt.xlabel("高风险组支持度")
    plt.ylabel("特征组合")
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
