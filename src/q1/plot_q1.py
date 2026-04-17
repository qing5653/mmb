#!/usr/bin/env python3
"""问题1结果可视化脚本。

读取 outputs/q1 下结果，输出专业图表到 outputs/q1/figures。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter


FEATURE_MAP = {
    "TG（甘油三酯）": "TG",
    "TC（总胆固醇）": "TC",
    "血尿酸": "Uric Acid",
    "LDL-C（低密度脂蛋白）": "LDL-C",
    "HDL-C（高密度脂蛋白）": "HDL-C",
    "空腹血糖": "Fasting Glucose",
    "BMI": "BMI",
    "ADL总分": "ADL Score",
    "IADL总分": "IADL Score",
    "活动量表总分（ADL总分+IADL总分）": "Total Activity Score",
    "活动量表总分": "Total Activity Score",
}

CONSTITUTION_MAP = {
    "平和质": "Balanced",
    "气虚质": "Qi-deficient",
    "阳虚质": "Yang-deficient",
    "阴虚质": "Yin-deficient",
    "痰湿质": "Phlegm-dampness",
    "湿热质": "Damp-heat",
    "血瘀质": "Blood-stasis",
    "气郁质": "Qi-depressed",
    "特禀质": "Special diathesis",
}


def _prepare_font() -> None:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "#fcfcfd"
    plt.rcParams["axes.edgecolor"] = "#d0d7de"
    plt.rcParams["grid.color"] = "#e5e7eb"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.45


def _beautify_axis() -> None:
    ax = plt.gca()
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def _save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close()


def _add_hbar_value_labels(ax: plt.Axes, fmt: str = "{:.3f}") -> None:
    for p in ax.patches:
        w = p.get_width()
        y = p.get_y() + p.get_height() / 2
        ax.text(w + 0.005 * max(1.0, ax.get_xlim()[1]), y, fmt.format(w), va="center", ha="left", fontsize=10)


def _map_feature_name(s: str) -> str:
    return FEATURE_MAP.get(s, s)


def _map_constitution_name(s: str) -> str:
    return CONSTITUTION_MAP.get(s, s)


def _load_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def plot_feature_votes(fs: pd.DataFrame, out: Path) -> None:
    view = fs.sort_values(["votes", "rf_importance"], ascending=[False, False]).copy()
    view["selected"] = view["final_selected"].map({True: "入选", False: "未入选"})
    view["feature_en"] = view["feature"].map(_map_feature_name)
    view["selected"] = view["selected"].map({"入选": "Selected", "未入选": "Not selected"})

    plt.figure(figsize=(11, 6.4))
    ax = sns.barplot(
        data=view,
        x="votes",
        y="feature_en",
        hue="selected",
        dodge=False,
        palette={"Selected": "#0f766e", "Not selected": "#94a3b8"},
        edgecolor="#334155",
        linewidth=0.5,
    )
    plt.title("Q1 Feature Voting Results", fontsize=18, weight="bold")
    plt.xlabel("Vote Count (Three-method Ensemble)")
    plt.ylabel("Candidate Feature")
    plt.xlim(0, 3.2)
    _add_hbar_value_labels(ax, "{:.0f}")
    _beautify_axis()
    _save_fig(out / "q1_feature_votes.png")


def plot_vote_heatmap(fs: pd.DataFrame, out: Path) -> None:
    view = fs.copy()
    view["feature_en"] = view["feature"].map(_map_feature_name)
    hm = view[["feature_en", "corr_selected", "lasso_selected", "rf_selected", "votes"]].copy()
    hm = hm.sort_values(["votes", "rf_selected", "lasso_selected", "corr_selected"], ascending=False)
    mat = hm[["corr_selected", "lasso_selected", "rf_selected"]].astype(int)
    mat.index = hm["feature_en"]

    plt.figure(figsize=(7.6, 6.2))
    ax = sns.heatmap(
        mat,
        cmap=sns.color_palette(["#f8fafc", "#0f766e"], as_cmap=True),
        cbar=False,
        linewidths=0.8,
        linecolor="#e2e8f0",
        annot=True,
        fmt="d",
        annot_kws={"fontsize": 11, "weight": "bold"},
    )
    ax.set_title("Method-level Selection Matrix", fontsize=16, weight="bold", pad=12)
    ax.set_xlabel("Selection Method")
    ax.set_ylabel("Feature")
    ax.set_xticklabels(["Correlation", "L1-Logistic", "Random Forest"], rotation=0)
    _beautify_axis()
    _save_fig(out / "q1_method_heatmap.png")


def plot_rf_importance(fs: pd.DataFrame, out: Path) -> None:
    view = fs.sort_values("rf_importance", ascending=False).copy()
    view["feature_en"] = view["feature"].map(_map_feature_name)

    plt.figure(figsize=(11, 6.4))
    palette = sns.color_palette("Blues_r", n_colors=len(view))
    ax = sns.barplot(
        data=view,
        x="rf_importance",
        y="feature_en",
        hue="feature_en",
        palette=palette,
        edgecolor="#334155",
        linewidth=0.4,
        legend=False,
    )
    mean_val = float(view["rf_importance_mean"].iloc[0]) if "rf_importance_mean" in view.columns else float(view["rf_importance"].mean())
    plt.axvline(mean_val, color="#ef4444", linestyle="--", linewidth=1.5, label=f"Mean={mean_val:.3f}")
    plt.title("Q1 Random Forest Feature Importance", fontsize=18, weight="bold")
    plt.xlabel("Gini-based Importance")
    plt.ylabel("Candidate Feature")
    _add_hbar_value_labels(ax, "{:.3f}")
    plt.legend(frameon=False, loc="lower right")
    _beautify_axis()
    _save_fig(out / "q1_rf_importance.png")


def plot_or_forest(or_df: pd.DataFrame, out: Path) -> None:
    view = or_df[(or_df["variable"] != "const")].copy()
    view = view.sort_values("or", ascending=False)
    view["variable_en"] = view["variable"].map(_map_constitution_name)
    view["sig"] = (view["or_ci_low"] > 1) | (view["or_ci_high"] < 1)

    plt.figure(figsize=(10.2, 6.8))
    y = range(len(view))
    for i, (_, row) in enumerate(view.iterrows()):
        line_color = "#dc2626" if bool(row["sig"]) else "#64748b"
        point_color = "#b91c1c" if bool(row["sig"]) else "#1d4ed8"
        plt.hlines(i, row["or_ci_low"], row["or_ci_high"], color=line_color, linewidth=2.2, alpha=0.9)
        plt.plot(row["or"], i, "o", color=point_color, markersize=7)

    plt.axvline(1.0, linestyle="--", color="#111827", linewidth=1.3)
    plt.yticks(list(y), view["variable_en"])
    plt.xscale("log")
    plt.xlabel("OR (log scale)")
    plt.title("OR and 95% CI of Nine Constitutions", fontsize=18, weight="bold")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}" if v >= 1 else f"{v:.2f}"))
    _beautify_axis()
    _save_fig(out / "q1_or_forest.png")


def plot_auc_compare(summary: dict, out: Path) -> None:
    auc_const = summary.get("diagnostics", {}).get("val_auc", float("nan"))
    auc_feat = summary.get("selected_feature_model", {}).get("val_auc", float("nan"))
    comp = pd.DataFrame(
        {
            "model": ["Constitution Logistic", "Selected-feature Risk Model"],
            "auc": [auc_const, auc_feat],
        }
    )

    plt.figure(figsize=(8.2, 5.6))
    sns.barplot(data=comp, x="model", y="auc", hue="model", palette=["#64748b", "#10b981"], legend=False)
    plt.ylim(0, 1.05)
    plt.ylabel("Validation AUC")
    plt.xlabel("")
    plt.title("Q1 Validation AUC Comparison", fontsize=18, weight="bold")
    for i, v in enumerate(comp["auc"]):
        plt.text(i, min(v + 0.02, 1.02), f"{v:.4f}", ha="center", fontsize=11, weight="bold")
    _beautify_axis()
    _save_fig(out / "q1_auc_compare.png")


def plot_selected_coef(coef_df: pd.DataFrame, out: Path) -> None:
    view = coef_df.sort_values("coef", ascending=False)
    view["feature_en"] = view["feature"].map(_map_feature_name)

    plt.figure(figsize=(9, 5.8))
    palette = sns.color_palette("YlOrBr", n_colors=len(view))
    ax = sns.barplot(
        data=view,
        x="coef",
        y="feature_en",
        hue="feature_en",
        palette=palette,
        edgecolor="#7c2d12",
        linewidth=0.4,
        legend=False,
    )
    plt.title("Coefficients of Selected-feature Risk Model", fontsize=18, weight="bold")
    plt.xlabel("Standardized Coefficient")
    plt.ylabel("Feature")
    _add_hbar_value_labels(ax, "{:.3f}")
    _beautify_axis()
    _save_fig(out / "q1_selected_model_coef.png")


def main() -> None:
    _prepare_font()

    q1_dir = Path("outputs/q1")
    fig_dir = q1_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fs = _load_table(q1_dir / "feature_selection_details.csv")
    or_df = _load_table(q1_dir / "OR_values_table.csv")
    coef_df = _load_table(q1_dir / "selected_feature_model_coef.csv")
    with open(q1_dir / "q1_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    plot_feature_votes(fs, fig_dir)
    plot_vote_heatmap(fs, fig_dir)
    plot_rf_importance(fs, fig_dir)
    plot_or_forest(or_df, fig_dir)
    plot_auc_compare(summary, fig_dir)
    plot_selected_coef(coef_df, fig_dir)

    print("Q1可视化完成")
    print(f"输出目录: {fig_dir}")


if __name__ == "__main__":
    main()
