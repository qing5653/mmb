#!/usr/bin/env python3
"""问题2：网文式双轨预警 + 规则分层。

输出：
1) q2_risk_predictions.csv：每个样本的风险等级与模型概率。
2) q2_thresholds.json：三级风险阈值与规则依据。
3) q2_feature_importance.csv：模型A（去血脂）特征重要性。
4) q2_high_risk_core_combos.csv：核心高风险组合与lift。
5) q2_summary.json：核心评估指标与分层结果汇总。
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


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
    bmi: str
    constitution_label: Optional[str]
    constitution_scores: List[str]
    core_lipids: List[str]
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
    activity_total = find_column(df, r"活动量表总分")
    bmi = find_column(df, r"BMI")

    candidate_features = [
        hdl,
        ldl,
        tg,
        tc,
        find_column(df, r"空腹血糖"),
        find_column(df, r"血尿酸"),
        bmi,
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
        bmi=bmi,
        constitution_label=find_column_optional(df, r"体质标签"),
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


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def cv_metrics(df: pd.DataFrame, features: List[str], label_col: str, seed: int) -> Dict[str, float]:
    x = df[features].values
    y = df[label_col].astype(int).values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    aucs, f1s, accs = [], [], []
    for tr, te in skf.split(x, y):
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            random_state=seed,
        )
        model.fit(x[tr], y[tr])
        prob = model.predict_proba(x[te])[:, 1]
        met = evaluate_binary(y[te], prob, threshold=0.5)
        aucs.append(met["auc"])
        f1s.append(met["f1"])
        accs.append(met["accuracy"])

    auc_mean = float(np.mean(aucs))
    auc_std = float(np.std(aucs, ddof=0))
    return {
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "f1_mean": float(np.mean(f1s)),
        "accuracy_mean": float(np.mean(accs)),
        "fold_aucs": [float(x) for x in aucs],
    }


def build_oof_scores(train_df: pd.DataFrame, features: List[str], label_col: str, seed: int) -> np.ndarray:
    """对训练集生成折外(O O F)概率，降低阈值搜索的乐观偏差。"""
    x = train_df[features].values
    y = train_df[label_col].astype(int).values
    oof = np.zeros(len(train_df), dtype=float)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in skf.split(x, y):
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            random_state=seed,
        )
        model.fit(x[tr], y[tr])
        oof[va] = model.predict_proba(x[va])[:, 1]
    return oof


def add_lipid_abnormal_flags(df: pd.DataFrame, col_map: ColumnMap) -> pd.DataFrame:
    out = df.copy()
    tc, tg, ldl, hdl = col_map.core_lipids
    out["abn_tc"] = (out[tc] > 6.2).astype(int)
    out["abn_tg"] = (out[tg] > 1.7).astype(int)
    out["abn_ldl"] = (out[ldl] > 3.1).astype(int)
    out["abn_hdl"] = (out[hdl] < 1.04).astype(int)
    out["abnormal_lipid"] = ((out["abn_tc"] + out["abn_tg"] + out["abn_ldl"] + out["abn_hdl"]) > 0).astype(int)
    out["abnormal_lipid_count"] = out[["abn_tc", "abn_tg", "abn_ldl", "abn_hdl"]].sum(axis=1)
    return out


def search_tier_thresholds(train_score: np.ndarray, train_label: np.ndarray) -> Tuple[float, float]:
    """在训练集上搜索三层阈值，保证阳性率呈 低<中<高 的递进关系。"""
    qs = np.arange(0.20, 0.91, 0.05)
    cand = [float(np.quantile(train_score, q)) for q in qs]

    best = None
    for low_th in cand:
        for high_th in cand:
            if high_th <= low_th:
                continue

            low_mask = train_score < low_th
            mid_mask = (train_score >= low_th) & (train_score < high_th)
            high_mask = train_score >= high_th

            n_low = int(low_mask.sum())
            n_mid = int(mid_mask.sum())
            n_high = int(high_mask.sum())

            # 每层至少占训练集10%，避免过小分层导致不稳定。
            if min(n_low, n_mid, n_high) < max(1, int(0.10 * len(train_score))):
                continue

            r_low = float(train_label[low_mask].mean())
            r_mid = float(train_label[mid_mask].mean())
            r_high = float(train_label[high_mask].mean())

            # 强制单调递进，并要求有最小分离度。
            if not (r_low + 0.03 <= r_mid and r_mid + 0.01 <= r_high):
                continue

            sep = (r_high - r_mid) + (r_mid - r_low)
            balance = -abs((n_low / len(train_score)) - (n_high / len(train_score)))
            score = sep + 0.05 * balance

            if best is None or score > best[0]:
                best = (score, low_th, high_th)

    if best is not None:
        return float(best[1]), float(best[2])

    # 若训练集分布极端导致无可行组合，则回退到稳定分位点。
    return float(np.quantile(train_score, 0.35)), float(np.quantile(train_score, 0.75))


def assign_score_tier(df: pd.DataFrame, score_col: str, low_threshold: float, high_threshold: float) -> pd.Series:
    score = df[score_col]
    risk = np.where(score >= high_threshold, "高风险", np.where(score < low_threshold, "低风险", "中风险"))
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
    return agg.sort_values("_order").drop(columns=["_order"])


def build_web_core_combos(df: pd.DataFrame, col_map: ColumnMap) -> pd.DataFrame:
    y_col = col_map.label
    tan = col_map.tan_score
    act = col_map.activity_total
    tc, tg, _, _ = col_map.core_lipids

    combos: List[Tuple[str, pd.Series]] = [
        ("痰湿积分>=60 & TG>=1.7", (df[tan] >= 60) & (df[tg] >= 1.7)),
        ("BMI>=25 & TG>=1.7", (df[col_map.bmi] >= 25) & (df[tg] >= 1.7)),
        ("TG>=1.7 & TC>=6.2", (df[tg] >= 1.7) & (df[tc] >= 6.2)),
        ("痰湿积分>=60 & 活动<40", (df[tan] >= 60) & (df[act] < 40)),
        ("痰湿积分>=60 & 年龄组>=3", (df[tan] >= 60) & (df[col_map.age_group] >= 3)),
        ("痰湿积分>=60 & BMI>=25", (df[tan] >= 60) & (df[col_map.bmi] >= 25)),
    ]

    if col_map.constitution_label is not None:
        cl = col_map.constitution_label
        combos.extend(
            [
                ("痰湿体质(标签=5) & 活动<40", (df[cl] == 5) & (df[act] < 40)),
                ("痰湿体质(标签=5) & BMI>=25", (df[cl] == 5) & (df[col_map.bmi] >= 25)),
            ]
        )

    base_rate = float(df[y_col].mean())
    rows = []
    for name, mask in combos:
        sub = df[mask]
        if len(sub) == 0:
            continue
        rate = float(sub[y_col].mean())
        lift = float(rate / (base_rate + 1e-12))
        if lift >= 1.2:
            risk_grade = "高"
        elif lift >= 1.0:
            risk_grade = "中"
        else:
            risk_grade = "低"

        rows.append(
            {
                "combo": name,
                "sample_count": int(len(sub)),
                "positive_rate": rate,
                "lift": lift,
                "risk_grade": risk_grade,
            }
        )

    out = pd.DataFrame(rows).sort_values(["lift", "sample_count"], ascending=[False, False])
    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="MathorCup C题-问题2网文式重算")
    parser.add_argument("--input-csv", type=str, default="附件1_样例数据.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/q2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    col_map = build_column_map(df)

    no_lipid_features = [f for f in col_map.candidate_features if f not in col_map.core_lipids]
    with_lipid_features = col_map.candidate_features.copy()

    numeric_cols = list(
        set(
            [
                col_map.label,
                col_map.sample_id,
                col_map.tan_score,
                col_map.activity_total,
                col_map.age_group,
                col_map.bmi,
            ]
            + with_lipid_features
            + ([col_map.constitution_label] if col_map.constitution_label else [])
        )
    )
    df = to_numeric_frame(df, numeric_cols)

    required_cols = [col_map.sample_id, col_map.label] + with_lipid_features
    df = df.dropna(subset=required_cols).copy().reset_index(drop=True)
    df[col_map.label] = df[col_map.label].astype(int)

    df = add_lipid_abnormal_flags(df, col_map)

    # 双轨5折评估
    track_a = cv_metrics(df, no_lipid_features, col_map.label, args.seed)
    track_b = cv_metrics(df, with_lipid_features, col_map.label, args.seed)

    # 用轨道A模型生成风险分值（用于输出与可视化）
    train_df, val_df, test_df = split_data(df, col_map.label, args.seed)

    # 训练集分值采用OOF，避免in-sample过乐观导致分层异常。
    train_oof_prob = build_oof_scores(train_df, no_lipid_features, col_map.label, args.seed)

    model_a = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=args.seed,
    )
    model_a.fit(train_df[no_lipid_features].values, train_df[col_map.label].values)

    val_prob = model_a.predict_proba(val_df[no_lipid_features].values)[:, 1]
    test_prob = model_a.predict_proba(test_df[no_lipid_features].values)[:, 1]

    # 按样本ID合并三段分值：train用OOF，val/test用留出预测。
    score_map: Dict[int, float] = {}
    for sid, p in zip(train_df[col_map.sample_id].astype(int).tolist(), train_oof_prob.tolist()):
        score_map[int(sid)] = float(p)
    for sid, p in zip(val_df[col_map.sample_id].astype(int).tolist(), val_prob.tolist()):
        score_map[int(sid)] = float(p)
    for sid, p in zip(test_df[col_map.sample_id].astype(int).tolist(), test_prob.tolist()):
        score_map[int(sid)] = float(p)

    pred_df = df[
        [
            col_map.sample_id,
            col_map.label,
            col_map.tan_score,
            col_map.activity_total,
            col_map.bmi,
            col_map.age_group,
            col_map.core_lipids[0],
            col_map.core_lipids[1],
            col_map.core_lipids[2],
            col_map.core_lipids[3],
            "abn_tc",
            "abn_tg",
            "abn_ldl",
            "abn_hdl",
            "abnormal_lipid",
            "abnormal_lipid_count",
        ]
    ].copy()

    if col_map.constitution_label is not None:
        pred_df[col_map.constitution_label] = df[col_map.constitution_label]

    pred_df["risk_score"] = pred_df[col_map.sample_id].astype(int).map(score_map)

    # 阈值仅在训练集分值上估计，避免泄漏。
    train_prob = train_oof_prob
    low_threshold, high_threshold = search_tier_thresholds(
        train_prob,
        train_df[col_map.label].astype(int).values,
    )
    pred_df["risk_level"] = assign_score_tier(
        pred_df,
        score_col="risk_score",
        low_threshold=float(low_threshold),
        high_threshold=float(high_threshold),
    )

    pred_df["risk_index"] = pred_df["risk_score"]

    train_ids = set(train_df[col_map.sample_id].tolist())
    val_ids = set(val_df[col_map.sample_id].tolist())
    pred_df["data_split"] = pred_df[col_map.sample_id].map(
        lambda x: "train" if x in train_ids else ("val" if x in val_ids else "test")
    )

    # 辅助解释标记（不参与风险层级计算）
    pred_df["profile_tan_ge_70_act_lt_40"] = (
        (pred_df[col_map.tan_score] >= 70) & (pred_df[col_map.activity_total] < 40)
    ).astype(int)
    pred_df["profile_tan_lt_50_act_ge_60"] = (
        (pred_df[col_map.tan_score] < 50) & (pred_df[col_map.activity_total] >= 60)
    ).astype(int)

    pred_df.to_csv(output_dir / "q2_risk_predictions.csv", index=False, encoding="utf-8-sig")

    # 模型A特征重要性
    importance_df = pd.DataFrame(
        {"feature": no_lipid_features, "importance": model_a.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(output_dir / "q2_feature_importance.csv", index=False, encoding="utf-8-sig")

    tier_df = summarize_risk_tiers(pred_df, col_map.label)
    tier_val_df = summarize_risk_tiers(pred_df[pred_df["data_split"] == "val"], col_map.label)
    tier_test_df = summarize_risk_tiers(pred_df[pred_df["data_split"] == "test"], col_map.label)
    tier_df.to_csv(output_dir / "q2_risk_tier_summary.csv", index=False, encoding="utf-8-sig")
    tier_val_df.to_csv(output_dir / "q2_risk_tier_summary_val.csv", index=False, encoding="utf-8-sig")
    tier_test_df.to_csv(output_dir / "q2_risk_tier_summary_test.csv", index=False, encoding="utf-8-sig")

    combo_df = build_web_core_combos(pred_df, col_map)
    combo_df.to_csv(output_dir / "q2_high_risk_core_combos.csv", index=False, encoding="utf-8-sig")

    thresholds = {
        "score_based_tier": {
            "base_rule": [
                f"risk_score < {float(low_threshold):.6f} -> 低风险",
                f"{float(low_threshold):.6f} <= risk_score < {float(high_threshold):.6f} -> 中风险",
                f"risk_score >= {float(high_threshold):.6f} -> 高风险",
            ],
            "threshold_selection": "在训练集上网格搜索分位点，约束三层阳性率呈低<中<高并保证每层最小样本占比。",
            "note": "分层仅依赖轨道A去血脂模型分值，不使用核心血脂阈值；痰湿/活动仅用于人群画像解释。",
        },
        "lipid_abnormal_definition": {
            "TC": ">6.2 mmol/L",
            "TG": ">1.7 mmol/L",
            "LDL-C": ">3.1 mmol/L",
            "HDL-C": "<1.04 mmol/L",
        },
        "model_track": {
            "A_no_lipid_features": no_lipid_features,
            "B_with_lipid_features": with_lipid_features,
        },
    }
    with open(output_dir / "q2_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)

    summary = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "sample_size": int(len(df)),
        "model": "GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1)",
        "track_A_no_lipid_5fold": {
            "auc_mean": track_a["auc_mean"],
            "auc_std": track_a["auc_std"],
            "f1_mean": track_a["f1_mean"],
            "accuracy_mean": track_a["accuracy_mean"],
        },
        "track_B_with_lipid_5fold": {
            "auc_mean": track_b["auc_mean"],
            "auc_std": track_b["auc_std"],
            "f1_mean": track_b["f1_mean"],
            "accuracy_mean": track_b["accuracy_mean"],
        },
        "risk_level_counts": pred_df["risk_level"].value_counts().to_dict(),
        "risk_level_positive_rate": tier_df.set_index("risk_level")["positive_rate"].to_dict(),
        "risk_level_positive_rate_val": tier_val_df.set_index("risk_level")["positive_rate"].to_dict(),
        "risk_level_positive_rate_test": tier_test_df.set_index("risk_level")["positive_rate"].to_dict(),
        "top_high_risk_combos": combo_df.head(8).to_dict(orient="records"),
        "thresholds_file": "q2_thresholds.json",
    }

    with open(output_dir / "q2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Q2运行完成（网文式重算）")
    print(f"- 轨道A去血脂 5-fold AUC: {track_a['auc_mean']:.4f}±{track_a['auc_std']:.4f}")
    print(f"- 轨道B含血脂 5-fold AUC: {track_b['auc_mean']:.4f}±{track_b['auc_std']:.4f}")
    print(f"- 低/中/高风险样本数: {pred_df['risk_level'].value_counts().to_dict()}")
    print(f"- 输出目录: {output_dir}")


if __name__ == "__main__":
    main()
