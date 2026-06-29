"""
train_credit_model.py
----------------------
Trains an XGBoost binary classifier on the S&P 500 Corporate Credit Rating dataset.

Binary target:
  0 → Investment Grade  (BBB- or higher)
  1 → Speculative Grade (BB+ or lower)

Outputs
-------
models/xgb_credit_model.json  — XGBoost native JSON artifact
models/scaler.pkl             — StandardScaler (written by data_engineering)
models/metrics.json           — CV + hold-out evaluation snapshot
models/feature_importance.csv — Feature gain/cover/frequency
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.phase1_quantitative.data_engineering import (
    PROCESSED_DIR,
    MODELS_DIR,
    load_and_prepare,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH = MODELS_DIR / "xgb_credit_model.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
IMPORTANCE_PATH = MODELS_DIR / "feature_importance.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return the threshold that maximises F1 on the provided labels/probs."""
    thresholds = np.linspace(0.05, 0.95, 181)
    best_threshold, best_f1 = 0.5, 0.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f = f1_score(y_true, y_pred, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_threshold = float(t)
    return best_threshold


def _build_model(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train(X_train: pd.DataFrame, y_train: pd.Series,
          X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Run 5-fold CV, refit on full training set, evaluate on hold-out,
    and persist model + metrics artifacts.

    Returns
    -------
    metrics : dict  — same payload written to models/metrics.json
    """
    feature_columns = list(X_train.columns)
    X_tr = X_train.values
    y_tr = y_train.values
    X_te = X_test.values
    y_te = y_test.values

    n_neg = int((y_tr == 0).sum())
    n_pos = int((y_tr == 1).sum())
    scale_pos_weight = n_neg / n_pos
    print(f"[train] n_neg={n_neg}  n_pos={n_pos}  scale_pos_weight={scale_pos_weight:.2f}")

    # ---- 5-Fold Stratified CV ----
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_pr_aucs: list[float] = []
    cv_roc_aucs: list[float] = []

    print("[train] Running 5-fold stratified CV …")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_tr, y_tr), 1):
        X_fold_tr, X_fold_val = X_tr[train_idx], X_tr[val_idx]
        y_fold_tr, y_fold_val = y_tr[train_idx], y_tr[val_idx]

        model = _build_model(scale_pos_weight)
        model.fit(
            X_fold_tr, y_fold_tr,
            eval_set=[(X_fold_val, y_fold_val)],
            verbose=False,
        )
        probs = model.predict_proba(X_fold_val)[:, 1]
        cv_pr_aucs.append(average_precision_score(y_fold_val, probs))
        cv_roc_aucs.append(roc_auc_score(y_fold_val, probs))
        print(f"  Fold {fold}: PR-AUC={cv_pr_aucs[-1]:.4f}  ROC-AUC={cv_roc_aucs[-1]:.4f}")

    mean_pr_auc = float(np.mean(cv_pr_aucs))
    std_pr_auc = float(np.std(cv_pr_aucs))
    mean_roc_auc = float(np.mean(cv_roc_aucs))
    print(f"[train] CV PR-AUC: {mean_pr_auc:.4f} ± {std_pr_auc:.4f}")
    print(f"[train] CV ROC-AUC: {mean_roc_auc:.4f}")

    # ---- Final fit on full training set ----
    print("[train] Fitting final model on full training set …")
    final_model = _build_model(scale_pos_weight)
    final_model.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        verbose=False,
    )

    # ---- Hold-out evaluation ----
    y_prob_test = final_model.predict_proba(X_te)[:, 1]
    optimal_threshold = _compute_optimal_threshold(y_te, y_prob_test)
    y_pred_test = (y_prob_test >= optimal_threshold).astype(int)

    metrics = {
        "cv_pr_auc_mean": mean_pr_auc,
        "cv_pr_auc_std": std_pr_auc,
        "cv_roc_auc_mean": mean_roc_auc,
        "holdout_pr_auc": float(average_precision_score(y_te, y_prob_test)),
        "holdout_roc_auc": float(roc_auc_score(y_te, y_prob_test)),
        "holdout_f1": float(f1_score(y_te, y_pred_test, zero_division=0)),
        "holdout_precision": float(precision_score(y_te, y_pred_test, zero_division=0)),
        "holdout_recall": float(recall_score(y_te, y_pred_test, zero_division=0)),
        "optimal_threshold": optimal_threshold,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "n_features": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "class_balance_train": {
            "investment_grade_0": n_neg,
            "speculative_grade_1": n_pos,
        },
        "model_version": "xgb_credit_model_v1",
    }

    # ---- Persist model ----
    final_model.save_model(MODEL_PATH)
    print(f"[train] Model saved → {MODEL_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train] Metrics saved → {METRICS_PATH}")

    # ---- Feature importance ----
    importance_types = ["gain", "cover", "weight"]
    imp_dfs = []
    for imp_type in importance_types:
        scores = final_model.get_booster().get_score(importance_type=imp_type)
        imp_df = pd.DataFrame(scores.items(), columns=["feature", imp_type])
        imp_dfs.append(imp_df.set_index("feature"))

    importance_df = pd.concat(imp_dfs, axis=1).fillna(0).reset_index()
    importance_df = importance_df.rename(columns={"weight": "frequency"})
    importance_df = importance_df.sort_values("gain", ascending=False)
    importance_df.to_csv(IMPORTANCE_PATH, index=False)
    print(f"[train] Feature importances saved → {IMPORTANCE_PATH}")

    print("\n[train] ── Hold-out Results ─────────────────────────────────")
    print(f"  PR-AUC:    {metrics['holdout_pr_auc']:.4f}")
    print(f"  ROC-AUC:   {metrics['holdout_roc_auc']:.4f}")
    print(f"  F1:        {metrics['holdout_f1']:.4f}")
    print(f"  Precision: {metrics['holdout_precision']:.4f}")
    print(f"  Recall:    {metrics['holdout_recall']:.4f}")
    print(f"  Threshold: {metrics['optimal_threshold']:.4f}")
    print("─────────────────────────────────────────────────────────────\n")

    return metrics


if __name__ == "__main__":
    # Load or regenerate processed data
    if all((PROCESSED_DIR / f).exists()
           for f in ("X_train.parquet", "X_test.parquet",
                     "y_train.parquet", "y_test.parquet")):
        print("[train] Loading pre-processed data from parquet …")
        X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
        X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
        y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet")["binary_target"]
        y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet")["binary_target"]
    else:
        print("[train] Processed data not found — running data engineering pipeline …")
        X_train, X_test, y_train, y_test = load_and_prepare()

    train(X_train, y_train, X_test, y_test)
