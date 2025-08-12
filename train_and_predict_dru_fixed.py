#!/usr/bin/env python3
"""
train_and_predict_dru.py  (fixed)

Train an ML model on rookie CBs (2016–2020) with second-contract labels,
then predict Dru Phillips' probability from 2024 stats.

Usage
-----
python train_and_predict_dru.py \
  --train rookie_cb_2016_2020_with_contracts.csv \
  --new defense_summary24.csv \
  --out ./out_ml \
  --andru_round 3

Outputs
-------
(out)/model_metrics.txt
(out)/roc_curve.png
(out)/calibration_curve.png
(out)/predictions_cv.csv
(out)/dru_prediction.csv
(out)/feature_importance_random_forest.csv
"""

import argparse, os, sys, math, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    roc_auc_score, roc_curve, brier_score_loss, log_loss, average_precision_score
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

TARGET_COL = "got_second_contract"

def norm_name(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
         .str.strip()
         .str.lower()
         .str.replace(r"[^a-z\s]", "", regex=True)
         .str.replace(r"\s+", " ", regex=True)
    )

def load_train(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training file not found: {path}")
    df = pd.read_csv(path)
    # Normalize player name for dedupe / info only
    if "player_norm" not in df.columns and "player" in df.columns:
        df["player_norm"] = norm_name(df["player"])
    # Ensure target exists
    if TARGET_COL not in df.columns:
        raise ValueError(f"Column '{TARGET_COL}' not found in {path}")
    return df

def pick_feature_columns(df: pd.DataFrame) -> list:
    # Numeric columns only, drop obvious leakage or IDs
    drop_like = [
        TARGET_COL, "apy", "apy_m", "apy_millions", "total_value", "total_value_m",
        "contract_year", "team_signed", "years", "player", "player_norm", "team"
    ]
    drop_cols = set()
    for c in df.columns:
        for kw in drop_like:
            if kw.lower() in c.lower():
                drop_cols.add(c)
                break
    feats = [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop_cols]
    return feats

def make_pipeline(num_cols: list) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                        ("sc", StandardScaler())]), num_cols)],
        remainder="drop"
    )
    # Use a simple, robust model
    lr = LogisticRegression(max_iter=2000, n_jobs=None)
    pipe_lr = Pipeline([("pre", pre), ("model", lr)])

    rf = RandomForestClassifier(
        n_estimators=600, max_depth=None, min_samples_leaf=3, random_state=42, n_jobs=-1
    )
    pipe_rf = Pipeline([("pre", pre), ("model", rf)])
    return pipe_lr, pipe_rf

def cross_validate(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, seed: int = 42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for train_idx, val_idx in skf.split(X, y):
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = pipe.predict_proba(X.iloc[val_idx])[:, 1]
    return oof

def plot_roc(y_true, y_prob, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_calibration(y_true, y_prob, out_path):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("Mean predicted prob"); plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve"); plt.tight_layout(); plt.savefig(out_path); plt.close()

def extract_rf_importance(pipe_rf: Pipeline, X_train: pd.DataFrame, out_csv: str, num_cols: list):
    # Fit RF on full training set to get importances on the same preprocessor
    pipe_rf.fit(X_train, X_train[TARGET_COL] if TARGET_COL in X_train else None)  # safeguard
    # ^ We'll actually re-fit later properly; feature names come from num_cols
    # Instead, fit on outer scope before calling this function. To keep this robust,
    # we assume the caller calls after rf is already fit. If not, the line above is a no-op.
    model = pipe_rf.named_steps["model"]
    imp = getattr(model, "feature_importances_", None)
    if imp is None or len(imp) != len(num_cols):
        # Fall back: derive importances by permutation style quickly (slower, but robust)
        try:
            from sklearn.inspection import permutation_importance
            # Need y to compute permutation importance; skip if missing
            raise RuntimeError
        except Exception:
            # As a last resort, write empty file
            pd.DataFrame({"feature": num_cols, "importance": np.nan}).to_csv(out_csv, index=False)
            return
    pd.DataFrame({"feature": num_cols, "importance": imp}).sort_values("importance", ascending=False)\
        .to_csv(out_csv, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="rookie_cb_2016_2020_with_contracts.csv")
    ap.add_argument("--new", required=True, help="defense_summary24.csv with Dru's rookie metrics")
    ap.add_argument("--out", default="./out_ml", help="output folder")
    ap.add_argument("--andru_round", type=int, default=3, help="Dru Phillips draft round (if missing in new csv)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load training
    df = load_train(args.train).copy()
    if df[TARGET_COL].isna().any():
        # drop unlabeled rows just in case
        df = df.loc[~df[TARGET_COL].isna()].copy()
    y = df[TARGET_COL].astype(int)

    num_cols = pick_feature_columns(df)
    if len(num_cols) < 3:
        raise ValueError(f"Too few numeric features selected ({len(num_cols)}).")

    pipe_lr, pipe_rf = make_pipeline(num_cols)

    # Cross-validated probabilities
    oof_lr = cross_validate(pipe_lr, df[num_cols], y, seed=args.seed)
    oof_rf = cross_validate(pipe_rf, df[num_cols], y, seed=args.seed)
    oof_avg = 0.5 * (oof_lr + oof_rf)

    # Metrics & plots
    auc = roc_auc_score(y, oof_avg)
    brier = brier_score_loss(y, oof_avg)
    ap = average_precision_score(y, oof_avg)
    with open(os.path.join(args.out, "model_metrics.txt"), "w") as f:
        f.write(f"AUC: {auc:.4f}\nBrier: {brier:.4f}\nAverage Precision: {ap:.4f}\nN: {len(y)}\n")

    plot_roc(y, oof_avg, os.path.join(args.out, "roc_curve.png"))
    plot_calibration(y, oof_avg, os.path.join(args.out, "calibration_curve.png"))
    pd.DataFrame({"player": df.get("player", np.arange(len(y))), "oof_prob": oof_avg, TARGET_COL: y})\
      .to_csv(os.path.join(args.out, "predictions_cv.csv"), index=False)

    # Fit models on full data for final prediction
    pipe_lr.fit(df[num_cols], y)
    pipe_rf.fit(df[num_cols], y)

    # Feature importance from RF
    rf = pipe_rf.named_steps["model"]
    importances = getattr(rf, "feature_importances_", None)
    if importances is not None and len(importances) == len(num_cols):
        pd.DataFrame({"feature": num_cols, "importance": importances})\
          .sort_values("importance", ascending=False)\
          .to_csv(os.path.join(args.out, "feature_importance_random_forest.csv"), index=False)
    else:
        pd.DataFrame({"feature": num_cols, "importance": np.nan})\
          .to_csv(os.path.join(args.out, "feature_importance_random_forest.csv"), index=False)

    # -------- New data (Dru) --------
    new_df = pd.read_csv(args.new)
    # Try to locate Dru by name; otherwise assume single-row file
    if "player" in new_df.columns:
        mask = new_df["player"].str.contains("andru", case=False, na=False) & new_df["player"].str.contains("phillips", case=False, na=False)
        if mask.any():
            new_row = new_df.loc[mask].copy()
        else:
            new_row = new_df.copy()
    else:
        new_row = new_df.copy()

    # If draft_round is missing, set it
    if "draft_round" not in new_row.columns:
        new_row["draft_round"] = args.andru_round

    # Align columns to training numeric features
    for c in num_cols:
        if c not in new_row.columns:
            new_row[c] = np.nan  # will be imputed
    X_new = new_row[num_cols].iloc[[0]]  # single row

    p_lr = pipe_lr.predict_proba(X_new)[0, 1]
    p_rf = pipe_rf.predict_proba(X_new)[0, 1]
    p_avg = 0.5 * (p_lr + p_rf)

    out_pred = pd.DataFrame({
        "player": new_row.get("player", pd.Series(["Andru Phillips"])).iloc[:1].values,
        "pred_prob_second_contract": [float(p_avg)],
        "prob_lr": [float(p_lr)],
        "prob_rf": [float(p_rf)],
        "andru_round_used": [int(new_row["draft_round"].iloc[0])]
    })
    out_pred.to_csv(os.path.join(args.out, "dru_prediction.csv"), index=False)

    print(f"\n=== Dru Phillips — predicted probability of a multi-year 2nd contract: {p_avg*100:.1f}% ===")
    print(f"(Saved details to: {args.out})")

if __name__ == "__main__":
    main()
