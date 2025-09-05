#!/usr/bin/env python
# -*- coding: utf-8 -*-
# training_pipeline.py
# Python 3.9–3.12 compatible
# End-to-end pipeline aligned with the notebook:
# - Download Microsoft Azure PdM dataset (kagglehub)
# - Assemble df_brute (telemetry + failures + one-hot errors + one-hot maint + machines)
# - make_features_14d: 14-day rolling stats and target fail_next_14d
# - Feature selection: VIF stepwise + low-variance pruning
# - (Optional) class 0 winsorization
# - Train Logistic Regression; pick KS threshold; export metrics & figures
# - Export artifacts to output_dir

import argparse
import json
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import kagglehub
from scipy.stats import ks_2samp

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# ---- Optional statsmodels import (needed for VIF) ----
try:
    import statsmodels.api as sm  # for add_constant
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_STATSMODELS = True
except Exception:
    sm = None
    variance_inflation_factor = None
    HAS_STATSMODELS = False


# =========================================================
# Dataset I/O & assembly
# =========================================================

def load_pdm_dataset() -> Path:
    """Download the dataset to local cache and return the base path."""
    slug = "arnabbiswas1/microsoft-azure-predictive-maintenance"
    path = kagglehub.dataset_download(slug)
    return Path(path)


def sanity_checks(machines, failures, errors, maint, telemetry) -> None:
    """Critical column checks, as in the notebook."""
    required_cols = {
        "machines":  {"machineID", "model", "age"},
        "failures":  {"machineID", "failure", "datetime"},
        "errors":    {"machineID", "errorID", "datetime"},
        "maint":     {"machineID", "comp", "datetime"},
        "telemetry": {"machineID", "datetime", "volt", "rotate", "pressure", "vibration"},
    }
    tables = {
        "machines": machines, "failures": failures, "errors": errors,
        "maint": maint, "telemetry": telemetry
    }
    for name, req in required_cols.items():
        missing = req - set(tables[name].columns)
        if missing:
            raise ValueError(f"Table '{name}' missing required columns: {missing}")
    print("All sanity checks passed.")


def assemble_df_brute(path: Path) -> pd.DataFrame:
    """Replicates the granular assembly from the notebook (telemetry as base)."""
    FILES = {
        "machines":  "PdM_machines.csv",
        "failures":  "PdM_failures.csv",
        "errors":    "PdM_errors.csv",
        "maint":     "PdM_maint.csv",
        "telemetry": "PdM_telemetry.csv",
    }
    machines  = pd.read_csv(path / FILES["machines"])
    failures  = pd.read_csv(path / FILES["failures"],  parse_dates=["datetime"])
    errors    = pd.read_csv(path / FILES["errors"],    parse_dates=["datetime"])
    maint     = pd.read_csv(path / FILES["maint"],     parse_dates=["datetime"])
    telemetry = pd.read_csv(path / FILES["telemetry"], parse_dates=["datetime"])

    print("Shapes -> machines:", machines.shape,
          " | failures:", failures.shape,
          " | errors:", errors.shape,
          " | maint:", maint.shape,
          " | telemetry:", telemetry.shape)

    sanity_checks(machines, failures, errors, maint, telemetry)

    # Telemetry is the time series base
    df_brute = telemetry.sort_values(["machineID", "datetime"]).reset_index(drop=True)

    # Failures: event flag + (optional) type
    failures_flag = failures.copy()
    failures_flag["failure_event"] = 1
    df_brute = df_brute.merge(
        failures_flag[["machineID", "datetime", "failure_event"]],
        on=["machineID", "datetime"], how="left"
    )
    df_brute["failure_event"] = df_brute["failure_event"].fillna(0).astype(int)
    df_brute = df_brute.merge(
        failures[["machineID", "datetime", "failure"]].drop_duplicates(),
        on=["machineID", "datetime"], how="left"
    )

    # Errors -> one-hot by timestamp + total count
    err_pivot = (
        errors.assign(count=1)
              .pivot_table(index=["machineID", "datetime"], columns="errorID",
                           values="count", aggfunc="sum", fill_value=0)
              .add_prefix("error_")
              .reset_index()
    )
    err_pivot["error_count"] = err_pivot.filter(like="error_").sum(axis=1)
    df_brute = df_brute.merge(err_pivot, on=["machineID", "datetime"], how="left")
    error_cols = [c for c in df_brute.columns if c.startswith("error_")]
    df_brute[error_cols] = df_brute[error_cols].fillna(0).astype(int)

    # Maintenance -> one-hot by timestamp + total count
    mnt_pivot = (
        maint.assign(count=1)
             .pivot_table(index=["machineID", "datetime"], columns="comp",
                          values="count", aggfunc="sum", fill_value=0)
             .add_prefix("maint_")
             .reset_index()
    )
    mnt_pivot["maint_count"] = mnt_pivot.filter(like="maint_").sum(axis=1)
    df_brute = df_brute.merge(mnt_pivot, on=["machineID", "datetime"], how="left")
    maint_cols = [c for c in df_brute.columns if c.startswith("maint_")]
    df_brute[maint_cols] = df_brute[maint_cols].fillna(0).astype(int)

    # Join with static info (model, age)
    df_brute = df_brute.merge(machines, on="machineID", how="left")

    # Sensor physical range guards (as in the notebook)
    assert df_brute["volt"].between(90, 270).all(), "Volt out of expected range"
    assert df_brute["rotate"].between(100, 800).all(), "Rotate out of expected range"
    assert df_brute["pressure"].between(40, 200).all(), "Pressure out of expected range"
    assert df_brute["vibration"].between(0, 100).all(), "Vibration out of expected range"

    return df_brute


# =========================================================
# Feature engineering
# =========================================================

def make_features_14d(
    df: pd.DataFrame,
    *,
    machine_col: str = "machineID",
    time_col: str = "datetime",
    failure_event_col: str = "failure_event",
    sensor_cols: Optional[List[str]] = None,
    error_prefix: str = "error_",
    maint_prefix: str = "maint_",
    window_hours: int = 14 * 24,   # 14 days * 24 hours
    build_target: bool = True
) -> Tuple[pd.DataFrame, List[str], Optional[str]]:
    """
    Build aggregated features over a rolling window and optionally the target fail_next_{suffix}.
    Produces:
      - Sensor rolling stats: min/max/mean/std/range/rstd per sensor
      - error_count_{suf}, maint_count_{suf} : total counts over the window
      - age_days : days since first observation per machine
      - past_failures : cumulative past failures (shifted by 1)
      - error_per_maint_{suf} : smoothed ratio error/maint
      - Target: fail_next_{suf} (future max within window) if build_target=True
    """
    df = df.copy()

    # Basic checks
    if time_col not in df.columns or machine_col not in df.columns:
        raise ValueError(f"Missing required columns: {time_col}, {machine_col}")
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    if df[time_col].isna().any():
        raise ValueError("There are NaT values in the time column.")

    df = df.sort_values([machine_col, time_col], kind="mergesort").reset_index(drop=True)

    # Suffix for column names
    suf = f"{window_hours//24}d" if window_hours % 24 == 0 else f"{window_hours}h"

    # Identify columns
    if sensor_cols is None:
        candidate_sensors = ["volt", "rotate", "pressure", "vibration"]
        sensor_cols = [c for c in candidate_sensors if c in df.columns]
    error_cols = [c for c in df.columns if c.startswith(error_prefix)]
    maint_cols = [c for c in df.columns if c.startswith(maint_prefix)]

    feature_cols: List[str] = []
    g = df.groupby(machine_col, sort=False)

    # Rolling statistics for sensor values
    for col in sensor_cols:
        roll = g[col].transform
        rmin  = roll(lambda x: x.rolling(window_hours, min_periods=1).min())
        rmax  = roll(lambda x: x.rolling(window_hours, min_periods=1).max())
        rmean = roll(lambda x: x.rolling(window_hours, min_periods=1).mean())
        rstd  = roll(lambda x: x.rolling(window_hours, min_periods=1).std(ddof=0))

        df[f"{col}_min_{suf}"]   = rmin
        df[f"{col}_max_{suf}"]   = rmax
        df[f"{col}_mean_{suf}"]  = rmean
        df[f"{col}_std_{suf}"]   = rstd.fillna(0.0)
        df[f"{col}_range_{suf}"] = rmax - rmin
        df[f"{col}_rstd_{suf}"]  = np.where(np.abs(rmean) > 1e-8, df[f"{col}_std_{suf}"] / rmean, 0.0)

        feature_cols += [
            f"{col}_min_{suf}", f"{col}_max_{suf}", f"{col}_mean_{suf}",
            f"{col}_std_{suf}", f"{col}_range_{suf}", f"{col}_rstd_{suf}"
        ]

    # Recent errors (total count across all one-hot error_* in window)
    if error_cols:
        roll_errors = g[error_cols].transform(lambda x: x.rolling(window_hours, min_periods=1).sum())
        df[f"error_count_{suf}"] = roll_errors.sum(axis=1).astype(float)
    else:
        df[f"error_count_{suf}"] = 0.0
    feature_cols += [f"error_count_{suf}"]

    # Recent maintenance (total count across all one-hot maint_* in window)
    if maint_cols:
        roll_maint = g[maint_cols].transform(lambda x: x.rolling(window_hours, min_periods=1).sum())
        df[f"maint_count_{suf}"] = roll_maint.sum(axis=1).astype(float)
    else:
        df[f"maint_count_{suf}"] = 0.0
    feature_cols += [f"maint_count_{suf}"]

    # Machine "age" in days since first observation (per machine)
    install_date = g[time_col].transform("min")
    df["age_days"] = (df[time_col] - install_date).dt.days.astype(float)
    feature_cols += ["age_days"]

    # Past cumulative failures (shifted by 1 so it is strictly past)
    if failure_event_col in df.columns:
        df["past_failures"] = g[failure_event_col].cumsum().shift(1).fillna(0).astype(float)
    else:
        df["past_failures"] = 0.0
    feature_cols += ["past_failures"]

    # Smoothed error/maint ratio
    df[f"error_per_maint_{suf}"] = df[f"error_count_{suf}"] / (df[f"maint_count_{suf}"] + 1.0)
    feature_cols += [f"error_per_maint_{suf}"]

    # Target: fail_next_{suf}
    target_col: Optional[str] = None
    if build_target:
        if failure_event_col not in df.columns:
            raise ValueError(f"Column '{failure_event_col}' not found to build target.")
        def _future_max(x: pd.Series) -> pd.Series:
            return x.shift(-1).rolling(window_hours, min_periods=1).max()
        df[f"fail_next_{suf}"] = g[failure_event_col].transform(_future_max).fillna(0).astype(int)
        target_col = f"fail_next_{suf}"

    return df, feature_cols, target_col


# =========================================================
# Outlier handling (optional) & metrics helpers
# =========================================================

def ks_best_threshold(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 200) -> Tuple[float, float]:
    """Return (ks_max, threshold_ks) by evaluating score percentiles."""
    order = np.argsort(y_proba)
    y_true_sorted = y_true[order]
    y_proba_sorted = y_proba[order]
    qs = np.linspace(0, 1, n_bins + 1)
    thr_list = np.quantile(y_proba_sorted, qs)

    ks_vals = []
    for t in thr_list:
        pred = (y_proba >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        tn = ((pred == 0) & (y_true == 0)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        ks_vals.append((t, tpr - fpr))
    threshold_ks, ks_max = max(ks_vals, key=lambda z: z[1])
    return ks_max, float(threshold_ks)


def clean_outliers_class0_with_ks(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    cap_quantile: float = 0.99
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """
    Winsorize class 0 at p=cap_quantile for 'features' and report KS before vs after.
    """
    df_clean = df.copy()
    mask0 = df_clean[target] == 0
    df0 = df_clean.loc[mask0, features]
    caps = {col: df0[col].quantile(cap_quantile) for col in features}

    original = {col: df0[col].values for col in features}
    for col, cap in caps.items():
        df_clean.loc[mask0 & (df_clean[col] > cap), col] = cap

    # KS pre vs post only on class 0
    rows = []
    for col in features:
        ks_stat, pval = ks_2samp(original[col], df_clean.loc[mask0, col].values)
        rows.append({"feature": col, "ks_stat_pre_vs_post_class0": float(ks_stat), "p_value": float(pval)})
    ks_df = pd.DataFrame(rows).sort_values("ks_stat_pre_vs_post_class0", ascending=False)
    return df_clean, caps, ks_df


# =========================================================
# VIF selection helpers
# =========================================================

def _require_statsmodels():
    if not HAS_STATSMODELS:
        raise RuntimeError(
            "statsmodels is required for VIF selection. Install it with:\n"
            "  pip install statsmodels==0.14.2"
        )

def compute_vif_series(df_num: pd.DataFrame) -> pd.Series:
    """Compute VIF for each column in df_num (requires statsmodels)."""
    _require_statsmodels()
    X = sm.add_constant(df_num, has_constant="add")
    vifs = pd.Series(
        [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        index=X.columns
    )
    return vifs.drop("const", errors="ignore")

def stepwise_vif(df_num: pd.DataFrame, threshold: float = 5.0, verbose: bool = True):
    """
    Iteratively drop the column with the highest VIF until all VIFs <= threshold.
    Returns: (kept_columns, final_vif_series, dropped_list)
    """
    _require_statsmodels()
    X = df_num.copy()
    dropped = []
    while X.shape[1] > 2:
        v = compute_vif_series(X)
        max_feat, max_val = v.idxmax(), float(v.max())
        if max_val <= threshold:
            return X.columns.tolist(), v.sort_values(ascending=False), dropped
        if verbose:
            print(f"Dropping {max_feat} (VIF={max_val:.2f})")
        dropped.append((max_feat, max_val))
        X = X.drop(columns=[max_feat])
    return X.columns.tolist(), compute_vif_series(X), dropped


# =========================================================
# Training
# =========================================================

def train(
    output_dir: Path,
    test_size: float,
    random_state: int,
    penalty: str,
    solver: str,
    max_iter: int,
    class_weight: Optional[str],
    tol: float,
    do_winsor_class0: bool,
    winsor_quantile: float
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load & assemble
    base_path = load_pdm_dataset()
    df_brute = assemble_df_brute(base_path)

    # 2) Features 14d + target
    df_14, feat_cols, target_col = make_features_14d(df_brute, build_target=True)
    if target_col is None:
        raise RuntimeError("Target column was not built. Check failure_event in input data.")
    target = target_col
    df_14_eda = df_14[feat_cols + [target]].copy()

    # ----------------------------
    # 3) VIF + low-variance pruning
    # ----------------------------
    # Build the full numeric candidate set
    candidates = (
        df_14_eda.drop(columns=[target], errors="ignore")
                 .select_dtypes(include=[np.number])
                 .copy()
    )
    candidates.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop quasi-constant columns
    zero_var_cols = [c for c in candidates.columns if candidates[c].nunique(dropna=True) <= 1]
    if zero_var_cols:
        candidates.drop(columns=zero_var_cols, inplace=True)

    # Drop columns with too many NaN
    MAX_NA_FRAC = 0.30
    na_frac = candidates.isna().mean()
    high_na_cols = na_frac[na_frac > MAX_NA_FRAC].index.tolist()
    if high_na_cols:
        candidates.drop(columns=high_na_cols, inplace=True)

    # Simple median imputation ONLY for the VIF math
    candidates_vif = candidates.fillna(candidates.median(numeric_only=True))

    # Run stepwise VIF
    VIF_THRESHOLD = 5.0
    keep_cols, final_vif, dropped = stepwise_vif(candidates_vif, threshold=VIF_THRESHOLD, verbose=True)

    # Low-variance pruning over original (non-imputed) candidates
    kept_numeric = candidates[keep_cols].copy()
    MIN_UNIQUE_FRAC = 0.01
    MIN_STD = 1e-3
    n_rows = len(kept_numeric)
    unique_frac = kept_numeric.nunique(dropna=True) / max(n_rows, 1)
    std_vals = kept_numeric.std(numeric_only=True)

    low_var_cols = sorted(list(set(
        unique_frac[unique_frac <= MIN_UNIQUE_FRAC].index.tolist()
        + std_vals[std_vals <= MIN_STD].index.tolist()
    )))
    final_cols = [c for c in keep_cols if c not in low_var_cols]

    print("\nFinal feature count:", len(final_cols))
    print("Final features:", final_cols)

    # Save selection artifacts
    (output_dir / "feature_selection").mkdir(parents=True, exist_ok=True)
    pd.Series(final_cols, name="selected_feature").to_csv(output_dir / "feature_selection" / "selected_features.csv", index=False)
    final_vif.to_csv(output_dir / "feature_selection" / "final_vif.csv")
    if dropped:
        pd.DataFrame(dropped, columns=["feature", "VIF_when_dropped"]).to_csv(
            output_dir / "feature_selection" / "dropped_by_vif.csv", index=False
        )
    with open(output_dir / "selected_features.json", "w") as f:
        json.dump(final_cols, f, indent=2)

    # 4) Build final modeling frame
    df_model = df_14_eda[final_cols + [target]].copy()

    # 5) Imputation & scaling (for the model)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler(with_mean=True, with_std=True)

    X_imp = imputer.fit_transform(df_model.drop(columns=[target]))
    X_scl = scaler.fit_transform(X_imp)
    y = df_model[target].astype(int).values
    feature_names = final_cols

    # 6) Optional winsorization on class 0 (post-scale, as in previous pipeline)
    if do_winsor_class0:
        X_df = pd.DataFrame(X_scl, columns=feature_names)
        X_df[target] = y
        X_df_w, caps, ks_df = clean_outliers_class0_with_ks(
            X_df, features=feature_names, target=target, cap_quantile=winsor_quantile
        )
        y = X_df_w[target].values
        X_scl = X_df_w.drop(columns=[target]).values
        ks_df.to_csv(output_dir / "ks_winsor_report.csv", index=False)
        with open(output_dir / "winsor_caps.json", "w") as f:
            json.dump(caps, f, indent=2)

    # 7) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scl, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 8) Model: Logistic Regression
    model = LogisticRegression(
        penalty=penalty,
        solver=solver,
        class_weight=class_weight,
        max_iter=max_iter,
        tol=tol
    )
    model.fit(X_train, y_train)

    # 9) Evaluation + KS threshold
    proba_train = model.predict_proba(X_train)[:, 1]
    proba_test  = model.predict_proba(X_test)[:, 1]

    auc_train = roc_auc_score(y_train, proba_train)
    auc_test  = roc_auc_score(y_test,  proba_test)

    ks_train, thr_ks_train = ks_best_threshold(y_train, proba_train)
    ks_test,  thr_ks_test  = ks_best_threshold(y_test,  proba_test)

    # With KS threshold (test)
    y_pred_ks = (proba_test >= thr_ks_test).astype(int)
    cm = confusion_matrix(y_test, y_pred_ks)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred_ks, average="binary", zero_division=0)

    metrics = {
        "auc_train": float(auc_train),
        "auc_test": float(auc_test),
        "ks_train": float(ks_train),
        "ks_test": float(ks_test),
        "thr_ks_train": float(thr_ks_train),
        "thr_ks_test": float(thr_ks_test),
        "precision_test": float(pr),
        "recall_test": float(rc),
        "f1_test": float(f1),
        "confusion_matrix_test": cm.tolist(),
        "n_features": len(feature_names),
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    # 10) ROC curve
    fpr, tpr, _ = roc_curve(y_test, proba_test)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_test:.3f}")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve (Test)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve_test.png", dpi=120)
    plt.close()

    # 11) KS curve (proper CDFs over score)
    #    CDF de positivos y negativos contra el score, con línea en el umbral KS
    mask = ~np.isnan(proba_test)
    y_test_plot = y_test[mask]
    proba_test_plot = proba_test[mask]

    df_scores = (
        pd.DataFrame({"y_true": y_test_plot, "proba": proba_test_plot})
        .sort_values("proba", kind="mergesort")
        .reset_index(drop=True)
    )

    pos_total = int(df_scores["y_true"].sum())
    neg_total = int(len(df_scores) - pos_total)

    cum_pos = np.cumsum(df_scores["y_true"].values) / max(pos_total, 1)
    cum_neg = np.cumsum((1 - df_scores["y_true"].values)) / max(neg_total, 1)

    plt.figure(figsize=(6, 5))
    plt.plot(df_scores["proba"].values, cum_pos, label="Positive CDF")
    plt.plot(df_scores["proba"].values, cum_neg, label="Negative CDF")
    plt.fill_between(df_scores["proba"].values, cum_pos, cum_neg, color="gray", alpha=0.2)
    plt.axvline(thr_ks_test, color="green", linestyle="--", label=f"KS Thr={thr_ks_test:.3f}")
    plt.title(f"KS Curve (Test) - KS={ks_test:.3f} @ thr={thr_ks_test:.3f}")
    plt.xlabel("Score (predicted probability)")
    plt.ylabel("Cumulative proportion")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "ks_curve_test.png", dpi=120)
    plt.close()

    # 12) Export artifacts
    joblib.dump(model, output_dir / "model.joblib")
    joblib.dump(imputer, output_dir / "imputer.joblib")
    joblib.dump(scaler, output_dir / "scaler.joblib")
    with open(output_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="artifacts/")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--penalty", type=str, default="l1")
    p.add_argument("--solver", type=str, default="liblinear")
    p.add_argument("--max_iter", type=int, default=3000)
    p.add_argument("--class_weight", type=str, default="balanced")
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--winsor_class0", type=int, default=0,
                   help="1 = enable winsorization for class 0")
    p.add_argument("--winsor_quantile", type=float, default=0.99)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    outdir = Path(args.output_dir)
    train(
        output_dir=outdir,
        test_size=args.test_size,
        random_state=args.random_state,
        penalty=args.penalty,
        solver=args.solver,
        max_iter=args.max_iter,
        class_weight=(args.class_weight if args.class_weight.lower() != "none" else None),
        tol=args.tol,
        do_winsor_class0=bool(args.winsor_class0),
        winsor_quantile=args.winsor_quantile
    )
