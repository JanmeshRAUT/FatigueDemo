from __future__ import annotations

from datetime import datetime
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from xgboost import XGBClassifier

from config import Config


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Feature engineering
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6
    df = df.copy()
    df["ear_mar_ratio"] = df["EAR"] / (df["MAR"] + eps)
    df["perclos_blink_interaction"] = df["PERCLOS"] * df["blink_rate"]
    df["head_motion_sum"] = df[["head_pitch", "head_yaw", "head_roll"]].abs().sum(axis=1)
    return df


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Banner helper
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _banner(text: str) -> None:
    w = 60
    print("\n" + "=" * w)
    print(f"  {text}")
    print("=" * w)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Plotting helpers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def save_confusion(cm, class_names, out_path: Path, title: str) -> None:
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_epoch_curve(history: list[dict], out_path: Path) -> None:
    epochs    = [h["epoch"]     for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc   = [h["val_acc"]   for h in history]
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, label="Train Accuracy", color="#2b6cb0")
    plt.plot(epochs, val_acc,   label="Val Accuracy",   color="#c05621")
    plt.xlabel("Boosting Round")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.05)
    plt.title("XGBoost Training Curve (Best Trial)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_trial_comparison(trial_results: list[dict], out_path: Path) -> None:
    df = pd.DataFrame(trial_results).sort_values("val_macro_f1", ascending=False).reset_index(drop=True)
    labels = [f"T{i+1}" for i in range(len(df))]
    colors = ["#276749" if i == 0 else "#2b6cb0" for i in range(len(df))]
    plt.figure(figsize=(max(7, len(df) * 1.5), 5))
    bars = plt.bar(labels, df["val_macro_f1"], color=colors)
    plt.axhline(df["val_macro_f1"].max(), color="red", linestyle="--",
                linewidth=1, label=f"Best: {df['val_macro_f1'].max():.4f}")
    for bar, val in zip(bars, df["val_macro_f1"]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    plt.ylim(df["val_macro_f1"].min() - 0.01, df["val_macro_f1"].max() + 0.02)
    plt.ylabel("Val Macro F1")
    plt.title("XGBoost Trial Comparison (sorted by F1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_feature_importance(model: XGBClassifier, feature_names: list[str], out_path: Path) -> None:
    fi = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_}
                      ).sort_values("importance", ascending=False)
    plt.figure(figsize=(9, 6))
    sns.barplot(data=fi, x="importance", y="feature", color="#c05621")
    plt.title("XGBoost Feature Importance (gain)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_class_metrics_bar(y_true, y_pred, class_names, out_path: Path) -> None:
    labels = list(range(len(class_names)))
    precision = precision_score(y_true, y_pred, labels=labels, average=None)
    recall    = recall_score(y_true,    y_pred, labels=labels, average=None)
    f1        = f1_score(y_true,        y_pred, labels=labels, average=None)
    x, width = np.arange(len(labels)), 0.26
    plt.figure(figsize=(8, 5))
    plt.bar(x - width, precision, width, label="Precision", color="#276749")
    plt.bar(x,         recall,    width, label="Recall",    color="#2b6cb0")
    plt.bar(x + width, f1,        width, label="F1",        color="#c05621")
    plt.xticks(x, class_names)
    plt.ylim(0, 1.05)
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.title("XGBoost Per-Class Precision / Recall / F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_roc_curves(model, X_test, y_test, class_names, out_path: Path) -> None:
    labels  = list(range(len(class_names)))
    y_bin   = label_binarize(y_test, classes=labels)
    y_proba = model.predict_proba(X_test)
    colors  = ["#276749", "#2b6cb0", "#c05621", "#6b21a8", "#b91c1c"]
    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
        plt.plot(fpr, tpr, label=f"Class {cls} (AUC={auc:.3f})", color=colors[i % len(colors)])
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("XGBoost ROC Curves (One-vs-Rest)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_class_distribution(y_full_enc, y_train, y_test, class_names, out_path: Path) -> None:
    n = len(class_names)
    idx = list(range(n))
    full_c  = [int((y_full_enc == i).sum()) for i in idx]
    train_c = [int((y_train    == i).sum()) for i in idx]
    test_c  = [int((y_test     == i).sum()) for i in idx]
    x, width = np.arange(n), 0.28
    plt.figure(figsize=(8, 5))
    plt.bar(x - width, full_c,  width, label="Full dataset", color="#4a5568")
    plt.bar(x,         train_c, width, label="Train split",  color="#2b6cb0")
    plt.bar(x + width, test_c,  width, label="Test split",   color="#c05621")
    plt.xticks(x, class_names)
    plt.xlabel("Class")
    plt.ylabel("Sample count")
    plt.title("Class Distribution: Full / Train / Test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_prediction_distribution(y_true, y_pred, class_names, out_path: Path) -> None:
    n = len(class_names)
    idx = list(range(n))
    actual_c = [int((y_true == i).sum()) for i in idx]
    pred_c   = [int((y_pred == i).sum()) for i in idx]
    x, width = np.arange(n), 0.38
    plt.figure(figsize=(7, 5))
    plt.bar(x - width / 2, actual_c, width, label="Actual",    color="#2b6cb0")
    plt.bar(x + width / 2, pred_c,   width, label="Predicted", color="#c05621")
    plt.xticks(x, class_names)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("XGBoost Prediction Distribution: Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_learning_curve(best_params: dict, X_train, y_train, cv, class_names, out_path: Path) -> None:
    print("  Computing learning curve (this may take ~30s)...")
    fractions = np.linspace(0.10, 1.0, 10)
    n_total   = len(X_train)
    train_f1s, val_f1s, sizes = [], [], []
    for frac in fractions:
        n   = max(len(class_names) * 2, int(n_total * frac))
        idx = np.random.default_rng(42).choice(n_total, size=n, replace=False)
        Xs, ys = X_train[idx], y_train[idx]
        clf = XGBClassifier(
            n_estimators=best_params["epochs"],
            early_stopping_rounds=None,
            max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            min_child_weight=best_params["min_child_weight"],
            subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.1, reg_lambda=1.2,
            objective="multi:softprob", eval_metric="mlogloss",
            random_state=99, n_jobs=-1, tree_method="hist", verbosity=0,
        )
        fold_val, fold_tr = [], []
        for tr_idx, val_idx in cv.split(Xs, ys):
            clf.fit(Xs[tr_idx], ys[tr_idx], verbose=False)
            fold_tr.append(f1_score(ys[tr_idx], clf.predict(Xs[tr_idx]), average="macro"))
            fold_val.append(f1_score(ys[val_idx], clf.predict(Xs[val_idx]), average="macro"))
        sizes.append(n)
        train_f1s.append(np.mean(fold_tr))
        val_f1s.append(np.mean(fold_val))
        print(f"    size={n:,}  train_f1={train_f1s[-1]:.4f}  val_f1={val_f1s[-1]:.4f}")
    plt.figure(figsize=(9, 6))
    plt.plot(sizes, train_f1s, "o-", color="#2b6cb0", label="Train macro F1")
    plt.plot(sizes, val_f1s,   "o-", color="#c05621", label="CV Val macro F1")
    plt.xlabel("Training samples")
    plt.ylabel("Macro F1")
    plt.ylim(0.0, 1.05)
    plt.title("XGBoost Learning Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Trial runner
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def run_trial(
    params: dict,
    X_train, y_train,
    X_val,   y_val,
    trial_no: int,
    total: int,
) -> tuple[XGBClassifier, list[dict], float]:
    print(f"\n  Trial [{trial_no}/{total}]  max_depth={params['max_depth']}  "
          f"lr={params['learning_rate']}  min_child_weight={params['min_child_weight']}  "
          f"epochs={params['epochs']}")
    t0 = time.time()
    model = XGBClassifier(
        n_estimators=params["epochs"],
        early_stopping_rounds=params["early_stopping_rounds"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        min_child_weight=params["min_child_weight"],
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.1, reg_lambda=1.2,
        objective="multi:softprob",
        eval_metric=["mlogloss", "merror"],
        random_state=99, n_jobs=-1, tree_method="hist", verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
    elapsed = time.time() - t0

    evals     = model.evals_result()
    train_err = evals["validation_0"]["merror"]
    val_err   = evals["validation_1"]["merror"]

    history = []
    for i, (te, ve) in enumerate(zip(train_err, val_err), start=1):
        history.append({"epoch": i, "train_acc": float(1 - te), "val_acc": float(1 - ve)})
        if i % 20 == 0 or i == 1 or i == len(train_err):
            print(f"    [epoch {i:03d}/{len(train_err)}]  "
                  f"train_acc={history[-1]['train_acc']:.4f}  val_acc={history[-1]['val_acc']:.4f}")

    val_f1 = float(f1_score(y_val, model.predict(X_val), average="macro"))
    print(f"  => val_macro_f1={val_f1:.4f}  actual_rounds={len(train_err)}  ({elapsed:.1f}s)")
    return model, history, val_f1


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Main
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def main() -> None:
    cfg = Config()
    out_dir = Path("outputs_independent/xgb_only")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_started_at = datetime.utcnow().isoformat() + "Z"
    t_total_start  = time.time()

    # â”€â”€ STEP 1 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _banner("STEP 1/6  Loading & engineering features")
    print(f"  Path : {cfg.data.dataset_path}")
    df = pd.read_csv(cfg.data.dataset_path)
    print(f"  Rows : {len(df):,}   Cols: {len(df.columns)}")
    df = make_features(df)
    base_features  = list(cfg.features.features)
    extra_features = ["ear_mar_ratio", "perclos_blink_interaction", "head_motion_sum"]
    feature_cols   = base_features + extra_features
    target_col     = cfg.data.target_col
    print(f"  Engineered features : {extra_features}")
    print(f"  Total features      : {len(feature_cols)}")

    # â”€â”€ STEP 2 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _banner("STEP 2/6  Cleaning & validating data")
    required = feature_cols + [target_col]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    initial_rows = len(df)
    df = df[required].copy()
    before_dedup = len(df)
    df = df.drop_duplicates()
    print(f"  Duplicates removed : {before_dedup - len(df)}")
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[target_col] = df[target_col].astype(str).str.strip()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    print(f"  Rows after cleaning: {len(df):,}  (dropped {initial_rows - len(df)})")
    if df.empty:
        raise ValueError("No valid rows after cleaning.")

    X_raw = df[feature_cols]
    y_raw = df[target_col]
    le    = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    class_names  = [str(c) for c in le.classes_]
    class_counts = pd.Series(y_enc).value_counts().sort_index()
    min_class    = int(class_counts.min())
    print(f"  Classes ({len(class_names)}): {dict(zip(class_names, class_counts.tolist()))}")
    if min_class < 2:
        raise ValueError("At least 2 samples per class required.")

    # â”€â”€ STEP 3 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _banner("STEP 3/6  Splitting & scaling")
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_raw, y_enc, test_size=cfg.data.test_size, random_state=99, stratify=y_enc,
    )
    val_ratio = 0.20 / (1 - cfg.data.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, random_state=99, stratify=y_trainval,
    )
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    cv_folds = max(2, min(5, min_class))
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=99)
    print(f"  StratifiedKFold: {cv_folds} folds (used for learning curve)")

    # â”€â”€ STEP 4 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _banner("STEP 4/6  Trial-based hyperparameter search")
    trial_space = [
        {"epochs": 200, "early_stopping_rounds": 20, "max_depth": 4, "learning_rate": 0.05, "min_child_weight": 1},
        {"epochs": 200, "early_stopping_rounds": 20, "max_depth": 4, "learning_rate": 0.08, "min_child_weight": 1},
        {"epochs": 200, "early_stopping_rounds": 20, "max_depth": 6, "learning_rate": 0.05, "min_child_weight": 1},
        {"epochs": 200, "early_stopping_rounds": 20, "max_depth": 6, "learning_rate": 0.08, "min_child_weight": 1},
        {"epochs": 200, "early_stopping_rounds": 20, "max_depth": 6, "learning_rate": 0.05, "min_child_weight": 3},
        {"epochs": 200, "early_stopping_rounds": 25, "max_depth": 8, "learning_rate": 0.05, "min_child_weight": 1},
    ]
    print(f"  Total trials: {len(trial_space)}")

    best_model   = None
    best_history: list[dict] = []
    best_params: dict        = {}
    best_score   = float("-inf")
    trial_results: list[dict] = []

    for i, params in enumerate(trial_space, start=1):
        model, history, score = run_trial(params, X_train_s, y_train, X_val_s, y_val, i, len(trial_space))
        trial_results.append({**params, "val_macro_f1": score})
        if score > best_score:
            best_score   = score
            best_model   = model
            best_history = history
            best_params  = params
            print(f"  *** New best: val_macro_f1={best_score:.4f}")

    print(f"\n  Best params      : {best_params}")
    print(f"  Best val macro F1: {best_score:.4f}")

    # â”€â”€ STEP 5 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _banner("STEP 5/6  Evaluating on held-out test set")
    pred      = best_model.predict(X_test_s)
    labels_idx = list(range(len(class_names)))
    cm         = confusion_matrix(y_test, pred, labels=labels_idx)
    report     = classification_report(y_test, pred, labels=labels_idx,
                                       target_names=class_names, digits=4)
    macro_f1   = float(f1_score(y_test, pred, average="macro"))
    print(f"  Test macro F1 : {macro_f1:.4f}")
    print()
    print(report)

    # â”€â”€ STEP 6 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _banner("STEP 6/6  Saving graphs & artifacts")
    t_total = time.time() - t_total_start
    metrics = {
        "model":                  "xgboost_independent",
        "run_started_at_utc":     run_started_at,
        "total_runtime_seconds":  round(t_total, 2),
        "dataset_path":           str(cfg.data.dataset_path),
        "rows_before_cleaning":   initial_rows,
        "rows_after_cleaning":    len(df),
        "dropped_rows":           initial_rows - len(df),
        "class_distribution":     {class_names[i]: int(v) for i, v in enumerate(class_counts.tolist())},
        "n_features":             len(feature_cols),
        "features":               feature_cols,
        "cv_folds":               cv_folds,
        "best_params":            best_params,
        "best_val_macro_f1":      float(best_score),
        "test_macro_f1":          macro_f1,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    pd.DataFrame(best_history).to_csv(out_dir / "epoch_history.csv", index=False)
    pd.DataFrame(trial_results).to_csv(out_dir / "trial_results.csv", index=False)

    save_confusion(cm, class_names, out_dir / "confusion_matrix.png", "XGB Confusion Matrix")
    print("    [1/9] confusion_matrix.png")
    save_epoch_curve(best_history, out_dir / "epoch_curve.png")
    print("    [2/9] epoch_curve.png")
    save_trial_comparison(trial_results, out_dir / "trial_comparison.png")
    print("    [3/9] trial_comparison.png")
    save_feature_importance(best_model, feature_cols, out_dir / "feature_importance.png")
    print("    [4/9] feature_importance.png")
    save_class_metrics_bar(y_test, pred, class_names, out_dir / "class_metrics_bar.png")
    print("    [5/9] class_metrics_bar.png")
    save_roc_curves(best_model, X_test_s, y_test, class_names, out_dir / "roc_curves.png")
    print("    [6/9] roc_curves.png")
    save_class_distribution(y_enc, y_train, y_test, class_names, out_dir / "class_distribution.png")
    print("    [7/9] class_distribution.png")
    save_prediction_distribution(y_test, pred, class_names, out_dir / "prediction_distribution.png")
    print("    [8/9] prediction_distribution.png")
    save_learning_curve(best_params, X_train_s, y_train, cv, class_names, out_dir / "learning_curve.png")
    print("    [9/9] learning_curve.png")

    joblib.dump(best_model, out_dir / "xgboost_independent.joblib")
    joblib.dump(scaler,     out_dir / "scaler.joblib")
    joblib.dump(le,         out_dir / "label_encoder.joblib")

    _banner("DONE")
    print(f"  Runtime          : {t_total:.1f}s")
    print(f"  Test macro F1    : {macro_f1:.4f}")
    print(f"  Best val macro F1: {best_score:.4f}")
    print(f"  Best params      : {best_params}")
    print(f"  Output dir       : {out_dir.resolve()}")


if __name__ == "__main__":
    main()
