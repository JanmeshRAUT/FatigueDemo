from __future__ import annotations

from datetime import datetime
import itertools
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve, train_test_split
from sklearn.preprocessing import label_binarize

from config import Config


def save_confusion(cm, labels, out_path: Path, title: str) -> None:
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_feature_importance(model: RandomForestClassifier, feature_names: list[str], out_path: Path) -> None:
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=fi, x="importance", y="feature", color="#2f855a")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_cv_tuning_curve(results: list[dict], out_path: Path) -> None:
    df = pd.DataFrame(results).sort_values("mean_f1", ascending=False).reset_index(drop=True)
    x = np.arange(len(df))
    plt.figure(figsize=(max(10, len(df) // 2), 5))
    plt.bar(x, df["mean_f1"], yerr=df["std_f1"], color="#2b6cb0", alpha=0.85,
            capsize=3, error_kw={"elinewidth": 1})
    plt.axhline(df["mean_f1"].max(), color="red", linestyle="--", linewidth=1,
                label=f"Best: {df['mean_f1'].max():.4f}")
    plt.xticks(x, [str(i + 1) for i in x], fontsize=7)
    plt.xlabel("Combination rank (sorted by F1)")
    plt.ylabel("CV macro F1")
    plt.title("RF Grid Search: CV Macro F1 per Combination")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_class_metrics_bar(y_true, y_pred, labels, out_path: Path) -> None:
    precision = precision_score(y_true, y_pred, labels=labels, average=None)
    recall = recall_score(y_true, y_pred, labels=labels, average=None)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None)
    x = np.arange(len(labels))
    width = 0.26
    plt.figure(figsize=(8, 5))
    plt.bar(x - width, precision, width, label="Precision", color="#2f855a")
    plt.bar(x,          recall,    width, label="Recall",    color="#2b6cb0")
    plt.bar(x + width,  f1,        width, label="F1",        color="#c05621")
    plt.xticks(x, [str(l) for l in labels])
    plt.ylim(0, 1.05)
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.title("RF Per-Class Precision / Recall / F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_roc_curves(model, X_test, y_test, labels, out_path: Path) -> None:
    y_bin = label_binarize(y_test, classes=labels)
    y_proba = model.predict_proba(X_test)
    plt.figure(figsize=(8, 6))
    colors = ["#2f855a", "#2b6cb0", "#c05621", "#6b21a8", "#b91c1c"]
    for i, cls in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
        plt.plot(fpr, tpr, label=f"Class {cls} (AUC={auc:.3f})",
                 color=colors[i % len(colors)])
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("RF ROC Curves (One-vs-Rest)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_class_distribution(y_full, y_train, y_test, out_path: Path) -> None:
    classes = sorted(y_full.unique())
    full_counts  = [int((y_full  == c).sum()) for c in classes]
    train_counts = [int((y_train == c).sum()) for c in classes]
    test_counts  = [int((y_test  == c).sum()) for c in classes]
    x = np.arange(len(classes))
    width = 0.28
    plt.figure(figsize=(8, 5))
    plt.bar(x - width, full_counts,  width, label="Full dataset", color="#4a5568")
    plt.bar(x,         train_counts, width, label="Train split",  color="#2b6cb0")
    plt.bar(x + width, test_counts,  width, label="Test split",   color="#c05621")
    plt.xticks(x, [str(c) for c in classes])
    plt.xlabel("Class")
    plt.ylabel("Sample count")
    plt.title("Class Distribution: Full / Train / Test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_prediction_distribution(y_true, y_pred, labels, out_path: Path) -> None:
    actual_counts = [int((y_true == c).sum()) for c in labels]
    pred_counts   = [int((y_pred == c).sum()) for c in labels]
    x = np.arange(len(labels))
    width = 0.38
    plt.figure(figsize=(7, 5))
    plt.bar(x - width / 2, actual_counts, width, label="Actual",    color="#2b6cb0")
    plt.bar(x + width / 2, pred_counts,   width, label="Predicted", color="#c05621")
    plt.xticks(x, [str(l) for l in labels])
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("RF Prediction Distribution: Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_learning_curve(model, X_train, y_train, cv, out_path: Path) -> None:
    print("  Computing learning curve (this may take ~30s)...")
    train_sizes = np.linspace(0.10, 1.0, 10)
    sizes, train_scores, val_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        train_sizes=train_sizes,
        n_jobs=-1,
        shuffle=True,
        random_state=42,
    )
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    plt.figure(figsize=(9, 6))
    plt.plot(sizes, train_mean, "o-", color="#2b6cb0", label="Train macro F1")
    plt.fill_between(sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.15, color="#2b6cb0")
    plt.plot(sizes, val_mean, "o-", color="#c05621", label="CV Val macro F1")
    plt.fill_between(sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.15, color="#c05621")
    plt.xlabel("Training samples")
    plt.ylabel("Macro F1")
    plt.ylim(0.0, 1.05)
    plt.title("Random Forest Learning Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _banner(text: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def main() -> None:
    cfg = Config()
    out_dir = Path("outputs_independent/rf_only")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_started_at = datetime.utcnow().isoformat() + "Z"
    t_total_start = time.time()

    _banner("STEP 1/5  Loading dataset")
    print(f"  Path : {cfg.data.dataset_path}")
    df = pd.read_csv(cfg.data.dataset_path)
    print(f"  Rows : {len(df):,}   Cols: {len(df.columns)}")
    feature_cols = list(cfg.features.features)
    target_col = cfg.data.target_col

    required = feature_cols + [target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    _banner("STEP 2/5  Cleaning & validating")
    initial_rows = int(len(df))
    df = df[required].copy()
    before_dedup = len(df)
    df = df.drop_duplicates()
    print(f"  Duplicates removed : {before_dedup - len(df)}")

    # Force numeric conversion for model features and remove invalid rows.
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[target_col] = df[target_col].astype(str).str.strip()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    print(f"  Rows after cleaning: {len(df):,}  (dropped {initial_rows - len(df)})")

    if df.empty:
        raise ValueError("No valid rows left after cleaning. Check dataset quality.")

    X = df[feature_cols]
    y = df[target_col]

    class_counts = y.value_counts().sort_index()
    min_class_count = int(class_counts.min())
    print(f"  Classes ({len(class_counts)}):")
    for cls, cnt in class_counts.items():
        print(f"    {cls}: {cnt:,} samples")
    if min_class_count < 2:
        raise ValueError("At least 2 samples per class are required for reliable train/test split.")

    _banner("STEP 3/5  Splitting dataset")
    # RF model is intentionally trained on unscaled raw features.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state,
        stratify=y,
    )
    print(f"  Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    print(f"  Features: {feature_cols}")

    cv_folds = max(2, min(5, min_class_count, len(y_train)))
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=cfg.rf.random_state)
    print(f"  StratifiedKFold: {cv_folds} folds")

    _banner("STEP 4/5  Hyperparameter tuning (manual grid search)")
    param_grid = {
        "n_estimators": [160, 240],
        "max_depth": [10, 14, None],
        "min_samples_split": [2, 4],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt"],
        "class_weight": ["balanced_subsample"],
    }

    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    total = len(combos)
    print(f"  Total combinations: {total}  |  Folds per combo: {cv_folds}  |  Total fits: {total * cv_folds}")

    best_score = float("-inf")
    best_params: dict = {}
    best_model: RandomForestClassifier | None = None
    all_results: list[dict] = []

    for i, combo in enumerate(combos, start=1):
        params = dict(zip(keys, combo))
        t0 = time.time()
        clf = RandomForestClassifier(
            random_state=cfg.rf.random_state,
            n_jobs=-1,
            bootstrap=True,
            **params,
        )
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        mean_f1 = float(scores.mean())
        std_f1 = float(scores.std())
        elapsed = time.time() - t0
        new_best = mean_f1 > best_score

        marker = "  <<< NEW BEST" if new_best else ""
        print(
            f"  [{i:02d}/{total}] "
            + "  ".join(f"{k}={v}" for k, v in params.items())
            + f"  =>  val_macro_f1={mean_f1:.4f} ± {std_f1:.4f}  ({elapsed:.1f}s){marker}"
        )

        all_results.append({**params, "mean_f1": mean_f1, "std_f1": std_f1})

        if new_best:
            best_score = mean_f1
            best_params = params
            clf.fit(X_train, y_train)          # refit on full train set to save
            best_model = clf

    # Ensure final model is always refit with best params on full train data.
    final = RandomForestClassifier(
        random_state=cfg.rf.random_state,
        n_jobs=-1,
        bootstrap=True,
        **best_params,
    )
    final.fit(X_train, y_train)
    model: RandomForestClassifier = final
    print(f"\n  Best params : {best_params}")
    print(f"  Best cv f1  : {best_score:.4f}")

    _banner("STEP 5/5  Evaluating on held-out test set")
    pred = model.predict(X_test)

    labels = sorted(y.unique().tolist())
    cm = confusion_matrix(y_test, pred, labels=labels)
    report = classification_report(y_test, pred, labels=labels, digits=4)
    macro_f1 = float(f1_score(y_test, pred, average="macro"))
    print(f"  Test macro F1 : {macro_f1:.4f}")
    print()
    print(report)

    t_total = time.time() - t_total_start

    metrics = {
        "model": "random_forest_independent",
        "run_started_at_utc": run_started_at,
        "total_runtime_seconds": round(t_total, 2),
        "dataset_path": str(cfg.data.dataset_path),
        "rows_before_cleaning": initial_rows,
        "rows_after_cleaning": int(len(df)),
        "dropped_rows": int(initial_rows - len(df)),
        "class_distribution": {str(k): int(v) for k, v in class_counts.to_dict().items()},
        "cv_folds": int(cv_folds),
        "best_params": best_params,
        "cv_best_macro_f1": float(best_score),
        "test_macro_f1": macro_f1,
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    cv_df = pd.DataFrame(all_results).sort_values("mean_f1", ascending=False)
    cv_df.to_csv(out_dir / "cv_top_results.csv", index=False)

    print("  Saving graphs...")
    save_confusion(cm, labels, out_dir / "confusion_matrix.png", "RF Independent Confusion Matrix")
    print("    [1/8] confusion_matrix.png")
    save_feature_importance(model, feature_cols, out_dir / "feature_importance.png")
    print("    [2/8] feature_importance.png")
    save_cv_tuning_curve(all_results, out_dir / "cv_tuning_curve.png")
    print("    [3/8] cv_tuning_curve.png")
    save_class_metrics_bar(y_test, pred, labels, out_dir / "class_metrics_bar.png")
    print("    [4/8] class_metrics_bar.png")
    save_roc_curves(model, X_test, y_test, labels, out_dir / "roc_curves.png")
    print("    [5/8] roc_curves.png")
    save_class_distribution(y, y_train, y_test, out_dir / "class_distribution.png")
    print("    [6/8] class_distribution.png")
    save_prediction_distribution(y_test, pred, labels, out_dir / "prediction_distribution.png")
    print("    [7/8] prediction_distribution.png")
    save_learning_curve(model, X_train, y_train, cv, out_dir / "learning_curve.png")
    print("    [8/8] learning_curve.png")
    save_learning_curve(model, X_train, y_train, cv, out_dir / "learning_curve.png")
    print("    [8/8] learning_curve.png")

    joblib.dump(model, out_dir / "random_forest_independent.joblib")

    _banner("DONE")
    print(f"  Runtime         : {t_total:.1f}s")
    print(f"  Test macro F1   : {macro_f1:.4f}")
    print(f"  CV best macro F1: {best_score:.4f}")
    print(f"  Best params     : {best_params}")
    print(f"  Output dir      : {out_dir.resolve()}")


if __name__ == "__main__":
    main()
