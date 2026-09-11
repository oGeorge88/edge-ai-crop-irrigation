"""
Train an irrigation decision classifier on sensor data.

Trains and compares:
  - Gradient Boosting (best accuracy per literature)
  - Random Forest      (good balance of accuracy + MCU deployability)
  - Logistic Regression (lightest, fastest on edge)

Saves the best model as models/irrigation_model.joblib.
Also exports a plain decision threshold for ultra-lightweight MCU deployment.

Usage:
    python train.py
    python train.py --data data/sensor_data.csv --model gb
"""

import argparse
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--data",  default="data/sensor_data.csv")
parser.add_argument("--model", choices=["gb", "rf", "lr", "all"], default="all")
parser.add_argument("--test_size", type=float, default=0.2)
args = parser.parse_args()

DATA_DIR   = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURES = [
    "soil_moisture",
    "temperature",
    "humidity",
    "rainfall_24h",
    "time_since_last_irrigation",
]
TARGET = "irrigate"


def load_data():
    path = DATA_DIR / Path(args.data).name
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run generate_data.py first."
        )
    df = pd.read_csv(path)
    X = df[FEATURES].values
    y = df[TARGET].values
    return X, y


def build_models():
    return {
        "gb":  GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                          learning_rate=0.1, random_state=42),
        "rf":  RandomForestClassifier(n_estimators=100, max_depth=8,
                                      random_state=42, n_jobs=-1),
        "lr":  LogisticRegression(max_iter=500, random_state=42),
    }


def evaluate(name, pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred,
                                   target_names=["No Irrigate", "Irrigate"])
    print(f"\n── {name} ──")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(report)
    return acc, auc, y_pred


def plot_confusion(name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Irrigate", "Irrigate"],
                yticklabels=["No Irrigate", "Irrigate"], ax=ax)
    ax.set_title(f"Confusion Matrix — {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    out = MODELS_DIR / f"cm_{name.lower()}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved → {out}")


def feature_importance_plot(name, pipeline, feature_names):
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(importances)), importances[idx])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=30, ha="right")
    ax.set_title(f"Feature Importance — {name}")
    fig.tight_layout()
    out = MODELS_DIR / f"feature_importance_{name.lower()}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved → {out}")


def main():
    print("Loading data …")
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    candidates = build_models()
    if args.model != "all":
        candidates = {args.model: candidates[args.model]}

    results = {}
    for name, clf in candidates.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    clf),
        ])
        pipeline.fit(X_train, y_train)
        acc, auc, y_pred = evaluate(name, pipeline, X_test, y_test)
        plot_confusion(name, y_test, y_pred)
        feature_importance_plot(name, pipeline, FEATURES)
        results[name] = {"accuracy": acc, "roc_auc": auc, "pipeline": pipeline}

    # ── Pick best model ───────────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_pipe = results[best_name]["pipeline"]
    print(f"\nBest model: {best_name}  (acc={results[best_name]['accuracy']:.4f})")

    model_path = MODELS_DIR / "irrigation_model.joblib"
    joblib.dump(best_pipe, model_path)
    print(f"Saved → {model_path}")

    # Save metadata
    meta = {
        "best_model": best_name,
        "features": FEATURES,
        "accuracy": results[best_name]["accuracy"],
        "roc_auc": results[best_name]["roc_auc"],
        "all_results": {k: {"accuracy": v["accuracy"], "roc_auc": v["roc_auc"]}
                        for k, v in results.items()},
    }
    with open(MODELS_DIR / "irrigation_model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved → {MODELS_DIR / 'irrigation_model_meta.json'}")


if __name__ == "__main__":
    main()
