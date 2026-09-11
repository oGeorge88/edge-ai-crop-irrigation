"""
Evaluate a trained disease detection model on the PlantVillage test split.

Produces:
  - Classification report (precision, recall, F1 per class)
  - Confusion matrix heatmap
  - Top-5 misclassified examples

Usage:
    python evaluate.py
    python evaluate.py --model_dir models/mobilenetv2_final
"""

import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="models/mobilenetv2_final")
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

MODELS_DIR = Path(__file__).parent / "models"
META_PATH  = MODELS_DIR / "model_meta.json"


def load_meta():
    with open(META_PATH) as f:
        return json.load(f)


def build_test_ds(img_size, batch_size):
    def preprocess(sample):
        image = tf.image.resize(sample["image"], (img_size, img_size))
        image = tf.cast(image, tf.float32) / 255.0
        return image, sample["label"]

    ds, info = tfds.load(
        "plant_village",
        split="train[90%:]",
        with_info=True,
        as_supervised=False,
    )
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, info


def main():
    meta = load_meta()
    class_names = meta["class_names"]
    img_size    = meta["img_size"]

    print(f"Loading model from: {args.model_dir}")
    model = tf.keras.models.load_model(args.model_dir)

    print("Building test dataset …")
    ds_test, _ = build_test_ds(img_size, args.batch_size)

    print("Running inference …")
    y_true, y_pred = [], []
    for images, labels in ds_test:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ── Classification report ─────────────────────────────────────────────
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\n── Classification Report ──")
    print(report)
    report_path = MODELS_DIR / "classification_report.txt"
    report_path.write_text(report)
    print(f"Saved → {report_path}")

    # ── Confusion matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
        cm, annot=False, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — PlantVillage Test Set")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0,  fontsize=7)
    fig.tight_layout()
    cm_path = MODELS_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    print(f"Saved → {cm_path}")
    plt.close(fig)

    # ── Overall accuracy ──────────────────────────────────────────────────
    overall = (y_true == y_pred).mean()
    print(f"\nOverall accuracy: {overall:.4f}")


if __name__ == "__main__":
    main()
