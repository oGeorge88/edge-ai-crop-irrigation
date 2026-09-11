"""
Disease Detection Training — MobileNetV2 transfer learning on PlantVillage.

Uses tensorflow-datasets to automatically download and split PlantVillage.
Trains a MobileNetV2 head, then fine-tunes the top layers.

Usage:
    python train.py
    python train.py --epochs 20 --batch_size 32 --img_size 128
"""

import argparse
import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── CLI ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--epochs",      type=int, default=15)
parser.add_argument("--finetune_epochs", type=int, default=10)
parser.add_argument("--batch_size",  type=int, default=32)
parser.add_argument("--img_size",    type=int, default=128)
parser.add_argument("--lr",          type=float, default=1e-3)
args = parser.parse_args()

IMG_SIZE   = (args.img_size, args.img_size)
AUTOTUNE   = tf.data.AUTOTUNE


def preprocess(sample):
    image = tf.image.resize(sample["image"], IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    label = sample["label"]
    return image, label


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.15)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    return image, label


def build_dataset():
    (ds_train, ds_val, ds_test), info = tfds.load(
        "plant_village",
        split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
        with_info=True,
        as_supervised=False,
    )
    num_classes = info.features["label"].num_classes
    class_names = info.features["label"].names

    ds_train = (
        ds_train
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .map(augment,    num_parallel_calls=AUTOTUNE)
        .shuffle(1000)
        .batch(args.batch_size)
        .prefetch(AUTOTUNE)
    )
    ds_val = (
        ds_val
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .batch(args.batch_size)
        .prefetch(AUTOTUNE)
    )
    ds_test = (
        ds_test
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .batch(args.batch_size)
        .prefetch(AUTOTUNE)
    )
    return ds_train, ds_val, ds_test, num_classes, class_names


def build_model(num_classes: int) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # freeze for transfer learning phase

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs), base


def plot_history(history, tag=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["accuracy", "loss"]):
        ax.plot(history.history[metric],     label=f"train {metric}")
        ax.plot(history.history[f"val_{metric}"], label=f"val {metric}")
        ax.set_title(f"{tag} {metric}")
        ax.legend()
    fig.tight_layout()
    out = MODELS_DIR / f"training_curve_{tag}.png"
    fig.savefig(out)
    print(f"Saved plot → {out}")
    plt.close(fig)


def main():
    print("Loading PlantVillage dataset …")
    ds_train, ds_val, ds_test, num_classes, class_names = build_dataset()
    print(f"  Classes: {num_classes}")

    # ── Phase 1: transfer learning (frozen base) ──────────────────────────
    model, base = build_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODELS_DIR / "mobilenetv2_phase1.keras"),
            save_best_only=True,
        ),
    ]

    print("\n── Phase 1: transfer learning ──")
    h1 = model.fit(ds_train, validation_data=ds_val,
                   epochs=args.epochs, callbacks=callbacks)
    plot_history(h1, "phase1")

    # ── Phase 2: fine-tune top 30 layers ─────────────────────────────────
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr / 10),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks[1] = tf.keras.callbacks.ModelCheckpoint(
        str(MODELS_DIR / "mobilenetv2_phase2.keras"),
        save_best_only=True,
    )

    print("\n── Phase 2: fine-tuning ──")
    h2 = model.fit(ds_train, validation_data=ds_val,
                   epochs=args.finetune_epochs, callbacks=callbacks)
    plot_history(h2, "phase2")

    # ── Evaluate on test set ──────────────────────────────────────────────
    print("\n── Test evaluation ──")
    loss, acc = model.evaluate(ds_test)
    print(f"  Test accuracy : {acc:.4f}")
    print(f"  Test loss     : {loss:.4f}")

    # Save class names for inference
    meta = {"class_names": class_names, "img_size": args.img_size,
            "num_classes": num_classes, "test_accuracy": round(acc, 4)}
    with open(MODELS_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata → {MODELS_DIR / 'model_meta.json'}")

    # Save final SavedModel for TFLite conversion
    model.save(MODELS_DIR / "mobilenetv2_final")
    print(f"Saved model    → {MODELS_DIR / 'mobilenetv2_final'}")


if __name__ == "__main__":
    main()
