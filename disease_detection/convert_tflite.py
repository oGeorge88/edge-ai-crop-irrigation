"""
Convert a trained Keras SavedModel to TensorFlow Lite.

Two outputs:
  - mobilenetv2_float16.tflite  : float16 quantization (~half size, negligible accuracy drop)
  - mobilenetv2_int8.tflite     : full int8 quantization (~4x smaller, fastest on MCU)

The int8 model is what you flash to an ESP32-S3 or Raspberry Pi for on-device inference.

Usage:
    python convert_tflite.py
    python convert_tflite.py --model_dir models/mobilenetv2_final --quant int8
"""

import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="models/mobilenetv2_final")
parser.add_argument("--quant", choices=["float16", "int8", "both"], default="both")
args = parser.parse_args()

MODELS_DIR = Path(__file__).parent / "models"
META_PATH  = MODELS_DIR / "model_meta.json"


def representative_dataset(img_size: int):
    """Calibration samples for int8 quantization — uses 200 real images."""
    def _gen():
        ds = tfds.load("plant_village", split="train[:200]", as_supervised=False)
        for sample in ds:
            image = tf.image.resize(sample["image"], (img_size, img_size))
            image = tf.cast(image, tf.float32) / 255.0
            yield [tf.expand_dims(image, 0)]
    return _gen


def convert_float16(converter: tf.lite.TFLiteConverter, out_path: Path):
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"  float16 → {out_path}  ({size_kb:.1f} KB)")
    return size_kb


def convert_int8(converter: tf.lite.TFLiteConverter, img_size: int, out_path: Path):
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset(img_size)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"  int8    → {out_path}  ({size_kb:.1f} KB)")
    return size_kb


def benchmark_tflite(tflite_path: Path, img_size: int, n: int = 20):
    """Quick latency benchmark on the host CPU to estimate edge performance."""
    import time
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    # build a dummy input with the right dtype
    dtype  = inp["dtype"]
    dummy  = np.random.randint(0, 255, (1, img_size, img_size, 3), dtype=np.uint8)
    if dtype == np.float32:
        dummy = dummy.astype(np.float32) / 255.0

    times = []
    for _ in range(n):
        interpreter.set_tensor(inp["index"], dummy)
        t0 = time.perf_counter()
        interpreter.invoke()
        times.append(time.perf_counter() - t0)

    mean_ms = np.mean(times) * 1000
    print(f"  Latency (CPU host, n={n}): {mean_ms:.1f} ms  "
          f"[note: MCU will be slower — divide by ~5–10x for estimate]")


def main():
    with open(META_PATH) as f:
        meta = json.load(f)
    img_size = meta["img_size"]

    model_path = Path(args.model_dir)
    print(f"Loading model from: {model_path}")

    sizes = {}

    if args.quant in ("float16", "both"):
        converter = tf.lite.TFLiteConverter.from_saved_model(str(model_path))
        out = MODELS_DIR / "mobilenetv2_float16.tflite"
        sizes["float16"] = convert_float16(converter, out)
        benchmark_tflite(out, img_size)

    if args.quant in ("int8", "both"):
        converter = tf.lite.TFLiteConverter.from_saved_model(str(model_path))
        out = MODELS_DIR / "mobilenetv2_int8.tflite"
        sizes["int8"] = convert_int8(converter, img_size, out)
        benchmark_tflite(out, img_size)

    print("\n── Size comparison ──")
    original_kb = sum(
        f.stat().st_size for f in model_path.rglob("*") if f.is_file()
    ) / 1024
    print(f"  Original SavedModel : {original_kb:.1f} KB")
    for name, kb in sizes.items():
        ratio = original_kb / kb if kb else 0
        print(f"  {name:<10} TFLite : {kb:.1f} KB  ({ratio:.1f}x smaller)")

    print("\nDone. Use the int8 .tflite file for embedded deployment.")


if __name__ == "__main__":
    main()
