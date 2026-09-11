"""
Crop disease inference using the int8 TFLite model.

Loads mobilenetv2_int8.tflite and model_meta.json from the models/ directory,
classifies a crop leaf image, and returns a structured result including a
recommended irrigation action.

Usage (CLI):
    python infer.py path/to/leaf.jpg
    python infer.py path/to/leaf.jpg --top 3

Usage (library):
    from disease_detection.infer import DiseaseDetector
    detector = DiseaseDetector()
    result = detector.predict("leaf.jpg")
    print(result["disease"], result["confidence"], result["irrigation_action"])
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

MODELS_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODELS_DIR / "mobilenetv2_int8.tflite"
META_PATH  = MODELS_DIR / "model_meta.json"

# Per-class irrigation recommendation.
# Fungal/bacterial diseases spread via wet foliage → reduce overhead watering.
# Spider mites prefer dry conditions → slight increase helps suppress them.
# Viral diseases are not affected by irrigation → keep normal schedule.
_IRRIGATION = {
    "Pepper__bell___Bacterial_spot":                    "REDUCE",
    "Pepper__bell___healthy":                           "NORMAL",
    "Potato___Early_blight":                            "REDUCE",
    "Potato___Late_blight":                             "REDUCE",
    "Potato___healthy":                                 "NORMAL",
    "Tomato_Bacterial_spot":                            "REDUCE",
    "Tomato_Early_blight":                              "REDUCE",
    "Tomato_Late_blight":                               "REDUCE",
    "Tomato_Leaf_Mold":                                 "REDUCE",
    "Tomato_Septoria_leaf_spot":                        "REDUCE",
    "Tomato_Spider_mites_Two_spotted_spider_mite":      "INCREASE",
    "Tomato__Target_Spot":                              "REDUCE",
    "Tomato__Tomato_YellowLeaf__Curl_Virus":            "NORMAL",
    "Tomato__Tomato_mosaic_virus":                      "NORMAL",
    "Tomato_healthy":                                   "NORMAL",
}


class DiseaseDetector:
    """Wraps the int8 TFLite model for single-image crop disease inference."""

    def __init__(self, model_path=MODEL_PATH, meta_path=META_PATH):
        try:
            import tflite_runtime.interpreter as tflite
            Interpreter = tflite.Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter

        with open(meta_path) as f:
            meta = json.load(f)

        self.class_names = meta["class_names"]
        self.img_size    = meta["img_size"]

        self._interp = Interpreter(model_path=str(model_path))
        self._interp.allocate_tensors()
        self._inp = self._interp.get_input_details()[0]
        self._out = self._interp.get_output_details()[0]

    def _preprocess(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)   # model expects uint8 input
        return arr[np.newaxis, ...]            # add batch dimension

    def predict(self, image_path: str, top_k: int = 1) -> dict:
        """
        Classify one image.

        Returns a dict:
            disease          – top predicted class name
            confidence       – float in [0, 1]
            irrigation_action – 'NORMAL' | 'REDUCE' | 'INCREASE'
            is_healthy       – bool
            top_k            – list of (class_name, confidence) for top_k classes
            latency_ms       – inference time in milliseconds
        """
        tensor = self._preprocess(image_path)
        self._interp.set_tensor(self._inp["index"], tensor)

        t0 = time.perf_counter()
        self._interp.invoke()
        latency_ms = (time.perf_counter() - t0) * 1000

        raw = self._interp.get_tensor(self._out["index"])[0]  # shape: (num_classes,)

        # Dequantise uint8 output to float probabilities
        scale, zero_point = self._out["quantization"]
        if scale > 0:
            probs = (raw.astype(np.float32) - zero_point) * scale
        else:
            probs = raw.astype(np.float32)

        # Normalise in case quantisation shifts the sum slightly off 1
        probs = probs / probs.sum()

        top_indices = np.argsort(probs)[::-1][:top_k]
        top_results = [(self.class_names[i], float(probs[i])) for i in top_indices]

        disease    = top_results[0][0]
        confidence = top_results[0][1]

        return {
            "disease":           disease,
            "confidence":        confidence,
            "irrigation_action": _IRRIGATION.get(disease, "NORMAL"),
            "is_healthy":        "healthy" in disease.lower(),
            "top_k":             top_results,
            "latency_ms":        round(latency_ms, 2),
        }


def main():
    parser = argparse.ArgumentParser(description="Crop disease inference (int8 TFLite)")
    parser.add_argument("image", help="Path to a leaf image (JPG/PNG)")
    parser.add_argument("--top", type=int, default=3,
                        help="Show top-N predictions (default: 3)")
    parser.add_argument("--model", default=str(MODEL_PATH),
                        help="Path to .tflite model file")
    parser.add_argument("--meta", default=str(META_PATH),
                        help="Path to model_meta.json")
    args = parser.parse_args()

    detector = DiseaseDetector(model_path=args.model, meta_path=args.meta)
    result   = detector.predict(args.image, top_k=args.top)

    status = "HEALTHY" if result["is_healthy"] else "DISEASED"
    print(f"\n{'='*50}")
    print(f"  Image  : {args.image}")
    print(f"  Status : {status}")
    print(f"  Disease: {result['disease']}")
    print(f"  Confidence   : {result['confidence']:.1%}")
    print(f"  Irrigation   : {result['irrigation_action']}")
    print(f"  Latency      : {result['latency_ms']} ms")
    print(f"{'='*50}")
    if args.top > 1:
        print("  Top predictions:")
        for name, conf in result["top_k"]:
            print(f"    {conf:.1%}  {name}")
    print()


if __name__ == "__main__":
    main()
