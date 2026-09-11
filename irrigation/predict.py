"""
Irrigation inference — simulates what runs on the edge device.

Loads the trained model and makes a real-time decision from sensor readings.
In production this script runs on the Raspberry Pi / ESP32 gateway,
reading from actual GPIO/I2C sensors.

Usage:
    python predict.py --soil 35 --temp 30 --humidity 55 --rain 0 --hours_since 18
    python predict.py  # uses default values to demo
"""

import argparse
import json
import joblib
import numpy as np
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"

parser = argparse.ArgumentParser()
parser.add_argument("--soil",        type=float, default=32.0,
                    help="Soil moisture 0-100%%")
parser.add_argument("--temp",        type=float, default=31.0,
                    help="Temperature in °C")
parser.add_argument("--humidity",    type=float, default=60.0,
                    help="Air humidity 0-100%%")
parser.add_argument("--rain",        type=float, default=1.5,
                    help="Rainfall in last 24h (mm)")
parser.add_argument("--hours_since", type=float, default=20.0,
                    help="Hours since last irrigation")
parser.add_argument("--model", default="models/irrigation_model.joblib")
args = parser.parse_args()


def load_model():
    path = MODELS_DIR / Path(args.model).name
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run train.py first."
        )
    return joblib.load(path)


def load_meta():
    path = MODELS_DIR / "irrigation_model_meta.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def decide(model, reading: dict) -> dict:
    features = [
        reading["soil_moisture"],
        reading["temperature"],
        reading["humidity"],
        reading["rainfall_24h"],
        reading["time_since_last_irrigation"],
    ]
    X = np.array(features).reshape(1, -1)
    decision   = int(model.predict(X)[0])
    confidence = float(model.predict_proba(X)[0][decision])
    return {"irrigate": bool(decision), "confidence": confidence}


def main():
    model = load_model()
    meta  = load_meta()

    reading = {
        "soil_moisture":               args.soil,
        "temperature":                 args.temp,
        "humidity":                    args.humidity,
        "rainfall_24h":                args.rain,
        "time_since_last_irrigation":  args.hours_since,
    }

    result = decide(model, reading)

    print("\n── Sensor Reading ──────────────────────")
    for k, v in reading.items():
        print(f"  {k:<32} {v}")

    print("\n── Irrigation Decision ─────────────────")
    action = "IRRIGATE NOW" if result["irrigate"] else "NO IRRIGATION NEEDED"
    print(f"  Decision   : {action}")
    print(f"  Confidence : {result['confidence']:.1%}")

    if meta:
        print(f"\n  Model      : {meta.get('best_model', 'unknown')}")
        print(f"  Trained acc: {meta.get('accuracy', '?'):.4f}")

    return result


if __name__ == "__main__":
    main()
