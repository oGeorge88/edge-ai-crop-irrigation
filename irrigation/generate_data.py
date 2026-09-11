"""
Generate synthetic sensor data for irrigation model training.

Simulates realistic ESP32 sensor readings:
  - soil_moisture  : 0–100 % (capacitive sensor)
  - temperature    : 15–45 °C
  - humidity       : 20–95 %
  - rainfall_24h   : 0–50 mm (last 24-hour rain gauge reading)
  - time_since_last_irrigation : 0–96 hours

Label: irrigate (1 = yes, 0 = no)
  Rule: irrigate if soil_moisture < 40 AND rainfall_24h < 5 AND
                   temperature > 20 AND time_since_last_irrigation > 12

Usage:
    python generate_data.py
    python generate_data.py --n_samples 5000 --out data/sensor_data.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--n_samples", type=int, default=3000)
parser.add_argument("--out", default="data/sensor_data.csv")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(args.seed)


def label_irrigate(df: pd.DataFrame) -> np.ndarray:
    """Decision rule derived from agronomic guidelines."""
    return (
        (df["soil_moisture"] < 40) &
        (df["rainfall_24h"] < 5) &
        (df["temperature"] > 20) &
        (df["time_since_last_irrigation"] > 12)
    ).astype(int)


def main():
    n = args.n_samples

    df = pd.DataFrame({
        "soil_moisture":               rng.uniform(10, 95, n).round(1),
        "temperature":                 rng.uniform(15, 45, n).round(1),
        "humidity":                    rng.uniform(20, 95, n).round(1),
        "rainfall_24h":                rng.exponential(3, n).clip(0, 50).round(1),
        "time_since_last_irrigation":  rng.uniform(0, 96, n).round(1),
    })

    # Add small Gaussian noise to make the boundary fuzzy (more realistic)
    df["soil_moisture"] += rng.normal(0, 2, n)
    df["soil_moisture"] = df["soil_moisture"].clip(0, 100).round(1)

    df["irrigate"] = label_irrigate(df)

    out = DATA_DIR / Path(args.out).name
    df.to_csv(out, index=False)

    pos = df["irrigate"].sum()
    print(f"Generated {n} samples → {out}")
    print(f"  Irrigate=1: {pos} ({100*pos/n:.1f}%)  |  Irrigate=0: {n-pos} ({100*(n-pos)/n:.1f}%)")
    print(df.describe().to_string())


if __name__ == "__main__":
    main()
