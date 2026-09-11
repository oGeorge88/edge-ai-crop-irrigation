"""
Edge AI Crop Disease Detection & Automated Irrigation — System Simulation
=========================================================================
Simulates a 24-hour farm day (one tick per hour).

Each tick:
  1. Sensor readings evolve realistically (soil moisture drifts down,
     resets after irrigation; temperature follows a diurnal curve).
  2. The irrigation ML model (GradientBoosting) makes a base decision.
  3. If a leaf image is supplied, the disease detector runs every
     DISEASE_CHECK_INTERVAL ticks and modifies the irrigation threshold:
       REDUCE  — only irrigate when critically dry  (soil < 25 %)
       INCREASE — irrigate more aggressively         (soil < 55 %)
       NORMAL  — follow the model as-is
  4. The pump fires (or stays off) and soil moisture is updated.
  5. A live table is printed and a CSV log is saved.

Usage:
    python simulation/demo.py
    python simulation/demo.py --hours 48 --leaf path/to/leaf.jpg
    python simulation/demo.py --hours 24 --leaf path/to/leaf.jpg --seed 7
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── CLI ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--hours",  type=int,   default=24,
                    help="Simulation length in hours (default: 24)")
parser.add_argument("--leaf",   default=None,
                    help="Path to a leaf image for disease detection")
parser.add_argument("--seed",   type=int,   default=42,
                    help="Random seed for sensor noise")
parser.add_argument("--out",    default="simulation/sim_log.csv",
                    help="Output CSV log path")
parser.add_argument("--disease_interval", type=int, default=6,
                    help="Run disease check every N hours (default: 6)")
args = parser.parse_args()

random.seed(args.seed)

# ── Load irrigation model ──────────────────────────────────────────────────
try:
    import joblib
    import numpy as np
    IRRIG_MODEL = joblib.load(ROOT / "irrigation" / "models" / "irrigation_model.joblib")
except Exception as e:
    print(f"[ERROR] Could not load irrigation model: {e}")
    print("        Run  python irrigation/train.py  first.")
    sys.exit(1)

# ── Load disease detector (optional) ──────────────────────────────────────
detector = None
if args.leaf:
    try:
        from disease_detection.infer import DiseaseDetector
        detector = DiseaseDetector()
        print(f"[INFO] Disease detector ready. Leaf image: {args.leaf}")
    except Exception as e:
        print(f"[WARN] Could not load disease detector: {e}")
        print("       Continuing without disease detection.")

# ── Sensor simulation ──────────────────────────────────────────────────────
SOIL_DRAIN_PER_HOUR  = 2.0    # % soil moisture lost per hour (evapotranspiration)
SOIL_GAIN_ON_PUMP    = 25.0   # % soil moisture gained when pump fires
SOIL_CRITICAL_LOW    = 20.0   # emergency threshold (always irrigate below this)
SOIL_RAIN_GAIN       = 8.0    # % gain per mm of rain

def diurnal_temp(hour: int) -> float:
    """Simple sinusoidal day temperature: min ~24°C at 06:00, max ~36°C at 14:00."""
    angle = math.pi * (hour - 6) / 12
    return 30.0 + 6.0 * math.sin(angle) + random.uniform(-1.0, 1.0)

def diurnal_humidity(temp: float) -> float:
    """Humidity inversely follows temperature."""
    base = 85.0 - (temp - 24.0) * 1.5
    return max(30.0, min(95.0, base + random.uniform(-5.0, 5.0)))

def rain_event(hour: int, seed: int) -> float:
    """Occasional short rain showers (probability 10% per hour, mostly at night)."""
    rng = random.Random(seed * 1000 + hour)
    night = 1.5 if (hour < 6 or hour > 20) else 1.0
    if rng.random() < 0.10 * night:
        return round(rng.uniform(0.5, 8.0), 1)
    return 0.0

# ── Disease modifier on irrigation threshold ───────────────────────────────
IRRIG_THRESHOLD = {
    "REDUCE":   25.0,   # only fire pump if critically dry
    "NORMAL":   None,   # use ML model decision as-is
    "INCREASE": 55.0,   # fire pump whenever below this level
}

def irrigation_decision(soil: float, temp: float, humidity: float,
                         rain: float, hours_since: float,
                         disease_action: str) -> bool:
    """Combine ML model + disease modifier into a single pump decision."""
    # Emergency: always water when critically dry regardless of disease
    if soil < SOIL_CRITICAL_LOW:
        return True

    threshold = IRRIG_THRESHOLD[disease_action]

    if threshold is not None:
        return soil < threshold

    # NORMAL: defer to ML model
    X = np.array([[soil, temp, humidity, rain, hours_since]])
    return bool(IRRIG_MODEL.predict(X)[0])

# ── Colour helpers (ANSI, graceful fallback on Windows without VT100) ─────
def _c(code, text):
    try:
        return f"\033[{code}m{text}\033[0m"
    except Exception:
        return text

GREEN  = lambda t: _c("32", t)
RED    = lambda t: _c("31", t)
YELLOW = lambda t: _c("33", t)
CYAN   = lambda t: _c("36", t)
BOLD   = lambda t: _c("1",  t)

# ── Main simulation loop ───────────────────────────────────────────────────
def run():
    os.makedirs(Path(args.out).parent, exist_ok=True)

    # State
    soil            = random.uniform(35.0, 55.0)   # start mid-range
    hours_since     = random.uniform(6.0, 18.0)
    disease_action  = "NORMAL"
    disease_name    = "—"
    disease_conf    = 0.0
    pump_count      = 0
    total_water     = 0.0          # arbitrary units (each pump event = 1 unit)
    log_rows        = []

    # Print header
    print()
    print(BOLD("=" * 78))
    print(BOLD("  Edge AI — Crop Disease Detection & Automated Irrigation  SIMULATION"))
    print(BOLD("=" * 78))
    print(f"  Duration : {args.hours} hours")
    print(f"  Leaf img : {args.leaf or 'not provided (disease detection disabled)'}")
    print(f"  Seed     : {args.seed}")
    print(BOLD("=" * 78))
    print()
    print(f"{'Hr':>3}  {'Soil%':>6}  {'Temp°C':>7}  {'RH%':>5}  {'Rain':>5}  "
          f"{'Disease':<34}  {'IrrigAct':>8}  {'Pump':>5}")
    print("-" * 90)

    for hour in range(args.hours):
        clock = hour % 24

        # 1. Evolve sensors
        rain   = rain_event(clock, args.seed)
        temp   = diurnal_temp(clock)
        humid  = diurnal_humidity(temp)
        soil   = max(5.0, soil - SOIL_DRAIN_PER_HOUR + rain * SOIL_RAIN_GAIN / 10)
        soil   = min(100.0, soil)
        hours_since += 1

        # 2. Disease check (every N hours, if detector available)
        if detector and hour % args.disease_interval == 0:
            try:
                res            = detector.predict(args.leaf, top_k=1)
                disease_name   = res["disease"]
                disease_conf   = res["confidence"]
                disease_action = res["irrigation_action"]
            except Exception as ex:
                print(f"[WARN] Disease check failed at hour {hour}: {ex}")

        # 3. Irrigation decision
        pump = irrigation_decision(soil, temp, humid, rain, hours_since, disease_action)

        # 4. Apply pump
        if pump:
            soil        = min(100.0, soil + SOIL_GAIN_ON_PUMP)
            hours_since = 0.0
            pump_count += 1
            total_water += 1.0

        # 5. Print row
        soil_str  = f"{soil:5.1f}%"
        pump_icon = GREEN("  ON ") if pump else RED(" OFF ")
        disease_col = f"{disease_name[:28]:<28} {disease_conf:4.0%}" if detector else f"{'(disabled)':<34}"

        action_col = {
            "REDUCE":   YELLOW("REDUCE  "),
            "INCREASE": CYAN(  "INCREASE"),
            "NORMAL":   "NORMAL  ",
        }.get(disease_action, disease_action)

        print(f"{clock:>3}h  {soil_str:>6}  {temp:>6.1f}  {humid:>5.1f}  "
              f"{rain:>4.1f}  {disease_col}  {action_col}  {pump_icon}")

        # 6. Log
        log_rows.append({
            "hour":                     hour,
            "clock_hour":               clock,
            "soil_moisture":            round(soil, 2),
            "temperature":              round(temp, 2),
            "humidity":                 round(humid, 2),
            "rainfall_mm":              rain,
            "hours_since_irrigation":   round(hours_since, 1),
            "disease":                  disease_name,
            "disease_confidence":       round(disease_conf, 4),
            "disease_irrigation_action": disease_action,
            "pump":                     int(pump),
        })

    # ── Summary ────────────────────────────────────────────────────────────
    print("-" * 90)
    print()
    print(BOLD("  SUMMARY"))
    print(f"  Pump activations : {pump_count} / {args.hours} hours "
          f"({pump_count/args.hours:.0%})")
    print(f"  Water units used : {total_water:.0f}")
    final_soil = log_rows[-1]["soil_moisture"]
    print(f"  Final soil moisture: {final_soil:.1f}%")
    if detector:
        last_disease = log_rows[-1]["disease"]
        last_action  = log_rows[-1]["disease_irrigation_action"]
        print(f"  Last disease check : {last_disease}  →  {last_action}")
    print()

    # ── Save CSV ───────────────────────────────────────────────────────────
    out_path = ROOT / args.out
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"  Log saved → {out_path}")
    print()


if __name__ == "__main__":
    run()
