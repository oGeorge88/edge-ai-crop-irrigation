"""
AI → Firmware Bridge — Edge AI Crop Disease + Irrigation System
Author: George Obinna Oguejiofor | KMITL

Runs MobileNetV2 disease inference on a leaf image, then sends the resulting
irrigation mode command to an ESP32-S3 over USB serial.

Usage (dry-run, no hardware):
    python pipeline.py leaf.jpg

Usage (with hardware):
    python pipeline.py leaf.jpg --port COM3
    python pipeline.py leaf.jpg --port /dev/ttyUSB0 --baud 115200

The ESP32-S3 firmware expects one of these commands (115200 baud, newline-terminated):
    MODE:NORMAL
    MODE:REDUCE
    MODE:INCREASE
"""

import argparse
import sys
import time
from pathlib import Path

# Resolve the disease_detection package relative to this file
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from disease_detection.infer import DiseaseDetector


def run_inference(image_path: str) -> dict:
    detector = DiseaseDetector()
    return detector.predict(image_path, top_k=3)


def send_mode(port: str, baud: int, mode: str) -> None:
    try:
        import serial
    except ImportError:
        print("[ERROR] pyserial is not installed.  Run:  pip install pyserial")
        sys.exit(1)

    command = f"MODE:{mode}\n"
    print(f"\n[serial] Opening {port} @ {baud} baud …")
    with serial.Serial(port, baud, timeout=2) as ser:
        time.sleep(1.5)          # wait for ESP32 reset after DTR toggle
        ser.write(command.encode())
        ser.flush()
        print(f"[serial] Sent: {command.strip()}")

        # Read back one telemetry block to confirm the mode was applied
        deadline = time.time() + 5
        buf = ""
        while time.time() < deadline:
            line = ser.readline().decode(errors="replace").strip()
            if not line:
                continue
            buf += line + "\n"
            if ">> PUMP" in line:
                break

    if buf:
        print("\n[serial] ESP32 response:")
        for ln in buf.strip().splitlines():
            print(f"         {ln}")
    else:
        print("[serial] No response received within 5 s (check baud / port).")


def print_result(result: dict, image_path: str) -> None:
    status = "HEALTHY" if result["is_healthy"] else "DISEASED"
    mode   = result["irrigation_action"]
    sep    = "=" * 54
    print(f"\n{sep}")
    print(f"  Image     : {image_path}")
    print(f"  Status    : {status}")
    print(f"  Disease   : {result['disease']}")
    print(f"  Confidence: {result['confidence']:.1%}")
    print(f"  Latency   : {result['latency_ms']} ms")
    print(f"  ▶ Irrigation mode: {mode}")
    print(f"{sep}")
    if result["top_k"]:
        print("  Top predictions:")
        for name, conf in result["top_k"]:
            marker = " ◀" if name == result["disease"] else ""
            print(f"    {conf:.1%}  {name}{marker}")
    print()

    cmd = f"MODE:{mode}"
    print(f"  Wokwi serial command: {cmd}")
    print(f"  (paste into Wokwi serial monitor and press Enter)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run disease inference and optionally push mode to ESP32-S3"
    )
    parser.add_argument("image",         help="Path to leaf image (JPG / PNG)")
    parser.add_argument("--port", "-p",  default=None,
                        help="Serial port (e.g. COM3 or /dev/ttyUSB0). "
                             "Omit for dry-run (print command only).")
    parser.add_argument("--baud", "-b",  type=int, default=115200,
                        help="Baud rate (default: 115200)")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    result = run_inference(args.image)
    print_result(result, args.image)

    if args.port:
        send_mode(args.port, args.baud, result["irrigation_action"])
    else:
        print("  (use --port <COMx> to send to ESP32-S3 hardware)\n")


if __name__ == "__main__":
    main()
