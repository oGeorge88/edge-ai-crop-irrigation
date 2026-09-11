# Edge AI — Crop Disease Detection & Automated Irrigation System

**Author:** George Obinna Oguejiofor | King Mongkut's University of Technology Thonburi (KMUTT)  
**Contact:** george.ogue@kmutt.ac.th

A software-first prototype of an edge-deployable system that combines computer-vision-based crop disease detection with an ML-driven automated irrigation controller. The two subsystems share a common inference pipeline so that detected disease conditions directly modify irrigation decisions.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     EDGE DEVICE                         │
│                                                         │
│  Camera ──► Disease Detector (MobileNetV2 int8 TFLite)  │
│                      │                                  │
│                       ▼  irrigation_action              │
│  Sensors ──► Irrigation Controller (GradientBoosting) ──► Pump ON/OFF
│              (soil, temp, humidity, rain)                │
└─────────────────────────────────────────────────────────┘
```

| Subsystem | Algorithm | Accuracy | Model size |
|-----------|-----------|----------|------------|
| Disease detection | MobileNetV2 (int8 TFLite) | **96.56 %** | ~2.7 MB |
| Irrigation control | GradientBoosting | **100 %** (synthetic) | — |

---

## Project Structure

```
KMUTL/
├── disease_detection/
│   ├── models/
│   │   ├── mobilenetv2_int8.tflite   # trained int8 model (ready to deploy)
│   │   └── model_meta.json           # class names, img_size, accuracy
│   ├── infer.py                      # inference script + DiseaseDetector class
│   ├── train.py                      # local training script (Keras)
│   ├── evaluate.py                   # evaluation on test split
│   ├── convert_tflite.py             # SavedModel → TFLite conversion
│   └── TrainDiseaseModel_Colab.ipynb # Colab notebook (T4 GPU, PlantVillage)
│
├── irrigation/
│   ├── models/
│   │   ├── irrigation_model.joblib   # trained GradientBoosting model
│   │   └── irrigation_model_meta.json
│   ├── generate_data.py              # synthetic sensor dataset generator
│   ├── train.py                      # trains GB / RF / LR, saves best
│   └── predict.py                    # single-reading inference CLI
│
├── simulation/
│   └── demo.py                       # 24-hour integrated farm simulation
│
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd KMUTL
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run disease inference on a leaf image

```bash
python disease_detection/infer.py path/to/leaf.jpg --top 3
```

Example output:
```
==================================================
  Image  : leaf.jpg
  Status : DISEASED
  Disease: Tomato_Late_blight
  Confidence   : 94.1%
  Irrigation   : REDUCE
  Latency      : 6.1 ms
==================================================
```

### 3. Run the irrigation controller

```bash
python irrigation/predict.py --soil 32 --temp 31 --humidity 60 --rain 0 --hours_since 18
```

### 4. Run the integrated 24-hour simulation

```bash
# Without disease detection (sensor-only)
python simulation/demo.py --hours 24

# With disease detection (full system)
python simulation/demo.py --hours 24 --leaf path/to/leaf.jpg
```

The simulation prints a live table and saves a CSV log to `simulation/sim_log.csv`.

---

## Disease Detection Model

### Dataset
- **PlantVillage** via Kaggle (`emmarex/plantdisease`)
- 15 classes: Pepper, Potato, Tomato (diseases + healthy)
- 80 / 10 / 10 % train / val / test split

### Classes
| # | Class | Irrigation action |
|---|-------|------------------|
| 0 | Pepper\_\_bell\_\_\_Bacterial\_spot | REDUCE |
| 1 | Pepper\_\_bell\_\_\_healthy | NORMAL |
| 2 | Potato\_\_\_Early\_blight | REDUCE |
| 3 | Potato\_\_\_Late\_blight | REDUCE |
| 4 | Potato\_\_\_healthy | NORMAL |
| 5 | Tomato\_Bacterial\_spot | REDUCE |
| 6 | Tomato\_Early\_blight | REDUCE |
| 7 | Tomato\_Late\_blight | REDUCE |
| 8 | Tomato\_Leaf\_Mold | REDUCE |
| 9 | Tomato\_Septoria\_leaf\_spot | REDUCE |
| 10 | Tomato\_Spider\_mites\_Two\_spotted\_spider\_mite | INCREASE |
| 11 | Tomato\_\_Target\_Spot | REDUCE |
| 12 | Tomato\_\_Tomato\_YellowLeaf\_\_Curl\_Virus | NORMAL |
| 13 | Tomato\_\_Tomato\_mosaic\_virus | NORMAL |
| 14 | Tomato\_healthy | NORMAL |

### Training (Colab T4 GPU)
Open `disease_detection/TrainDiseaseModel_Colab.ipynb` in Google Colab:

1. `Runtime → Change runtime type → T4 GPU`
2. Add your Kaggle API token to **Colab Secrets** as `KAGGLE_API_TOKEN`
3. Run all cells (~20 min)

Training uses two phases:
- **Phase 1** — frozen MobileNetV2 base, 15 epochs, lr = 1e-3
- **Phase 2** — top 30 layers unfrozen, 10 epochs, lr = 1e-4

### Irrigation logic
| Disease status | Action | Threshold |
|----------------|--------|-----------|
| Fungal / bacterial | `REDUCE` | Irrigate only below 25 % soil moisture |
| Healthy / viral | `NORMAL` | Follow ML model |
| Spider mites | `INCREASE` | Irrigate below 55 % soil moisture |
| Any, critically dry | Emergency | Always irrigate below 20 % |

---

## Irrigation Model

### Features
`soil_moisture`, `temperature`, `humidity`, `rainfall_24h`, `time_since_last_irrigation`

### Training
```bash
python irrigation/generate_data.py   # creates irrigation/data/sensor_data.csv
python irrigation/train.py           # trains GB, RF, LR; saves best to models/
```

---

## Requirements

```
tensorflow-cpu>=2.15
Pillow>=10.0
numpy>=1.24
scikit-learn>=1.3
joblib>=1.3
matplotlib>=3.7
seaborn>=0.12
```

Install with `pip install -r requirements.txt`.

For edge deployment on Raspberry Pi / ESP32-S3, replace `tensorflow-cpu` with `tflite-runtime`.

---

## Deployment Notes

The int8 TFLite model (`disease_detection/models/mobilenetv2_int8.tflite`) runs on:

| Platform | Estimated latency |
|----------|------------------|
| Colab CPU (T4 host) | ~6 ms |
| Raspberry Pi 4 | ~12 ms |
| ESP32-S3 (PSRAM) | ~60 ms |

Input: **128 × 128 RGB uint8**  
Output: softmax probabilities over 15 classes (uint8 dequantised)

---

## License

This project is part of ongoing academic research at KMUTT. All rights reserved.
