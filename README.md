# SmartTraffic-AI

**Intelligent Urban Traffic Congestion Management & Driver Safety AI System**

SmartTraffic-AI is a modular, production-ready computer vision and machine learning application integrating 5 core intelligence modules into a unified dashboard.

---

## 🚀 The 5 Core Modules

1. **Indian Route Planner & Resistance Index**
   - Dynamic urban road network graph routing using **NetworkX**.
   - Multi-horizon congestion forecasting (+10m, +30m, +60m) powered by **Random Forest** and **XGBoost**.
   - Resistance Index combining live speed, incident penalties, and waterlogging risks.

2. **Vehicle Type & Dedicated License Plate Detection**
   - Dual-stage pure vision pipeline: **YOLOv8** vehicle classification (`Car`, `Bus`, `Truck`, `Motorcycle`).
   - Dedicated YOLOv8 license plate detector (`best.pt`, 96.14% Precision, 94.16% mAP50) highlighting plates in **bright RED**.
   - Real-time base64 physical plate crop evidence in the UI table.
   - Zero OCR text guessing and zero speed clutter.

3. **Crash Predictor & Accident Detection**
   - Real-time accident identification trained on surveillance collision footage.
   - Immediate audio-visual emergency alert dispatch.

4. **Driver Eye Sensor & Fatigue Monitor**
   - Real-time driver eyelid tracking directly via **browser webcam** using **MediaPipe 468 FaceMesh**.
   - Computes Eye Aspect Ratio (EAR), tracks blink frequency, and triggers warnings upon prolonged eye closure.

5. **Model Evaluation Metrics**
   - Live benchmark viewer displaying measured Accuracy, Precision, Recall, F1-scores, and confusion matrices across all deployed models.

---

## 📁 Clean Project Structure

```text
SmartTraffic-AI/
│
├── backend/
│   ├── app.py                     # Flask web server & streaming endpoints
│   ├── services/
│   │   ├── route_service.py       # SmartRouter & CongestionPredictor
│   │   ├── plate_service.py       # Vehicle & Red Plate Detection Engine
│   │   ├── crash_service.py       # Accident Detection AI Engine
│   │   └── retina_service.py      # MediaPipe Eye Tracking Engine
│   └── utils/
│       └── helpers.py             # JSON serialization & array utilities
│
├── frontend/
│   ├── templates/
│   │   └── index.html             # Clean modern UI dashboard
│   └── static/
│       ├── css/
│       │   └── style.css          # Responsive dashboard styling
│       └── js/
│           ├── main.js            # Video streams, telemetry, & webcam logic
│           └── congestion.js      # Map routing & forecasting visualization
│
├── ml_models/
│   ├── route_resistance/          # 5 Joblib models for congestion & speed
│   ├── vehicle_detection/         # yolov8n.pt (6.5 MB)
│   ├── license_plate_detection/   # Retrained best.pt (6.2 MB)
│   ├── crash_prediction/          # crash_accident_best.pt (22.5 MB)
│   └── retina_ear/                # Caffe face detector models
│
├── demo_data/
│   ├── vehicle_anpr_demo.webm     # Demo highway footage for Module 2
│   ├── crash_demo.mp4             # Demo accident footage for Module 3
│   └── traffic_historical.csv     # Historical route resistance dataset
│
├── evaluation/
│   ├── evaluate_all.py            # Automated 5-module test & benchmark suite
│   └── benchmarks.json            # Measured metrics across all models
│
├── config/
│   └── settings.py                # Dynamic path resolution relative to project root
│
├── run.bat                        # One-click Windows launch script
├── requirements.txt               # Minimal verified dependencies
├── .gitignore                     # Standard repository ignore file
└── README.md                      # Documentation
```

---

## ⚡ Quick Start

### 1. Requirements
Ensure Python 3.10+ is installed with CUDA/PyTorch if GPU acceleration is available:
```powershell
pip install -r requirements.txt
```

### 2. Launch the Application
Simply double-click:
```powershell
run.bat
```
Or start via terminal:
```powershell
python backend/app.py
```
Open your browser at: **`http://localhost:5000`**

### 3. Run Automated System Verification
```powershell
python evaluation/evaluate_all.py
```
All 5 modules will execute live functional tests and output a 100% PASS verification report.
