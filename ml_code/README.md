# SmartTraffic-AI: Standalone Machine Learning Models

This directory contains clean, standalone, decoupled Python scripts for all 4 core Machine Learning pipelines in the project. Each file can be run directly from the command line without launching the full web dashboard.

---

## Model Scripts Directory

| Script | Module | Description | Key Tech |
| :--- | :--- | :--- | :--- |
| [`01_route_resistance_model.py`](01_route_resistance_model.py) | **Module 1** | Route Congestion & Multi-Horizon Speed Predictor | Random Forest, XGBoost, Scikit-Learn |
| [`02_license_plate_detection_yolo.py`](02_license_plate_detection_yolo.py) | **Module 2** | Dedicated License Plate Detection & Vehicle Tracking | YOLOv8 (Ultralytics), PyTorch, OpenCV |
| [`03_crash_accident_detection.py`](03_crash_accident_detection.py) | **Module 3** | Highway Crash & Accident Collision Detection | YOLOv8 Collision Detector |
| [`04_driver_eye_fatigue_sensor.py`](04_driver_eye_fatigue_sensor.py) | **Module 4** | Driver Eye Aspect Ratio (EAR) & Fatigue Monitor | MediaPipe 468-pt FaceMesh, SciPy |
| [`inference_all_models.py`](inference_all_models.py) | **All** | Unified benchmark testing all 4 models sequentially | All Frameworks |

---

## How to Run Each ML Model Independently

### 1. Route Resistance & Speed Forecasting
Trains the Random Forest and XGBoost models on historical traffic data and evaluates a sample prediction:
```powershell
python ml_code/01_route_resistance_model.py
```

### 2. Dedicated License Plate Detection
Runs two-stage vehicle tracking and plate detection on the demo video, saving an annotated frame:
```powershell
# Run inference test:
python ml_code/02_license_plate_detection_yolo.py

# Retrain YOLOv8 detector (optional):
python ml_code/02_license_plate_detection_yolo.py --train --data path/to/data.yaml --epochs 25
```

### 3. Crash & Accident Detection
Processes collision footage, checks accident thresholds, and saves an emergency-annotated output:
```powershell
python ml_code/03_crash_accident_detection.py
```

### 4. Driver Eye Fatigue & EAR Sensor
Computes 6-point Euclidean Eye Aspect Ratio and tracks blinks/drowsiness:
```powershell
# Headless test:
python ml_code/04_driver_eye_fatigue_sensor.py

# Live webcam monitoring:
python ml_code/04_driver_eye_fatigue_sensor.py --webcam
```

### 5. Unified Test
Runs an automated health check across all models:
```powershell
python ml_code/inference_all_models.py
```
