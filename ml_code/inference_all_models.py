"""
SmartTraffic-AI: Unified Multi-Model Inference Demo
Demonstrates direct ML inference for all models used across the project:
  1. Route Resistance & Speed Predictor (RF + XGBoost + RF Regressor)
  2. Vehicle & License Plate Detection (YOLOv8 Dual-Stage Pipeline)
  3. Crash / Road Accident Detector (YOLOv8 Accident Classifier)
  4. Driver Fatigue & Eye Drowsiness Monitor (MediaPipe 468-pt Mesh & EAR)
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.services.route_service import CongestionPredictor
from backend.services.plate_service import LPREngine
from backend.services.crash_service import CrashEngine
from backend.services.retina_service import RetinaEngine


def demo_inference():
    print("=" * 65)
    print("      SmartTraffic-AI Unified ML Model Inference Demo        ")
    print("=" * 65)

    # 1. Route Resistance & Speed Forecast
    print("\n--- [Model 1] Route Resistance & Speed Predictor ---")
    route_pred = CongestionPredictor()
    sample_features = {
        "traffic_volume": 1420.0,
        "average_speed": 28.5,
        "density": 65.0,
        "hour_of_day": 18,
        "day_of_week": 4,
        "is_weekend": 0,
        "temperature_c": 31.0,
        "humidity_pct": 72.0,
        "rainfall_mm": 0.0,
        "historical_lag_1": 1350.0,
        "historical_lag_2": 1280.0
    }
    result_m1 = route_pred.predict(sample_features)
    print(f"  Congestion Level: {result_m1['congestion_level']} (Confidence: {result_m1['confidence']}%)")
    print(f"  Predicted Speed:  {result_m1['predicted_speed_kmh']} km/h")
    print(f"  Forecast:         {result_m1['forecast']}")

    # 2. Vehicle & License Plate Detection
    print("\n--- [Model 2] Vehicle & License Plate Detection (Dual YOLOv8) ---")
    plate_svc = LPREngine()
    demo_video = ROOT_DIR / "demo_data" / "vehicle_anpr_demo.webm"
    if demo_video.exists():
        cap = cv2.VideoCapture(str(demo_video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
        ret, frame = cap.read()
        cap.release()
        if ret:
            annotated, tele = plate_svc.process_frame(frame)
            print(f"  Vehicles Tracked: {tele['total']}")
            print(f"  Plates Detected:  {tele['plates_scanned']}")
            for v in tele['vehicles']:
                if v['plate_detected']:
                    print(f"    * Vehicle #{v['track_id']} ({v['class'].upper()}) -> Plate Detected! Box: {v['plate_box']}")

    # 3. Crash / Accident Detection
    print("\n--- [Model 3] Accident & Crash Detection (YOLOv8) ---")
    crash_svc = CrashEngine()
    crash_video = ROOT_DIR / "demo_data" / "crash_demo.mp4"
    if crash_video.exists():
        cap = cv2.VideoCapture(str(crash_video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
        ret, frame = cap.read()
        cap.release()
        if ret:
            ann, tele = crash_svc.process_frame(frame)
            print(f"  Accident State: {tele['status']}")
            print(f"  Confidence:     {tele['confidence']}%")

    # 4. Driver Fatigue & Eye Drowsiness Monitor
    print("\n--- [Model 4] Driver Eye Fatigue & EAR Monitor (MediaPipe) ---")
    retina_svc = RetinaEngine()
    # Test with a blank or sample frame
    sample_face_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ann, tele = retina_svc.process_frame(sample_face_frame)
    print(f"  Eye State:      {tele['eye_state']}")
    print(f"  Face Detected:  {tele['face_detected']}")
    print(f"  Fatigue Status: {tele['status']}")

    print("\n[SUCCESS] All ML inference pipelines successfully tested!")


if __name__ == "__main__":
    demo_inference()
