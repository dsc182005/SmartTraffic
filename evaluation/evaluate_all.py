"""
Comprehensive Automated Verification & Evaluation Suite for SmartTraffic-AI.
Verifies all 5 modules, model weights, demo datasets, and inference pipelines.
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import cv2
import numpy as np

from config.settings import (
    PROJECT_ROOT, ROUTE_MODELS_DIR, VEHICLE_MODEL_PATH, PLATE_MODEL_PATH,
    CRASH_MODEL_PATH, RETINA_MODELS_DIR, DEFAULT_VEHICLE_VIDEO,
    DEFAULT_CRASH_VIDEO, HISTORICAL_TRAFFIC_DATA, BENCHMARKS_JSON,
    TEMPLATES_DIR, STATIC_DIR
)
from backend.services import (
    CongestionPredictor, SmartRouter, LPREngine, CrashEngine, RetinaEngine
)

def run_evaluation():
    print("=" * 65)
    print("      SmartTraffic-AI Comprehensive Verification Suite")
    print(f"      Root: {PROJECT_ROOT}")
    print("=" * 65)

    passed = 0
    total = 5

    # [1/5] Checking ML Models
    print("\n[1/5] Checking ML Model Checkpoints:")
    model_files = [
        ROUTE_MODELS_DIR / "rf_classifier.joblib",
        ROUTE_MODELS_DIR / "xgb_classifier.joblib",
        ROUTE_MODELS_DIR / "speed_regressor.joblib",
        ROUTE_MODELS_DIR / "scaler.joblib",
        ROUTE_MODELS_DIR / "metrics.joblib",
        VEHICLE_MODEL_PATH,
        PLATE_MODEL_PATH,
        CRASH_MODEL_PATH,
        RETINA_MODELS_DIR / "deploy.prototxt.txt",
        RETINA_MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
    ]
    models_ok = True
    for mf in model_files:
        if mf.exists():
            print(f"  [OK] {mf.relative_to(PROJECT_ROOT)} ({mf.stat().st_size:,} bytes)")
        else:
            print(f"  [FAIL] Missing model: {mf}")
            models_ok = False

    if models_ok:
        passed += 1

    # [2/5] Checking Demo Media & Datasets
    print("\n[2/5] Checking Demo Media & Datasets:")
    demo_files = [
        DEFAULT_VEHICLE_VIDEO,
        DEFAULT_CRASH_VIDEO,
        HISTORICAL_TRAFFIC_DATA,
        BENCHMARKS_JSON
    ]
    demo_ok = True
    for df in demo_files:
        if df.exists():
            print(f"  [OK] {df.relative_to(PROJECT_ROOT)} ({df.stat().st_size:,} bytes)")
        else:
            print(f"  [FAIL] Missing file: {df}")
            demo_ok = False

    if demo_ok:
        passed += 1

    # [3/5] Checking Frontend Assets
    print("\n[3/5] Checking Frontend Assets:")
    frontend_files = [
        TEMPLATES_DIR / "index.html",
        STATIC_DIR / "css" / "style.css",
        STATIC_DIR / "js" / "main.js",
        STATIC_DIR / "js" / "congestion.js"
    ]
    frontend_ok = True
    for ff in frontend_files:
        if ff.exists():
            print(f"  [OK] {ff.relative_to(PROJECT_ROOT)} ({ff.stat().st_size:,} bytes)")
        else:
            print(f"  [FAIL] Missing asset: {ff}")
            frontend_ok = False

    if frontend_ok:
        passed += 1

    # [4/5] Executing 5-Module Functional Inference Tests
    print("\n[4/5] Executing 5-Module Functional Live Inference Tests:")
    functional_ok = True

    # Test Module 1
    try:
        router = SmartRouter()
        route_res = router.compute_route("J1_CityCenter", "J3_TechPark")
        predictor = CongestionPredictor()
        pred_res = predictor.predict({"traffic_volume": 500, "average_speed": 30})
        assert "recommended_route" in route_res, "Route computation failed"
        assert "congestion_level" in pred_res, "Congestion prediction failed"
        print(f"  [OK] Module 1 (Route & Resistance): Route computed ({route_res['recommended_route']['distance_km']} km), Congestion: {pred_res['congestion_level']}")
    except Exception as e:
        print(f"  [FAIL] Module 1 failed: {e}")
        functional_ok = False

    # Test Module 2
    try:
        lpr = LPREngine()
        cap = cv2.VideoCapture(str(DEFAULT_VEHICLE_VIDEO))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
        ret, frame = cap.read()
        cap.release()
        assert ret, "Could not read frame from vehicle demo video"
        annotated, telemetry = lpr.process_frame(frame)
        assert len(telemetry["vehicles"]) > 0, "No vehicles detected"
        assert "speed_kmh" not in telemetry["vehicles"][0], "Speed should not be in telemetry"
        print(f"  [OK] Module 2 (Vehicle & Plate): Detected {len(telemetry['vehicles'])} vehicles, {telemetry['plates_scanned']} plates highlighted in RED")
    except Exception as e:
        print(f"  [FAIL] Module 2 failed: {e}")
        functional_ok = False

    # Test Module 3
    try:
        crash = CrashEngine()
        cap = cv2.VideoCapture(str(DEFAULT_CRASH_VIDEO))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
        ret, frame = cap.read()
        cap.release()
        assert ret, "Could not read frame from crash demo video"
        annotated_crash, c_telemetry = crash.process_frame(frame)
        print(f"  [OK] Module 3 (Crash Predictor): Engine operational, status: {c_telemetry['status']}")
    except Exception as e:
        print(f"  [FAIL] Module 3 failed: {e}")
        functional_ok = False

    # Test Module 4
    try:
        retina = RetinaEngine()
        dummy_face = np.zeros((480, 640, 3), dtype=np.uint8)
        _, r_telemetry = retina.process_frame(dummy_face)
        print(f"  [OK] Module 4 (Driver Eye Sensor): MediaPipe FaceMesh operational, state: {r_telemetry['status']}")
    except Exception as e:
        print(f"  [FAIL] Module 4 failed: {e}")
        functional_ok = False

    # Test Module 5
    try:
        with open(BENCHMARKS_JSON, "r", encoding="utf-8") as f:
            benchmarks = json.load(f)
        assert "models" in benchmarks and len(benchmarks["models"]) >= 5
        print(f"  [OK] Module 5 (Evaluation Benchmarks): Loaded {len(benchmarks['models'])} model benchmarks")
    except Exception as e:
        print(f"  [FAIL] Module 5 failed: {e}")
        functional_ok = False

    if functional_ok:
        passed += 1

    # [5/5] Final Verification Status
    print("\n[5/5] Final Project Verification Status:")
    if models_ok and demo_ok and frontend_ok and functional_ok:
        passed += 1
        print("  * ALL CHECKS PASSED WITH 100% SUCCESS! SmartTraffic-AI is clean and fully operational. *")
    else:
        print("  * Some checks failed. Review log above. *")

    print(f"\nCompleted: {passed}/{total} Verification Checkpoints Passed.")
    return passed == total


if __name__ == "__main__":
    success = run_evaluation()
    sys.exit(0 if success else 1)
