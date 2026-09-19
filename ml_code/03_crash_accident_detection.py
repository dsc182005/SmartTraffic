"""
================================================================================
SMARTTRAFFIC-AI : ML MODEL 3 - ACCIDENT & CRASH DETECTOR (YOLOv8)
================================================================================
Components:
  1. YOLOv8 Deep Learning Object Detection for Road Traffic Collisions
  2. Multi-Class Confidence Thresholding & IoU Suppression
  3. Real-Time Emergency Audio-Visual Alarm Triggers
  4. Video Stream Processing & Incident Telemetry Logging

Usage:
  python 03_crash_accident_detection.py
================================================================================
"""

import os
import sys
import time
from pathlib import Path
import cv2
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
CRASH_MODEL_PATH = ROOT_DIR / "ml_models" / "crash_prediction" / "crash_accident_best.pt"
DEMO_VIDEO_PATH = ROOT_DIR / "demo_data" / "crash_demo.mp4"


class CrashAccidentDetector:
    def __init__(self, model_path=CRASH_MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Crash model weights not found at: {self.model_path}")

        print(f"[INIT] Loading YOLOv8 Crash Detector: {self.model_path}")
        self.model = YOLO(str(self.model_path))

    def detect_frame(self, frame, conf_thresh=0.35, iou_thresh=0.4):
        """
        Runs collision detection inference on a single video frame.
        Returns annotated frame and detection telemetry.
        """
        h, w = frame.shape[:2]
        annotated = frame.copy()

        results = self.model.predict(frame, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]
        boxes = results.boxes

        is_accident = False
        max_conf = 0.0
        accidents_detected = []

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            name = self.model.names.get(cls_id, f"Class_{cls_id}")
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Identify accident class (class 0 or name containing accident/crash)
            if "accident" in name.lower() or "crash" in name.lower() or cls_id == 0:
                is_accident = True
                max_conf = max(max_conf, conf)
                accidents_detected.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": round(conf * 100, 1)
                })

                # Draw high-visibility emergency bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)

                label = f"ACCIDENT DETECTED ({conf*100:.1f}%)"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 10, y1), (0, 0, 255), -1)
                cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 50), 2)
                cv2.putText(annotated, f"{name.upper()} {conf*100:.0f}%", (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 50), 1)

        # Header Status Banner
        if is_accident:
            cv2.rectangle(annotated, (0, 0), (w, 42), (0, 0, 220), -1)
            cv2.putText(annotated, f"EMERGENCY WARNING: ROAD ACCIDENT DETECTED! (Conf: {max_conf*100:.1f}%)",
                        (max(15, w // 2 - 290), 28), cv2.FONT_HERSHEY_DUPLEX, 0.58, (255, 255, 255), 2)
        else:
            cv2.rectangle(annotated, (0, 0), (w, 36), (20, 20, 20), -1)
            cv2.putText(annotated, "CRASH PREDICTOR: NORMAL TRAFFIC FLOW - NO ACCIDENT",
                        (max(15, w // 2 - 240), 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 150), 1)

        telemetry = {
            "status": "ACCIDENT DETECTED" if is_accident else "NORMAL TRAFFIC",
            "is_accident": is_accident,
            "max_confidence": round(max_conf * 100, 1),
            "accidents_count": len(accidents_detected),
            "accidents": accidents_detected
        }

        return annotated, telemetry


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print(" [TEST INFERENCE] Testing YOLOv8 Crash Detector on Video")
    print("=" * 65)

    detector = CrashAccidentDetector()

    if DEMO_VIDEO_PATH.exists():
        cap = cv2.VideoCapture(str(DEMO_VIDEO_PATH))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
        ret, frame = cap.read()
        cap.release()

        if ret:
            annotated, tele = detector.detect_frame(frame)
            print(f"\nCrash Detection Results for Frame 30:")
            print(f"  Traffic Status: {tele['status']}")
            print(f"  Is Accident:    {tele['is_accident']}")
            print(f"  Confidence:     {tele['max_confidence']}%")
            print(f"  Collision Hits: {tele['accidents_count']}")

            out_path = ROOT_DIR / "evaluation" / "crash_demo_output.jpg"
            cv2.imwrite(str(out_path), annotated)
            print(f"\n[SUCCESS] Annotated accident frame saved to: {out_path}")
    else:
        print(f"[WARN] Demo video not found at: {DEMO_VIDEO_PATH}")
