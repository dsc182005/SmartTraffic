"""
Module 3: Crash Predictor & Accident Detection Service.
Runs YOLOv8 accident detection on surveillance collision footage.
"""

import os
import time
import cv2
from ultralytics import YOLO

from config.settings import CRASH_MODEL_PATH, DEFAULT_CRASH_VIDEO


class CrashEngine:
    def __init__(self, model_path=None, default_video=None):
        self.model_path = str(model_path or CRASH_MODEL_PATH)
        self.default_video = str(default_video or DEFAULT_CRASH_VIDEO)
        self.current_video = self.default_video
        self.model = None

        if os.path.exists(self.model_path):
            try:
                self.model = YOLO(self.model_path)
                print(f"[CrashEngine] Loaded crash model from {self.model_path}")
            except Exception as e:
                print(f"[CrashEngine] Error loading crash model: {e}")
        else:
            print(f"[CrashEngine] Warning: Crash model path {self.model_path} not found.")

        self.last_telemetry = {
            "status": "NORMAL TRAFFIC",
            "is_accident": False,
            "confidence": 0.0,
            "detected_objects": 0,
            "time_str": "--"
        }

    def set_video(self, video_path):
        if os.path.exists(video_path):
            self.current_video = video_path
            return True
        return False

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        annotated = frame.copy()

        is_accident = False
        max_conf = 0.0
        det_count = 0

        if self.model is not None:
            results = self.model.predict(frame, conf=0.35, iou=0.4, verbose=False)[0]
            boxes = results.boxes
            det_count = len(boxes)

            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                name = self.model.names.get(cls_id, f"Class {cls_id}")
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                if "accident" in name.lower() or "crash" in name.lower() or cls_id == 0:
                    is_accident = True
                    max_conf = max(max_conf, conf)

                    # Red bounding box for accident
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)

                    lbl = f"ACCIDENT {conf*100:.1f}%"
                    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), (0, 0, 255), -1)
                    cv2.putText(annotated, lbl, (x1 + 4, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    # Non-accident objects
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)
                    lbl = f"{name} {conf*100:.0f}%"
                    cv2.putText(annotated, lbl, (x1, max(15, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

        # Header status banner
        if is_accident:
            cv2.rectangle(annotated, (0, 0), (w, 40), (0, 0, 200), -1)
            cv2.putText(annotated, f"CRITICAL ALERT: ACCIDENT DETECTED! ({max_conf*100:.1f}% CONFIDENCE)",
                        (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(annotated, (0, 0), (w, 36), (15, 23, 42), -1)
            cv2.putText(annotated, "SURVEILLANCE ACCIDENT AI | TRAFFIC NORMAL | NO HAZARDS DETECTED",
                        (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 120), 1, cv2.LINE_AA)

        self.last_telemetry = {
            "status": "ACCIDENT DETECTED" if is_accident else "NORMAL TRAFFIC",
            "is_accident": is_accident,
            "confidence": round(max_conf * 100, 1),
            "detected_objects": det_count,
            "time_str": time.strftime("%H:%M:%S")
        }

        return annotated, self.last_telemetry
