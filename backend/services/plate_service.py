"""
Module 2: Vehicle Type & License Plate Detection Service.
Pure Vision Pipeline:
Video Frame -> Vehicle Detection (YOLOv8) -> License Plate Detection (Dedicated YOLOv8 best.pt)
-> High-Visibility Red Bounding Box Highlighting -> Base64 Physical Plate Evidence Crops.

Completely eliminates OCR text, synthetic plate strings, and speed estimation.
"""

import os
import time
import base64
import cv2
import numpy as np
from collections import Counter
from ultralytics import YOLO

from config.settings import VEHICLE_MODEL_PATH, PLATE_MODEL_PATH, DEFAULT_VEHICLE_VIDEO


class LPREngine:
    def __init__(self,
                 vehicle_model_path=None,
                 plate_model_path=None,
                 default_video=None):
        self.default_video = str(default_video or DEFAULT_VEHICLE_VIDEO)
        self.current_video = self.default_video

        # Vehicle classes from COCO dataset
        self.vehicle_classes = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        self.class_colors = {
            "car": (0, 220, 50),
            "motorcycle": (0, 230, 255),
            "bicycle": (50, 200, 255),
            "bus": (0, 165, 255),
            "truck": (255, 220, 0)
        }

        # 1. Load vehicle detector model
        v_path = str(vehicle_model_path or VEHICLE_MODEL_PATH)
        self.vehicle_model = None
        if os.path.exists(v_path):
            try:
                self.vehicle_model = YOLO(v_path)
                print(f"[LPREngine] Loaded vehicle model: {v_path}")
            except Exception as e:
                print(f"[LPREngine] Error loading vehicle model: {e}")
        else:
            print(f"[LPREngine] Warning: Vehicle model not found at {v_path}")

        # 2. Load dedicated license plate detector model
        p_path = str(plate_model_path or PLATE_MODEL_PATH)
        self.lpr_model = None
        if os.path.exists(p_path):
            try:
                self.lpr_model = YOLO(p_path)
                print(f"[LPREngine] Loaded dedicated license plate model: {p_path}")
            except Exception as e:
                print(f"[LPREngine] Error loading license plate model: {e}")
        else:
            print(f"[LPREngine] Warning: Plate model not found at {p_path}")

        # Track caches
        self.track_best_crops = {}     # track_id -> best numpy crop
        self.track_crop_b64 = {}       # track_id -> base64 data URI
        self.track_plate_conf = {}     # track_id -> highest detection confidence
        self.track_timestamps = {}     # track_id -> first seen timestamp
        self.last_telemetry = {"vehicles": [], "total": 0, "plates_scanned": 0, "counts": {}}

    def set_video(self, video_path):
        if os.path.exists(video_path):
            self.current_video = video_path
            self.track_best_crops.clear()
            self.track_crop_b64.clear()
            self.track_plate_conf.clear()
            self.track_timestamps.clear()
            return True
        return False

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        annotated = frame.copy()
        time_str = time.strftime("%H:%M:%S")

        vehicles = []
        counts = Counter()

        # Step 1: Vehicle Detection & Tracking
        if self.vehicle_model is not None:
            try:
                results = self.vehicle_model.track(frame, persist=True, conf=0.35, verbose=False)[0]
                boxes = results.boxes
            except Exception:
                try:
                    results = self.vehicle_model(frame, conf=0.35, verbose=False)[0]
                    boxes = results.boxes
                except Exception:
                    boxes = []

            for idx, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                if cls_id in self.vehicle_classes:
                    vclass = self.vehicle_classes[cls_id]
                    track_id = int(box.id[0].item()) if (hasattr(box, 'id') and box.id is not None) else (idx + 1)
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)

                    counts[vclass] += 1
                    if track_id not in self.track_timestamps:
                        self.track_timestamps[track_id] = time_str

                    vehicles.append({
                        "track_id": track_id,
                        "class": vclass,
                        "box": [x1, y1, x2, y2]
                    })

        # Step 2: Dedicated License Plate Detection
        matched_plates = {}
        if self.lpr_model is not None:
            try:
                lpr_res = self.lpr_model.predict(frame, conf=0.35, verbose=False)[0]
                for pbox in lpr_res.boxes:
                    px1, py1, px2, py2 = map(int, pbox.xyxy[0].tolist())
                    pconf = float(pbox.conf[0].item())
                    pcx = (px1 + px2) // 2
                    pcy = (py1 + py2) // 2

                    best_tid = None
                    min_dist = float('inf')
                    for v in vehicles:
                        vx1, vy1, vx2, vy2 = v["box"]
                        if vx1 - 15 <= pcx <= vx2 + 15 and vy1 - 15 <= pcy <= vy2 + 15:
                            dist = abs(pcx - (vx1 + vx2) // 2) + abs(pcy - (vy1 + vy2) // 2)
                            if dist < min_dist:
                                min_dist = dist
                                best_tid = v["track_id"]

                    if best_tid is not None:
                        if best_tid not in matched_plates or pconf > matched_plates[best_tid][1]:
                            matched_plates[best_tid] = ([px1, py1, px2, py2], pconf)
            except Exception:
                pass

        # Step 3: Draw Bounding Boxes & Extract Plate Crops
        detected_list = []
        plates_detected_count = 0

        for v in vehicles:
            track_id = v["track_id"]
            vclass = v["class"]
            vx1, vy1, vx2, vy2 = v["box"]
            has_plate = False
            plate_conf = 0.0
            plate_box = []

            if track_id in matched_plates:
                (px1, py1, px2, py2), pconf = matched_plates[track_id]
                has_plate = True
                plate_conf = pconf
                plate_box = [px1, py1, px2, py2]
                self.track_plate_conf[track_id] = max(self.track_plate_conf.get(track_id, 0.0), pconf)

                # Bright RED Bounding Box on Physical Plate
                cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 0, 255), 3)

                # Red tag badge above plate
                plate_tag = "PLATE DETECTED"
                (ptw, pth), _ = cv2.getTextSize(plate_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                tag_y1 = max(0, py1 - pth - 6)
                cv2.rectangle(annotated, (px1, tag_y1), (px1 + ptw + 8, py1), (0, 0, 255), -1)
                cv2.putText(annotated, plate_tag, (px1 + 4, py1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

                # Extract crisp plate crop
                px1_c, py1_c = max(0, px1), max(0, py1)
                px2_c, py2_c = min(w, px2), min(h, py2)
                crop = frame[py1_c:py2_c, px1_c:px2_c]

                if crop.size > 0 and crop.shape[0] >= 12 and crop.shape[1] >= 30:
                    if (track_id not in self.track_best_crops or
                            crop.size > self.track_best_crops[track_id].size):
                        self.track_best_crops[track_id] = crop.copy()
                        ret_enc, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                        if ret_enc:
                            b64 = base64.b64encode(buf).decode("utf-8")
                            self.track_crop_b64[track_id] = f"data:image/jpeg;base64,{b64}"

            ever_detected = (track_id in self.track_crop_b64) or has_plate
            if ever_detected:
                plates_detected_count += 1
                status_str = "License Plate Detected"
                best_conf = self.track_plate_conf.get(track_id, plate_conf)
            else:
                status_str = "Scanning..."
                best_conf = 0.0

            crop_b64 = self.track_crop_b64.get(track_id, "")

            # Vehicle Bounding Box
            box_col = self.class_colors.get(vclass, (0, 220, 50))
            cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), box_col, 2)

            # Top label: Vehicle ID and Class (NO SPEED)
            top_lbl = f"#{track_id} {vclass.upper()}"
            (tw, th), _ = cv2.getTextSize(top_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(annotated, (vx1, max(0, vy1 - th - 6)), (vx1 + tw + 8, vy1), box_col, -1)
            cv2.putText(annotated, top_lbl, (vx1 + 4, vy1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
            detected_list.append({
                "track_id": track_id,
                "class": vclass,
                "plate_detected": ever_detected,
                "status": status_str,
                "confidence": round(best_conf * 100, 1) if best_conf > 0 else 0,
                "plate_crop": crop_b64,
                "box": [vx1, vy1, vx2, vy2],
                "plate_box": plate_box
            })

        # Header overlay
        cv2.rectangle(annotated, (0, 0), (w, 36), (15, 23, 42), -1)
        cv2.putText(annotated,
                    f"SMART TRAFFIC AI | VEHICLE & LICENSE PLATE DETECTOR | {len(vehicles)} VEHICLES | {plates_detected_count} PLATES HIGHLIGHTED",
                    (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1)

        self.last_telemetry = {
            "vehicles": detected_list,
            "counts": dict(counts),
            "total": len(detected_list),
            "plates_scanned": plates_detected_count
        }

        return annotated, self.last_telemetry
