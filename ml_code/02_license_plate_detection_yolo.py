"""
================================================================================
SMARTTRAFFIC-AI : ML MODEL 2 - DEDICATED LICENSE PLATE DETECTOR (YOLOv8)
================================================================================
Components:
  1. YOLOv8 Nano Dedicated Model Training (AdamW, Mosaic, HSV augmentations)
  2. Dual-Stage Vision Pipeline:
       Stage A: Vehicle Detection & Tracking (yolov8n.pt)
       Stage B: Dedicated License Plate Detection (fine-tuned best.pt)
  3. Spatial Bounding Box Association (Pairs physical plate to vehicle)
  4. Red Box Highlighting & Clean Base64 Evidence Cropping

Usage:
  # Run inference on demo video:
  python 02_license_plate_detection_yolo.py

  # Train the YOLOv8 detector (if dataset available):
  python 02_license_plate_detection_yolo.py --train --data path/to/data.yaml
================================================================================
"""

import os
import sys
import base64
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
VEHICLE_MODEL_PATH = ROOT_DIR / "ml_models" / "vehicle_detection" / "yolov8n.pt"
PLATE_MODEL_PATH = ROOT_DIR / "ml_models" / "license_plate_detection" / "best.pt"
DEMO_VIDEO_PATH = ROOT_DIR / "demo_data" / "vehicle_anpr_demo.webm"


# ==============================================================================
# 1. TRAINING PIPELINE (YOLOv8)
# ==============================================================================
def train_license_plate_detector(data_yaml="data.yaml", epochs=25, imgsz=640, batch=16, weights="yolov8n.pt"):
    """
    Trains dedicated YOLOv8 Nano detector on license plate bounding boxes.
    """
    device = "0" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 65)
    print(" [TRAINING] Dedicated YOLOv8 License Plate Detection Model")
    print("=" * 65)
    print(f"Device:    {device}")
    print(f"Epochs:    {epochs} | Batch: {batch} | ImgSz: {imgsz}")
    print(f"Dataset:   {data_yaml}")

    model = YOLO(weights)
    train_results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=4,
        patience=8,
        save=True,
        project="runs/detect",
        name="license_plate_run",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",     # High-stability optimizer
        lr0=0.002,             # Initial learning rate
        lrf=0.01,              # Final learning rate fraction
        mosaic=1.0,            # Mosaic data augmentation for multi-scale plates
        fliplr=0.5,            # Horizontal flip
        hsv_h=0.015,           # Color & lighting invariant augmentations
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.5,             # Multi-distance plate scale jitter
        verbose=True
    )
    print("[SUCCESS] Training completed. Best weights saved to runs/detect/license_plate_run/weights/best.pt")


# ==============================================================================
# 2. DUAL-STAGE INFERENCE PIPELINE
# ==============================================================================
class DualStagePlateDetector:
    def __init__(self, vehicle_weights=VEHICLE_MODEL_PATH, plate_weights=PLATE_MODEL_PATH):
        print("\n[INIT] Loading Dual-Stage Vision Models:")
        print(f"  -> Vehicle Detector: {vehicle_weights}")
        self.vehicle_model = YOLO(str(vehicle_weights))

        print(f"  -> License Plate Detector: {plate_weights}")
        self.plate_model = YOLO(str(plate_weights))

        # COCO Vehicle class IDs: bicycle=1, car=2, motorcycle=3, bus=5, truck=7
        self.vehicle_classes = [1, 2, 3, 5, 7]
        self.class_names = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def process_frame(self, frame, conf_thresh=0.35):
        """
        Executes dual-stage inference:
          1. Detects & tracks vehicles
          2. Detects license plates
          3. Associates plate to closest vehicle
          4. Draws single bright red box directly on plate
          5. Extracts crisp plate crop
        """
        h, w = frame.shape[:2]
        annotated = frame.copy()

        # Step 1: Vehicle Tracking
        vehicles = []
        v_results = self.vehicle_model.track(
            frame, persist=True, conf=conf_thresh, classes=self.vehicle_classes, verbose=False
        )[0]

        if v_results.boxes and v_results.boxes.id is not None:
            for box, track_id, cls_id in zip(v_results.boxes.xyxy, v_results.boxes.id, v_results.boxes.cls):
                x1, y1, x2, y2 = map(int, box.tolist())
                vclass = self.class_names.get(int(cls_id.item()), "vehicle")
                vehicles.append({
                    "track_id": int(track_id.item()),
                    "class": vclass,
                    "box": [x1, y1, x2, y2]
                })

        # Step 2: Dedicated License Plate Detection
        matched_plates = {}
        lpr_res = self.plate_model.predict(frame, conf=conf_thresh, verbose=False)[0]

        for pbox in lpr_res.boxes:
            px1, py1, px2, py2 = map(int, pbox.xyxy[0].tolist())
            pconf = float(pbox.conf[0].item())
            pcx = (px1 + px2) // 2
            pcy = (py1 + py2) // 2

            # Find matching vehicle by geometric containment / proximity
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

        # Step 3: Draw Annotations (Single label on plate, NO duplicate vehicle badge)
        telemetry = []
        for v in vehicles:
            tid = v["track_id"]
            vclass = v["class"]
            vx1, vy1, vx2, vy2 = v["box"]

            has_plate = tid in matched_plates
            plate_box = []
            plate_crop_b64 = ""

            # Vehicle Box (Green/Cyan)
            cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (0, 220, 50), 2)
            cv2.putText(annotated, f"#{tid} {vclass.upper()}", (vx1 + 4, max(15, vy1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 50), 2)

            if has_plate:
                (px1, py1, px2, py2), pconf = matched_plates[tid]
                plate_box = [px1, py1, px2, py2]

                # BRIGHT RED BOX on physical plate (Guaranteed single label)
                cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 0, 255), 3)

                tag = "PLATE DETECTED"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated, (px1, max(0, py1 - th - 6)), (px1 + tw + 8, py1), (0, 0, 255), -1)
                cv2.putText(annotated, tag, (px1 + 4, py1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

                # Extract Crop
                crop = frame[max(0, py1):min(h, py2), max(0, px1):min(w, px2)]
                if crop.size > 0:
                    ret_enc, buf = cv2.imencode(".jpg", crop)
                    if ret_enc:
                        plate_crop_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"

            telemetry.append({
                "track_id": tid,
                "class": vclass,
                "plate_detected": has_plate,
                "plate_box": plate_box,
                "plate_crop": plate_crop_b64
            })

        return annotated, telemetry


# ==============================================================================
# 3. MAIN RUNNER
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="License Plate Detector Pipeline")
    parser.add_argument("--train", action="store_true", help="Run YOLOv8 model training")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=25, help="Epochs to train")
    args = parser.parse_args()

    if args.train:
        train_license_plate_detector(data_yaml=args.data, epochs=args.epochs)
    else:
        print("\n" + "=" * 65)
        print(" [TEST INFERENCE] Testing Dual-Stage Plate Detection on Video Frame")
        print("=" * 65)

        pipeline = DualStagePlateDetector()
        if DEMO_VIDEO_PATH.exists():
            cap = cv2.VideoCapture(str(DEMO_VIDEO_PATH))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
            ret, frame = cap.read()
            cap.release()

            if ret:
                annotated, tele = pipeline.process_frame(frame)
                print(f"\nDetection Results for Frame 60:")
                print(f"  Total Vehicles Tracked: {len(tele)}")
                for v in tele:
                    status = f"PLATE DETECTED at {v['plate_box']}" if v['plate_detected'] else "Scanning..."
                    print(f"    * Vehicle #{v['track_id']} ({v['class'].upper()}) -> {status}")

                out_path = ROOT_DIR / "evaluation" / "plate_demo_output.jpg"
                cv2.imwrite(str(out_path), annotated)
                print(f"\n[SUCCESS] Annotated test image saved to: {out_path}")
        else:
            print(f"[WARN] Demo video not found at: {DEMO_VIDEO_PATH}")
