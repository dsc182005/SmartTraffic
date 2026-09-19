"""
SmartTraffic-AI: Module 2 Dedicated License Plate Detector (YOLOv8) Training Pipeline
Trains a dedicated YOLOv8 Nano object detector specifically on vehicle license plates.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "ml_models" / "license_plate_detection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train License Plate YOLO Model")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml dataset config")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image resolution")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="", help="Device: '0' for CUDA GPU, 'cpu' for CPU")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Pretrained base YOLO weights")
    return parser.parse_args()


def train_plate_model():
    args = parse_args()

    # Determine hardware device
    if not args.device:
        device = "0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 65)
    print("      YOLOv8 Dedicated License Plate Detector Training        ")
    print("=" * 65)
    dev_name = torch.cuda.get_device_name(0) if (torch.cuda.is_available() and device != "cpu") else "CPU"
    print(f"Device:   {device} ({dev_name})")
    print(f"Epochs:   {args.epochs} | Batch: {args.batch} | ImgSz: {args.imgsz}")
    print(f"Base:     {args.weights} | Patience: {args.patience}")

    # Load base model
    model = YOLO(args.weights)

    # Train dedicated license plate detector
    train_results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=4,
        patience=args.patience,
        save=True,
        project="runs/detect",
        name="license_plate_detector",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",     # Robust adaptive optimizer
        lr0=0.002,             # Initial learning rate
        lrf=0.01,              # Final learning rate fraction
        mosaic=1.0,            # Mosaic data augmentation
        fliplr=0.5,            # Horizontal flip
        hsv_h=0.015,           # Hue augmentation
        hsv_s=0.7,             # Saturation augmentation
        hsv_v=0.4,             # Value/brightness augmentation
        scale=0.5,             # Scale jitter
        verbose=True
    )

    # Export best model checkpoint
    best_weights = Path("runs/detect/license_plate_detector/weights/best.pt")
    if best_weights.exists():
        target_path = OUTPUT_DIR / "best.pt"
        shutil.copy(best_weights, target_path)
        print(f"\n[SUCCESS] Exported best trained model to: {target_path}")

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    train_plate_model()
