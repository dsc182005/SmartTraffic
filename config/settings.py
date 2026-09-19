"""
Central Configuration & Dynamic Path Resolver for SmartTraffic-AI.
Resolves all project directories and assets relative to the project root.
"""

from pathlib import Path
import os

# Project Root: C:\Users\331lo\OneDrive\Desktop\SmartTraffic-AI
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Backend Directories
BACKEND_DIR = PROJECT_ROOT / "backend"
SERVICES_DIR = BACKEND_DIR / "services"
UTILS_DIR = BACKEND_DIR / "utils"

# Frontend Directories
FRONTEND_DIR = PROJECT_ROOT / "frontend"
TEMPLATES_DIR = FRONTEND_DIR / "templates"
STATIC_DIR = FRONTEND_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ML Models Directories
MODELS_DIR = PROJECT_ROOT / "ml_models"
ROUTE_MODELS_DIR = MODELS_DIR / "route_resistance"
VEHICLE_MODEL_PATH = MODELS_DIR / "vehicle_detection" / "yolov8n.pt"
PLATE_MODEL_PATH = MODELS_DIR / "license_plate_detection" / "best.pt"
CRASH_MODEL_PATH = MODELS_DIR / "crash_prediction" / "crash_accident_best.pt"
RETINA_MODELS_DIR = MODELS_DIR / "retina_ear"

# Demo Media & Data
DEMO_DIR = PROJECT_ROOT / "demo_data"
DEFAULT_VEHICLE_VIDEO = DEMO_DIR / "vehicle_anpr_demo.webm"
DEFAULT_CRASH_VIDEO = DEMO_DIR / "crash_demo.mp4"
HISTORICAL_TRAFFIC_DATA = DEMO_DIR / "traffic_historical_dataset.csv"

# Evaluation Benchmarks
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
BENCHMARKS_JSON = EVALUATION_DIR / "benchmarks.json"

# Server Settings
HOST = "0.0.0.0"
PORT = 5000
DEBUG = False
