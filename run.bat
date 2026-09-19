@echo off
title SmartTraffic-AI Management System
cd /d "%~dp0"

echo =====================================================================
echo   Starting SmartTraffic-AI Intelligent System
echo   5 Dedicated Modules Active:
echo     1. Indian Route Planner & Resistance Index
echo     2. Vehicle & Dedicated License Plate Detector (Red Box Highlight)
echo     3. Crash Predictor (Accident Detection AI)
echo     4. Driver Eye Sensor (MediaPipe Live Fatigue Tracking)
echo     5. Model Evaluation Metrics
echo =====================================================================
echo   Dashboard will open at: http://localhost:5000
echo =====================================================================

start "" http://localhost:5000
python backend/app.py
pause
