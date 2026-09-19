"""
================================================================================
SMARTTRAFFIC-AI : ML MODEL 4 - DRIVER EYE FATIGUE & EAR SENSOR
================================================================================
Components:
  1. Google MediaPipe 468-point 3D FaceMesh Landmark Topology
  2. Soukupová & Čech Eye Aspect Ratio (EAR) Mathematical Formulation:
       EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
  3. Real-time Eye Aperture Smoothing & Blink Counter
  4. Temporal Closure Monitoring (Drowsiness Alert on >12 consecutive closed frames)

Usage:
  # Test with camera or synthetic frame:
  python 04_driver_eye_fatigue_sensor.py

  # Run live webcam monitoring:
  python 04_driver_eye_fatigue_sensor.py --webcam
================================================================================
"""

import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import distance as dist
import mediapipe as mp


class DriverFatigueSensor:
    def __init__(self, closure_thresh=0.23, consec_frames=12):
        self.CLOSURE_THRESH = closure_thresh
        self.CONSEC_FRAMES = consec_frames

        # Standard 6-point Landmark indices in MediaPipe 468-point FaceMesh
        # Left eye: [outer corner, top-outer, top-inner, inner corner, bottom-inner, bottom-outer]
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        # Right eye: [inner corner, top-inner, top-outer, outer corner, bottom-outer, bottom-inner]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

        # Full eye contour landmarks for visual overlay
        self.LEFT_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

        # Initialize MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Dynamic state
        self.consec_closed_frames = 0
        self.total_blinks = 0
        self.eye_closed_prev = False
        self.openness_smoothed = 0.32

    def calculate_ear(self, landmarks, indices, w, h):
        """
        Computes 6-point Euclidean Eye Aspect Ratio (EAR).
        """
        pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in indices]
        # Vertical distances
        A = dist.euclidean(pts[1], pts[5])
        B = dist.euclidean(pts[2], pts[4])
        # Horizontal distance
        C = dist.euclidean(pts[0], pts[3])
        if C == 0:
            return 0.30
        return (A + B) / (2.0 * C)

    def process_frame(self, frame):
        """
        Analyzes a single frame for driver eye state, blinks, and fatigue.
        """
        h, w = frame.shape[:2]
        annotated = frame.copy()

        # MediaPipe expects RGB format
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        face_detected = False
        raw_openness = 0.32

        if results.multi_face_landmarks:
            face_detected = True
            mesh = results.multi_face_landmarks[0].landmark

            # Calculate EAR for both eyes
            left_ear = self.calculate_ear(mesh, self.LEFT_EYE_INDICES, w, h)
            right_ear = self.calculate_ear(mesh, self.RIGHT_EYE_INDICES, w, h)
            raw_openness = (left_ear + right_ear) / 2.0

            # Draw eye contour lines
            left_pts = np.array([[int(mesh[i].x * w), int(mesh[i].y * h)] for i in self.LEFT_CONTOUR], np.int32)
            right_pts = np.array([[int(mesh[i].x * w), int(mesh[i].y * h)] for i in self.RIGHT_CONTOUR], np.int32)

            contour_col = (0, 0, 255) if raw_openness < self.CLOSURE_THRESH else (0, 255, 120)
            cv2.polylines(annotated, [left_pts], True, contour_col, 1, cv2.LINE_AA)
            cv2.polylines(annotated, [right_pts], True, contour_col, 1, cv2.LINE_AA)

        # Smooth aperture
        if face_detected:
            self.openness_smoothed = round(0.4 * raw_openness + 0.6 * self.openness_smoothed, 3)
        else:
            self.openness_smoothed = 0.30

        # State evaluation
        is_closed = face_detected and (self.openness_smoothed < self.CLOSURE_THRESH)

        # Blink Detection Logic
        if is_closed and not self.eye_closed_prev:
            self.total_blinks += 1
        self.eye_closed_prev = is_closed

        # Drowsiness Consecutive Frame Logic
        if is_closed:
            self.consec_closed_frames += 1
        else:
            self.consec_closed_frames = 0

        is_drowsy = self.consec_closed_frames >= self.CONSEC_FRAMES

        if not face_detected:
            status = "NO FACE DETECTED"
            eye_state = "UNKNOWN"
        elif is_drowsy:
            status = "DROWSINESS DETECTED - WAKE UP!"
            eye_state = "PROLONGED CLOSURE"
        elif is_closed:
            status = "EYES CLOSED"
            eye_state = "CLOSED"
        else:
            status = "EYES OPEN"
            eye_state = "OPEN"

        # Annotation Overlay
        if is_drowsy:
            cv2.rectangle(annotated, (0, 0), (w, 42), (0, 0, 220), -1)
            cv2.putText(annotated, f"WAKE UP! DROWSINESS ALERT! (Closed: {self.consec_closed_frames} frames)",
                        (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(annotated, (0, 0), (w, 36), (15, 23, 42), -1)
            cv2.putText(annotated, f"DRIVER EYE SENSOR | {status} | EAR: {self.openness_smoothed:.3f} | BLINKS: {self.total_blinks}",
                        (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 120), 1, cv2.LINE_AA)

        telemetry = {
            "status": status,
            "eye_state": eye_state,
            "ear_openness": self.openness_smoothed,
            "is_drowsy": is_drowsy,
            "is_closed": is_closed,
            "total_blinks": self.total_blinks,
            "consec_closed_frames": self.consec_closed_frames,
            "face_detected": face_detected
        }

        return annotated, telemetry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver Fatigue & Eye Sensor")
    parser.add_argument("--webcam", action="store_true", help="Launch live webcam monitoring loop")
    args = parser.parse_args()

    sensor = DriverFatigueSensor()

    if args.webcam:
        print("[INFO] Launching live webcam test (Press 'q' to exit)...")
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            ann, tele = sensor.process_frame(frame)
            cv2.imshow("SmartTraffic-AI Driver Eye Sensor", ann)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("\n" + "=" * 65)
        print(" [TEST INFERENCE] Testing Driver Fatigue Sensor Pipeline")
        print("=" * 65)
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ann, tele = sensor.process_frame(test_frame)
        print(f"  Face Detected:        {tele['face_detected']}")
        print(f"  Eye State:            {tele['eye_state']}")
        print(f"  EAR Openness:         {tele['ear_openness']}")
        print(f"  Drowsiness Alert:     {tele['is_drowsy']}")
        print(f"  Total Blinks:         {tele['total_blinks']}")
        print("\n[SUCCESS] Driver fatigue pipeline tested successfully.")
        print("Tip: Run with --webcam to test live using your laptop camera.")
