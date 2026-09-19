"""
Module 4: Driver Retina & EAR Sensor Service.
Real-time driver fatigue monitoring using MediaPipe 468 FaceMesh landmarks.
Accurately computes Eye Aspect Ratio (EAR), tracks blink count,
and triggers audio-visual alarms upon prolonged eye closure.
"""

import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist


class RetinaEngine:
    def __init__(self, closure_thresh=0.23, consec_frames=12):
        self.CLOSURE_THRESH = closure_thresh
        self.CONSEC_FRAMES = consec_frames

        # Eye landmark indices in MediaPipe FaceMesh
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

        self.LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

        # Initialize MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # State tracking
        self.consec_closed_frames = 0
        self.total_blinks = 0
        self.eye_closed_prev = False
        self.openness_smoothed = 0.32

        self.last_telemetry = {
            "status": "EYES OPEN",
            "eye_state": "OPEN",
            "is_drowsy": False,
            "is_closed": False,
            "blink_count": 0,
            "consec_closed": 0,
            "face_detected": False,
            "alert": "NORMAL"
        }

    def reset_state(self):
        self.consec_closed_frames = 0
        self.total_blinks = 0
        self.eye_closed_prev = False
        self.openness_smoothed = 0.32

    def _eye_aspect_ratio(self, eye_pts):
        # Vertical distances
        a = dist.euclidean(eye_pts[1], eye_pts[5])
        b = dist.euclidean(eye_pts[2], eye_pts[4])
        # Horizontal distance
        c = dist.euclidean(eye_pts[0], eye_pts[3])
        return (a + b) / (2.0 * max(1e-6, c))

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        annotated = frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        face_detected = False
        is_closed = False
        is_drowsy = False
        ear = 0.0

        if results.multi_face_landmarks:
            face_detected = True
            landmarks = results.multi_face_landmarks[0].landmark

            # Extract left and right eye coordinates
            left_pts = np.array([(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in self.LEFT_EYE_INDICES])
            right_pts = np.array([(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in self.RIGHT_EYE_INDICES])

            left_ear = self._eye_aspect_ratio(left_pts)
            right_ear = self._eye_aspect_ratio(right_pts)
            ear = (left_ear + right_ear) / 2.0
            self.openness_smoothed = 0.4 * ear + 0.6 * self.openness_smoothed

            is_closed = self.openness_smoothed < self.CLOSURE_THRESH

            # Blink counting
            if is_closed and not self.eye_closed_prev:
                self.total_blinks += 1
            self.eye_closed_prev = is_closed

            # Prolonged closure drowsiness tracking
            if is_closed:
                self.consec_closed_frames += 1
            else:
                self.consec_closed_frames = max(0, self.consec_closed_frames - 2)

            is_drowsy = self.consec_closed_frames >= self.CONSEC_FRAMES

            # Draw eye contours
            draw_col = (0, 0, 255) if is_drowsy else ((0, 165, 255) if is_closed else (0, 255, 120))
            left_hull = cv2.convexHull(np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in self.LEFT_EYE_CONTOUR]))
            right_hull = cv2.convexHull(np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in self.RIGHT_EYE_CONTOUR]))

            cv2.polylines(annotated, [left_hull], True, draw_col, 2)
            cv2.polylines(annotated, [right_hull], True, draw_col, 2)

        # Telemetry & Status
        if not face_detected:
            status = "NO FACE DETECTED"
            eye_state = "UNKNOWN"
            alert = "WARNING"
        elif is_drowsy:
            status = "DROWSINESS DETECTED"
            eye_state = "CLOSED (ALERT)"
            alert = "DROWSY_ALERT"
        elif is_closed:
            status = "EYES CLOSED"
            eye_state = "CLOSED"
            alert = "NORMAL"
        else:
            status = "EYES OPEN"
            eye_state = "OPEN"
            alert = "NORMAL"

        self.last_telemetry = {
            "status": status,
            "eye_state": eye_state,
            "is_drowsy": is_drowsy,
            "is_closed": is_closed,
            "blink_count": self.total_blinks,
            "consec_closed": self.consec_closed_frames,
            "face_detected": face_detected,
            "alert": alert
        }

        # Overlay banner
        if is_drowsy:
            cv2.rectangle(annotated, (0, 0), (w, 42), (0, 0, 220), -1)
            cv2.putText(annotated, f"WAKE UP! DROWSINESS ALERT! (Closed: {self.consec_closed_frames} frames)",
                        (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(annotated, (0, 0), (w, 36), (15, 23, 42), -1)
            cv2.putText(annotated, f"DRIVER EYE SENSOR | {status} | BLINKS: {self.total_blinks}",
                        (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 120), 1, cv2.LINE_AA)

        return annotated, self.last_telemetry
