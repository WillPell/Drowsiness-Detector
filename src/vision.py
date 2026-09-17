import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")

LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

HEAD_POSE_IDX = [1, 152, 33, 263, 61, 291]

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, 330.0, 65.0),
    (-225.0, -170.0, 135.0),
    (225.0, -170.0, 135.0),
    (-150.0, 150.0, 125.0),
    (150.0, 150.0, 125.0),
], dtype=np.float64)


def ensure_model() -> str:
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Downloading face landmarker model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")
    return MODEL_PATH


def _wrap_angle(angle: float) -> float:
    if angle > 90.0:
        return angle - 180.0
    if angle < -90.0:
        return angle + 180.0
    return angle


class VisionProcessor:
    def __init__(self, min_detection_confidence=0.6, min_tracking_confidence=0.6):
        model_path = ensure_model()

        base_options = mp.tasks.BaseOptions
        face_landmarker = mp.tasks.vision.FaceLandmarker
        landmarker_options = mp.tasks.vision.FaceLandmarkerOptions
        running_mode = mp.tasks.vision.RunningMode

        options = landmarker_options(
            base_options=base_options(model_asset_path=model_path),
            running_mode=running_mode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = face_landmarker.create_from_options(options)
        self._start_time = time.monotonic()
        self._last_timestamp_ms = -1

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((time.monotonic() - self._start_time) * 1000)
        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            return None

        return [(lm.x * w, lm.y * h) for lm in result.face_landmarks[0]]

    @staticmethod
    def compute_ear(landmarks, eye_indices) -> float:
        pts = np.array([landmarks[i] for i in eye_indices], dtype=np.float64)
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        h = np.linalg.norm(pts[0] - pts[3])
        if h < 1e-6:
            return 0.0
        return (v1 + v2) / (2.0 * h)

    def get_ear(self, landmarks):
        left = self.compute_ear(landmarks, LEFT_EYE_IDX)
        right = self.compute_ear(landmarks, RIGHT_EYE_IDX)
        return left, right, (left + right) / 2.0

    @staticmethod
    def estimate_head_pose(landmarks, frame_shape):
        h, w = frame_shape[:2]
        image_points = np.array([landmarks[i] for i in HEAD_POSE_IDX], dtype=np.float64)

        focal_length = float(w)
        camera_matrix = np.array([
            [focal_length, 0, w / 2.0],
            [0, focal_length, h / 2.0],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rvec, _ = cv2.solvePnP(
            MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return 0.0, 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rvec)
        angles = cv2.RQDecomp3x3(rmat)[0]
        return tuple(_wrap_angle(a) for a in angles[:3])

    @staticmethod
    def draw_eye_landmarks(frame, landmarks, color=(0, 255, 128)):
        for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
            cv2.circle(frame, (int(landmarks[idx][0]), int(landmarks[idx][1])), 2, color, -1)

    def release(self):
        self._landmarker.close()
