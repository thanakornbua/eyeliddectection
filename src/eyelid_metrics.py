from dataclasses import dataclass
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class EyeMeasurement:
    eyelid_distance: float
    eye_aspect_ratio: float
    is_closed: bool


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    v1_u = v1 / (np.linalg.norm(v1) + 1e-6)
    v2_u = v2 / (np.linalg.norm(v2) + 1e-6)
    dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


@dataclass
class FaceMeasurement:
    eye: EyeMeasurement
    pitch_deg: float
    landmarks_px: Optional[np.ndarray]


class MediapipeEyelidDetector:
    def __init__(
        self,
        detection_confidence: float = 0.6,
        tracking_confidence: float = 0.6,
        eyelid_threshold: float = 0.21,
            pitch_alpha: float = 0.2,
    ) -> None:
        self.eyelid_threshold = eyelid_threshold
        self.pitch_alpha = pitch_alpha
        self._pitch_estimate: Optional[float] = None
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def close(self) -> None:
        self.face_mesh.close()

    def process_frame(self, frame: np.ndarray, include_landmarks: bool = False) -> Optional[FaceMeasurement]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]
        coords = np.array(
            [(lm.x, lm.y, lm.z) for lm in landmarks.landmark],
            dtype=np.float32,
        )
        eye_indices = {
            "left": [33, 160, 158, 133, 153, 144],
            "right": [263, 387, 385, 362, 380, 373],
        }

        def calc_ratio(indices: Iterable[int]) -> Tuple[float, float]:
            pts = coords[indices][:, :2]
            vertical = np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])
            horizontal = np.linalg.norm(pts[0] - pts[3])
            ratio = vertical / (2.0 * horizontal + 1e-6)
            return vertical, ratio

        lid_distances, ratios = [], []
        for eye in eye_indices.values():
            vertical, ratio = calc_ratio(eye)
            lid_distances.append(vertical)
            ratios.append(ratio)

        eyelid_distance = float(np.mean(lid_distances))
        ear = float(np.mean(ratios))
        is_closed = ear < self.eyelid_threshold
        eye_measurement = EyeMeasurement(
            eyelid_distance=eyelid_distance,
            eye_aspect_ratio=ear,
            is_closed=is_closed,
        )

        forehead = coords[10]
        chin = coords[152]
        pitch_value = _angle_between(chin - coords[1], np.array([0.0, 1.0, 0.0]))

        landmarks_px = None
        if include_landmarks:
            h, w = frame.shape[:2]
            landmarks_px = np.column_stack((coords[:, 0] * w, coords[:, 1] * h))

        if self._pitch_estimate is None:
            self._pitch_estimate = pitch_value
        else:
            self._pitch_estimate = (
                    self.pitch_alpha * pitch_value + (1.0 - self.pitch_alpha) * self._pitch_estimate
            )

        return FaceMeasurement(
            eye=eye_measurement,
            pitch_deg=float(self._pitch_estimate),
            landmarks_px=landmarks_px,
        )
