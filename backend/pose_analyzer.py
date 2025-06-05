#The code is according to PEP 8 coding styles standards
import logging
from typing import Dict, Any, Tuple, Optional, List

import cv2
import mediapipe as mp
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PoseAnalyzer")


class PoseAnalyzer:
    """Analyzes human pose using MediaPipe for exercise detection."""

    def __init__(self):
        """Initialize MediaPipe models and configuration parameters."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
            enable_segmentation=False,
            smooth_landmarks=False
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.alignment_thresholds = {
            'vertical_threshold': 0.1,
            'min_alignment_time': 0.5,
            'cooldown_time': 1.0
        }

        self.form_thresholds = {
            'max_shoulder_asymmetry': 0.2,
            'min_back_angle': 150,
            'max_knee_lift': 0.1
        }

        self.gesture_params = {
            'reset_finger_count': 7,         # Trigger gesture if 7 fingers raised
            'min_gesture_confidence': 0.7,   # Confidence threshold for hand detection
            'gesture_cooldown': 1.5          # Cooldown to avoid repeated detections
        }

    def analyze_pose(self, frame_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Analyze pose from an RGB frame.

        Args:
            frame_rgb: Input RGB image frame

        Returns:
            Dictionary containing pose analysis results or None
        """
        results = self.pose.process(frame_rgb)
        if not results.pose_landmarks:
            return None

        return {
            'landmarks': results.pose_landmarks.landmark,
            'pose_results': results,
            'hand_results': self.hands.process(frame_rgb)
        }

    def is_shoulder_ear_aligned(self, landmarks: list) -> bool:
        """
        Check if shoulders and ears are vertically aligned.

        Args:
            landmarks: List of pose landmarks

        Returns:
            True if aligned, False otherwise
        """
        left_shoulder = self._get_landmark(
            landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = self._get_landmark(
            landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_ear = self._get_landmark(
            landmarks, self.mp_pose.PoseLandmark.LEFT_EAR)
        right_ear = self._get_landmark(
            landmarks, self.mp_pose.PoseLandmark.RIGHT_EAR)

        if not all([left_shoulder, right_shoulder, left_ear, right_ear]):
            return False

        left_diff = abs(left_shoulder[1] - left_ear[1])
        right_diff = abs(right_shoulder[1] - right_ear[1])

        return (
            left_diff < self.alignment_thresholds['vertical_threshold']
            and right_diff < self.alignment_thresholds['vertical_threshold']
        )

    def calculate_metrics(self, landmarks: list) -> Dict[str, float]:
        """
        Calculate pose metrics from detected landmarks.

        Args:
            landmarks: List of pose landmarks

        Returns:
            Dictionary of calculated metrics
        """
        nose = self._get_landmark(
            landmarks, self.mp_pose.PoseLandmark.NOSE)
        left_shoulder = self._get_landmark(
            landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = self._get_landmark(
            landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_hip = self._get_landmark(
            landmarks, self.mp_pose.PoseLandmark.LEFT_HIP)
        left_ear = self._get_landmark(
            landmarks, self.mp_pose.PoseLandmark.LEFT_EAR)
        left_knee = self._get_landmark(
            landmarks, self.mp_pose.PoseLandmark.LEFT_KNEE)

        metrics = {}
        critical_points = [left_shoulder, left_knee, left_hip]
        if not all(critical_points):
            return metrics # Return empty metrics if any critical point is missing

        if all([nose, left_shoulder, left_hip]):
            shoulder_y = left_shoulder[1]
            hip_y = left_hip[1]
            metrics.update({
                'head_lift': hip_y - nose[1],
                'shoulder_lift': hip_y - shoulder_y,
                'torso_angle': (
                    self._calculate_angle(left_ear, left_shoulder, left_hip)
                    if left_ear else 180
                ),
                'back_angle': self._calculate_back_angle(
                    left_shoulder, left_hip),
                'shoulder_symmetry': (
                    abs(left_shoulder[1] - right_shoulder[1])
                    if right_shoulder else 0
                ),
                'alignment': self.is_shoulder_ear_aligned(landmarks)
            })

            if left_knee:
                metrics['knee_lift'] = hip_y - left_knee[1]
                metrics['knee_shoulder_distance'] = self._calculate_distance(
                    left_shoulder, left_knee)

        return metrics

    def draw_keypoints(self, frame: np.ndarray, landmarks: list) -> np.ndarray:
        """
        Draw keypoints and connections on the frame.

        Args:
            frame: Input BGR image frame
            landmarks: List of pose landmarks

        Returns:
            Annotated frame with keypoints
        """
        keypoints = [
            (self._get_landmark(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER), (0, 0, 255)),
            (self._get_landmark(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER), (0, 0, 255)),
            (self._get_landmark(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP), (0, 255, 0)),
            (self._get_landmark(landmarks, self.mp_pose.PoseLandmark.LEFT_EAR), (255, 0, 0)),
            (self._get_landmark(landmarks, self.mp_pose.PoseLandmark.LEFT_KNEE), (255, 255, 0))
        ]

        h, w = frame.shape[:2]
        for point, color in keypoints:
            if point:
                x, y = int(point[0] * w), int(point[1] * h)
                cv2.circle(frame, (x, y), 8, color, -1)

        return frame

    def detect_reset_gesture(
        self,
        hand_results: Any,
        current_time: float,
        last_gesture_time: float
    ) -> bool:
        """
        Detect reset gesture (7 fingers shown).

        Args:
            hand_results: MediaPipe hand results
            current_time: Current timestamp
            last_gesture_time: Time of last gesture

        Returns:
            True if reset gesture detected
        """
        if (
            hand_results.multi_hand_landmarks
            and current_time - last_gesture_time
            > self.gesture_params['gesture_cooldown']
        ):
            total_fingers = 0
            for hand_landmarks, handedness in zip(
                hand_results.multi_hand_landmarks,
                hand_results.multi_handedness
            ):
                confidence = handedness.classification[0].score
                if confidence >= self.gesture_params['min_gesture_confidence']:
                    total_fingers += self._count_fingers(
                        hand_landmarks,
                        handedness.classification[0].label
                    )

            return total_fingers == self.gesture_params['reset_finger_count']
        return False

    def _get_landmark(
        self, landmarks: list, landmark_type: Any
    ) -> Optional[Tuple[float, float]]:
        """
        Get landmark coordinates if visible.

        Args:
            landmarks: List of pose landmarks
            landmark_type: Landmark type

        Returns:
            Tuple of (x, y) or None
        """
        landmark = landmarks[landmark_type.value]
        if landmark.visibility > 0.5:
            return (landmark.x, landmark.y)
        return None

    def _calculate_angle(
        self, a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]
    ) -> float:
        """Calculate angle between three points in degrees."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b

        if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
            return 180.0

        cosine_angle = np.dot(ba, bc) / (
            np.linalg.norm(ba) * np.linalg.norm(bc)
        )
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def _calculate_back_angle(
        self, shoulder: Tuple[float, float], hip: Tuple[float, float]
    ) -> float:
        """Calculate back angle relative to vertical."""
        vertical = np.array([0, -1])
        body_vector = np.array([
            shoulder[0] - hip[0],
            shoulder[1] - hip[1]
        ])

        if np.linalg.norm(body_vector) == 0:
            return 180.0

        cosine = np.dot(vertical, body_vector) / (
            np.linalg.norm(vertical) * np.linalg.norm(body_vector)
        )
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def _calculate_distance(
        self, point1: Tuple[float, float], point2: Tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt(
            (point1[0] - point2[0]) ** 2 +
            (point1[1] - point2[1]) ** 2
        )

    def _count_fingers(
        self, hand_landmarks: Any, hand_label: str
    ) -> int:
        """Count raised fingers on a hand."""
        landmarks = hand_landmarks.landmark
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [2, 6, 10, 14, 18]
        fingers = []

        # Thumb
        if (
            (hand_label == 'Right' and landmarks[4].x < landmarks[2].x) or
            (hand_label == 'Left' and landmarks[4].x > landmarks[2].x)
        ):
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers
        for tip, pip in zip(finger_tips[1:], finger_pips[1:]):
            fingers.append(1 if landmarks[tip].y < landmarks[pip].y else 0)

        return sum(fingers)
