"""Human pose analysis module using MediaPipe.

This module provides comprehensive body pose analysis capabilities
integrated with hand gesture recognition for exercise tracking.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np
from hand_analyzer import HandGestureAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PoseAnalyzer")


class PoseAnalyzer:
    """Analyzes human body pose using MediaPipe with integrated hand tracking.

    Attributes:
        mp_pose: MediaPipe Pose solution module
        pose: Initialized MediaPipe Pose instance
        hand_analyzer: Hand gesture recognition instance
        mp_drawing: MediaPipe drawing utilities
        mp_drawing_styles: MediaPipe drawing styles
        alignment_thresholds: Thresholds for posture alignment detection
        form_thresholds: Thresholds for exercise form evaluation
    """

    def __init__(self) -> None:
        """Initialize pose analysis models and configuration."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )

        self.hand_analyzer = HandGestureAnalyzer()

        # Initialize visualization utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Configure analysis thresholds
        self.alignment_thresholds: Dict[str, float] = {
            'vertical_threshold': 0.1,
            'min_alignment_time': 0.5,
            'cooldown_time': 1.0
        }

        self.form_thresholds: Dict[str, float] = {
            'max_shoulder_asymmetry': 0.2,
            'min_back_angle': 150,
            'max_knee_lift': 0.1
        }

    def analyze_pose(self, frame_rgb: np.ndarray) -> Optional[Dict[str, Any]]:
        """Analyze body pose and hand gestures from an RGB frame.

        Args:
            frame_rgb: Input frame in RGB format with shape (H, W, 3)

        Returns:
            Dictionary containing:
                - landmarks: Normalized pose landmark coordinates
                - pose_results: Raw MediaPipe pose results
                - hand_results: Hand detection results
            Returns None if no pose detected
        """
        results = self.pose.process(frame_rgb)
        if not results.pose_landmarks:
            return None

        hand_results = self.hand_analyzer.process_frame(frame_rgb)

        return {
            'landmarks': results.pose_landmarks.landmark,
            'pose_results': results,
            'hand_results': hand_results
        }

    def draw_keypoints(self, frame: np.ndarray, landmarks: List[Any]) -> np.ndarray:
        """Draw key anatomical points on frame with color coding.

        Args:
            frame: Input BGR frame (will be modified in-place)
            landmarks: List of MediaPipe pose landmarks

        Returns:
            Annotated frame with drawn keypoints
        """
        keypoints = [
            (self._get_landmark(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER),
            (0, 0, 255)),  # Red
            (self._get_landmark(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (0, 0, 255)),  # Red
            (self._get_landmark(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP),
            (0, 255, 0)),  # Green
            (self._get_landmark(landmarks, self.mp_pose.PoseLandmark.LEFT_EAR),
            (255, 0, 0)),  # Blue
            (self._get_landmark(landmarks, self.mp_pose.PoseLandmark.LEFT_KNEE),
            (255, 255, 0))  # Cyan
        ]

        frame_height, frame_width = frame.shape[:2]
        for point, color in keypoints:
            if point:
                x, y = int(point[0] * frame_width), int(point[1] * frame_height)
                cv2.circle(frame, (x, y), 8, color, -1)

        return frame

    def is_shoulder_ear_aligned(self, landmarks: List[Any]) -> bool:
        """Check vertical alignment between shoulders and ears.

        Args:
            landmarks: List of MediaPipe pose landmarks

        Returns:
            True if both shoulder-ear pairs are vertically aligned within threshold
        """
        left_shoulder = self._get_landmark(landmarks,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = self._get_landmark(landmarks,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_ear = self._get_landmark(landmarks,
            self.mp_pose.PoseLandmark.LEFT_EAR)
        right_ear = self._get_landmark(landmarks,
            self.mp_pose.PoseLandmark.RIGHT_EAR)

        if not all([left_shoulder, right_shoulder, left_ear, right_ear]):
            return False

        left_diff = abs(left_shoulder[1] - left_ear[1])
        right_diff = abs(right_shoulder[1] - right_ear[1])

        return (left_diff < self.alignment_thresholds['vertical_threshold'] and
                right_diff < self.alignment_thresholds['vertical_threshold'])

    def calculate_metrics(self, landmarks: List[Any]) -> Dict[str, float]:
        """Calculate comprehensive pose metrics for exercise analysis.

        Args:
            landmarks: List of MediaPipe pose landmarks

        Returns:
            Dictionary of calculated metrics including:
                - shoulder_lift: Vertical shoulder displacement
                - back_angle: Torso lean angle
                - alignment: Shoulder-ear alignment status
                - head_lift: Head elevation
                - torso_angle: Upper body angle
                - shoulder_symmetry: Bilateral shoulder level difference
                - knee_lift: Knee elevation
                - knee_shoulder_distance: Limb extension metric
        """
        metrics: Dict[str, float] = {}

        # Extract key landmarks
        required_landmarks = {
            'nose': self.mp_pose.PoseLandmark.NOSE,
            'left_shoulder': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            'right_shoulder': self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP,
            'left_ear': self.mp_pose.PoseLandmark.LEFT_EAR,
            'left_knee': self.mp_pose.PoseLandmark.LEFT_KNEE
        }

        landmark_coords = {
            name: self._get_landmark(landmarks, landmark)
            for name, landmark in required_landmarks.items()
        }

        # Calculate core metrics if base landmarks visible
        if all([landmark_coords['left_shoulder'], landmark_coords['left_hip']]):
            shoulder_y = landmark_coords['left_shoulder'][1]
            hip_y = landmark_coords['left_hip'][1]

            metrics.update({
                'shoulder_lift': hip_y - shoulder_y,
                'back_angle': self._calculate_back_angle(
                    landmark_coords['left_shoulder'],
                    landmark_coords['left_hip']
                ),
                'alignment': self.is_shoulder_ear_aligned(landmarks)
            })

            # Optional metrics
            if landmark_coords['nose']:
                metrics['head_lift'] = hip_y - landmark_coords['nose'][1]

            if landmark_coords['left_ear']:
                metrics['torso_angle'] = self._calculate_angle(
                    landmark_coords['left_ear'],
                    landmark_coords['left_shoulder'],
                    landmark_coords['left_hip']
                )

            if landmark_coords['right_shoulder']:
                metrics['shoulder_symmetry'] = abs(
                    landmark_coords['left_shoulder'][1] -
                    landmark_coords['right_shoulder'][1]
                )

            if landmark_coords['left_knee']:
                metrics['knee_lift'] = hip_y - landmark_coords['left_knee'][1]
                metrics['knee_shoulder_distance'] = self._calculate_distance(
                    landmark_coords['left_shoulder'],
                    landmark_coords['left_knee']
                )

        return metrics

    def detect_reset_gesture(
        self,
        hand_results: Any,
        current_time: float,
        last_gesture_time: float
    ) -> bool:
        """Detect reset gesture through hand analyzer delegation.

        Args:
            hand_results: MediaPipe hand detection results
            current_time: Current timestamp in seconds
            last_gesture_time: Previous gesture detection time

        Returns:
            True if valid reset gesture detected
        """
        return self.hand_analyzer.detect_reset_gesture(
            hand_results,
            current_time,
            last_gesture_time
        )

    def _get_landmark(
        self,
        landmarks: List[Any],
        landmark_type: Any
    ) -> Optional[Tuple[float, float]]:
        """Extract visible landmark coordinates.

        Args:
            landmarks: List of MediaPipe landmarks
            landmark_type: Specific landmark to extract

        Returns:
            (x, y) coordinates if landmark visible, else None
        """
        landmark = landmarks[landmark_type.value]
        return (landmark.x, landmark.y) if landmark.visibility > 0.5 else None

    def _calculate_angle(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        c: Tuple[float, float]
    ) -> float:
        """Calculate angle between three points (ABC with B as vertex).

        Args:
            a: First point coordinates
            b: Vertex point coordinates
            c: Third point coordinates

        Returns:
            Angle in degrees (0-180)
        """
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b

        if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
            return 180.0

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def _calculate_back_angle(
        self,
        shoulder: Tuple[float, float],
        hip: Tuple[float, float]
    ) -> float:
        """Calculate torso angle relative to vertical axis.

        Args:
            shoulder: Shoulder joint coordinates
            hip: Hip joint coordinates

        Returns:
            Angle in degrees between torso and vertical (0=upright, 180=fully bent)
        """
        vertical = np.array([0, -1])
        body_vector = np.array([shoulder[0] - hip[0], shoulder[1] - hip[1]])

        if np.linalg.norm(body_vector) == 0:
            return 180.0

        cosine = np.dot(vertical, body_vector) / (
            np.linalg.norm(vertical) * np.linalg.norm(body_vector))
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def _calculate_distance(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance between two points.

        Args:
            point1: First point coordinates
            point2: Second point coordinates

        Returns:
            Straight-line distance between points
        """
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def close(self) -> None:
        """Release all MediaPipe resources."""
        self.pose.close()
        self.hand_analyzer.close()
