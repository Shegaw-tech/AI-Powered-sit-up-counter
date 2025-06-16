"""Hand gesture recognition module using MediaPipe.

This module provides hand tracking and gesture recognition capabilities
using Google's MediaPipe library.
"""

import logging
from typing import Any, Dict, Optional, Tuple
import mediapipe as mp
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HandGestureAnalyzer")


class HandGestureAnalyzer:
    """Hand gesture recognition using MediaPipe Hands solution.

    Attributes:
        mp_hands: MediaPipe Hands solution module
        hands: Initialized MediaPipe Hands instance
        mp_drawing: MediaPipe drawing utilities
        mp_drawing_styles: MediaPipe drawing styles
        gesture_params: Configuration parameters for gesture detection
        last_gesture_time: Timestamp of last detected gesture
    """

    def __init__(self) -> None:
        """Initialize hand tracking model with default parameters."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # Initialize drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Configure gesture detection parameters
        self.gesture_params: Dict[str, Any] = {
            'reset_finger_count': 7,
            'min_gesture_confidence': 0.7,
            'gesture_cooldown': 1.5
        }

        self.last_gesture_time: float = 0

    def process_frame(self, frame_rgb: np.ndarray) -> Any:
        """Process an RGB frame for hand detection.

        Args:
            frame_rgb: Input frame in RGB format

        Returns:
            MediaPipe hand detection results
        """
        return self.hands.process(frame_rgb)

    def detect_reset_gesture(
        self,
        hand_results: Any,
        current_time: float,
        last_gesture_time: float
    ) -> bool:
        """Detect if a reset gesture (7 fingers) is shown.

        Args:
            hand_results: MediaPipe hand detection results
            current_time: Current timestamp in seconds
            last_gesture_time: Time of last detected gesture

        Returns:
            True if reset gesture is detected, False otherwise
        """
        if (hand_results.multi_hand_landmarks and
                current_time - last_gesture_time > self.gesture_params['gesture_cooldown']):

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

            if total_fingers == self.gesture_params['reset_finger_count']:
                self.last_gesture_time = current_time
                return True

        return False

    def draw_landmarks(self, frame: np.ndarray, hand_results: Any) -> np.ndarray:
        """Draw hand landmarks and connections on the frame.

        Args:
            frame: Input BGR frame
            hand_results: MediaPipe hand detection results

        Returns:
            Frame with drawn landmarks
        """
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        return frame

    def _count_fingers(self, hand_landmarks: Any, hand_label: str) -> int:
        """Count number of extended fingers on a hand.

        Args:
            hand_landmarks: Detected hand landmarks
            hand_label: 'Left' or 'Right' hand designation

        Returns:
            Count of extended fingers (0-5)
        """
        landmarks = hand_landmarks.landmark
        finger_tips = [4, 8, 12, 16, 20]  # Thumb to pinky tip indices
        finger_pips = [2, 6, 10, 14, 18]  # PIP joint indices
        fingers = []

        # Thumb detection (different logic than other fingers)
        thumb_tip = landmarks[4]
        thumb_pip = landmarks[2]
        if (hand_label == 'Right' and thumb_tip.x < thumb_pip.x) or \
                (hand_label == 'Left' and thumb_tip.x > thumb_pip.x):
            fingers.append(1)  # Thumb extended
        else:
            fingers.append(0)  # Thumb not extended

        # Other four fingers
        for tip_idx, pip_idx in zip(finger_tips[1:], finger_pips[1:]):
            if landmarks[tip_idx].y < landmarks[pip_idx].y:
                fingers.append(1)  # Finger extended
            else:
                fingers.append(0)  # Finger not extended

        return sum(fingers)

    def close(self) -> None:
        """Release resources and clean up MediaPipe instances."""
        self.hands.close()


