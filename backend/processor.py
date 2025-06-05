#The code is according to PEP 8 coding styles standards
import cv2
import numpy as np
from typing import Dict, Any
import time
from backend.pose_analyzer import PoseAnalyzer


class FrameProcessor:
    """Handles frame processing for sit-up detection using pose analysis."""

    def __init__(self):
        """Initialize the pose analyzer and tracking variables."""
        self.pose_analyzer = PoseAnalyzer()
        self.last_count_time = 0  # Time of last completed repetition
        self.alignment_start_time = None  # Start time of correct alignment
        self.last_gesture_time = 0  # Time of last reset gesture
        self.gesture_detected = False

    def process_frame(self, frame: np.ndarray, session_data: Any) -> Dict[str, Any]:
        """
        Process a video frame for sit-up detection and analysis.

        Args:
            frame: Input BGR image frame
            session_data: Session data object containing tracking state

        Returns:
            Dictionary containing processing results and annotations
        """
        self._init_session_data(session_data)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_time = time.time()
        annotated_frame = frame.copy()

        # Initialize response structure
        response = {
            'annotated_frame': annotated_frame,
            'metrics': {},
            'feedback': [],
            'completed_rep': False,
            'count': session_data.count,
            'debug': {
                'timestamp': current_time,
                'is_in_rep': session_data.is_in_rep,
                'landmarks_visible': False,
                'alignment': False,
                'gesture_detected': False
            }
        }

        # Analyze pose and hands from the frame
        analysis_results = self.pose_analyzer.analyze_pose(frame_rgb)

        # Draw hand landmarks if hands are detected
        if analysis_results and analysis_results.get('hand_results'):
            annotated_frame = self._draw_hand_landmarks(
                annotated_frame,
                analysis_results['hand_results']
            )

            # Check for reset gesture to reset repetition count
            if self.pose_analyzer.detect_reset_gesture(
                    analysis_results['hand_results'],
                    current_time,
                    self.last_gesture_time
            ):
                session_data.count = 0
                session_data.is_in_rep = False
                self.last_gesture_time = current_time
                response['debug']['gesture_detected'] = True
                response['count'] = 0

        # Exit early if pose landmarks are not detected
        if not analysis_results or not analysis_results.get('pose_results'):
            return response

        # Pose landmarks found — extract and draw them
        landmarks = analysis_results['pose_results'].pose_landmarks.landmark
        response['debug']['landmarks_visible'] = True

        # Draw full-body pose landmarks
        self.pose_analyzer.mp_drawing.draw_landmarks(
            annotated_frame,
            analysis_results['pose_results'].pose_landmarks,
            self.pose_analyzer.mp_pose.POSE_CONNECTIONS,
            self.pose_analyzer.mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # Draw keypoints (head, shoulders, hips, etc.)
        annotated_frame = self.pose_analyzer.draw_keypoints(annotated_frame, landmarks)

        # Extract form metrics from landmarks
        metrics = self.pose_analyzer.calculate_metrics(landmarks)
        if metrics:
            response['metrics'] = metrics
            response['debug']['alignment'] = metrics.get('alignment', False)

            # Save selected form metrics in session data
            session_data.form_metrics = {
                'shoulder_symmetry': metrics.get('shoulder_symmetry', 0),
                'back_angle': metrics.get('back_angle', 180),
                'torso_angle': metrics.get('torso_angle', 180),
                'knee_lift': metrics.get('knee_lift', 0)
            }

            # Check for rep completion using alignment logic
            self._handle_alignment_counting(metrics, current_time, session_data, response)

            # Generate user feedback about their form
            self._generate_form_feedback(metrics, response)

        # Finalize response
        response['count'] = session_data.count
        response['annotated_frame'] = annotated_frame
        response['debug'].update({
            'head_lift': metrics.get('head_lift', 0),
            'shoulder_lift': metrics.get('shoulder_lift', 0),
            'torso_angle': metrics.get('torso_angle', 180),
            'back_angle': metrics.get('back_angle', 180),
            'shoulder_symmetry': metrics.get('shoulder_symmetry', 0)
        })

        return response

    def _init_session_data(self, session_data: Any) -> None:
        """Initialize session attributes with defaults if not present."""
        if not hasattr(session_data, 'count'):
            session_data.count = 0
        if not hasattr(session_data, 'is_in_rep'):
            session_data.is_in_rep = False
        if not hasattr(session_data, 'last_rep_time'):
            session_data.last_rep_time = 0

    def _draw_hand_landmarks(self, frame: np.ndarray, hand_results: Any) -> np.ndarray:
        """Draw hand landmarks on the frame using MediaPipe utilities."""
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.pose_analyzer.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.pose_analyzer.mp_hands.HAND_CONNECTIONS,
                    self.pose_analyzer.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.pose_analyzer.mp_drawing_styles.get_default_hand_connections_style()
                )
        return frame

    def _handle_alignment_counting(self, metrics: Dict[str, float],
                                   current_time: float,
                                   session_data: Any,
                                   response: Dict[str, Any]) -> None:
        """
        Count sit-ups by tracking correct alignment over time.
        A rep is counted when alignment ends after sufficient duration.
        """
        is_aligned = metrics.get('alignment', False)
        cooldown = self.pose_analyzer.alignment_thresholds['cooldown_time']
        min_alignment_time = self.pose_analyzer.alignment_thresholds['min_alignment_time']

        if is_aligned:
            if self.alignment_start_time is None:
                self.alignment_start_time = current_time
            elif not session_data.is_in_rep:
                if current_time - self.alignment_start_time >= min_alignment_time:
                    session_data.is_in_rep = True
        else:
            if session_data.is_in_rep:
                if current_time - self.last_count_time > cooldown:
                    session_data.count += 1
                    session_data.is_in_rep = False
                    self.last_count_time = current_time
                    response['completed_rep'] = True
            self.alignment_start_time = None

    def _generate_form_feedback(self, metrics: Dict[str, float], response: Dict[str, Any]) -> None:
        """
        Evaluate posture metrics and generate improvement suggestions.
        Adds suggestions to the `feedback` field in the response.
        """
        if not metrics:
            response['feedback'].append("Critical points not visible")
            return

        feedback = []
        if metrics.get('shoulder_symmetry', 0) > self.pose_analyzer.form_thresholds['max_shoulder_asymmetry']:
            feedback.append("Keep shoulders level")
        if metrics.get('back_angle', 180) < self.pose_analyzer.form_thresholds['min_back_angle']:
            feedback.append("Maintain straight back")
        if metrics.get('knee_lift', 0) > self.pose_analyzer.form_thresholds['max_knee_lift']:
            feedback.append("Keep knees down")

        response['feedback'] = feedback
