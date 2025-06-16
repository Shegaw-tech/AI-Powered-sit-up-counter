"""Frame processing pipeline for sit-up analysis.

This module handles the complete frame processing pipeline including:
- Pose detection and analysis
- Hand gesture recognition
- Repetition counting
- Form feedback generation
"""

import time
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from pose_analyzer import PoseAnalyzer


class FrameProcessor:
    """Processes video frames for exercise analysis with pose tracking.

    Attributes:
        pose_analyzer: Pose analysis instance
        last_count_time: Timestamp of last counted repetition
        alignment_start_time: When current alignment period began
        last_gesture_time: When last reset gesture was detected
        gesture_detected: Flag for recent gesture detection
    """

    def __init__(self) -> None:
        """Initialize processing pipeline and tracking variables."""
        self.pose_analyzer = PoseAnalyzer()
        self.last_count_time: float = 0
        self.alignment_start_time: Optional[float] = None
        self.last_gesture_time: float = 0
        self.gesture_detected: bool = False

    def process_frame(
        self,
        frame: np.ndarray,
        session_data: Any
    ) -> Dict[str, Any]:
        """Process a video frame for exercise analysis.

        Args:
            frame: Input BGR frame (H, W, 3)
            session_data: Session tracking object with attributes:
                - count: Current rep count
                - is_in_rep: Whether mid-repetition
                - last_rep_time: Timestamp of last rep
                - form_metrics: Dictionary of posture metrics

        Returns:
            Dictionary containing:
                - annotated_frame: Visualized analysis results
                - metrics: Calculated pose metrics
                - feedback: Form improvement suggestions
                - completed_rep: Whether rep was just counted
                - count: Updated rep count
                - debug: Processing metadata
        """
        self._init_session_data(session_data)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_time = time.time()
        annotated_frame = frame.copy()

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

        # Perform pose and hand analysis
        analysis_results = self.pose_analyzer.analyze_pose(frame_rgb)

        # Process hand gestures if detected
        if analysis_results and analysis_results.get('hand_results'):
            annotated_frame = self._process_hand_gestures(
                analysis_results['hand_results'],
                current_time,
                annotated_frame,
                session_data,
                response
            )

        # Exit early if no pose detected
        if not analysis_results or not analysis_results.get('pose_results'):
            return response

        # Process pose analysis results
        landmarks = analysis_results['pose_results'].pose_landmarks.landmark
        response['debug']['landmarks_visible'] = True

        # Visualize pose landmarks
        self._visualize_pose(
            annotated_frame,
            analysis_results['pose_results'].pose_landmarks,
            landmarks
        )

        # Calculate and process metrics
        metrics = self.pose_analyzer.calculate_metrics(landmarks)
        if metrics:
            self._process_metrics(
                metrics,
                current_time,
                session_data,
                response
            )

        # Finalize response
        response.update({
            'count': session_data.count,
            'annotated_frame': annotated_frame
        })
        return response

    def _init_session_data(self, session_data: Any) -> None:
        """Ensure required session attributes exist with defaults.

        Args:
            session_data: Session tracking object to initialize
        """
        if not hasattr(session_data, 'count'):
            session_data.count = 0
        if not hasattr(session_data, 'is_in_rep'):
            session_data.is_in_rep = False
        if not hasattr(session_data, 'last_rep_time'):
            session_data.last_rep_time = 0
        if not hasattr(session_data, 'form_metrics'):
            session_data.form_metrics = {}

    def _process_hand_gestures(
        self,
        hand_results: Any,
        current_time: float,
        frame: np.ndarray,
        session_data: Any,
        response: Dict[str, Any]
    ) -> np.ndarray:
        """Process hand gestures and draw landmarks.

        Args:
            hand_results: MediaPipe hand detection results
            current_time: Current timestamp
            frame: Frame to draw on
            session_data: Session tracking object
            response: Response dictionary to update

        Returns:
            Frame with hand landmarks drawn
        """
        # Draw hand landmarks
        frame = self.pose_analyzer.hand_analyzer.draw_landmarks(
            frame,
            hand_results
        )

        # Check for reset gesture
        if self.pose_analyzer.detect_reset_gesture(
            hand_results,
            current_time,
            self.last_gesture_time
        ):
            session_data.count = 0
            session_data.is_in_rep = False
            self.last_gesture_time = current_time
            response['debug']['gesture_detected'] = True
            response['count'] = 0

        return frame

    def _visualize_pose(
        self,
        frame: np.ndarray,
        pose_landmarks: Any,
        landmarks: List[Any]
    ) -> None:
        """Visualize pose landmarks and keypoints.

        Args:
            frame: Frame to draw on
            pose_landmarks: MediaPipe pose landmarks
            landmarks: List of landmark coordinates
        """
        # Draw full pose connections
        self.pose_analyzer.mp_drawing.draw_landmarks(
            frame,
            pose_landmarks,
            self.pose_analyzer.mp_pose.POSE_CONNECTIONS,
            self.pose_analyzer.mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # Draw key anatomical points
        self.pose_analyzer.draw_keypoints(frame, landmarks)

    def _process_metrics(
        self,
        metrics: Dict[str, float],
        current_time: float,
        session_data: Any,
        response: Dict[str, Any]
    ) -> None:
        """Process calculated pose metrics and generate feedback.

        Args:
            metrics: Dictionary of calculated pose metrics
            current_time: Current timestamp
            session_data: Session tracking object
            response: Response dictionary to update
        """
        response['metrics'] = metrics
        response['debug']['alignment'] = metrics.get('alignment', False)

        # Store form metrics
        session_data.form_metrics.update({
            'shoulder_symmetry': metrics.get('shoulder_symmetry', 0),
            'back_angle': metrics.get('back_angle', 180),
            'torso_angle': metrics.get('torso_angle', 180),
            'knee_lift': metrics.get('knee_lift', 0)
        })

        # Handle repetition counting
        self._handle_alignment_counting(
            metrics,
            current_time,
            session_data,
            response
        )

        # Generate form feedback
        self._generate_form_feedback(metrics, response)

        # Update  info
        response['debug'].update({
            'head_lift': metrics.get('head_lift', 0),
            'shoulder_lift': metrics.get('shoulder_lift', 0),
            'torso_angle': metrics.get('torso_angle', 180),
            'back_angle': metrics.get('back_angle', 180),
            'shoulder_symmetry': metrics.get('shoulder_symmetry', 0)
        })

    def _handle_alignment_counting(
        self,
        metrics: Dict[str, float],
        current_time: float,
        session_data: Any,
        response: Dict[str, Any]
    ) -> None:
        """Track exercise repetitions through alignment timing.

        Args:
            metrics: Dictionary of pose metrics
            current_time: Current timestamp
            session_data: Session tracking object
            response: Response dictionary to update
        """
        is_aligned = metrics.get('alignment', False)
        cooldown = self.pose_analyzer.alignment_thresholds['cooldown_time']
        min_align_time = self.pose_analyzer.alignment_thresholds['min_alignment_time']

        if is_aligned:
            if self.alignment_start_time is None:
                self.alignment_start_time = current_time
            elif not session_data.is_in_rep:
                if current_time - self.alignment_start_time >= min_align_time:
                    session_data.is_in_rep = True
        else:
            if session_data.is_in_rep:
                if current_time - self.last_count_time > cooldown:
                    session_data.count += 1
                    session_data.is_in_rep = False
                    self.last_count_time = current_time
                    response['completed_rep'] = True
            self.alignment_start_time = None

    def _generate_form_feedback(
        self,
        metrics: Dict[str, float],
        response: Dict[str, Any]
    ) -> None:
        """Generate user feedback based on form metrics.

        Args:
            metrics: Dictionary of pose metrics
            response: Response dictionary to update
        """
        if not metrics:
            response['feedback'].append("Critical points not visible")
            return

        feedback = []
        thresholds = self.pose_analyzer.form_thresholds

        if metrics.get('shoulder_symmetry', 0) > thresholds['max_shoulder_asymmetry']:
            feedback.append("Keep shoulders level")
        if metrics.get('back_angle', 180) < thresholds['min_back_angle']:
            feedback.append("Maintain straight back")
        if metrics.get('knee_lift', 0) > thresholds['max_knee_lift']:
            feedback.append("Keep knees down")

        response['feedback'] = feedback

    def close(self) -> None:
        """Clean up analysis resources."""
        self.pose_analyzer.close()
