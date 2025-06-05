#The code is according to PEP 8 Coding styles standards
from typing import Dict, Any
from dataclasses import dataclass, field

@dataclass
class SessionData:
    """
    Stores per-user session state for sit-up tracking.
    Tracks count, rep status, gesture timing, and form metrics.
    """
    count: int = 0  # Number of sit-ups completed
    is_in_rep: bool = False  # Whether user is currently in a repetition
    last_gesture_time: float = 0  # Last time a reset gesture was detected
    rep_start_time: float = 0  # Timestamp when current rep started
    form_metrics: Dict[str, Any] = field(default_factory=dict)  # Stores real-time form analysis metrics

    def __post_init__(self):
        """
        Ensures form_metrics is initialized properly to avoid shared mutable defaults.
        Called automatically after dataclass initialization.
        """
        # Initialize form metrics only if not already provided
        if not self.form_metrics:
            self.form_metrics = {
                'shoulder_symmetry': 0.0,
                'back_angle': 180.0,
                'torso_angle': 180.0
            }
