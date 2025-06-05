#The code is according to PEP 8 coding styles standards
from flask import Flask, render_template, jsonify, request
from backend.processor import FrameProcessor
from backend.health_recommender import HealthRecommender
from backend.models import SessionData
import cv2
import os
import numpy as np
import base64
from threading import Lock
import json

# Initialize core backend modules
frame_processor = FrameProcessor()
health_recommender = HealthRecommender()

# Base directory setup for resolving template and static paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Flask app initialization
app = Flask(
    __name__,
    template_folder=os.path.join(base_dir, 'frontend/templates'),
    static_folder=os.path.join(base_dir, 'frontend/static')
)

# Thread-safe dictionary to store session-specific state
sessions = {}
session_lock = Lock()


@app.route('/')
def index():
    """Render the main UI page (index.html)."""
    return render_template('index.html')


@app.route('/process_frame', methods=['POST'])
def process_frame():
    """
    Process an incoming video frame sent via POST request.

    Expects:
        JSON with a base64-encoded image under 'image' key.
        Optional session ID in 'X-Session-ID' header.

    Returns:
        JSON containing:
        - updated rep count
        - feedback & recommendations
        - pose metrics
        - debug data for frontend
        - (optional) annotated frame
    """
    with session_lock:
        # Get or create session data for the user
        session_id = request.headers.get('X-Session-ID', 'default')
        if session_id not in sessions:
            sessions[session_id] = SessionData()

        # Decode base64-encoded image
        frame_data = request.json.get('image', '').split(',')[1]
        img_bytes = base64.b64decode(frame_data)
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Perform frame analysis (pose detection, form check, gesture detection)
        results = frame_processor.process_frame(frame, sessions[session_id])

        # Build structured response for frontend
        response = {
            'count': sessions[session_id].count,
            'feedback': results.get('feedback', []),
            'completed_rep': results.get('completed_rep', False),
            'recommendations': [],
            'situp_type': 'full',  # Default sit-up type; change logic elsewhere
            'type_changed': False,
            'form_metrics': sessions[session_id].form_metrics,
            'metrics': results.get('metrics', {}),
            'debug': {
                'landmarks_visible': results.get('debug', {}).get('landmarks_visible', False),
                'is_in_rep': sessions[session_id].is_in_rep,
                'rep_start': results.get('debug', {}).get('rep_start', False),
                'rep_end': results.get('debug', {}).get('rep_end', False),
                'gesture_detected': results.get('debug', {}).get('gesture_detected', False),
                # Pose metrics for live display
                'head_lift': results.get('metrics', {}).get('head_lift', 0),
                'shoulder_lift': results.get('metrics', {}).get('shoulder_lift', 0),
                'torso_angle': results.get('metrics', {}).get('torso_angle', 180),
                'back_angle': results.get('metrics', {}).get('back_angle', 180),
                'shoulder_symmetry': results.get('metrics', {}).get('shoulder_symmetry', 0),
                'alignment': results.get('metrics', {}).get('alignment', False)
            }
        }

        # If rep completed, recommend exercises or tips
        if response['completed_rep']:
            response['recommendations'] = health_recommender.get_recommendations(
                sessions[session_id].count,
                sessions[session_id].form_metrics
            )

        # Optionally include processed frame with annotations (base64)
        if 'annotated_frame' in results:
            _, buffer = cv2.imencode('.jpg', results['annotated_frame'])
            response['image'] = base64.b64encode(buffer).decode('utf-8')

        return jsonify(response)


@app.errorhandler(500)
def handle_server_error(e):
    """
    Global error handler for unhandled internal server errors (HTTP 500).
    """
    return jsonify({
        'error': 'Internal server error',
        'message': str(e)
    }), 500


# App entry point
if __name__ == '__main__':
    # Run server on 0.0.0.0 to allow external access (e.g., mobile testing)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
