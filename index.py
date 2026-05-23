from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
from blink import APILivenessDetector

app = Flask(__name__)

detector = APILivenessDetector()

@app.route('/')
def index():
    """Serve the webcam detection page"""
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process a frame sent from the browser"""
    try:
        # Get the frame from the request
        frame = request.files['frame'].read()
        np_frame = np.frombuffer(frame, np.uint8)
        image = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

        # Detect blink and face
        blink_detected, ear_value, face_detected = detector.detect_blink(image)

        return jsonify({
            "blink_detected": blink_detected,
            "ear_value": ear_value,
            "face_detected": face_detected
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/start_detection', methods=['GET'])
def start_detection():
    """Start real-time detection for 5 seconds and check authenticity"""
    try:
        results = detector.analyze_webcam_with_authenticity_check(duration=5)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting webcam-based face and blink detection server...")
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5555)))
