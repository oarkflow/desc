from flask import Flask, send_file, render_template, request, jsonify
import os
import cv2
import numpy as np
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

@app.route('/video')
def get_video():
    """Serve the test video file"""
    # Put your video file in the same directory as this script
    video_file = "test_video.mp4"  # Change this to your video filename

    if os.path.exists(video_file):
        return send_file(video_file, as_attachment=True)
    else:
        return {"error": f"Video file '{video_file}' not found"}, 404

@app.route('/start_detection', methods=['GET'])
def start_detection():
    """Start real-time detection for 5 seconds"""
    try:
        cap = cv2.VideoCapture(0)  # Open webcam
        if not cap.isOpened():
            return jsonify({"error": "Cannot access webcam"}), 500

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        max_frames = int(fps * 5)  # 5 seconds duration

        detector.blink_count = 0
        detector.consecutive_low_ear = 0

        frame_count = 0
        frames_with_face = 0

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            blink_detected, ear_value, face_detected = detector.detect_blink(frame)

            if face_detected:
                frames_with_face += 1

        cap.release()

        face_detection_rate = frames_with_face / max(frame_count, 1)
        is_live = (detector.blink_count >= 4 and face_detection_rate > 0.5)

        results = {
            "is_live": is_live,
            "blink_count": detector.blink_count,
            "face_detection_rate": face_detection_rate,
            "frames_processed": frame_count
        }

        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting webcam-based face and blink detection server...")
    app.run(debug=True, host='0.0.0.0', port=5555)
