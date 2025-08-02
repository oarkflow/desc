from flask import Flask, send_file, render_template, request, jsonify
import os
import cv2
import numpy as np
from blink import APILivenessDetector
import pytesseract

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
    """Start real-time detection for 5 seconds and check authenticity"""
    try:
        results = detector.analyze_webcam_with_authenticity_check(duration=5)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload_document', methods=['POST'])
def upload_document():
    """Perform OCR on uploaded document"""
    try:
        # Get the uploaded file
        file = request.files['document']
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # Read the image
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Preprocess the image for better OCR results
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Perform OCR
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed_image, config=custom_config, lang='eng+nep')

        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload')
def upload():
    """Serve the document upload page"""
    return render_template('upload.html')

if __name__ == '__main__':
    print("Starting webcam-based face and blink detection server...")
    app.run(debug=True, host='0.0.0.0', port=5555)
