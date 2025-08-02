import cv2
import mediapipe as mp
import numpy as np
import requests
import tempfile
import os
from scipy.spatial import distance as dist

class APILivenessDetector:
    def __init__(self, ear_threshold=0.18, consecutive_frames=1):
        self.EAR_THRESHOLD = ear_threshold
        self.CONSECUTIVE_FRAMES = consecutive_frames

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Eye landmark indices (6 key points per eye for EAR calculation)
        self.LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]

        # Counters
        self.blink_count = 0
        self.consecutive_low_ear = 0

    def download_video_from_api(self, api_url, headers=None):

        try:
            print(f"Downloading video from API: {api_url}")
            response = requests.get(api_url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()

            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')

            # Download video in chunks
            total_size = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
                    total_size += len(chunk)

            temp_file.close()
            print(f"Downloaded {total_size} bytes to {temp_file.name}")
            return temp_file.name

        except requests.exceptions.RequestException as e:
            print(f"Error downloading video: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    def calculate_ear(self, eye_points):
        """Calculate Eye Aspect Ratio"""
        # Vertical distances
        v1 = dist.euclidean(eye_points[1], eye_points[5])
        v2 = dist.euclidean(eye_points[2], eye_points[4])

        # Horizontal distance
        h = dist.euclidean(eye_points[0], eye_points[3])

        if h == 0:
            return 0

        return (v1 + v2) / (2.0 * h)

    def get_eye_landmarks(self, face_landmarks, image_shape):
        """Extract eye landmark coordinates"""
        h, w = image_shape[:2]

        left_eye = []
        right_eye = []

        for idx in self.LEFT_EYE_POINTS:
            landmark = face_landmarks.landmark[idx]
            left_eye.append([int(landmark.x * w), int(landmark.y * h)])

        for idx in self.RIGHT_EYE_POINTS:
            landmark = face_landmarks.landmark[idx]
            right_eye.append([int(landmark.x * w), int(landmark.y * h)])

        return np.array(left_eye), np.array(right_eye)

    def detect_blink(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return False, 0, False

        # Get first face
        face_landmarks = results.multi_face_landmarks[0]
        left_eye, right_eye = self.get_eye_landmarks(face_landmarks, frame.shape)

        # Calculate EAR for both eyes
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        print(f"Frame EAR values - Left: {left_ear:.2f}, Right: {right_ear:.2f}, Average: {avg_ear:.2f}")
        print(f"Consecutive low EAR count: {self.consecutive_low_ear}")

        # Blink detection logic
        blink_detected = False
        if avg_ear < self.EAR_THRESHOLD:
            self.consecutive_low_ear += 1
        else:
            if self.consecutive_low_ear >= self.CONSECUTIVE_FRAMES:
                self.blink_count += 1
                blink_detected = True
            self.consecutive_low_ear = 0

        return blink_detected, avg_ear, True

    def analyze_video_from_api(self, api_url, headers=None, min_blinks=4, duration=10):

        # Download video from API
        temp_file_path = self.download_video_from_api(api_url, headers)
        if temp_file_path is None:
            return {"error": "Failed to download video from API"}

        try:
            # Open video file
            cap = cv2.VideoCapture(temp_file_path)
            if not cap.isOpened():
                return {"error": "Cannot open downloaded video"}

            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            max_frames = int(fps * duration)

            # Reset counters
            self.blink_count = 0
            self.consecutive_low_ear = 0

            frame_count = 0
            frames_with_face = 0

            print(f"Analyzing video... (max {duration}s)")

            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                blink_detected, ear_value, face_detected = self.detect_blink(frame)

                if face_detected:
                    frames_with_face += 1

                # Early exit if enough blinks detected
                if self.blink_count >= min_blinks:
                    print(f"Early exit: Found {min_blinks} blinks")
                    break

            cap.release()

            # Calculate results
            face_detection_rate = frames_with_face / max(frame_count, 1)
            processing_time = frame_count / fps

            is_live = (self.blink_count >= min_blinks and
                      face_detection_rate > 0.5)

            results = {
                "is_live": is_live,
                "blink_count": self.blink_count,
                "face_detection_rate": face_detection_rate,
                "processing_time": processing_time,
                "frames_processed": frame_count,
                "api_url": api_url
            }

            print(f"Analysis complete: {'LIVE' if is_live else 'NOT LIVE'} ({self.blink_count} blinks)")
            return results

        finally:
            # Always clean up temporary file
            try:
                os.unlink(temp_file_path)
                print(f"Cleaned up temporary file")
            except:
                pass

    def analyze_webcam(self, duration=10):
        """Analyze blinks using webcam"""
        cap = cv2.VideoCapture(0)  # Open webcam
        if not cap.isOpened():
            return {"error": "Cannot access webcam"}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        max_frames = int(fps * duration)

        # Reset counters
        self.blink_count = 0
        self.consecutive_low_ear = 0

        frame_count = 0
        frames_with_face = 0

        print(f"Analyzing webcam feed... (max {duration}s)")

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            blink_detected, ear_value, face_detected = self.detect_blink(frame)

            if face_detected:
                frames_with_face += 1

            # Early exit if enough blinks detected
            if self.blink_count >= 4:  # Default minimum blinks
                print(f"Early exit: Found 4 blinks")
                break

        cap.release()

        # Calculate results
        face_detection_rate = frames_with_face / max(frame_count, 1)
        processing_time = frame_count / fps

        is_live = (self.blink_count >= 4 and face_detection_rate > 0.5)

        results = {
            "is_live": is_live,
            "blink_count": self.blink_count,
            "face_detection_rate": face_detection_rate,
            "processing_time": processing_time,
            "frames_processed": frame_count
        }

        print(f"Analysis complete: {'LIVE' if is_live else 'NOT LIVE'} ({self.blink_count} blinks)")
        return results

# Simple usage function
def analyze_api_video(api_url, headers=None, min_blinks=4, duration=10, ear_threshold=0.2):

    detector = APILivenessDetector(ear_threshold=ear_threshold)
    result = detector.analyze_video_from_api(api_url, headers, min_blinks, duration)

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return result

    print(f"\n{'='*50}")
    print(f"LIVENESS DETECTION RESULTS")
    print(f"{'='*50}")
    print(f"Result: {'✅ LIVE' if result['is_live'] else '❌ NOT LIVE'}")
    print(f"Blinks detected: {result['blink_count']}")
    print(f"Face detection rate: {result['face_detection_rate']:.1%}")
    print(f"Processing time: {result['processing_time']:.2f}s")
    print(f"Frames processed: {result['frames_processed']}")

    return result

if __name__ == "__main__":
    # Example usage for webcam
    detector = APILivenessDetector(ear_threshold=0.18, consecutive_frames=1)
    result = detector.analyze_webcam(duration=10)
    print(result)

    print("API Liveness Detector ready!")
    print("Use: analyze_api_video('your_api_url') to analyze videos")
    print(result)
