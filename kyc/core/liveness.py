import cv2
import numpy as np

from kyc.blink import APILivenessDetector

class LivenessService:
    def __init__(self, detector=None):
        self.detector = detector or APILivenessDetector()
        self.session_states = {}

    def analyze_frame_bytes(self, data, session_id=None, challenge=None):
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid image frame")
        blink, value, face = self.detector.detect_blink(image)
        points = self.detector.detect_face_points(image) if face else None
        center_x = None
        head_position = "unknown"
        if points:
            xs = [point[0] for point in points]
            center_x = sum(xs) / len(xs)
            if center_x < 0.43:
                head_position = "left"
            elif center_x > 0.57:
                head_position = "right"
            else:
                head_position = "center"

        result = {
            "blink_detected": blink,
            "eye_value": value,
            "face_detected": face,
            "center_x": round(center_x, 4) if center_x is not None else None,
            "head_position": head_position,
            "backend": self.detector.backend,
        }
        if session_id:
            result["liveness_state"] = self.update_session_state(session_id, result, challenge or [])
        return result

    def update_session_state(self, session_id, frame_result, challenge):
        state = self.session_states.setdefault(
            session_id,
            {
                "frames": 0,
                "frames_with_face": 0,
                "blink_count": 0,
                "centers": [],
                "completed": {action: False for action in challenge},
            },
        )
        for action in challenge:
            state["completed"].setdefault(action, False)

        state["frames"] += 1
        if frame_result["face_detected"]:
            state["frames_with_face"] += 1
        if frame_result["blink_detected"]:
            state["blink_count"] += 1
            state["completed"]["blink"] = True
        if frame_result["center_x"] is not None:
            state["centers"].append(frame_result["center_x"])
        if frame_result["head_position"] == "left":
            state["completed"]["turn_left"] = True
        if frame_result["head_position"] == "right":
            state["completed"]["turn_right"] = True
        if frame_result["head_position"] == "center":
            state["completed"]["look_center"] = True

        face_detection_rate = state["frames_with_face"] / max(state["frames"], 1)
        center_range = max(state["centers"]) - min(state["centers"]) if len(state["centers"]) > 1 else 0
        if center_range > 0.12:
            state["completed"]["turn_left"] = True
            state["completed"]["turn_right"] = True

        completed = {action: state["completed"].get(action, False) for action in challenge}
        risk_status = self.score_state(completed, face_detection_rate)
        return {
            "risk_status": risk_status,
            "completed": completed,
            "blink_count": state["blink_count"],
            "face_detection_rate": round(face_detection_rate, 4),
            "movement_detected": center_range > 0.08,
            "frames_processed": state["frames"],
        }

    def score_state(self, completed, face_detection_rate):
        if completed and all(completed.values()) and face_detection_rate >= 0.5:
            return "pass"
        if face_detection_rate >= 0.35:
            return "warn"
        return "fail"

    def merge_results(self, video_result, live_result):
        challenge = live_result.get("challenge") or video_result.get("challenge") or []
        video_completed = video_result.get("completed") or {}
        live_completed = live_result.get("completed") or {}
        completed = {
            action: bool(video_completed.get(action) or live_completed.get(action))
            for action in challenge
        }
        face_detection_rate = max(
            float(video_result.get("face_detection_rate") or 0),
            float(live_result.get("face_detection_rate") or 0),
        )
        risk_status = self.score_state(completed, face_detection_rate)
        return {
            **video_result,
            "risk_status": risk_status,
            "challenge": challenge,
            "completed": completed,
            "blink_count": max(int(video_result.get("blink_count") or 0), int(live_result.get("blink_count") or 0)),
            "face_detection_rate": round(face_detection_rate, 4),
            "movement_detected": bool(video_result.get("movement_detected") or live_result.get("movement_detected")),
            "frames_processed": int(video_result.get("frames_processed") or 0) + int(live_result.get("frames_processed") or 0),
            "backend": live_result.get("backend") or video_result.get("backend"),
            "video_result": video_result,
            "live_frame_result": live_result,
        }

    def finalize_session(self, session_id, challenge):
        state = self.session_states.get(session_id)
        if not state:
            return {
                "risk_status": "needs_manual_review",
                "challenge": challenge,
                "completed": {action: False for action in challenge},
                "blink_count": 0,
                "face_detection_rate": 0,
                "movement_detected": False,
                "frames_processed": 0,
                "backend": self.detector.backend,
                "reason": "No live frame state was available; use recorded proof video for manual review.",
            }

        completed = {action: state["completed"].get(action, False) for action in challenge}
        face_detection_rate = state["frames_with_face"] / max(state["frames"], 1)
        center_range = max(state["centers"]) - min(state["centers"]) if len(state["centers"]) > 1 else 0
        result = {
            "risk_status": self.score_state(completed, face_detection_rate),
            "challenge": challenge,
            "completed": completed,
            "blink_count": state["blink_count"],
            "face_detection_rate": round(face_detection_rate, 4),
            "movement_detected": center_range > 0.08,
            "frames_processed": state["frames"],
            "backend": self.detector.backend,
        }
        self.session_states.pop(session_id, None)
        return result

    def analyze_video_file(self, path, challenge):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return {"risk_status": "fail", "error": "Cannot open liveness video", "challenge": challenge}

        self.detector.blink_count = 0
        self.detector.consecutive_low_ear = 0
        frame_count = 0
        frames_with_face = 0
        centers = []
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        max_frames = int(min(fps * 12, 420))

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 3 != 0:
                continue
            blink, _, face = self.detector.detect_blink(frame)
            if face:
                frames_with_face += 1
                points = self.detector.detect_face_points(frame)
                if points:
                    xs = [point[0] for point in points]
                    centers.append(sum(xs) / len(xs))

        cap.release()
        face_detection_rate = frames_with_face / max(frame_count // 3, 1)
        center_range = max(centers) - min(centers) if len(centers) > 1 else 0
        movement_detected = center_range > 0.04
        checks = {
            "blink": self.detector.blink_count >= 1,
            "turn_left": movement_detected,
            "turn_right": movement_detected,
            "look_center": face_detection_rate >= 0.5,
        }
        completed = {action: checks.get(action, False) for action in challenge}
        passed = all(completed.values()) and face_detection_rate >= 0.4
        risk_status = "pass" if passed else "warn" if face_detection_rate >= 0.3 else "fail"
        return {
            "risk_status": risk_status,
            "challenge": challenge,
            "completed": completed,
            "blink_count": self.detector.blink_count,
            "face_detection_rate": round(face_detection_rate, 4),
            "movement_detected": movement_detected,
            "frames_processed": frame_count,
            "backend": self.detector.backend,
        }
