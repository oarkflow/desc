import os
from pathlib import Path

import cv2
import numpy as np

from kyc.blink import APILivenessDetector


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_size(value, default=(80, 80)):
    if not value:
        return default
    parts = [part.strip() for part in value.replace("x", ",").split(",") if part.strip()]
    if len(parts) != 2:
        return default
    try:
        return (int(parts[0]), int(parts[1]))
    except ValueError:
        return default


def _sigmoid(value):
    return float(1.0 / (1.0 + np.exp(-value)))


def _softmax(values):
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


class AntiSpoofingProvider:
    provider_name = "onnx_anti_spoof"

    def __init__(self, model_path=None, threshold=None, input_size=None):
        self.model_path = Path(model_path or os.environ.get("ANTI_SPOOF_MODEL_PATH", MODEL_DIR / "anti_spoof.onnx"))
        self.threshold = float(threshold or os.environ.get("ANTI_SPOOF_LIVE_THRESHOLD", "0.65"))
        self.input_size = input_size or _parse_size(os.environ.get("ANTI_SPOOF_INPUT_SIZE"), (80, 80))
        self.live_index = int(os.environ.get("ANTI_SPOOF_LIVE_INDEX", "1"))
        self.enabled = _env_flag("ANTI_SPOOF_ENABLED", self.model_path.exists())
        self.session = None
        self.load_error = None
        if not self.enabled:
            return
        if not self.model_path.exists():
            self.load_error = f"Anti-spoofing model is missing at {self.model_path}."
            return
        try:
            import onnxruntime as ort
        except Exception as error:
            self.load_error = f"onnxruntime is required for anti-spoofing: {error}"
            return
        try:
            self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        except Exception as error:
            self.load_error = f"Anti-spoofing model could not be loaded: {error}"

    @property
    def available(self):
        return self.session is not None

    @property
    def model_version(self):
        return self.model_path.name

    def analyze(self, image, face_box=None):
        if not self.enabled:
            return {
                "enabled": False,
                "available": False,
                "status": "not_configured",
                "provider": self.provider_name,
                "model_version": self.model_version,
            }
        if not self.available:
            return {
                "enabled": True,
                "available": False,
                "status": "needs_manual_review",
                "provider": self.provider_name,
                "model_version": self.model_version,
                "reason": self.load_error or "Anti-spoofing model is not available.",
            }
        crop = self.crop_face(image, face_box)
        if crop is None or crop.size == 0:
            return {
                "enabled": True,
                "available": True,
                "status": "needs_manual_review",
                "provider": self.provider_name,
                "model_version": self.model_version,
                "reason": "No usable face crop was available for anti-spoofing.",
            }
        try:
            blob = self.preprocess(crop)
            input_name = self.session.get_inputs()[0].name
            output = self.session.run(None, {input_name: blob})[0].reshape(-1).astype(np.float32)
        except Exception as error:
            return {
                "enabled": True,
                "available": True,
                "status": "needs_manual_review",
                "provider": self.provider_name,
                "model_version": self.model_version,
                "reason": f"Anti-spoofing inference failed: {error}",
            }
        live_score = self.live_score(output)
        status = "live" if live_score >= self.threshold else "spoof"
        return {
            "enabled": True,
            "available": True,
            "status": status,
            "live_score": round(live_score, 4),
            "threshold": self.threshold,
            "provider": self.provider_name,
            "model_version": self.model_version,
        }

    def crop_face(self, image, face_box=None):
        if image is None:
            return None
        if not face_box:
            return image
        image_h, image_w = image.shape[:2]
        x = max(int(face_box.get("x", 0)), 0)
        y = max(int(face_box.get("y", 0)), 0)
        width = max(int(face_box.get("width", image_w)), 1)
        height = max(int(face_box.get("height", image_h)), 1)
        pad = int(max(width, height) * 0.18)
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + width + pad, image_w)
        y2 = min(y + height + pad, image_h)
        return image[y1:y2, x1:x2]

    def preprocess(self, crop):
        resized = cv2.resize(crop, self.input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.transpose(rgb, (2, 0, 1))[None, :, :, :]

    def live_score(self, output):
        if output.size == 1:
            value = float(output[0])
            return value if 0.0 <= value <= 1.0 else _sigmoid(value)
        probabilities = _softmax(output)
        live_index = self.live_index if self.live_index < probabilities.size else probabilities.size - 1
        return float(probabilities[live_index])


class LivenessService:
    def __init__(self, detector=None, anti_spoofing=None):
        self.detector = detector or APILivenessDetector()
        self.anti_spoofing = anti_spoofing or AntiSpoofingProvider()
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
        result["anti_spoofing"] = self.analyze_anti_spoofing(image, face_detected=face)
        if session_id:
            result["liveness_state"] = self.update_session_state(session_id, result, challenge or [])
        return result

    def analyze_anti_spoofing(self, image, face_detected=True, face_box=None):
        if not face_detected:
            return {
                "enabled": getattr(self.anti_spoofing, "enabled", False),
                "available": getattr(self.anti_spoofing, "available", False),
                "status": "not_run",
                "provider": getattr(self.anti_spoofing, "provider_name", "none"),
                "reason": "No face was detected in the frame.",
            }
        return self.anti_spoofing.analyze(image, face_box=face_box)

    def update_session_state(self, session_id, frame_result, challenge):
        state = self.session_states.setdefault(
            session_id,
            {
                "frames": 0,
                "frames_with_face": 0,
                "blink_count": 0,
                "centers": [],
                "completed": {action: False for action in challenge},
                "anti_spoofing": {},
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
        anti_spoofing = frame_result.get("anti_spoofing") or {}
        if anti_spoofing:
            state["anti_spoofing"] = anti_spoofing
        if anti_spoofing.get("status") == "spoof":
            risk_status = "fail"
        elif anti_spoofing.get("status") == "needs_manual_review" and risk_status == "pass":
            risk_status = "warn"
        return {
            "risk_status": risk_status,
            "completed": completed,
            "blink_count": state["blink_count"],
            "face_detection_rate": round(face_detection_rate, 4),
            "movement_detected": center_range > 0.08,
            "frames_processed": state["frames"],
            "anti_spoofing": anti_spoofing,
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
        anti_spoofing = live_result.get("anti_spoofing") or video_result.get("anti_spoofing") or {}
        if anti_spoofing.get("status") == "spoof":
            risk_status = "fail"
        elif anti_spoofing.get("status") == "needs_manual_review" and risk_status == "pass":
            risk_status = "warn"
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
            "anti_spoofing": anti_spoofing,
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
            "anti_spoofing": state.get("anti_spoofing", {}),
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
        anti_spoofing_results = []
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
                if len(anti_spoofing_results) < 12:
                    anti_spoofing_results.append(self.analyze_anti_spoofing(frame, face_detected=True))
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
        anti_spoofing = self.aggregate_anti_spoofing(anti_spoofing_results)
        if anti_spoofing.get("status") == "spoof":
            risk_status = "fail"
        elif anti_spoofing.get("status") == "needs_manual_review" and risk_status == "pass":
            risk_status = "warn"
        return {
            "risk_status": risk_status,
            "challenge": challenge,
            "completed": completed,
            "blink_count": self.detector.blink_count,
            "face_detection_rate": round(face_detection_rate, 4),
            "movement_detected": movement_detected,
            "frames_processed": frame_count,
            "backend": self.detector.backend,
            "anti_spoofing": anti_spoofing,
        }

    def aggregate_anti_spoofing(self, results):
        configured = {
            "enabled": getattr(self.anti_spoofing, "enabled", False),
            "available": getattr(self.anti_spoofing, "available", False),
            "status": "not_run",
            "provider": getattr(self.anti_spoofing, "provider_name", "none"),
        }
        if not results:
            return configured
        spoof_count = sum(1 for result in results if result.get("status") == "spoof")
        live_scores = [result["live_score"] for result in results if result.get("live_score") is not None]
        review = next((result for result in results if result.get("status") == "needs_manual_review"), None)
        if spoof_count:
            status = "spoof"
        elif live_scores:
            status = "live"
        elif review:
            status = "needs_manual_review"
        else:
            status = results[-1].get("status", "not_run")
        aggregate = {**results[-1], "status": status, "frames_checked": len(results)}
        if live_scores:
            aggregate["live_score"] = round(float(np.mean(live_scores)), 4)
        return aggregate
