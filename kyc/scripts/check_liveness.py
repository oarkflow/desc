import argparse
from pathlib import Path

import cv2

from check_common import default_test_image, existing_path, print_json, require


def default_liveness_image():
    path = Path("/tmp/kyc_identity_selfie.jpg")
    if path.exists():
        return path
    from PIL import Image
    from skimage import data

    Image.fromarray(data.astronaut()).save(path, format="JPEG")
    return path


def main():
    parser = argparse.ArgumentParser(description="Check liveness frame/video processing and challenge state.")
    parser.add_argument("--image", type=existing_path, default=default_liveness_image())
    parser.add_argument("--video", type=existing_path)
    parser.add_argument("--challenge", action="append", default=["look_center"])
    parser.add_argument("--require-face", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-live-state", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    from kyc.core.liveness import LivenessService

    image = cv2.imread(str(args.image))
    require(image is not None, f"Could not read liveness image: {args.image}")
    ok, encoded = cv2.imencode(".jpg", image)
    require(ok, f"Could not encode liveness image: {args.image}")

    service = LivenessService()
    frame_result = service.analyze_frame_bytes(
        encoded.tobytes(),
        session_id="script-check",
        challenge=args.challenge,
    )
    if args.require_face:
        require(frame_result.get("face_detected"), "Liveness frame check did not detect a face.")
    if args.require_live_state:
        require(frame_result.get("liveness_state"), "Liveness frame check did not return session state.")

    video_result = None
    if args.video:
        video_result = service.analyze_video_file(args.video, args.challenge)
        require(video_result.get("frames_processed", 0) > 0, "Liveness video check did not process frames.")

    print_json(
        {
            "check": "liveness",
            "status": "pass",
            "image": str(args.image),
            "video": str(args.video) if args.video else None,
            "frame_result": frame_result,
            "video_result": video_result,
        }
    )


if __name__ == "__main__":
    main()
