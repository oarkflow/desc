import argparse
import io
import tempfile
from pathlib import Path

from check_common import default_test_image, existing_path, image_mime, print_json, require


def default_selfie_image():
    path = Path("/tmp/kyc_identity_selfie.jpg")
    if path.exists():
        return path
    from PIL import Image
    from skimage import data

    Image.fromarray(data.astronaut()).save(path, format="JPEG")
    return path


class LocalFakeOCRGateway:
    def extract(self, *args, **kwargs):
        return {
            "engine": "script_fake",
            "response": {
                "document_type": "nepali_national_id",
                "values": {"nid_number": "123-456-7890"},
                "meta": {"document_type": "nepali_national_id", "document_type_confidence": 1.0},
                "objects": [
                    {
                        "label": "id_card",
                        "confidence": 0.91,
                        "box": {"pixel": {"x": 1, "y": 2, "width": 100, "height": 60}},
                        "source": "script_fake",
                    }
                ],
                "object_summary": {"has_id_card": True, "id_card_confidence": 0.91, "face_count": 1},
            },
        }


def main():
    parser = argparse.ArgumentParser(description="Check KYC session, document storage, OCR mapping, selfie upload, and liveness flow.")
    parser.add_argument("--document", type=existing_path, default=default_test_image("national-id.webp"))
    parser.add_argument("--selfie", type=existing_path, default=default_selfie_image())
    parser.add_argument("--use-real-ocr-gateway", action="store_true")
    args = parser.parse_args()

    import kyc.ocr.service as service
    from kyc.core.repository import KYCRepository
    from kyc.core.storage import LocalEvidenceStorage
    from kyc.core.liveness import LivenessService
    from fastapi.testclient import TestClient

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        repo = KYCRepository(str(root / "kyc.db"))
        storage = LocalEvidenceStorage(str(root / "evidence"))
        created = repo.create_demo_session()
        session_id = created["id"]

        repo.save_profile(
            session_id,
            {
                "full_name": "Script Check",
                "date_of_birth": "1990-01-01",
                "nationality": "Nepal",
                "address": "Kathmandu",
                "document_type": "national_id",
                "document_number": "123-456-7890",
            },
        )

        document_bytes = Path(args.document).read_bytes()
        if args.use_real_ocr_gateway:
            service.OCREngine._instances.clear()
            client = TestClient(service.app)
            ocr_response = client.post(
                "/ocr",
                params={"values_only": "false", "include_stats": "true", "accuracy_mode": "fast", "retry": "false"},
                files={"file": (Path(args.document).name, io.BytesIO(document_bytes), image_mime(args.document))},
            )
            require(ocr_response.status_code == 200, f"Real OCR failed: {ocr_response.text[:500]}")
            gateway_result = {"engine": "in_process_real_ocr", "response": ocr_response.json()}
        else:
            gateway_result = LocalFakeOCRGateway().extract(document_bytes)
        document_info = storage.save_bytes(session_id, "documents", Path(args.document).name, document_bytes)
        repo.add_document(session_id, "national_id", "front", document_info, image_mime(args.document), gateway_result)

        selfie_bytes = Path(args.selfie).read_bytes()
        selfie_info = storage.save_bytes(session_id, "selfies", Path(args.selfie).name, selfie_bytes)
        repo.add_selfie(session_id, selfie_info, image_mime(args.selfie))

        frame_result = LivenessService().analyze_frame_bytes(
            selfie_bytes,
            session_id=f"script-{session_id}",
            challenge=["look_center"],
        )
        liveness_state = dict(frame_result.get("liveness_state") or {})
        require(liveness_state.get("risk_status"), "Liveness did not return a risk status.")
        liveness_state.update(
            {
                "backend": frame_result.get("backend"),
                "face_detected": frame_result.get("face_detected"),
                "frame_result": frame_result,
            }
        )
        repo.add_liveness(session_id, selfie_info, liveness_state)

        case_payload = repo.get_case(session_id)
        require(case_payload.get("documents"), "KYC case did not retain the uploaded document.")
        require(case_payload.get("selfies"), "KYC case did not retain the uploaded selfie.")
        require(case_payload.get("liveness_checks"), "KYC case did not retain the liveness check.")

    print_json(
        {
            "check": "kyc_flow",
            "status": "pass",
            "session_id": session_id,
            "document_status": case_payload["documents"][0]["risk_status"],
            "liveness": liveness_state,
            "summary": {
                "documents": len(case_payload.get("documents", [])),
                "selfies": len(case_payload.get("selfies", [])),
                "liveness_checks": len(case_payload.get("liveness_checks", [])),
                "face_embeddings": len(case_payload.get("face_embeddings", [])),
            },
        }
    )


if __name__ == "__main__":
    main()
