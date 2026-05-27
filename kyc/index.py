import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from starlette.datastructures import FormData, UploadFile as StarletteUploadFile

from kyc.kyc import (
    DOCUMENT_TYPES,
    FaceMatchService,
    KYCRepository,
    LivenessService,
    LocalEvidenceStorage,
    OCRGatewayClient,
    OCRProfileMapper,
    decode_data_url,
)
from kyc.core.liveness import AntiSpoofingProvider


templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

repo = KYCRepository(os.environ.get("KYC_DB", "kyc.db"))
storage = LocalEvidenceStorage(os.environ.get("EVIDENCE_DIR", "evidence"))
ocr_gateway = OCRGatewayClient()
ocr_mapper = OCRProfileMapper()
face_match_service = FaceMatchService(repo)
liveness_service = LivenessService()
identity_liveness_service = LivenessService()


def default_identity_challenge() -> list[str]:
    return ["blink", "turn_left", "turn_right", "look_center"]


def json_error(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def applicant_session(request: Request, session_id: str):
    token = (
        request.headers.get("X-Session-Token")
        or request.query_params.get("token")
        or request.session.get("applicant_session_token")
    )
    return repo.verify_session_token(session_id, token)


async def uploaded_bytes(request: Request, form: FormData | None = None, name: str = "file"):
    form = form if form is not None else await request.form()
    upload = form.get(name)
    if isinstance(upload, StarletteUploadFile):
        return await upload.read(), upload.filename, upload.content_type

    payload: dict[str, Any] = {}
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        payload = await request.json()
    data_url = payload.get("data_url") or payload.get("frame")
    if data_url:
        return (
            decode_data_url(data_url),
            payload.get("filename", "capture.jpg"),
            payload.get("content_type", "image/jpeg"),
        )

    raw = await request.body()
    if raw:
        return raw, request.headers.get("X-Filename", "upload.bin"), request.headers.get("content-type")
    raise ValueError("No upload provided")


def gateway_object_detection(gateway_result):
    response = gateway_result.get("response", {}) if isinstance(gateway_result, dict) else {}
    if not isinstance(response, dict):
        return [], {"has_id_card": False, "id_card_confidence": 0.0, "face_count": 0, "text_region_count": 0}
    objects = response.get("objects") if isinstance(response.get("objects"), list) else []
    summary = response.get("object_summary") if isinstance(response.get("object_summary"), dict) else {}
    if not summary and isinstance(response.get("meta"), dict):
        summary = response["meta"].get("object_summary") or {}
    return objects, {
        "has_id_card": bool(summary.get("has_id_card")),
        "id_card_confidence": float(summary.get("id_card_confidence") or 0.0),
        "face_count": int(summary.get("face_count") or 0),
        "text_region_count": int(summary.get("text_region_count") or 0),
    }


def verification_summary(verification):
    safe = dict(verification)
    safe["document_types"] = DOCUMENT_TYPES
    for collection in ("documents", "selfies", "liveness_checks", "face_matches", "face_embeddings", "face_search_results", "audit_events"):
        safe[collection] = [dict(item) for item in safe[collection]]
    for document in safe["documents"]:
        try:
            document["normalized"] = json.loads(document.get("normalized_json") or "{}")
        except json.JSONDecodeError:
            document["normalized"] = {}
    if safe["session"]:
        safe["session"]["challenge"] = json.loads(safe["session"]["challenge_json"])
    return safe


async def index(request: Request):
    session_token = request.session.get("applicant_session_token")
    if session_token and repo.get_session_by_token(session_token):
        return RedirectResponse(str(request.url_for("applicant_kyc", session_token=session_token)), status_code=302)
    created = repo.create_demo_session()
    request.session["applicant_session_token"] = created["session_token"]
    return RedirectResponse(str(request.url_for("applicant_kyc", session_token=created["session_token"])), status_code=302)


async def applicant_kyc(request: Request, session_token: str):
    kyc_session = repo.get_session_by_token(session_token)
    if not kyc_session:
        return JSONResponse({"detail": "Not found"}, status_code=404)
    request.session["applicant_session_token"] = session_token
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "document_types": DOCUMENT_TYPES,
            "session_id": kyc_session["id"],
            "session_token": session_token,
        },
    )


async def identity_portrait(request: Request):
    try:
        data, filename, content_type = await uploaded_bytes(request)
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return json_error("Invalid portrait image")

        from kyc.face import FacePlatform

        model_dir = Path(__file__).resolve().parent / "models"
        mediapipe_model = model_dir / "face_landmarker.task"
        yunet_model = model_dir / "face_detection_yunet_2023mar.onnx"
        platform = FacePlatform(
            mediapipe_model_path=str(mediapipe_model) if mediapipe_model.exists() else None,
            yunet_model_path=str(yunet_model) if yunet_model.exists() else None,
            detection_mode="yunet" if yunet_model.exists() else "multiscale",
            landmark_mode="mediapipe" if mediapipe_model.exists() else "auto",
            recognition_enabled=False,
            demographic_enabled=False,
        )
        try:
            face_result = platform.analyze(image, return_annotated=False).to_dict()
        finally:
            platform.close()

        anti_spoofing = AntiSpoofingProvider().analyze(image)
        return {
            "filename": filename,
            "content_type": content_type,
            "face": face_result,
            "anti_spoofing": anti_spoofing,
        }
    except ValueError as error:
        return json_error(str(error))


async def identity_liveness_frame(request: Request):
    try:
        data, _, _ = await uploaded_bytes(request)
        session_id = request.query_params.get("session_id") or "identity-workbench"
        challenge = request.query_params.getlist("challenge") or default_identity_challenge()
        return identity_liveness_service.analyze_frame_bytes(data, session_id=session_id, challenge=challenge)
    except ValueError as error:
        return json_error(str(error))


async def identity_liveness_complete(request: Request):
    session_id = request.query_params.get("session_id") or "identity-workbench"
    challenge = request.query_params.getlist("challenge") or default_identity_challenge()
    return {
        "liveness": identity_liveness_service.finalize_session(session_id, challenge),
        "session_id": session_id,
    }


async def document_types():
    return DOCUMENT_TYPES


async def create_kyc_session(request: Request):
    created = repo.create_demo_session()
    request.session["applicant_session_token"] = created["session_token"]
    applicant_url = str(request.url_for("applicant_kyc", session_token=created["session_token"]))
    return JSONResponse(
        {
            "session_id": created["id"],
            "session_token": created["session_token"],
            "applicant_url": applicant_url,
            "verification": verification_summary(repo.get_case(created["id"])),
        },
        status_code=201,
    )


async def create_demo_kyc_session(request: Request):
    created = repo.create_demo_session()
    return {
        "session_id": created["id"],
        "session_token": created["session_token"],
        "applicant_url": str(request.url_for("applicant_kyc", session_token=created["session_token"])),
        "verification": verification_summary(repo.get_case(created["id"])),
    }


async def get_kyc_session(request: Request, session_id: str):
    try:
        applicant_session(request, session_id)
        return verification_summary(repo.get_case(session_id))
    except ValueError as error:
        return json_error(str(error), 404)


async def save_profile(request: Request, session_id: str):
    try:
        applicant_session(request, session_id)
        content_type = request.headers.get("content-type", "")
        data = await request.json() if content_type.startswith("application/json") else dict(await request.form())
        repo.save_profile(session_id, data)
        return verification_summary(repo.get_case(session_id))
    except ValueError as error:
        return json_error(str(error))


async def upload_document(request: Request, session_id: str):
    try:
        applicant_session(request, session_id)
        form = await request.form()
        data, filename, content_type = await uploaded_bytes(request, form)
        selected_document_type = form.get("document_type") or request.query_params.get("document_type")
        side = form.get("side") or request.query_params.get("side", "front")
        file_info = storage.save_bytes(session_id, "documents", filename, data)
        try:
            gateway_result = ocr_gateway.extract(
                file_info["path"],
                content_type=content_type,
                filename=file_info["filename"],
            )
        except ValueError as error:
            gateway_result = {
                "engine": "http_gateway",
                "error": str(error),
                "response": {
                    "document_type": "unknown",
                    "values": {},
                    "meta": {"document_type": "unknown", "document_type_confidence": 0.0},
                },
            }
        document_type = ocr_mapper.resolve_document_type(gateway_result, fallback=selected_document_type)
        gateway_result["document_type_resolution"] = {
            "selected_document_type": selected_document_type,
            "resolved_document_type": document_type,
            "used_fallback": ocr_mapper.uses_fallback(gateway_result),
        }
        suggested_profile = ocr_mapper.map(gateway_result, document_type=document_type)
        if side not in DOCUMENT_TYPES.get(document_type, {}).get("sides", []):
            side = "front"
        repo.add_document(session_id, document_type, side, file_info, content_type, gateway_result)
        face_match_service.enroll_source(session_id, "document", file_info["path"], source_id=side)
        objects, object_summary = gateway_object_detection(gateway_result)
        return {
            "document": file_info,
            "gateway": gateway_result,
            "objects": objects,
            "object_summary": object_summary,
            "suggested_profile": suggested_profile,
            "verification": verification_summary(repo.get_case(session_id)),
        }
    except ValueError as error:
        return json_error(str(error))


async def upload_selfie(request: Request, session_id: str):
    try:
        applicant_session(request, session_id)
        data, filename, content_type = await uploaded_bytes(request)
        file_info = storage.save_bytes(session_id, "selfies", filename, data)
        repo.add_selfie(session_id, file_info, content_type)
        face_match_service.enroll_source(session_id, "selfie", file_info["path"])
        return {"selfie": file_info, "verification": verification_summary(repo.get_case(session_id))}
    except ValueError as error:
        return json_error(str(error))


async def get_challenge(request: Request, session_id: str):
    try:
        applicant_session(request, session_id)
        verification = repo.get_case(session_id)
        return {"challenge": json.loads(verification["session"]["challenge_json"])}
    except ValueError as error:
        return json_error(str(error), 404)


async def process_liveness_frame(request: Request, session_id: str):
    try:
        applicant_session(request, session_id)
        data, _, _ = await uploaded_bytes(request)
        verification = repo.get_case(session_id)
        challenge = json.loads(verification["session"]["challenge_json"])
        result = liveness_service.analyze_frame_bytes(data, session_id=session_id, challenge=challenge)
        repo.audit(session_id, "liveness_frame_processed", result)
        return result
    except ValueError as error:
        return json_error(str(error))


async def complete_liveness(request: Request, session_id: str):
    try:
        applicant_session(request, session_id)
        verification = repo.get_case(session_id)
        challenge = json.loads(verification["session"]["challenge_json"])
        result = liveness_service.finalize_session(session_id, challenge)
        video_check = repo.latest_liveness_with_file(session_id)
        if video_check:
            try:
                video_result = json.loads(video_check["result_json"])
            except json.JSONDecodeError:
                video_result = {}
            merged = liveness_service.merge_results(video_result, result)
            repo.update_liveness_result(video_check["id"], merged)
            repo.audit(session_id, "liveness_merged", {"liveness_check_id": video_check["id"], "status": merged["risk_status"]})
            result = merged
        else:
            repo.add_liveness(session_id, {}, result)
        return {"liveness": result, "verification": verification_summary(repo.get_case(session_id))}
    except ValueError as error:
        return json_error(str(error))


async def upload_liveness_video(request: Request, session_id: str):
    try:
        applicant_session(request, session_id)
        data, filename, content_type = await uploaded_bytes(request)
        file_info = storage.save_bytes(session_id, "liveness", filename or "liveness.webm", data)
        verification = repo.get_case(session_id)
        challenge = json.loads(verification["session"]["challenge_json"])
        result = liveness_service.analyze_video_file(file_info["path"], challenge)
        result["content_type"] = content_type
        repo.add_liveness(session_id, file_info, result)
        return {"liveness": result, "verification": verification_summary(repo.get_case(session_id))}
    except ValueError as error:
        return json_error(str(error))


def add_kyc_routes(target: FastAPI) -> None:
    if getattr(target.state, "kyc_routes_added", False):
        return
    target.state.kyc_routes_added = True
    target.add_middleware(SessionMiddleware, secret_key=os.environ.get("SECRET_KEY", "dev-kyc-secret-change-me"))
    target.add_api_route("/", index, methods=["GET"], name="index")
    target.add_api_route("/kyc/{session_token}", applicant_kyc, methods=["GET"], name="applicant_kyc", response_class=HTMLResponse)
    target.add_api_route("/api/document-types", document_types, methods=["GET"])
    target.add_api_route("/api/kyc/sessions", create_kyc_session, methods=["POST"])
    target.add_api_route("/api/kyc/demo-session", create_demo_kyc_session, methods=["POST"])
    target.add_api_route("/api/kyc/sessions/{session_id}", get_kyc_session, methods=["GET"])
    target.add_api_route("/api/kyc/sessions/{session_id}/profile", save_profile, methods=["POST"])
    target.add_api_route("/api/kyc/sessions/{session_id}/documents", upload_document, methods=["POST"])
    target.add_api_route("/api/kyc/sessions/{session_id}/selfie", upload_selfie, methods=["POST"])
    target.add_api_route("/api/kyc/sessions/{session_id}/challenge", get_challenge, methods=["GET"])
    target.add_api_route("/api/kyc/sessions/{session_id}/liveness/frame", process_liveness_frame, methods=["POST"])
    target.add_api_route("/api/kyc/sessions/{session_id}/liveness/complete", complete_liveness, methods=["POST"])
    target.add_api_route("/api/kyc/sessions/{session_id}/liveness/video", upload_liveness_video, methods=["POST"])
    target.add_api_route("/identity/api/portrait", identity_portrait, methods=["POST"])
    target.add_api_route("/identity/api/liveness/frame", identity_liveness_frame, methods=["POST"])
    target.add_api_route("/identity/api/liveness/complete", identity_liveness_complete, methods=["POST"])


if __name__ == "__main__":
    import uvicorn
    from kyc.ocr.service import app

    print("Starting KYC verification server...")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5555)))
