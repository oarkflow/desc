#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import mimetypes
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
KYC_ROOT = Path(__file__).resolve().parents[1]
TESTDATA_DIR = KYC_ROOT / "testdata"
RUN_ROOT = REPO_ROOT / "test-results" / "image-matrix"
REPORT_TEMPLATE = KYC_ROOT / "reports" / "image-feature-test-report.md"

os.environ.setdefault("OCR_CACHE_DIR", str(KYC_ROOT / ".ocr_cache"))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class Fixture:
    name: str
    path: Path
    image_type: str
    source: str
    feature_group: str
    mime_type: str
    width: int | None = None
    height: int | None = None


@dataclass
class ScenarioResult:
    feature: str
    scenario: str
    image_type: str
    input_path: str | None = None
    status: str = "pass"
    duration_ms: int = 0
    assertion: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    traceback: str | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real-model image feature matrix and write Markdown reports.")
    parser.add_argument("--output-dir", default=str(RUN_ROOT), help="Per-run artifact directory")
    parser.add_argument("--report-template", default=str(REPORT_TEMPLATE), help="Stable Markdown report path")
    parser.add_argument("--skip-gateway-smoke", action="store_true", help="Skip live gateway proxy/admin smoke")
    parser.add_argument("--all-formats", action="store_true", help="Include broad BMP/TIFF/PDF conversion stress cases in the OCR matrix.")
    parser.add_argument("--gateway-timeout", type=int, default=180, help="Seconds for live gateway smoke")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    report_template = Path(args.report_template)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fixtures").mkdir(exist_ok=True)
    (output_dir / "responses").mkdir(exist_ok=True)
    (output_dir / "face").mkdir(exist_ok=True)
    report_template.parent.mkdir(parents=True, exist_ok=True)

    fixtures = generate_fixtures(output_dir / "fixtures", all_formats=args.all_formats)
    runner = MatrixRunner(output_dir=output_dir, fixtures=fixtures, skip_gateway_smoke=args.skip_gateway_smoke, gateway_timeout=args.gateway_timeout)
    payload = runner.run()

    results_path = output_dir / "results.json"
    report_path = output_dir / "report.md"
    results_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    report = render_markdown(payload, results_path)
    report_path.write_text(report, encoding="utf-8")
    report_template.write_text(render_template_markdown(payload, report_path, results_path), encoding="utf-8")

    print(f"Markdown report: {report_path}")
    print(f"JSON results: {results_path}")
    print(f"Stable summary: {report_template}")

    failed = [r for r in payload["results"] if r["status"] == "fail"]
    if failed:
        raise SystemExit(1)


class MatrixRunner:
    def __init__(self, output_dir: Path, fixtures: list[Fixture], skip_gateway_smoke: bool, gateway_timeout: int):
        self.output_dir = output_dir
        self.fixtures = fixtures
        self.skip_gateway_smoke = skip_gateway_smoke
        self.gateway_timeout = gateway_timeout
        self.results: list[ScenarioResult] = []
        self.client = None

    def run(self) -> dict[str, Any]:
        metadata = collect_metadata()
        self._add_model_dependency_results(metadata)

        self._run_group("ocr", self._run_ocr_matrix)
        self._run_group("describe", self._run_describe_matrix)
        self._run_group("face", self._run_face_matrix)
        self._run_group("liveness", self._run_liveness_matrix)
        self._run_group("kyc_flow", self._run_kyc_flow)
        self._run_group("gateway", self._run_gateway_matrix)
        self._run_group("negative", self._run_negative_matrix)

        result_dicts = [result.__dict__ for result in self.results]
        return {
            "metadata": metadata,
            "summary": summarize(result_dicts),
            "fixtures": [fixture.__dict__ | {"path": str(fixture.path)} for fixture in self.fixtures],
            "results": result_dicts,
        }

    def _run_group(self, feature: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:
            self.results.append(ScenarioResult(
                feature=feature,
                scenario="group_setup",
                image_type="n/a",
                status="fail",
                assertion=f"{feature} group failed before scenario execution",
                error=str(exc),
                traceback=traceback.format_exc(),
            ))

    def _client(self):
        if self.client is None:
            from fastapi.testclient import TestClient
            from kyc.ocr.service import app
            self.client = TestClient(app)
        return self.client

    def _record(self, feature: str, scenario: str, image_type: str, fn: Callable[[], dict[str, Any]], input_path: Path | None = None, artifacts: dict[str, str] | None = None) -> None:
        print(f"[matrix] {feature}/{scenario}/{image_type}", flush=True)
        started = time.perf_counter()
        try:
            details = fn()
            self.results.append(ScenarioResult(
                feature=feature,
                scenario=scenario,
                image_type=image_type,
                input_path=str(input_path) if input_path else None,
                status=details.pop("status", "pass"),
                duration_ms=int((time.perf_counter() - started) * 1000),
                assertion=details.pop("assertion", "completed"),
                details=details,
                artifacts=artifacts or {},
            ))
        except BlockedScenario as exc:
            self.results.append(ScenarioResult(
                feature=feature,
                scenario=scenario,
                image_type=image_type,
                input_path=str(input_path) if input_path else None,
                status=exc.status,
                duration_ms=int((time.perf_counter() - started) * 1000),
                assertion=exc.assertion,
                error=str(exc),
                artifacts=artifacts or {},
            ))
        except Exception as exc:
            self.results.append(ScenarioResult(
                feature=feature,
                scenario=scenario,
                image_type=image_type,
                input_path=str(input_path) if input_path else None,
                status="fail",
                duration_ms=int((time.perf_counter() - started) * 1000),
                assertion="scenario failed",
                error=str(exc),
                traceback=traceback.format_exc(),
                artifacts=artifacts or {},
            ))

    def _run_ocr_matrix(self) -> None:
        for fixture in self._fixtures_for("service"):
            self._record("ocr", "values_only_with_stats", fixture.image_type, lambda f=fixture: self._ocr_request(f, {"values_only": "true", "include_stats": "true"}), fixture.path)
            self._record("ocr", "full_response_with_tamper", fixture.image_type, lambda f=fixture: self._ocr_request(f, {"values_only": "false", "include_stats": "true"}), fixture.path)
            self._record("ocr", "fields_only", fixture.image_type, lambda f=fixture: self._ocr_request(f, {"values_only": "false", "fields_only": "true"}), fixture.path)

    def _ocr_request(self, fixture: Fixture, params: dict[str, str]) -> dict[str, Any]:
        with fixture.path.open("rb") as handle:
            response = self._client().post("/ocr", params=params, files={"file": (fixture.path.name, handle, fixture.mime_type)})
        response_path = self._response_path("ocr", fixture.name, params)
        response_path.write_text(response.text, encoding="utf-8")
        require(response.status_code == 200, f"OCR HTTP {response.status_code}: {response.text[:500]}")
        payload = response.json()
        values = payload.get("values") or payload.get("response", {}).get("values") or {}
        if params.get("values_only") == "false":
            require("tamper" in payload, "full/fields OCR response did not include tamper")
            require("objects" in payload, "full/fields OCR response did not include detected objects")
            if not params.get("fields_only"):
                require("items" in payload, "full OCR response did not include OCR items")
        return {
            "assertion": "OCR returned HTTP 200 and expected response shape",
            "http_status": response.status_code,
            "value_count": len(values),
            "item_count": len(payload.get("items") or []),
            "object_count": len(payload.get("objects") or []),
            "document_type": payload.get("document_type") or payload.get("meta", {}).get("document_type"),
            "artifact_response": str(response_path),
        }

    def _run_describe_matrix(self) -> None:
        for fixture in self._fixtures_for("describe"):
            self._record("describe", "caption_tags_text_tamper", fixture.image_type, lambda f=fixture: self._describe_request(f), fixture.path)

    def _describe_request(self, fixture: Fixture) -> dict[str, Any]:
        with fixture.path.open("rb") as handle:
            response = self._client().post("/describe", files={"file": (fixture.path.name, handle, fixture.mime_type)})
        response_path = self._response_path("describe", fixture.name, {})
        response_path.write_text(response.text, encoding="utf-8")
        require(response.status_code == 200, f"Describe HTTP {response.status_code}: {response.text[:500]}")
        payload = response.json()
        require(payload.get("caption"), "describe response missing caption")
        require(payload.get("tags") is not None, "describe response missing tags")
        require(payload.get("tamper") is not None, "describe response missing tamper")
        return {
            "assertion": "Describe returned caption, tags, dimensions, and tamper",
            "caption": payload.get("caption"),
            "tag_count": len(payload.get("tags") or []),
            "object_count": payload.get("object_count"),
            "width": payload.get("width"),
            "height": payload.get("height"),
            "artifact_response": str(response_path),
        }

    def _run_face_matrix(self) -> None:
        for fixture in self._fixtures_for("face"):
            self._record("face", "loader", fixture.image_type, lambda f=fixture: self._face_loader(f), fixture.path)
        model = KYC_ROOT / "models" / "face_landmarker.task"
        if not model.exists():
            raise RuntimeError(f"Required MediaPipe face landmark model missing: {model}")
        for fixture in self._fixtures_for("face_detection"):
            self._record("face", "detect_strict_478_json_overlay", fixture.image_type, lambda f=fixture: self._face_analyze(f, model), fixture.path)
        self._record("face", "sface_real_recognition_search", "jpg", self._face_sface_search)
        self._record("face", "cli_detect_all", "mixed", lambda: self._face_cli_detect_all(model))

    def _face_loader(self, fixture: Fixture) -> dict[str, Any]:
        from kyc.face.image_loader import image_info, load_image
        image = load_image(str(fixture.path))
        info = image_info(image)
        return {"assertion": "face image loader returned a 3-channel image", **info}

    def _face_analyze(self, fixture: Fixture, model: Path) -> dict[str, Any]:
        from kyc.face import FacePlatform
        yunet = KYC_ROOT / "models" / "face_detection_yunet_2023mar.onnx"
        platform = FacePlatform(
            mediapipe_model_path=str(model),
            yunet_model_path=str(yunet) if yunet.exists() else None,
            detection_mode="yunet" if yunet.exists() else "multiscale",
            landmark_mode="mediapipe",
            recognition_enabled=False,
        )
        try:
            result = platform.analyze(str(fixture.path), save_annotated=str(self.output_dir / "face" / f"{fixture.name}.jpg"), return_annotated=False)
        finally:
            platform.close()
        payload = result.to_dict()
        response_path = self.output_dir / "responses" / f"face_{fixture.name}.json"
        response_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if result.num_faces > 0:
            require(any((face.get("landmarks") or {}).get("mode") == "mediapipe_478" and (face.get("landmarks") or {}).get("num_points") == 478 for face in payload["faces"]), "face detected without 478 MediaPipe landmarks")
        return {
            "assertion": "Face analysis completed and any detected face has 478 landmarks",
            "num_faces": result.num_faces,
            "landmark_counts": [(face.get("landmarks") or {}).get("num_points") for face in payload["faces"]],
            "artifact_response": str(response_path),
            "artifact_overlay": str(self.output_dir / "face" / f"{fixture.name}.jpg"),
        }

    def _face_cli_detect_all(self, model: Path) -> dict[str, Any]:
        target = self.output_dir / "fixtures"
        out = self.output_dir / "face" / "detect_all"
        command = [
            sys.executable, "-m", "kyc.face.cli",
            "--landmark-mode", "mediapipe",
            "--mediapipe-model", str(model),
            "detect-all", str(target),
            "--output-dir", str(out),
            "--overlay", "both",
            "--crop",
        ]
        completed = subprocess.run(command, cwd=str(REPO_ROOT), text=True, capture_output=True, timeout=180)
        log_path = self.output_dir / "responses" / "face_cli_detect_all.log"
        log_path.write_text(completed.stdout + "\n--- stderr ---\n" + completed.stderr, encoding="utf-8")
        require(completed.returncode == 0, f"face detect-all exited {completed.returncode}: {completed.stderr[:500]}")
        return {"assertion": "face CLI detect-all completed", "command": command, "artifact_log": str(log_path), "artifact_output_dir": str(out)}

    def _face_sface_search(self) -> dict[str, Any]:
        from kyc.face.recognizer import SFaceSearcher
        yunet = KYC_ROOT / "models" / "face_detection_yunet_2023mar.onnx"
        sface = KYC_ROOT / "models" / "face_recognition_sface_2021dec.onnx"
        require(yunet.exists(), f"required YuNet model missing: {yunet}")
        require(sface.exists(), f"required SFace model missing: {sface}")
        candidates = [fixture for fixture in self._fixtures_for("face_detection") if fixture.image_type in {"jpg", "jpeg", "webp", "png"}]
        require(bool(candidates), "no face detection fixtures available for SFace search")
        candidates.sort(key=lambda fixture: (0 if "citizenship.jpg" in str(fixture.path) else 1, str(fixture.path)))
        query = candidates[0]
        searcher = SFaceSearcher(str(yunet), str(sface), cosine_threshold=0.30)
        matches = searcher.search(str(query.path), [str(query.path)])
        response_path = self.output_dir / "responses" / "face_sface_search.json"
        response_path.write_text(json.dumps([match.to_dict() for match in matches], indent=2), encoding="utf-8")
        require(matches, "SFace search detected no faces")
        require(any(match.is_match for match in matches), "SFace did not match the query image against itself")
        return {
            "assertion": "SFace matched a real query image against itself",
            "query": str(query.path),
            "match_count": len(matches),
            "best_cosine": round(matches[0].cosine, 4),
            "artifact_response": str(response_path),
        }

    def _run_liveness_matrix(self) -> None:
        for fixture in self._fixtures_for("liveness"):
            self._record("liveness", "frame_state", fixture.image_type, lambda f=fixture: self._liveness_frame(f), fixture.path)

    def _liveness_frame(self, fixture: Fixture) -> dict[str, Any]:
        import cv2
        from kyc.core.liveness import LivenessService
        image = cv2.imread(str(fixture.path))
        require(image is not None, "cv2 could not read liveness fixture")
        ok, encoded = cv2.imencode(".jpg", image)
        require(ok, "could not encode liveness frame")
        service = LivenessService()
        result = service.analyze_frame_bytes(encoded.tobytes(), session_id=f"matrix-{fixture.name}", challenge=["look_center"])
        response_path = self.output_dir / "responses" / f"liveness_{fixture.name}.json"
        response_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        require(result.get("liveness_state") is not None, "liveness did not return session state")
        return {
            "assertion": "liveness frame returned session state",
            "face_detected": result.get("face_detected"),
            "risk_status": (result.get("liveness_state") or {}).get("risk_status"),
            "backend": result.get("backend"),
            "artifact_response": str(response_path),
        }

    def _run_kyc_flow(self) -> None:
        document = self._first_fixture("service", preferred="webp")
        selfie = self._first_fixture("liveness", preferred="jpg")
        self._record("kyc_flow", "real_session_document_selfie_liveness", f"{document.image_type}+{selfie.image_type}", lambda: self._kyc_real_flow(document, selfie), document.path)

    def _kyc_real_flow(self, document: Fixture, selfie: Fixture) -> dict[str, Any]:
        import kyc.index as index
        import kyc.ocr.service as service
        from fastapi.testclient import TestClient
        from kyc.core.repository import KYCRepository
        from kyc.core.storage import LocalEvidenceStorage

        class InProcessRealOCRGateway:
            def __init__(self, client):
                self.client = client

            def extract(self, image_bytes, filename, content_type=None, **kwargs):
                response = self.client.post(
                    "/ocr",
                    params={"values_only": "false", "include_stats": "true"},
                    files={"file": (filename, io.BytesIO(image_bytes), content_type or mime_for(filename))},
                )
                response.raise_for_status()
                return {"engine": "in_process_real_ocr", "response": response.json()}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_repo, old_storage, old_gateway = index.repo, index.storage, index.ocr_gateway
            index.repo = KYCRepository(str(root / "kyc.db"))
            index.storage = LocalEvidenceStorage(str(root / "evidence"))
            client = TestClient(service.app)
            index.ocr_gateway = InProcessRealOCRGateway(client)
            try:
                created = client.post("/api/kyc/sessions")
                require(created.status_code == 201, f"session create failed: {created.text[:300]}")
                session = created.json()
                headers = {"X-Session-Token": session["session_token"]}
                session_id = session["session_id"]
                profile = client.post(f"/api/kyc/sessions/{session_id}/profile", headers=headers, json={
                    "full_name": "Matrix Test",
                    "date_of_birth": "1990-01-01",
                    "nationality": "Nepal",
                    "address": "Kathmandu",
                    "document_type": "national_id",
                    "document_number": "123-456-7890",
                })
                require(profile.status_code == 200, f"profile failed: {profile.text[:300]}")
                doc_response = client.post(
                    f"/api/kyc/sessions/{session_id}/documents",
                    headers=headers,
                    data={"document_type": "national_id", "side": "front"},
                    files={"file": (document.path.name, io.BytesIO(document.path.read_bytes()), document.mime_type)},
                )
                require(doc_response.status_code == 200, f"document upload failed: {doc_response.text[:500]}")
                selfie_response = client.post(
                    f"/api/kyc/sessions/{session_id}/selfie",
                    headers=headers,
                    files={"file": (selfie.path.name, io.BytesIO(selfie.path.read_bytes()), selfie.mime_type)},
                )
                require(selfie_response.status_code == 200, f"selfie upload failed: {selfie_response.text[:500]}")
                live_response = client.post(
                    f"/api/kyc/sessions/{session_id}/liveness/frame",
                    headers=headers,
                    files={"file": (selfie.path.name, io.BytesIO(selfie.path.read_bytes()), selfie.mime_type)},
                )
                require(live_response.status_code == 200, f"liveness frame failed: {live_response.text[:500]}")
                complete = client.post(f"/api/kyc/sessions/{session_id}/liveness/complete", headers=headers)
                require(complete.status_code == 200, f"liveness complete failed: {complete.text[:500]}")
                case = client.get(f"/api/kyc/sessions/{session_id}", headers=headers)
                require(case.status_code == 200, f"case fetch failed: {case.text[:500]}")
                payload = case.json()
            finally:
                index.repo, index.storage, index.ocr_gateway = old_repo, old_storage, old_gateway
        response_path = self.output_dir / "responses" / "kyc_flow_case.json"
        response_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        require(payload.get("documents"), "case has no documents")
        require(payload.get("selfies"), "case has no selfies")
        require(payload.get("liveness_checks"), "case has no liveness checks")
        return {
            "assertion": "KYC flow retained document, selfie, and liveness records using real OCR output",
            "documents": len(payload.get("documents", [])),
            "selfies": len(payload.get("selfies", [])),
            "liveness_checks": len(payload.get("liveness_checks", [])),
            "face_embeddings": len(payload.get("face_embeddings", [])),
            "artifact_response": str(response_path),
        }

    def _run_gateway_matrix(self) -> None:
        self._record("gateway", "go_unit_tests", "n/a", self._gateway_go_tests)
        if self.skip_gateway_smoke:
            self.results.append(ScenarioResult(
                feature="gateway",
                scenario="live_proxy_admin_metrics",
                image_type="n/a",
                status="blocked_missing_dependency",
                assertion="gateway smoke skipped by user",
                error="live gateway smoke disabled by --skip-gateway-smoke",
            ))
            return
        fixture = self._first_fixture("service", preferred="jpg")
        self._record("gateway", "live_proxy_admin_metrics", fixture.image_type, lambda: self._gateway_live_smoke(fixture), fixture.path)

    def _gateway_go_tests(self) -> dict[str, Any]:
        command = ["go", "test", "./cmd/gateway"]
        completed = subprocess.run(command, cwd=str(KYC_ROOT), text=True, capture_output=True, timeout=180)
        log_path = self.output_dir / "responses" / "gateway_go_test.log"
        log_path.write_text(completed.stdout + "\n--- stderr ---\n" + completed.stderr, encoding="utf-8")
        require(completed.returncode == 0, f"go gateway tests exited {completed.returncode}: {completed.stderr[:500]}")
        return {"assertion": "Go gateway unit tests passed", "command": command, "artifact_log": str(log_path)}

    def _gateway_live_smoke(self, fixture: Fixture) -> dict[str, Any]:
        import requests
        ocr_port = free_port()
        gateway_port = free_port()
        env = os.environ.copy()
        env.update({
            "MPLCONFIGDIR": str(KYC_ROOT / ".cache" / "matplotlib"),
            "OCR_PORT": str(ocr_port),
            "GATEWAY_PORT": str(gateway_port),
            "GATEWAY_OCR_UPSTREAM": f"http://127.0.0.1:{ocr_port}",
            "GATEWAY_CONFIG_DIR": str(KYC_ROOT / "config"),
            "GATEWAY_DATA_DIR": str(KYC_ROOT / "data"),
        })
        ocr_log = self.output_dir / "responses" / "gateway_ocr_service.log"
        gw_log = self.output_dir / "responses" / "gateway_live.log"
        with ocr_log.open("w", encoding="utf-8") as ocr_handle, gw_log.open("w", encoding="utf-8") as gw_handle:
            ocr_proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "kyc.ocr_service:app", "--host", "127.0.0.1", "--port", str(ocr_port)], cwd=str(REPO_ROOT), env=env, stdout=ocr_handle, stderr=subprocess.STDOUT, text=True)
            try:
                wait_http(f"http://127.0.0.1:{ocr_port}/healthz", timeout=self.gateway_timeout)
                gw_proc = subprocess.Popen(["go", "run", "./cmd/gateway"], cwd=str(KYC_ROOT), env=env, stdout=gw_handle, stderr=subprocess.STDOUT, text=True)
                try:
                    wait_http(f"http://127.0.0.1:{gateway_port}/healthz", timeout=self.gateway_timeout)
                    with fixture.path.open("rb") as handle:
                        ocr_response = requests.post(
                            f"http://127.0.0.1:{gateway_port}/ocr",
                            params={"values_only": "true"},
                            files={"file": (fixture.path.name, handle, fixture.mime_type)},
                            timeout=self.gateway_timeout,
                        )
                    require(ocr_response.status_code == 200, f"gateway /ocr HTTP {ocr_response.status_code}: {ocr_response.text[:500]}")
                    with fixture.path.open("rb") as handle:
                        preview_response = requests.post(
                            f"http://127.0.0.1:{gateway_port}/admin/api/preview",
                            data={"values_only": "false"},
                            files={"file": (fixture.path.name, handle, fixture.mime_type)},
                            timeout=self.gateway_timeout,
                        )
                    require(preview_response.status_code == 200, f"gateway preview HTTP {preview_response.status_code}: {preview_response.text[:500]}")
                    metrics_response = requests.get(f"http://127.0.0.1:{gateway_port}/metrics", timeout=30)
                    require(metrics_response.status_code == 200, f"gateway metrics HTTP {metrics_response.status_code}")
                    require("kyc_gateway_ocr_requests_total" in metrics_response.text, "gateway metrics missing OCR counter")
                finally:
                    terminate_process(gw_proc)
            finally:
                terminate_process(ocr_proc)
        return {
            "assertion": "live gateway proxied OCR/admin preview and exposed metrics",
            "gateway_port": gateway_port,
            "ocr_port": ocr_port,
            "artifact_gateway_log": str(gw_log),
            "artifact_ocr_log": str(ocr_log),
        }

    def _run_negative_matrix(self) -> None:
        self._record("negative", "invalid_image_bytes", "png", self._negative_invalid_image)
        self._record("negative", "unsupported_extension", "txt", self._negative_unsupported_extension)
        self._record("negative", "empty_upload", "png", self._negative_empty_upload)
        self._record("negative", "oversized_upload", "png", self._negative_oversized_upload)
        self._record("negative", "corrupt_pdf", "pdf", self._negative_corrupt_pdf)

    def _negative_invalid_image(self) -> dict[str, Any]:
        response = self._client().post("/ocr", files={"file": ("bad.png", b"not-an-image", "image/png")})
        require(response.status_code == 400, f"expected 400, got {response.status_code}")
        return {"assertion": "invalid image bytes rejected", "http_status": response.status_code}

    def _negative_unsupported_extension(self) -> dict[str, Any]:
        response = self._client().post("/ocr", files={"file": ("document.txt", b"hello", "text/plain")})
        require(response.status_code == 415, f"expected 415, got {response.status_code}")
        return {"assertion": "unsupported file type rejected", "http_status": response.status_code}

    def _negative_empty_upload(self) -> dict[str, Any]:
        response = self._client().post("/ocr", files={"file": ("empty.png", b"", "image/png")})
        require(response.status_code == 400, f"expected 400, got {response.status_code}")
        return {"assertion": "empty upload rejected", "http_status": response.status_code}

    def _negative_oversized_upload(self) -> dict[str, Any]:
        from kyc.ocr import service
        old = service.settings.MAX_FILE_MB
        service.settings.MAX_FILE_MB = 0
        try:
            response = self._client().post("/ocr", files={"file": ("small.png", make_png_bytes(), "image/png")})
        finally:
            service.settings.MAX_FILE_MB = old
        require(response.status_code == 413, f"expected 413, got {response.status_code}")
        return {"assertion": "oversized upload rejected by configured limit", "http_status": response.status_code}

    def _negative_corrupt_pdf(self) -> dict[str, Any]:
        response = self._client().post("/ocr", files={"file": ("bad.pdf", b"%PDF-1.4\nnot really", "application/pdf")})
        require(response.status_code == 400, f"expected 400, got {response.status_code}")
        return {"assertion": "corrupt PDF rejected", "http_status": response.status_code}

    def _fixtures_for(self, group: str) -> list[Fixture]:
        return [fixture for fixture in self.fixtures if fixture.feature_group == group]

    def _first_fixture(self, group: str, preferred: str | None = None) -> Fixture:
        fixtures = self._fixtures_for(group)
        if preferred:
            for fixture in fixtures:
                if fixture.image_type == preferred:
                    return fixture
        require(bool(fixtures), f"no fixture for group {group}")
        return fixtures[0]

    def _response_path(self, feature: str, fixture_name: str, params: dict[str, str]) -> Path:
        suffix = "_".join(f"{key}-{value}" for key, value in sorted(params.items())) or "response"
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in f"{feature}_{fixture_name}_{suffix}")
        return self.output_dir / "responses" / f"{safe}.json"

    def _add_model_dependency_results(self, metadata: dict[str, Any]) -> None:
        for name, item in metadata["model_dependencies"].items():
            status = "pass" if item["available"] else ("blocked_missing_dependency" if item.get("optional") else "fail")
            self.results.append(ScenarioResult(
                feature="model_dependency",
                scenario=name,
                image_type="n/a",
                status=status,
                assertion=item["message"],
                details=item,
            ))


class BlockedScenario(Exception):
    def __init__(self, message: str, assertion: str = "blocked"):
        super().__init__(message)
        self.status = "blocked_missing_dependency"
        self.assertion = assertion


def generate_fixtures(output_dir: Path, all_formats: bool = False) -> list[Fixture]:
    from PIL import Image, ImageDraw
    output_dir.mkdir(parents=True, exist_ok=True)
    base = Image.new("RGB", (900, 580), "white")
    draw = ImageDraw.Draw(base)
    draw.rectangle((70, 70, 830, 500), outline=(20, 40, 80), width=6)
    draw.rectangle((110, 130, 420, 185), fill=(235, 241, 255), outline=(30, 80, 160), width=2)
    draw.text((130, 145), "NATIONAL IDENTITY CARD", fill=(20, 30, 60))
    draw.text((130, 230), "Name: MATRIX TEST", fill=(0, 0, 0))
    draw.text((130, 275), "ID No: 123-456-7890", fill=(0, 0, 0))
    draw.text((130, 320), "DOB: 1990-01-01", fill=(0, 0, 0))
    draw.ellipse((610, 145, 740, 275), fill=(210, 170, 140), outline=(80, 60, 50), width=3)
    draw.ellipse((645, 185, 662, 202), fill=(20, 20, 20))
    draw.ellipse((692, 185, 709, 202), fill=(20, 20, 20))
    draw.arc((650, 215, 710, 250), 0, 180, fill=(80, 30, 30), width=3)
    draw.rectangle((585, 120, 765, 355), outline=(80, 90, 120), width=3)

    fixtures: list[Fixture] = []
    service_specs = [
        ("generated_jpeg", "JPEG", ".jpg", "image/jpeg"),
        ("generated_png", "PNG", ".png", "image/png"),
        ("generated_webp", "WEBP", ".webp", "image/webp"),
    ]
    if all_formats:
        service_specs.extend([
            ("generated_bmp", "BMP", ".bmp", "image/bmp"),
            ("generated_tiff", "TIFF", ".tiff", "image/tiff"),
        ])
    for name, fmt, ext, mime in service_specs:
        path = output_dir / f"{name}{ext}"
        base.save(path, format=fmt)
        if all_formats:
            fixtures.append(make_fixture(name, path, ext.lstrip("."), "generated", "service", mime))
        fixtures.append(make_fixture(f"{name}_describe", path, ext.lstrip("."), "generated", "describe", mime))

    if all_formats:
        pdf_path = output_dir / "generated_multipage.pdf"
        second = base.copy()
        ImageDraw.Draw(second).text((130, 370), "Page 2 Address: Kathmandu", fill=(0, 0, 0))
        base.save(pdf_path, format="PDF", save_all=True, append_images=[second])
        fixtures.append(make_fixture("generated_pdf", pdf_path, "pdf", "generated", "service", "application/pdf"))

    try:
        from skimage import data
        face_image = Image.fromarray(data.astronaut()).convert("RGB")
        face_path = output_dir / "generated_selfie.jpg"
        face_image.save(face_path, format="JPEG")
        fixtures.append(make_fixture("generated_selfie_face", face_path, "jpg", "skimage.data.astronaut", "face_detection", "image/jpeg"))
        fixtures.append(make_fixture("generated_selfie_liveness", face_path, "jpg", "skimage.data.astronaut", "liveness", "image/jpeg"))
    except Exception:
        pass

    face_specs = [
        ("generated_gif", "GIF", ".gif", "image/gif"),
        ("generated_ppm", "PPM", ".ppm", "image/x-portable-pixmap"),
        ("generated_pgm", "PPM", ".pgm", "image/x-portable-graymap"),
        ("generated_pbm", "PPM", ".pbm", "image/x-portable-bitmap"),
    ]
    for name, fmt, ext, mime in face_specs:
        path = output_dir / f"{name}{ext}"
        if ext == ".pgm":
            base.convert("L").save(path)
        elif ext == ".pbm":
            base.convert("1").save(path)
        else:
            base.save(path, format=fmt)
        fixtures.append(make_fixture(name, path, ext.lstrip("."), "generated", "face", mime))

    required_service_names = {
        "national-id.webp",
    }
    for src in sorted(TESTDATA_DIR.glob("*")):
        if src.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
            continue
        image = Image.open(src).convert("RGB")
        safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in src.stem)
        if src.name not in required_service_names and not all_formats:
            continue
        conversions = [(src.suffix.lower(), None, mime_for(src))]
        if all_formats:
            conversions = [
                (".jpg", "JPEG", "image/jpeg"),
                (".png", "PNG", "image/png"),
                (".webp", "WEBP", "image/webp"),
                (".bmp", "BMP", "image/bmp"),
                (".tiff", "TIFF", "image/tiff"),
            ]
        for ext, fmt, mime in conversions:
            path = output_dir / f"real_{safe_stem}{ext}"
            if fmt is None:
                path.write_bytes(src.read_bytes())
            else:
                image.save(path, format=fmt)
            fixtures.append(make_fixture(f"real_{safe_stem}_{ext.lstrip('.')}", path, ext.lstrip("."), str(src), "service", mime))
        if all_formats:
            fixtures.append(make_fixture(f"real_{safe_stem}_face", src, src.suffix.lower().lstrip("."), str(src), "face_detection", mime_for(src)))
            fixtures.append(make_fixture(f"real_{safe_stem}_liveness", src, src.suffix.lower().lstrip("."), str(src), "liveness", mime_for(src)))

    return fixtures


def make_fixture(name: str, path: Path, image_type: str, source: str, feature_group: str, mime_type: str) -> Fixture:
    width = None
    height = None
    if image_type != "pdf":
        try:
            from PIL import Image
            with Image.open(path) as image:
                width, height = image.size
        except Exception:
            pass
    return Fixture(
        name=name,
        path=path,
        image_type=image_type,
        source=source,
        feature_group=feature_group,
        mime_type=mime_type,
        width=width,
        height=height,
    )


def collect_metadata() -> dict[str, Any]:
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "model_dependencies": model_dependencies(),
    }


def model_dependencies() -> dict[str, dict[str, Any]]:
    models = {
        "mediapipe_face_landmarker": KYC_ROOT / "models" / "face_landmarker.task",
        "yunet_face_detection": KYC_ROOT / "models" / "face_detection_yunet_2023mar.onnx",
        "sface_recognition": KYC_ROOT / "models" / "face_recognition_sface_2021dec.onnx",
    }
    deps = {
        name: {
            "available": path.exists(),
            "path": str(path),
            "optional": False,
            "message": f"{'found' if path.exists() else 'missing'} required model {path}",
        }
        for name, path in models.items()
    }
    for module_name, label in [("pillow_heif", "heic_heif_avif"), ("rawpy", "camera_raw")]:
        available = module_available(module_name)
        deps[label] = {
            "available": available,
            "optional": True,
            "module": module_name,
            "message": f"{'found' if available else 'missing'} optional dependency {module_name}",
        }
    return deps


def render_markdown(payload: dict[str, Any], results_path: Path) -> str:
    lines = [
        "# Image Feature Matrix Report",
        "",
        f"- Started: `{payload['metadata']['started_at']}`",
        f"- Git commit: `{payload['metadata'].get('git_commit') or 'unknown'}`",
        f"- Python: `{payload['metadata']['python'].split()[0]}`",
        f"- Platform: `{payload['metadata']['platform']}`",
        f"- JSON results: `{results_path}`",
        "",
        "## Summary By Feature",
        "",
        "| Feature | Pass | Fail | Blocked | Total |",
        "|---|---:|---:|---:|---:|",
    ]
    for feature, counts in sorted(payload["summary"]["by_feature"].items()):
        lines.append(f"| {feature} | {counts.get('pass', 0)} | {counts.get('fail', 0)} | {counts.get('blocked_missing_dependency', 0)} | {sum(counts.values())} |")
    lines.extend(["", "## Matrix", "", "| Feature | Scenario | Image Type | Status | Duration | Assertion | Artifact |", "|---|---|---|---|---:|---|---|"])
    for result in payload["results"]:
        artifact = result.get("details", {}).get("artifact_response") or result.get("details", {}).get("artifact_log") or next(iter((result.get("artifacts") or {}).values()), "")
        lines.append(
            f"| {md(result['feature'])} | {md(result['scenario'])} | {md(result['image_type'])} | {status_label(result['status'])} | {result['duration_ms']} | {md(result.get('assertion') or '')} | `{artifact}` |"
        )
    lines.extend(["", "## Model Dependencies", "", "| Dependency | Status | Detail |", "|---|---|---|"])
    for name, item in sorted(payload["metadata"]["model_dependencies"].items()):
        status = "pass" if item["available"] else ("blocked" if item.get("optional") else "fail")
        lines.append(f"| {md(name)} | {status} | {md(item['message'])} |")
    failures = [result for result in payload["results"] if result["status"] == "fail"]
    if failures:
        lines.extend(["", "## Failure Appendix", ""])
        for result in failures:
            lines.extend([
                f"### {result['feature']} / {result['scenario']}",
                "",
                f"- Image type: `{result['image_type']}`",
                f"- Input: `{result.get('input_path') or ''}`",
                f"- Error: `{md(result.get('error') or '')}`",
                "",
            ])
            if result.get("traceback"):
                lines.extend(["```text", result["traceback"][-4000:], "```", ""])
    return "\n".join(lines) + "\n"


def render_template_markdown(payload: dict[str, Any], report_path: Path, results_path: Path) -> str:
    return "\n".join([
        "# Image Feature Test Report",
        "",
        "This file is the stable entry point for the real-model image feature matrix.",
        "",
        f"- Latest generated Markdown report: `{report_path}`",
        f"- Latest generated JSON summary: `{results_path}`",
        f"- Last run: `{payload['metadata']['started_at']}`",
        f"- Last git commit: `{payload['metadata'].get('git_commit') or 'unknown'}`",
        "",
        "## Last Run Summary",
        "",
        "| Status | Count |",
        "|---|---:|",
        *[f"| {status} | {count} |" for status, count in sorted(payload["summary"]["overall"].items())],
        "",
        "Run with:",
        "",
        "```bash",
        "make -C kyc image-matrix",
        "```",
        "",
        "The per-run report contains the complete feature/image matrix, response artifact paths, and failure appendix.",
        "",
    ])


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    overall = Counter(result["status"] for result in results)
    by_feature: dict[str, Counter] = defaultdict(Counter)
    for result in results:
        by_feature[result["feature"]][result["status"]] += 1
    return {
        "overall": dict(overall),
        "by_feature": {feature: dict(counts) for feature, counts in by_feature.items()},
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def mime_for(path: str | Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def make_png_bytes() -> bytes:
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def git_commit() -> str | None:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True, capture_output=True)
    return completed.stdout.strip() if completed.returncode == 0 else None


def free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_http(url: str, timeout: int) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return
        except Exception as exc:
            last_error = exc
            time.sleep(1)
    raise TimeoutError(f"timed out waiting for {url}: {last_error}")


def terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def status_label(status: str) -> str:
    return {
        "pass": "pass",
        "fail": "FAIL",
        "blocked_missing_dependency": "blocked",
    }.get(status, status)


def md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
