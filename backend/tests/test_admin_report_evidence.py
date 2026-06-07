import io
import wave
from pathlib import Path

from fastapi.testclient import TestClient
from sqlmodel import Session

from konopro_backend.app import create_app
from konopro_backend.config import BackendSettings
from konopro_backend.repositories import replace_session_analysis


def _wav_bytes(duration_s: float = 2.0, sample_rate: int = 8000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * int(duration_s * sample_rate))
    return buffer.getvalue()


def _client(tmp_path: Path) -> TestClient:
    app = create_app(
        BackendSettings(
            database_url=f"sqlite:///{tmp_path / 'admin-evidence-test.db'}",
            storage_root=tmp_path / "storage",
            processing_root=tmp_path / "processing",
            environment="test",
            max_upload_mb=1,
            admin_api_key="secret",
        )
    )
    return TestClient(app)


def _headers(user: str = "tester") -> dict[str, str]:
    return {"X-Konopro-Beta-User": user}


def _admin_headers() -> dict[str, str]:
    return {"X-Konopro-Admin-Key": "secret"}


def _session_report_and_analysis(client: TestClient):
    upload = client.post(
        "/v1/sessions",
        headers=_headers(),
        files={"file": ("song.wav", _wav_bytes(), "audio/wav")},
    )
    assert upload.status_code == 201
    session_id = upload.json()["session"]["id"]
    job_id = upload.json()["job"]["id"]
    report = client.post(
        f"/v1/sessions/{session_id}/report-requests",
        headers=_headers(),
        json={"request_type": "paid"},
    )
    assert report.status_code == 201
    with Session(client.app.state.engine) as db:
        replace_session_analysis(
            db,
            session_id=session_id,
            job_id=job_id,
            analysis_payload={
                "provider": "fake",
                "status": "completed",
                "provider_status": "fake matched",
                "recording_duration_s": 2.0,
                "window_s": 1.0,
                "hop_s": 1.0,
                "windows": [
                    {
                        "provider": "fake",
                        "window_start_s": 0,
                        "window_end_s": 1,
                        "status": "matched",
                        "recognized": True,
                        "matched_title": "Demo Song",
                    }
                ],
                "intervals": [
                    {
                        "index": 1,
                        "song": "Demo Song",
                        "artist": "Demo Artist",
                        "identity_key": "isrc:demo",
                        "start_s": 0.0,
                        "end_s": 1.0,
                        "duration_s": 1.0,
                        "confidence": 90.0,
                        "confidence_level": "high",
                        "recognized_windows": 1,
                        "total_windows": 1,
                        "provider_confidence": 0.9,
                    }
                ],
                "weak_candidates": [],
                "diagnostic": {
                    "provider": "fake",
                    "profile": {"tested_windows": 1},
                    "flags": [],
                    "recommendations": [],
                    "can_segment": True,
                    "confidence_level": "recoverable",
                },
            },
        )
    return session_id, report.json()["id"]


def test_admin_evidence_bundle_includes_analysis_audio_and_interval_clips(tmp_path):
    client = _client(tmp_path)
    session_id, request_id = _session_report_and_analysis(client)

    response = client.get(
        f"/v1/admin/report-requests/{request_id}/evidence",
        headers=_admin_headers(),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["report_request"]["id"] == request_id
    assert payload["analysis"]["provider"] == "fake"
    assert payload["original_audio_url"] == f"/v1/admin/sessions/{session_id}/audio"
    assert len(payload["interval_clips"]) == 1
    assert payload["interval_clips"][0]["visibility"] == "internal"
    assert payload["interval_clips"][0]["download_url"].startswith(
        "/v1/admin/report-artifacts/"
    )

    audio = client.get(payload["original_audio_url"], headers=_admin_headers())
    clip = client.get(payload["interval_clips"][0]["download_url"], headers=_admin_headers())

    assert audio.status_code == 200
    assert clip.status_code == 200
    assert audio.content
    assert clip.content


def test_admin_evidence_requires_admin_key(tmp_path):
    client = _client(tmp_path)
    _session_id, request_id = _session_report_and_analysis(client)

    response = client.get(f"/v1/admin/report-requests/{request_id}/evidence")

    assert response.status_code == 401
