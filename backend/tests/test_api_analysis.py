from pathlib import Path

from fastapi.testclient import TestClient
from sqlmodel import Session

from konopro_backend.app import create_app
from konopro_backend.config import BackendSettings
from konopro_backend.repositories import replace_session_analysis


def _client(tmp_path: Path) -> TestClient:
    app = create_app(
        BackendSettings(
            database_url=f"sqlite:///{tmp_path / 'api-analysis-test.db'}",
            storage_root=tmp_path / "storage",
            processing_root=tmp_path / "processing",
            environment="test",
            max_upload_mb=1,
        )
    )
    return TestClient(app)


def _headers(user: str = "tester") -> dict[str, str]:
    return {"X-Konopro-Beta-User": user}


def _upload(client: TestClient):
    response = client.post(
        "/v1/sessions",
        headers=_headers(),
        files={"file": ("song.wav", b"fake audio", "audio/wav")},
    )
    assert response.status_code == 201
    return response.json()


def _persist_analysis(client: TestClient, session_id: str, job_id: str, *, weak: bool = False):
    payload = {
        "provider": "acrcloud",
        "status": "completed",
        "provider_status": "ACRCloud fingerprinting found matches.",
        "summary": {"windows_tested": 2, "recognized_windows": 1},
        "warnings": ["experimental"],
        "windows": [
            {
                "provider": "acrcloud",
                "window_start_s": 0.0,
                "window_end_s": 10.0,
                "status": "matched",
                "recognized": True,
                "matched_title": "Demo Song",
                "matched_artist": "Demo Artist",
                "identity_key": "isrc:demo",
                "confidence": 0.9,
            }
        ],
        "intervals": []
        if weak
        else [
            {
                "index": 1,
                "song": "Demo Song",
                "artist": "Demo Artist",
                "identity_key": "isrc:demo",
                "start_s": 0.0,
                "end_s": 10.0,
                "duration_s": 10.0,
                "confidence": 90.0,
                "confidence_level": "high",
                "recognized_windows": 2,
                "total_windows": 2,
                "provider_confidence": 0.9,
            }
        ],
        "weak_candidates": [
            {
                "index": 1,
                "song": "Maybe Song",
                "artist": "Maybe Artist",
                "identity_key": "title:maybe",
                "start_s": 0.0,
                "end_s": 10.0,
                "duration_s": 10.0,
                "recognized_windows": 1,
                "total_windows": 2,
                "provider_confidence": 0.4,
                "reason": "singleton_match",
            }
        ]
        if weak
        else [],
        "diagnostic": {
            "provider": "acrcloud",
            "profile": {"tested_windows": 2, "recognized_windows": 1},
            "flags": [{"code": "singleton_match", "message": "Only one window matched."}]
            if weak
            else [],
            "recommendations": [{"action": "retry_dense"}],
            "can_segment": False,
            "confidence_level": "weak_clues" if weak else "recoverable",
        },
        "recovery_sweeps": [{"name": "Dense retry", "priority": 10}],
    }
    with Session(client.app.state.engine) as db:
        replace_session_analysis(
            db,
            session_id=session_id,
            job_id=job_id,
            analysis_payload=payload,
        )


def test_get_session_analysis_returns_persisted_evidence(tmp_path):
    client = _client(tmp_path)
    upload = _upload(client)
    session_id = upload["session"]["id"]
    job_id = upload["job"]["id"]
    _persist_analysis(client, session_id, job_id)

    response = client.get(f"/v1/sessions/{session_id}/analysis", headers=_headers())

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "acrcloud"
    assert payload["result_summary"]["status"] == "accepted_intervals"
    assert payload["result_summary"]["accepted_interval_count"] == 1
    assert payload["windows"][0]["matched_title"] == "Demo Song"
    assert payload["intervals"][0]["song"] == "Demo Song"
    assert payload["diagnostic"]["recommendations"][0]["action"] == "retry_dense"


def test_session_analysis_is_owner_scoped(tmp_path):
    client = _client(tmp_path)
    upload = _upload(client)
    session_id = upload["session"]["id"]
    _persist_analysis(client, session_id, upload["job"]["id"])

    response = client.get(f"/v1/sessions/{session_id}/analysis", headers=_headers("other"))

    assert response.status_code == 404


def test_missing_analysis_returns_not_ready(tmp_path):
    client = _client(tmp_path)
    upload = _upload(client)
    session_id = upload["session"]["id"]

    response = client.get(f"/v1/sessions/{session_id}/analysis", headers=_headers())

    assert response.status_code == 404
    assert response.json()["detail"] == "Analysis not ready"


def test_weak_analysis_summary_does_not_claim_confirmed_match(tmp_path):
    client = _client(tmp_path)
    upload = _upload(client)
    session_id = upload["session"]["id"]
    _persist_analysis(client, session_id, upload["job"]["id"], weak=True)

    response = client.get(f"/v1/sessions/{session_id}/analysis", headers=_headers())

    assert response.status_code == 200
    summary = response.json()["result_summary"]
    assert summary["status"] == "weak_candidates"
    assert summary["accepted_interval_count"] == 0
    assert "do not treat them as confirmed matches" in summary["message"]
