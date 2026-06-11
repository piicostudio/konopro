from pathlib import Path

from fastapi.testclient import TestClient

from konopro_backend.app import create_app
from konopro_backend.config import BackendSettings


def _client(tmp_path: Path, *, admin_key: str | None = "secret") -> TestClient:
    app = create_app(
        BackendSettings(
            database_url=f"sqlite:///{tmp_path / 'api-admin-reports-test.db'}",
            storage_root=tmp_path / "storage",
            processing_root=tmp_path / "processing",
            environment="test",
            max_upload_mb=1,
            admin_api_key=admin_key,
        )
    )
    return TestClient(app)


def _headers(user: str = "tester") -> dict[str, str]:
    return {"X-Konopro-Beta-User": user}


def _admin_headers(key: str = "secret") -> dict[str, str]:
    return {"X-Konopro-Admin-Key": key}


def _upload_and_request(client: TestClient, *, request_type: str = "free"):
    upload = client.post(
        "/v1/sessions",
        headers=_headers(),
        files={"file": ("song.wav", b"fake audio", "audio/wav")},
    )
    assert upload.status_code == 201
    session_id = upload.json()["session"]["id"]
    report = client.post(
        f"/v1/sessions/{session_id}/report-requests",
        headers=_headers(),
        json={"request_type": request_type},
    )
    assert report.status_code == 201
    return report.json()


def test_admin_endpoints_require_configured_key(tmp_path):
    missing_key_client = _client(tmp_path / "missing", admin_key=None)
    wrong_key_client = _client(tmp_path / "wrong", admin_key="secret")

    missing_response = missing_key_client.get(
        "/v1/admin/report-requests",
        headers=_admin_headers(),
    )
    wrong_response = wrong_key_client.get(
        "/v1/admin/report-requests",
        headers=_admin_headers("wrong"),
    )

    assert missing_response.status_code == 503
    assert wrong_response.status_code == 401


def test_admin_can_list_queue_and_filter_by_priority(tmp_path):
    client = _client(tmp_path)
    free = _upload_and_request(client, request_type="free")
    paid_upload = client.post(
        "/v1/sessions",
        headers=_headers(),
        files={"file": ("paid.wav", b"fake audio", "audio/wav")},
    ).json()
    paid = client.post(
        f"/v1/sessions/{paid_upload['session']['id']}/report-requests",
        headers=_headers(),
        json={"request_type": "paid"},
    ).json()

    queue = client.get("/v1/admin/report-requests", headers=_admin_headers())
    high_only = client.get(
        "/v1/admin/report-requests?priority=high",
        headers=_admin_headers(),
    )

    assert queue.status_code == 200
    assert [item["id"] for item in queue.json()] == [paid["id"], free["id"]]
    assert [item["id"] for item in high_only.json()] == [paid["id"]]


def test_admin_can_fetch_detail_with_events_user_and_session(tmp_path):
    client = _client(tmp_path)
    report = _upload_and_request(client)

    response = client.get(f"/v1/admin/report-requests/{report['id']}", headers=_admin_headers())

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == report["id"]
    assert payload["session"]["id"] == report["session_id"]
    assert payload["user"]["external_key"] == "tester"
    assert [event["event_type"] for event in payload["events"]] == ["created"]


def test_admin_can_update_status_notes_priority_and_due_time(tmp_path):
    client = _client(tmp_path)
    report = _upload_and_request(client)

    response = client.patch(
        f"/v1/admin/report-requests/{report['id']}",
        headers=_admin_headers(),
        json={
            "status": "blocked",
            "priority": "normal",
            "admin_notes": "Need a cleaner recording.",
            "blocker_reason": "low confidence evidence",
            "message": "Blocked for recording quality.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "blocked"
    assert payload["priority"] == "normal"
    assert payload["admin_notes"] == "Need a cleaner recording."
    assert payload["blocker_reason"] == "low confidence evidence"
    assert payload["events"][-1]["event_type"] == "updated"
    assert payload["events"][-1]["message"] == "Blocked for recording quality."


def test_admin_cannot_deliver_without_user_visible_artifact(tmp_path):
    client = _client(tmp_path)
    report = _upload_and_request(client)

    response = client.patch(
        f"/v1/admin/report-requests/{report['id']}",
        headers=_admin_headers(),
        json={"status": "delivered"},
    )

    assert response.status_code == 400
    assert "user-visible report artifact" in response.json()["detail"]
