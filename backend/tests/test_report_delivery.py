from pathlib import Path

from fastapi.testclient import TestClient

from konopro_backend.app import create_app
from konopro_backend.config import BackendSettings


def _client(tmp_path: Path) -> TestClient:
    app = create_app(
        BackendSettings(
            database_url=f"sqlite:///{tmp_path / 'report-delivery-test.db'}",
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


def _report_request(client: TestClient):
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
        json={"request_type": "paid"},
    )
    assert report.status_code == 201
    return report.json()


def test_admin_can_create_publish_and_deliver_report_artifact(tmp_path):
    client = _client(tmp_path)
    report = _report_request(client)
    request_id = report["id"]

    draft = client.post(
        f"/v1/admin/report-requests/{request_id}/artifacts",
        headers=_admin_headers(),
        json={
            "artifact_type": "report_markdown",
            "title": "Draft",
            "body_text": "Internal only",
            "visibility": "internal",
        },
    )
    user_before_publish = client.get(f"/v1/report-requests/{request_id}", headers=_headers())
    published = client.patch(
        f"/v1/admin/report-artifacts/{draft.json()['id']}",
        headers=_admin_headers(),
        json={
            "title": "Verified Report",
            "body_text": "# Verified Report\n\nYou improved your chorus control.",
            "visibility": "user_visible",
        },
    )
    delivered = client.patch(
        f"/v1/admin/report-requests/{request_id}",
        headers=_admin_headers(),
        json={"status": "delivered"},
    )
    user_after_publish = client.get(f"/v1/report-requests/{request_id}", headers=_headers())
    other_user = client.get(f"/v1/report-requests/{request_id}", headers=_headers("other"))

    assert draft.status_code == 201
    assert draft.json()["visibility"] == "internal"
    assert user_before_publish.json()["artifacts"] == []
    assert published.status_code == 200
    assert published.json()["visibility"] == "user_visible"
    assert published.json()["published_at"] is not None
    assert delivered.status_code == 200
    assert delivered.json()["status"] == "delivered"
    assert user_after_publish.status_code == 200
    assert user_after_publish.json()["status"] == "delivered"
    assert [artifact["title"] for artifact in user_after_publish.json()["artifacts"]] == [
        "Verified Report"
    ]
    assert "chorus control" in user_after_publish.json()["artifacts"][0]["body_text"]
    assert other_user.status_code == 404


def test_admin_can_create_already_user_visible_report_artifact(tmp_path):
    client = _client(tmp_path)
    report = _report_request(client)

    response = client.post(
        f"/v1/admin/report-requests/{report['id']}/artifacts",
        headers=_admin_headers(),
        json={
            "title": "Verified Report",
            "body_text": "Ready.",
            "visibility": "user_visible",
        },
    )

    assert response.status_code == 201
    assert response.json()["visibility"] == "user_visible"
    assert response.json()["published_at"] is not None
