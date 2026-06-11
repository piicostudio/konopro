from pathlib import Path

from fastapi.testclient import TestClient
from sqlmodel import Session

from konopro_backend.app import create_app
from konopro_backend.config import BackendSettings
from konopro_backend.models import ReportArtifactVisibility, ReportRequestStatus
from konopro_backend.repositories import add_report_artifact, update_report_request


def _client(tmp_path: Path) -> TestClient:
    app = create_app(
        BackendSettings(
            database_url=f"sqlite:///{tmp_path / 'api-reports-test.db'}",
            storage_root=tmp_path / "storage",
            processing_root=tmp_path / "processing",
            environment="test",
            max_upload_mb=1,
            paid_report_turnaround_hours=24,
            free_report_turnaround_hours=72,
            manual_comp_report_turnaround_hours=48,
        )
    )
    return TestClient(app)


def _headers(user: str = "tester") -> dict[str, str]:
    return {"X-Konopro-Beta-User": user}


def _upload(client: TestClient, user: str = "tester"):
    response = client.post(
        "/v1/sessions",
        headers=_headers(user),
        files={"file": ("song.wav", b"fake audio", "audio/wav")},
    )
    assert response.status_code == 201
    return response.json()


def test_user_can_request_verified_report_for_owned_session(tmp_path):
    client = _client(tmp_path)
    upload = _upload(client)
    session_id = upload["session"]["id"]

    response = client.post(
        f"/v1/sessions/{session_id}/report-requests",
        headers=_headers(),
        json={"request_type": "paid", "user_notes": "Please check chorus."},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["session_id"] == session_id
    assert payload["request_type"] == "paid"
    assert payload["priority"] == "high"
    assert payload["target_turnaround_hours"] == 24
    assert payload["status"] == "requested"
    assert payload["user_notes"] == "Please check chorus."


def test_duplicate_active_report_request_is_rejected(tmp_path):
    client = _client(tmp_path)
    upload = _upload(client)
    session_id = upload["session"]["id"]

    first = client.post(
        f"/v1/sessions/{session_id}/report-requests",
        headers=_headers(),
        json={"request_type": "free"},
    )
    second = client.post(
        f"/v1/sessions/{session_id}/report-requests",
        headers=_headers(),
        json={"request_type": "free"},
    )

    assert first.status_code == 201
    assert second.status_code == 409


def test_user_can_list_and_fetch_own_report_requests(tmp_path):
    client = _client(tmp_path)
    upload = _upload(client)
    session_id = upload["session"]["id"]
    created = client.post(
        f"/v1/sessions/{session_id}/report-requests",
        headers=_headers(),
        json={"request_type": "free"},
    ).json()

    list_response = client.get("/v1/report-requests", headers=_headers())
    detail_response = client.get(f"/v1/report-requests/{created['id']}", headers=_headers())

    assert list_response.status_code == 200
    assert [item["id"] for item in list_response.json()] == [created["id"]]
    assert detail_response.status_code == 200
    assert detail_response.json()["id"] == created["id"]


def test_report_request_access_is_user_scoped(tmp_path):
    client = _client(tmp_path)
    upload = _upload(client)
    session_id = upload["session"]["id"]
    created = client.post(
        f"/v1/sessions/{session_id}/report-requests",
        headers=_headers(),
        json={"request_type": "free"},
    ).json()

    cross_user_detail = client.get(f"/v1/report-requests/{created['id']}", headers=_headers("other"))
    cross_user_create = client.post(
        f"/v1/sessions/{session_id}/report-requests",
        headers=_headers("other"),
        json={"request_type": "free"},
    )

    assert cross_user_detail.status_code == 404
    assert cross_user_create.status_code == 404


def test_user_detail_includes_only_user_visible_artifacts(tmp_path):
    client = _client(tmp_path)
    upload = _upload(client)
    session_id = upload["session"]["id"]
    created = client.post(
        f"/v1/sessions/{session_id}/report-requests",
        headers=_headers(),
        json={"request_type": "free"},
    ).json()
    request_id = created["id"]

    with Session(client.app.state.engine) as db:
        add_report_artifact(
            db,
            report_request_id=request_id,
            session_id=session_id,
            artifact_type="report_markdown",
            title="Internal Draft",
            body_text="private",
            visibility=ReportArtifactVisibility.internal,
        )
        add_report_artifact(
            db,
            report_request_id=request_id,
            session_id=session_id,
            artifact_type="report_markdown",
            title="Verified Report",
            body_text="public",
            visibility=ReportArtifactVisibility.user_visible,
        )
        update_report_request(db, request_id, status=ReportRequestStatus.delivered)

    response = client.get(f"/v1/report-requests/{request_id}", headers=_headers())

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "delivered"
    assert [artifact["title"] for artifact in payload["artifacts"]] == ["Verified Report"]
