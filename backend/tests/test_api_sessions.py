from pathlib import Path

from fastapi.testclient import TestClient

from konopro_backend.app import create_app
from konopro_backend.config import BackendSettings


def _client(tmp_path: Path) -> TestClient:
    app = create_app(
        BackendSettings(
            database_url=f"sqlite:///{tmp_path / 'api-test.db'}",
            storage_root=tmp_path / "storage",
            environment="test",
            max_upload_mb=1,
        )
    )
    return TestClient(app)


def _headers(user: str = "tester") -> dict[str, str]:
    return {"X-Konopro-Beta-User": user}


def _upload(client: TestClient, filename: str = "song.wav", content_type: str = "audio/wav"):
    return client.post(
        "/v1/sessions",
        headers=_headers(),
        files={"file": (filename, b"fake audio", content_type)},
        data={"client_duration_s": "12.5", "source": "karaoke_room"},
    )


def test_upload_session_creates_session_job_and_stored_file(tmp_path):
    client = _client(tmp_path)

    response = _upload(client)

    assert response.status_code == 201
    payload = response.json()
    assert payload["session"]["original_filename"] == "song.wav"
    assert payload["session"]["status"] == "queued"
    assert payload["session"]["size_bytes"] == len(b"fake audio")
    assert payload["job"]["status"] == "queued"
    assert payload["job"]["job_type"] == "fingerprint_segmentation"
    assert len(list((tmp_path / "storage").rglob("*.wav"))) == 1


def test_upload_requires_beta_user_header(tmp_path):
    client = _client(tmp_path)

    response = client.post(
        "/v1/sessions",
        files={"file": ("song.wav", b"fake audio", "audio/wav")},
    )

    assert response.status_code == 401


def test_upload_rejects_invalid_content_type(tmp_path):
    client = _client(tmp_path)

    response = client.post(
        "/v1/sessions",
        headers=_headers(),
        files={"file": ("song.txt", b"not audio", "text/plain")},
    )

    assert response.status_code == 400
    assert "Unsupported audio content type" in response.json()["detail"]


def test_list_detail_and_cross_user_access(tmp_path):
    client = _client(tmp_path)
    upload = _upload(client).json()
    session_id = upload["session"]["id"]

    list_response = client.get("/v1/sessions", headers=_headers())
    detail_response = client.get(f"/v1/sessions/{session_id}", headers=_headers())
    cross_user_response = client.get(f"/v1/sessions/{session_id}", headers=_headers("other"))

    assert list_response.status_code == 200
    assert [item["id"] for item in list_response.json()] == [session_id]
    assert detail_response.status_code == 200
    assert detail_response.json()["id"] == session_id
    assert cross_user_response.status_code == 404


def test_download_session_audio_is_owner_scoped(tmp_path):
    client = _client(tmp_path)
    upload = _upload(client).json()
    session_id = upload["session"]["id"]

    owner_response = client.get(f"/v1/sessions/{session_id}/audio", headers=_headers())
    cross_user_response = client.get(f"/v1/sessions/{session_id}/audio", headers=_headers("other"))

    assert owner_response.status_code == 200
    assert owner_response.content == b"fake audio"
    assert owner_response.headers["content-type"].startswith("audio/wav")
    assert cross_user_response.status_code == 404


def test_job_status_is_scoped_to_session_owner(tmp_path):
    client = _client(tmp_path)
    upload = _upload(client).json()
    job_id = upload["job"]["id"]

    owner_response = client.get(f"/v1/jobs/{job_id}", headers=_headers())
    cross_user_response = client.get(f"/v1/jobs/{job_id}", headers=_headers("other"))

    assert owner_response.status_code == 200
    assert owner_response.json()["id"] == job_id
    assert cross_user_response.status_code == 404


def test_delete_session_removes_file_and_hides_session(tmp_path):
    client = _client(tmp_path)
    upload = _upload(client).json()
    session_id = upload["session"]["id"]

    assert len(list((tmp_path / "storage").rglob("*.wav"))) == 1
    delete_response = client.delete(f"/v1/sessions/{session_id}", headers=_headers())
    list_response = client.get("/v1/sessions", headers=_headers())

    assert delete_response.status_code == 204
    assert list_response.json() == []
    assert list((tmp_path / "storage").rglob("*.wav")) == []
