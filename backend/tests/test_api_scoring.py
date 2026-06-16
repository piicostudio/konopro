from pathlib import Path

from fastapi.testclient import TestClient

from konopro_backend.app import create_app
from konopro_backend.config import BackendSettings


def _client(tmp_path: Path, *, max_pending_jobs: int = 10) -> TestClient:
    app = create_app(
        BackendSettings(
            database_url=f"sqlite:///{tmp_path / 'api-scoring-test.db'}",
            storage_root=tmp_path / "storage",
            processing_root=tmp_path / "processing",
            environment="test",
            max_upload_mb=1,
            reference_scoring_max_pending_jobs=max_pending_jobs,
        )
    )
    return TestClient(app)


def _headers(user: str = "tester") -> dict[str, str]:
    return {"X-Konopro-Beta-User": user}


def _create_scoring_job(client: TestClient, user: str = "tester"):
    return client.post(
        "/v1/scoring-jobs",
        headers=_headers(user),
        data={"youtube_url": "https://www.youtube.com/watch?v=demo"},
        files={
            "take_audio": ("take.wav", b"fake take", "audio/wav"),
            "reference_audio": ("reference.wav", b"fake reference", "audio/wav"),
        },
    )


def test_create_scoring_job_stores_take_reference_and_job(tmp_path):
    client = _client(tmp_path)

    response = _create_scoring_job(client)

    assert response.status_code == 201
    payload = response.json()
    assert payload["session"]["source"] == "web_reference_scoring"
    assert payload["session"]["status"] == "queued"
    assert payload["job"]["job_type"] == "reference_scoring"
    assert payload["job"]["status"] == "queued"
    assert payload["queue"]["people_ahead_count"] == 0
    assert payload["queue"]["queue_position"] == 1
    assert payload["queue"]["pending_count"] == 1
    assert payload["scoring_run"]["status"] == "queued"
    assert payload["scoring_run"]["youtube_url"] == "https://www.youtube.com/watch?v=demo"
    assert payload["scoring_run"]["reference_source"] == "upload"
    assert payload["scoring_run"]["reference_original_filename"] == "reference.wav"
    assert len(list((tmp_path / "storage").rglob("*.wav"))) == 2


def test_scoring_job_status_and_result_are_owner_scoped(tmp_path):
    client = _client(tmp_path)
    created = _create_scoring_job(client).json()
    job_id = created["job"]["id"]

    owner_status = client.get(f"/v1/scoring-jobs/{job_id}", headers=_headers())
    owner_result = client.get(f"/v1/scoring-jobs/{job_id}/result", headers=_headers())
    cross_user_status = client.get(f"/v1/scoring-jobs/{job_id}", headers=_headers("other"))
    cross_user_result = client.get(f"/v1/scoring-jobs/{job_id}/result", headers=_headers("other"))

    assert owner_status.status_code == 200
    assert owner_status.json()["job"]["id"] == job_id
    assert owner_result.status_code == 200
    assert owner_result.json()["job_id"] == job_id
    assert cross_user_status.status_code == 404
    assert cross_user_result.status_code == 404


def test_scoring_job_reports_queue_position_for_later_jobs(tmp_path):
    client = _client(tmp_path)
    first = _create_scoring_job(client).json()
    second = _create_scoring_job(client).json()

    first_status = client.get(
        f"/v1/scoring-jobs/{first['job']['id']}",
        headers=_headers(),
    )
    second_status = client.get(
        f"/v1/scoring-jobs/{second['job']['id']}",
        headers=_headers(),
    )

    assert first_status.status_code == 200
    assert first_status.json()["queue"]["people_ahead_count"] == 0
    assert first_status.json()["queue"]["queue_position"] == 1
    assert second_status.status_code == 200
    assert second_status.json()["queue"]["queued_ahead_count"] == 1
    assert second_status.json()["queue"]["people_ahead_count"] == 1
    assert second_status.json()["queue"]["queue_position"] == 2
    assert second_status.json()["queue"]["pending_count"] == 2


def test_scoring_job_rejects_when_pending_queue_is_full(tmp_path):
    client = _client(tmp_path, max_pending_jobs=1)

    accepted = _create_scoring_job(client)
    rejected = _create_scoring_job(client, user="tester-2")

    assert accepted.status_code == 201
    assert rejected.status_code == 429
    assert rejected.json()["detail"] == "현재 체험 대기열이 가득 찼어요. 잠시 후 다시 시도해주세요."


def test_scoring_job_requires_beta_user_header(tmp_path):
    client = _client(tmp_path)

    response = client.post(
        "/v1/scoring-jobs",
        data={"youtube_url": "https://www.youtube.com/watch?v=demo"},
        files={"take_audio": ("take.wav", b"fake take", "audio/wav")},
    )

    assert response.status_code == 401


def test_scoring_job_rejects_blank_youtube_url(tmp_path):
    client = _client(tmp_path)

    response = client.post(
        "/v1/scoring-jobs",
        headers=_headers(),
        data={"youtube_url": "   "},
        files={"take_audio": ("take.wav", b"fake take", "audio/wav")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "YouTube URL is required"


def test_invalid_reference_upload_cleans_up_take_file(tmp_path):
    client = _client(tmp_path)

    response = client.post(
        "/v1/scoring-jobs",
        headers=_headers(),
        data={"youtube_url": "https://www.youtube.com/watch?v=demo"},
        files={
            "take_audio": ("take.wav", b"fake take", "audio/wav"),
            "reference_audio": ("reference.txt", b"bad reference", "text/plain"),
        },
    )

    assert response.status_code == 400
    assert list((tmp_path / "storage").rglob("*")) == []
