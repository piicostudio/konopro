from pathlib import Path

from fastapi.testclient import TestClient

from konopro_backend.app import create_app
from konopro_backend.config import BackendSettings


def _client(tmp_path: Path) -> TestClient:
    app = create_app(
        BackendSettings(
            database_url=f"sqlite:///{tmp_path / 'presence-test.db'}",
            storage_root=tmp_path / "storage",
            processing_root=tmp_path / "processing",
            environment="test",
        )
    )
    return TestClient(app)


def test_presence_heartbeat_counts_active_visitors(tmp_path):
    client = _client(tmp_path)

    first = client.post(
        "/v1/presence/heartbeat",
        json={"visitor_id": "visitor-1", "path": "/"},
    )
    second = client.post(
        "/v1/presence/heartbeat",
        json={"visitor_id": "visitor-2", "path": "/#analyze"},
    )
    repeat = client.post(
        "/v1/presence/heartbeat",
        json={"visitor_id": "visitor-1", "path": "/#analyze"},
    )

    assert first.status_code == 200
    assert first.json()["active_visitor_count"] == 1
    assert first.json()["queued_scoring_count"] == 0
    assert first.json()["processing_scoring_count"] == 0
    assert first.json()["pending_scoring_count"] == 0
    assert second.status_code == 200
    assert second.json()["active_visitor_count"] == 2
    assert repeat.status_code == 200
    assert repeat.json()["active_visitor_count"] == 2
