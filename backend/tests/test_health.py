from fastapi.testclient import TestClient

from konopro_backend.app import create_app
from konopro_backend.config import BackendSettings


def test_health_endpoint_returns_status_and_environment(tmp_path):
    app = create_app(
        BackendSettings(
            database_url=f"sqlite:///{tmp_path / 'test.db'}",
            storage_root=tmp_path / "storage",
            environment="test",
        )
    )

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "environment": "test"}
