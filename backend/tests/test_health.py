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


def test_local_cors_preflight_allows_any_frontend_origin(tmp_path):
    app = create_app(
        BackendSettings(
            database_url=f"sqlite:///{tmp_path / 'test.db'}",
            storage_root=tmp_path / "storage",
            environment="local",
        )
    )

    response = TestClient(app).options(
        "/health",
        headers={
            "Origin": "https://demo-frontend.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
