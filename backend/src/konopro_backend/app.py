from fastapi import FastAPI

from konopro_backend.api.admin_reports import router as admin_reports_router
from konopro_backend.api.analysis import router as analysis_router
from konopro_backend.api.jobs import router as jobs_router
from konopro_backend.api.reports import router as reports_router
from konopro_backend.api.sessions import router as sessions_router
from konopro_backend.config import BackendSettings
from konopro_backend.db import create_db_and_tables, create_engine_from_settings
from konopro_backend.storage import LocalAudioStorage


def create_app(settings: BackendSettings | None = None) -> FastAPI:
    app_settings = settings or BackendSettings()
    app = FastAPI(title="Konopro Backend")
    app.state.settings = app_settings
    app.state.engine = create_engine_from_settings(app_settings)
    app.state.storage = LocalAudioStorage(app_settings.storage_root)
    create_db_and_tables(app.state.engine)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "environment": app_settings.environment}

    app.include_router(sessions_router)
    app.include_router(jobs_router)
    app.include_router(analysis_router)
    app.include_router(reports_router)
    app.include_router(admin_reports_router)

    return app
