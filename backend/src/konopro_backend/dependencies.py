from collections.abc import Generator
from hmac import compare_digest

from fastapi import Depends, Header, HTTPException, Request, status
from sqlmodel import Session

from konopro_backend.config import BackendSettings
from konopro_backend.storage import LocalAudioStorage


def get_settings(request: Request) -> BackendSettings:
    return request.app.state.settings


def get_db(request: Request) -> Generator[Session, None, None]:
    with Session(request.app.state.engine) as db:
        yield db


def get_storage(request: Request) -> LocalAudioStorage:
    return request.app.state.storage


def get_beta_user_key(
    x_konopro_beta_user: str | None = Header(default=None, alias="X-Konopro-Beta-User"),
) -> str:
    if not x_konopro_beta_user or not x_konopro_beta_user.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Konopro-Beta-User header",
        )
    return x_konopro_beta_user.strip()


def get_admin_access(
    settings: BackendSettings = Depends(get_settings),
    x_konopro_admin_key: str | None = Header(default=None, alias="X-Konopro-Admin-Key"),
) -> str:
    expected_key = (settings.admin_api_key or "").strip()
    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin API key is not configured",
        )
    provided_key = (x_konopro_admin_key or "").strip()
    if not provided_key or not compare_digest(provided_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin API key",
        )
    return provided_key
