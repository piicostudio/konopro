from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy.engine import Engine
from sqlmodel import SQLModel, Session, create_engine

from konopro_backend.config import BackendSettings


def _ensure_sqlite_parent(database_url: str) -> None:
    if not database_url.startswith("sqlite:///"):
        return
    raw_path = database_url.removeprefix("sqlite:///")
    if raw_path in {":memory:", ""}:
        return
    Path(raw_path).expanduser().parent.mkdir(parents=True, exist_ok=True)


def create_engine_from_settings(settings: BackendSettings) -> Engine:
    _ensure_sqlite_parent(settings.database_url)
    connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
    return create_engine(settings.database_url, connect_args=connect_args)


def create_db_and_tables(engine: Engine) -> None:
    SQLModel.metadata.create_all(engine)


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    with Session(engine) as db:
        yield db

