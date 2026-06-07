from pathlib import Path

from sqlmodel import Session

from konopro_backend.config import BackendSettings
from konopro_backend.db import create_db_and_tables, create_engine_from_settings
from konopro_backend.jobs import run_next_job
from konopro_backend.models import JobStatus, SessionStatus
from konopro_backend.repositories import (
    create_audio_session,
    create_processing_job,
    get_audio_session,
    get_or_create_beta_user,
    get_processing_job,
)
from konopro_backend.worker import main


def _settings(tmp_path: Path) -> BackendSettings:
    return BackendSettings(
        database_url=f"sqlite:///{tmp_path / 'jobs-test.db'}",
        storage_root=tmp_path / "storage",
        environment="test",
    )


def _create_queued_job(settings: BackendSettings) -> tuple[str, str]:
    engine = create_engine_from_settings(settings)
    create_db_and_tables(engine)
    with Session(engine) as db:
        user = get_or_create_beta_user(db, "tester")
        audio_session = create_audio_session(
            db,
            user_id=user.id,
            original_filename="song.wav",
            content_type="audio/wav",
            storage_key="audio/song.wav",
            sha256="abc",
            size_bytes=3,
        )
        job = create_processing_job(db, audio_session.id, "fingerprint_segmentation")
        return audio_session.id, job.id


def test_run_next_job_returns_none_when_queue_empty(tmp_path):
    assert run_next_job(_settings(tmp_path)) is None


def test_run_next_job_marks_job_and_session_completed(tmp_path):
    settings = _settings(tmp_path)
    session_id, job_id = _create_queued_job(settings)
    seen: dict[str, str] = {}

    def processor(job, audio_session):
        seen["job_id"] = job.id
        seen["session_id"] = audio_session.id

    processed = run_next_job(settings, processor=processor)

    engine = create_engine_from_settings(settings)
    with Session(engine) as db:
        job = get_processing_job(db, job_id)
        audio_session = get_audio_session(db, session_id)

    assert processed is not None
    assert seen == {"job_id": job_id, "session_id": session_id}
    assert job is not None
    assert job.status == JobStatus.completed
    assert job.started_at is not None
    assert job.finished_at is not None
    assert audio_session is not None
    assert audio_session.status == SessionStatus.processed


def test_run_next_job_marks_failure_and_stores_error(tmp_path):
    settings = _settings(tmp_path)
    session_id, job_id = _create_queued_job(settings)

    def processor(_job, _audio_session):
        raise RuntimeError("provider exploded")

    processed = run_next_job(settings, processor=processor)

    engine = create_engine_from_settings(settings)
    with Session(engine) as db:
        job = get_processing_job(db, job_id)
        audio_session = get_audio_session(db, session_id)

    assert processed is not None
    assert job is not None
    assert job.status == JobStatus.failed
    assert job.attempt_count == 1
    assert job.error_message == "provider exploded"
    assert audio_session is not None
    assert audio_session.status == SessionStatus.failed


def test_worker_once_exits_zero_on_empty_queue(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KONOPRO_DATABASE_URL", f"sqlite:///{tmp_path / 'worker.db'}")
    monkeypatch.setenv("KONOPRO_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("KONOPRO_ENVIRONMENT", "test")

    exit_code = main(["--once"])

    assert exit_code == 0
    assert "No queued jobs found." in capsys.readouterr().out
