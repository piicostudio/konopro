from pathlib import Path

from sqlmodel import Session

from konopro_backend.config import BackendSettings
from konopro_backend.db import create_db_and_tables, create_engine_from_settings
from konopro_backend.models import JobStatus, SessionStatus
from konopro_backend.repositories import (
    create_audio_session,
    create_processing_job,
    get_audio_session,
    get_next_queued_job,
    get_or_create_beta_user,
    get_processing_job,
    list_audio_sessions_for_user,
    soft_delete_audio_session,
    update_job_status,
)


def _settings(tmp_path: Path) -> BackendSettings:
    return BackendSettings(
        database_url=f"sqlite:///{tmp_path / 'konopro-test.db'}",
        storage_root=tmp_path / "storage",
        environment="test",
    )


def _session(tmp_path: Path) -> Session:
    engine = create_engine_from_settings(_settings(tmp_path))
    create_db_and_tables(engine)
    return Session(engine)


def test_beta_user_upsert_reuses_existing_user(tmp_path):
    with _session(tmp_path) as db:
        first = get_or_create_beta_user(db, external_key="tester")
        second = get_or_create_beta_user(db, external_key="tester")

    assert first.id == second.id
    assert first.external_key == "tester"
    assert first.created_at is not None


def test_session_and_initial_job_are_persisted(tmp_path):
    with _session(tmp_path) as db:
        user = get_or_create_beta_user(db, external_key="tester")
        audio_session = create_audio_session(
            db,
            user_id=user.id,
            original_filename="song.wav",
            content_type="audio/wav",
            storage_key="audio/abc.wav",
            sha256="abc123",
            size_bytes=123,
            duration_s=12.5,
        )
        job = create_processing_job(
            db,
            session_id=audio_session.id,
            job_type="fingerprint_segmentation",
        )
        loaded_session = get_audio_session(db, audio_session.id, user_id=user.id)
        loaded_job = get_processing_job(db, job.id)

    assert loaded_session is not None
    assert loaded_session.processing_job_id == job.id
    assert loaded_session.status == SessionStatus.queued
    assert loaded_job is not None
    assert loaded_job.session_id == audio_session.id
    assert loaded_job.status == JobStatus.queued


def test_list_sessions_excludes_deleted_sessions(tmp_path):
    with _session(tmp_path) as db:
        user = get_or_create_beta_user(db, external_key="tester")
        kept = create_audio_session(
            db,
            user_id=user.id,
            original_filename="kept.wav",
            content_type="audio/wav",
            storage_key="audio/kept.wav",
            sha256="kept",
            size_bytes=10,
        )
        deleted = create_audio_session(
            db,
            user_id=user.id,
            original_filename="deleted.wav",
            content_type="audio/wav",
            storage_key="audio/deleted.wav",
            sha256="deleted",
            size_bytes=10,
        )
        soft_delete_audio_session(db, deleted.id, user_id=user.id)

        sessions = list_audio_sessions_for_user(db, user.id)

    assert [session.id for session in sessions] == [kept.id]


def test_job_status_transitions_update_timestamps(tmp_path):
    with _session(tmp_path) as db:
        user = get_or_create_beta_user(db, external_key="tester")
        audio_session = create_audio_session(
            db,
            user_id=user.id,
            original_filename="song.wav",
            content_type="audio/wav",
            storage_key="audio/abc.wav",
            sha256="abc123",
            size_bytes=123,
        )
        job = create_processing_job(db, audio_session.id, "fingerprint_segmentation")

        processing = update_job_status(db, job.id, JobStatus.processing)
        completed = update_job_status(db, job.id, JobStatus.completed)

    assert processing.started_at is not None
    assert completed.finished_at is not None
    assert completed.status == JobStatus.completed


def test_next_queued_job_returns_oldest_queued_job(tmp_path):
    with _session(tmp_path) as db:
        user = get_or_create_beta_user(db, external_key="tester")
        first_session = create_audio_session(
            db,
            user_id=user.id,
            original_filename="first.wav",
            content_type="audio/wav",
            storage_key="audio/first.wav",
            sha256="first",
            size_bytes=10,
        )
        second_session = create_audio_session(
            db,
            user_id=user.id,
            original_filename="second.wav",
            content_type="audio/wav",
            storage_key="audio/second.wav",
            sha256="second",
            size_bytes=10,
        )
        first_job = create_processing_job(db, first_session.id, "fingerprint_segmentation")
        create_processing_job(db, second_session.id, "fingerprint_segmentation")

        next_job = get_next_queued_job(db)

    assert next_job is not None
    assert next_job.id == first_job.id
