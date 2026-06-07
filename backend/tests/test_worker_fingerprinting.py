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
    get_session_analysis,
    replace_session_analysis,
)


def _settings(tmp_path: Path) -> BackendSettings:
    return BackendSettings(
        database_url=f"sqlite:///{tmp_path / 'worker-fingerprint-test.db'}",
        storage_root=tmp_path / "storage",
        processing_root=tmp_path / "processing",
        environment="test",
    )


def _create_queued_job(settings: BackendSettings, job_type: str = "fingerprint_segmentation"):
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
        job = create_processing_job(db, audio_session.id, job_type)
        return audio_session.id, job.id


class FakeFingerprintProcessor:
    def __init__(self, settings: BackendSettings):
        self.settings = settings

    def process(self, job, audio_session) -> None:
        engine = create_engine_from_settings(self.settings)
        with Session(engine) as db:
            replace_session_analysis(
                db,
                session_id=audio_session.id,
                job_id=job.id,
                analysis_payload={
                    "provider": "fake",
                    "status": "completed",
                    "provider_status": "fake matched",
                    "windows": [
                        {
                            "provider": "fake",
                            "window_start_s": 0,
                            "window_end_s": 10,
                            "status": "matched",
                            "recognized": True,
                            "matched_title": "Fake Song",
                        }
                    ],
                    "intervals": [],
                    "weak_candidates": [],
                    "diagnostic": {
                        "provider": "fake",
                        "can_segment": False,
                        "confidence_level": "failed",
                    },
                },
            )


class ExplodingFingerprintProcessor:
    def __init__(self, _settings: BackendSettings):
        pass

    def process(self, _job, _audio_session) -> None:
        raise RuntimeError("processor exploded")


def test_run_next_job_dispatches_fingerprint_job_and_persists_analysis(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    session_id, job_id = _create_queued_job(settings)
    monkeypatch.setattr("konopro_backend.jobs.FingerprintSegmentationProcessor", FakeFingerprintProcessor)

    processed = run_next_job(settings)

    engine = create_engine_from_settings(settings)
    with Session(engine) as db:
        job = get_processing_job(db, job_id)
        audio_session = get_audio_session(db, session_id)
        analysis = get_session_analysis(db, session_id)

    assert processed is not None
    assert job is not None
    assert job.status == JobStatus.completed
    assert audio_session is not None
    assert audio_session.status == SessionStatus.processed
    assert analysis is not None
    assert analysis.windows[0].matched_title == "Fake Song"


def test_run_next_job_marks_fingerprint_processor_failure(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    session_id, job_id = _create_queued_job(settings)
    monkeypatch.setattr(
        "konopro_backend.jobs.FingerprintSegmentationProcessor",
        ExplodingFingerprintProcessor,
    )

    run_next_job(settings)

    engine = create_engine_from_settings(settings)
    with Session(engine) as db:
        job = get_processing_job(db, job_id)
        audio_session = get_audio_session(db, session_id)

    assert job is not None
    assert job.status == JobStatus.failed
    assert job.error_message == "processor exploded"
    assert audio_session is not None
    assert audio_session.status == SessionStatus.failed


def test_run_next_job_fails_unknown_job_type(tmp_path):
    settings = _settings(tmp_path)
    _session_id, job_id = _create_queued_job(settings, job_type="unknown")

    run_next_job(settings)

    engine = create_engine_from_settings(settings)
    with Session(engine) as db:
        job = get_processing_job(db, job_id)

    assert job is not None
    assert job.status == JobStatus.failed
    assert job.error_message == "Unsupported job type: unknown"
