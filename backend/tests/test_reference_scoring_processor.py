from pathlib import Path

from sqlmodel import Session

from konopro_backend.config import BackendSettings
from konopro_backend.db import create_db_and_tables, create_engine_from_settings
from konopro_backend.models import JobStatus, SessionStatus
from konopro_backend.processing.scoring import ReferenceScoringProcessor
from konopro_backend.repositories import (
    create_audio_session,
    create_processing_job,
    create_reference_scoring_run,
    get_audio_session,
    get_or_create_beta_user,
    get_processing_job,
    get_reference_scoring_run_by_job,
    reference_scoring_run_payload,
)
from konopro_backend.storage import LocalAudioStorage
from konopro_research.demo_data import ensure_demo_data


def _settings(tmp_path: Path) -> BackendSettings:
    return BackendSettings(
        database_url=f"sqlite:///{tmp_path / 'reference-scoring-test.db'}",
        storage_root=tmp_path / "storage",
        processing_root=tmp_path / "processing",
        environment="test",
    )


def _create_scoring_job_with_reference(settings: BackendSettings, tmp_path: Path) -> tuple[str, str]:
    demo = ensure_demo_data(tmp_path / "demo")
    storage = LocalAudioStorage(settings.storage_root)
    with demo["current"].open("rb") as fileobj:
        stored_take = storage.save(fileobj, "current_take.wav", "audio/wav", settings.max_upload_bytes)
    with demo["reference"].open("rb") as fileobj:
        stored_reference = storage.save(
            fileobj,
            "reference_melody.wav",
            "audio/wav",
            settings.max_upload_bytes,
        )

    engine = create_engine_from_settings(settings)
    create_db_and_tables(engine)
    with Session(engine) as db:
        user = get_or_create_beta_user(db, "tester")
        audio_session = create_audio_session(
            db,
            user_id=user.id,
            original_filename="current_take.wav",
            content_type=stored_take.content_type,
            storage_key=stored_take.storage_key,
            sha256=stored_take.sha256,
            size_bytes=stored_take.size_bytes,
            source="web_reference_scoring",
        )
        job = create_processing_job(db, audio_session.id, "reference_scoring")
        create_reference_scoring_run(
            db,
            user_id=user.id,
            session_id=audio_session.id,
            job_id=job.id,
            youtube_url="https://www.youtube.com/watch?v=demo",
            reference_source="upload",
            reference_storage_key=stored_reference.storage_key,
            reference_original_filename="reference_melody.wav",
            reference_content_type=stored_reference.content_type,
        )
        return audio_session.id, job.id


def test_reference_scoring_processor_persists_scores_with_uploaded_reference(tmp_path):
    settings = _settings(tmp_path)
    session_id, job_id = _create_scoring_job_with_reference(settings, tmp_path)

    engine = create_engine_from_settings(settings)
    with Session(engine) as db:
        audio_session = get_audio_session(db, session_id)
        job = get_processing_job(db, job_id)
        assert audio_session is not None
        assert job is not None
        processor = ReferenceScoringProcessor(settings)
        processor.process(job, audio_session, db=db)

        scoring_run = get_reference_scoring_run_by_job(db, job_id)

    assert scoring_run is not None
    payload = reference_scoring_run_payload(scoring_run)
    assert payload["status"] == "completed"
    assert payload["scores"]["overall_score"] > 0
    assert payload["scores"]["pitch_accuracy_score"] > 0
    assert payload["reference_summary"]["source"] == "upload"
    assert payload["feedback"]


class FakeReferenceScoringProcessor:
    def __init__(self, _settings: BackendSettings):
        pass

    def process(self, _job, _audio_session) -> None:
        return None


def test_worker_dispatches_reference_scoring_jobs(tmp_path, monkeypatch):
    from konopro_backend.jobs import run_next_job

    settings = _settings(tmp_path)
    session_id, job_id = _create_scoring_job_with_reference(settings, tmp_path)
    monkeypatch.setattr(
        "konopro_backend.jobs.ReferenceScoringProcessor",
        FakeReferenceScoringProcessor,
    )

    processed = run_next_job(settings)

    engine = create_engine_from_settings(settings)
    with Session(engine) as db:
        job = get_processing_job(db, job_id)
        audio_session = get_audio_session(db, session_id)

    assert processed is not None
    assert job is not None
    assert job.status == JobStatus.completed
    assert audio_session is not None
    assert audio_session.status == SessionStatus.processed
