from collections.abc import Callable

from sqlmodel import Session

from konopro_backend.config import BackendSettings
from konopro_backend.db import create_db_and_tables, create_engine_from_settings
from konopro_backend.models import AudioSession, JobStatus, ProcessingJob
from konopro_backend.processing import FingerprintSegmentationProcessor, ReferenceScoringProcessor
from konopro_backend.repositories import get_audio_session, get_next_queued_job, update_job_status

JobProcessor = Callable[[ProcessingJob, AudioSession], None]


def default_processor(settings: BackendSettings) -> JobProcessor:
    def process(job: ProcessingJob, audio_session: AudioSession) -> None:
        if job.job_type == "fingerprint_segmentation":
            FingerprintSegmentationProcessor(settings).process(job, audio_session)
            return
        if job.job_type == "reference_scoring":
            ReferenceScoringProcessor(settings).process(job, audio_session)
            return
        raise ValueError(f"Unsupported job type: {job.job_type}")

    return process


def run_next_job(
    settings: BackendSettings,
    processor: JobProcessor | None = None,
) -> ProcessingJob | None:
    engine = create_engine_from_settings(settings)
    create_db_and_tables(engine)
    active_processor = processor or default_processor(settings)

    with Session(engine) as db:
        queued_job = get_next_queued_job(db)
        if queued_job is None:
            return None

        processing_job = update_job_status(db, queued_job.id, JobStatus.processing)
        audio_session = get_audio_session(db, processing_job.session_id, include_deleted=True)
        if audio_session is None:
            return update_job_status(
                db,
                processing_job.id,
                JobStatus.failed,
                error_message="Audio session not found for job",
                increment_attempt=True,
            )

        try:
            active_processor(processing_job, audio_session)
        except Exception as exc:
            return update_job_status(
                db,
                processing_job.id,
                JobStatus.failed,
                error_message=str(exc),
                increment_attempt=True,
            )

        return update_job_status(db, processing_job.id, JobStatus.completed)
