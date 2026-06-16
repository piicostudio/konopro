from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlmodel import Session

from konopro_backend.config import BackendSettings
from konopro_backend.dependencies import get_beta_user_key, get_db, get_settings, get_storage
from konopro_backend.models import AudioSession, ProcessingJob, ReferenceScoringRun
from konopro_backend.repositories import (
    count_pending_jobs,
    create_audio_session,
    create_processing_job,
    create_reference_scoring_run,
    get_audio_session,
    get_or_create_beta_user,
    get_processing_job,
    get_queue_status,
    get_reference_scoring_run_by_job,
    queue_status_payload,
    reference_scoring_run_payload,
)
from konopro_backend.schemas import ReferenceScoringRunResponse, ScoringJobResponse
from konopro_backend.storage import LocalAudioStorage, StoredAudio, StorageValidationError

router = APIRouter(prefix="/v1/scoring-jobs", tags=["scoring"])


@router.post("", response_model=ScoringJobResponse, status_code=status.HTTP_201_CREATED)
def create_scoring_job(
    take_audio: UploadFile = File(...),
    youtube_url: str = Form(...),
    reference_audio: UploadFile | None = File(default=None),
    beta_user_key: str = Depends(get_beta_user_key),
    db: Session = Depends(get_db),
    storage: LocalAudioStorage = Depends(get_storage),
    settings: BackendSettings = Depends(get_settings),
) -> ScoringJobResponse:
    normalized_youtube_url = youtube_url.strip()
    if not normalized_youtube_url:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="YouTube URL is required")
    if count_pending_jobs(db, "reference_scoring") >= settings.reference_scoring_max_pending_jobs:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="현재 체험 대기열이 가득 찼어요. 잠시 후 다시 시도해주세요.",
        )

    stored_take = _store_upload(take_audio, storage, settings)
    stored_reference: StoredAudio | None = None
    try:
        if reference_audio is not None and reference_audio.filename:
            stored_reference = _store_upload(reference_audio, storage, settings)
    except Exception:
        storage.delete(stored_take.storage_key)
        raise

    try:
        user = get_or_create_beta_user(db, beta_user_key)
        audio_session = create_audio_session(
            db,
            user_id=user.id,
            original_filename=take_audio.filename or "take-audio",
            content_type=stored_take.content_type,
            storage_key=stored_take.storage_key,
            sha256=stored_take.sha256,
            size_bytes=stored_take.size_bytes,
            source="web_reference_scoring",
        )
        job = create_processing_job(db, audio_session.id, "reference_scoring")
        scoring_run = create_reference_scoring_run(
            db,
            user_id=user.id,
            session_id=audio_session.id,
            job_id=job.id,
            youtube_url=normalized_youtube_url,
            reference_source="upload" if stored_reference else "youtube",
            reference_storage_key=stored_reference.storage_key if stored_reference else None,
            reference_original_filename=(
                reference_audio.filename if stored_reference and reference_audio else None
            ),
            reference_content_type=stored_reference.content_type if stored_reference else None,
        )
    except Exception:
        storage.delete(stored_take.storage_key)
        if stored_reference is not None:
            storage.delete(stored_reference.storage_key)
        raise

    refreshed_session = get_audio_session(db, audio_session.id, user_id=user.id)
    refreshed_job = get_processing_job(db, job.id)
    refreshed_run = get_reference_scoring_run_by_job(db, scoring_run.job_id, user_id=user.id)
    if refreshed_session is None or refreshed_job is None or refreshed_run is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Scoring job was created but metadata could not be loaded",
        )
    return _response(db, refreshed_session, refreshed_job, refreshed_run)


@router.get("/{job_id}", response_model=ScoringJobResponse)
def get_scoring_job(
    job_id: str,
    beta_user_key: str = Depends(get_beta_user_key),
    db: Session = Depends(get_db),
) -> ScoringJobResponse:
    audio_session, job, scoring_run = _load_owned_scoring_job(db, beta_user_key, job_id)
    return _response(db, audio_session, job, scoring_run)


@router.get("/{job_id}/result", response_model=ReferenceScoringRunResponse)
def get_scoring_result(
    job_id: str,
    beta_user_key: str = Depends(get_beta_user_key),
    db: Session = Depends(get_db),
) -> ReferenceScoringRunResponse:
    _audio_session, _job, scoring_run = _load_owned_scoring_job(db, beta_user_key, job_id)
    return ReferenceScoringRunResponse(**reference_scoring_run_payload(scoring_run))


def _store_upload(
    upload: UploadFile,
    storage: LocalAudioStorage,
    settings: BackendSettings,
) -> StoredAudio:
    try:
        return storage.save(
            upload.file,
            original_filename=upload.filename or "upload",
            content_type=upload.content_type or "",
            max_bytes=settings.max_upload_bytes,
        )
    except StorageValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


def _load_owned_scoring_job(
    db: Session,
    beta_user_key: str,
    job_id: str,
) -> tuple[AudioSession, ProcessingJob, ReferenceScoringRun]:
    user = get_or_create_beta_user(db, beta_user_key)
    job = get_processing_job(db, job_id)
    if job is None or job.job_type != "reference_scoring":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scoring job not found")
    audio_session = get_audio_session(db, job.session_id, user_id=user.id, include_deleted=True)
    scoring_run = get_reference_scoring_run_by_job(db, job.id, user_id=user.id)
    if audio_session is None or scoring_run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scoring job not found")
    return audio_session, job, scoring_run


def _response(
    db: Session,
    audio_session: AudioSession,
    job: ProcessingJob,
    scoring_run: ReferenceScoringRun,
) -> ScoringJobResponse:
    return ScoringJobResponse(
        session=audio_session,
        job=job,
        scoring_run=ReferenceScoringRunResponse(**reference_scoring_run_payload(scoring_run)),
        queue=queue_status_payload(get_queue_status(db, job)),
    )
