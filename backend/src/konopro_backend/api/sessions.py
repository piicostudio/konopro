from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from sqlmodel import Session

from konopro_backend.config import BackendSettings
from konopro_backend.dependencies import get_beta_user_key, get_db, get_settings, get_storage
from konopro_backend.repositories import (
    create_audio_session,
    create_processing_job,
    get_audio_session,
    get_or_create_beta_user,
    get_processing_job,
    list_audio_sessions_for_user,
    soft_delete_audio_session,
)
from konopro_backend.schemas import AudioSessionResponse, UploadSessionResponse
from konopro_backend.storage import LocalAudioStorage, StorageValidationError

router = APIRouter(prefix="/v1/sessions", tags=["sessions"])


@router.post("", response_model=UploadSessionResponse, status_code=status.HTTP_201_CREATED)
def upload_session(
    file: UploadFile = File(...),
    client_duration_s: float | None = Form(default=None),
    client_recorded_at: datetime | None = Form(default=None),
    source: str | None = Form(default=None),
    beta_user_key: str = Depends(get_beta_user_key),
    db: Session = Depends(get_db),
    storage: LocalAudioStorage = Depends(get_storage),
    settings: BackendSettings = Depends(get_settings),
) -> UploadSessionResponse:
    try:
        stored = storage.save(
            file.file,
            original_filename=file.filename or "upload",
            content_type=file.content_type or "",
            max_bytes=settings.max_upload_bytes,
        )
    except StorageValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    try:
        user = get_or_create_beta_user(db, beta_user_key)
        audio_session = create_audio_session(
            db,
            user_id=user.id,
            original_filename=file.filename or "upload",
            content_type=stored.content_type,
            storage_key=stored.storage_key,
            sha256=stored.sha256,
            size_bytes=stored.size_bytes,
            duration_s=client_duration_s,
            client_recorded_at=client_recorded_at,
            source=source,
        )
        job = create_processing_job(
            db,
            session_id=audio_session.id,
            job_type="fingerprint_segmentation",
        )
    except Exception:
        storage.delete(stored.storage_key)
        raise

    refreshed_session = get_audio_session(db, audio_session.id, user_id=user.id)
    refreshed_job = get_processing_job(db, job.id)
    if refreshed_session is None or refreshed_job is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Upload was stored but session metadata could not be loaded",
        )
    return UploadSessionResponse(session=refreshed_session, job=refreshed_job)


@router.get("", response_model=list[AudioSessionResponse])
def list_sessions(
    beta_user_key: str = Depends(get_beta_user_key),
    db: Session = Depends(get_db),
) -> list[AudioSessionResponse]:
    user = get_or_create_beta_user(db, beta_user_key)
    return list_audio_sessions_for_user(db, user.id)


@router.get("/{session_id}", response_model=AudioSessionResponse)
def get_session(
    session_id: str,
    beta_user_key: str = Depends(get_beta_user_key),
    db: Session = Depends(get_db),
) -> AudioSessionResponse:
    user = get_or_create_beta_user(db, beta_user_key)
    audio_session = get_audio_session(db, session_id, user_id=user.id)
    if audio_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return audio_session


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(
    session_id: str,
    beta_user_key: str = Depends(get_beta_user_key),
    db: Session = Depends(get_db),
    storage: LocalAudioStorage = Depends(get_storage),
) -> Response:
    user = get_or_create_beta_user(db, beta_user_key)
    audio_session = get_audio_session(db, session_id, user_id=user.id)
    if audio_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    storage.delete(audio_session.storage_key)
    soft_delete_audio_session(db, session_id, user_id=user.id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

