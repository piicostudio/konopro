from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from konopro_backend.dependencies import get_beta_user_key, get_db
from konopro_backend.repositories import get_audio_session, get_or_create_beta_user, get_processing_job
from konopro_backend.schemas import ProcessingJobResponse

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=ProcessingJobResponse)
def get_job(
    job_id: str,
    beta_user_key: str = Depends(get_beta_user_key),
    db: Session = Depends(get_db),
) -> ProcessingJobResponse:
    user = get_or_create_beta_user(db, beta_user_key)
    job = get_processing_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    audio_session = get_audio_session(db, job.session_id, user_id=user.id, include_deleted=True)
    if audio_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job

