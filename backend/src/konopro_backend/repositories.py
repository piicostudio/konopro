from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func
from sqlmodel import Session, select

from konopro_backend.models import (
    AudioSession,
    BetaUser,
    FingerprintAnalysisRun,
    FingerprintDiagnosticRecord,
    FingerprintWindowRecord,
    JobStatus,
    ProcessingJob,
    ReferenceScoringRun,
    ReportArtifact,
    ReportArtifactVisibility,
    ReportEvent,
    ReportPriority,
    ReportRequest,
    ReportRequestStatus,
    SessionFeedback,
    SessionStatus,
    SongIntervalRecord,
    WeakCandidateRecord,
    utc_now,
)


TERMINAL_JOB_STATUSES = {JobStatus.completed, JobStatus.failed, JobStatus.cancelled}
TERMINAL_REPORT_STATUSES = {
    ReportRequestStatus.delivered,
    ReportRequestStatus.cancelled,
    ReportRequestStatus.unable_to_complete,
}
_UNSET = object()
REPORT_PRIORITY_ORDER = {
    ReportPriority.high: 0,
    ReportPriority.normal: 1,
    ReportPriority.low: 2,
}


@dataclass(frozen=True)
class SessionAnalysis:
    run: FingerprintAnalysisRun
    windows: list[FingerprintWindowRecord]
    intervals: list[SongIntervalRecord]
    weak_candidates: list[WeakCandidateRecord]
    diagnostic: FingerprintDiagnosticRecord | None


@dataclass(frozen=True)
class ReportRequestDetail:
    request: ReportRequest
    user: BetaUser | None
    audio_session: AudioSession | None
    artifacts: list[ReportArtifact]
    events: list[ReportEvent]


@dataclass(frozen=True)
class QueueStatus:
    job_id: str
    job_type: str
    status: JobStatus
    queued_ahead_count: int
    active_processing_count: int
    people_ahead_count: int
    queue_position: int | None
    pending_count: int


def _commit_refresh(db: Session, obj):
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def _json_dumps(value: Any, default: Any) -> str:
    data = default if value is None else value
    return json.dumps(data, sort_keys=True)


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _warnings(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(";") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def get_or_create_beta_user(
    db: Session,
    external_key: str,
    display_name: str | None = None,
) -> BetaUser:
    existing = db.exec(select(BetaUser).where(BetaUser.external_key == external_key)).first()
    if existing is not None:
        if display_name and existing.display_name != display_name:
            existing.display_name = display_name
            existing.updated_at = utc_now()
            return _commit_refresh(db, existing)
        return existing

    user = BetaUser(external_key=external_key, display_name=display_name)
    return _commit_refresh(db, user)


def create_audio_session(
    db: Session,
    *,
    user_id: str,
    original_filename: str,
    content_type: str,
    storage_key: str,
    sha256: str,
    size_bytes: int,
    duration_s: float | None = None,
    client_recorded_at: datetime | None = None,
    source: str | None = None,
) -> AudioSession:
    audio_session = AudioSession(
        user_id=user_id,
        original_filename=original_filename,
        content_type=content_type,
        storage_key=storage_key,
        sha256=sha256,
        size_bytes=size_bytes,
        duration_s=duration_s,
        client_recorded_at=client_recorded_at,
        source=source,
    )
    return _commit_refresh(db, audio_session)


def get_audio_session(
    db: Session,
    session_id: str,
    *,
    user_id: str | None = None,
    include_deleted: bool = False,
) -> AudioSession | None:
    query = select(AudioSession).where(AudioSession.id == session_id)
    if user_id is not None:
        query = query.where(AudioSession.user_id == user_id)
    if not include_deleted:
        query = query.where(AudioSession.deleted_at.is_(None))
    return db.exec(query).first()


def list_audio_sessions_for_user(db: Session, user_id: str) -> list[AudioSession]:
    query = (
        select(AudioSession)
        .where(AudioSession.user_id == user_id)
        .where(AudioSession.deleted_at.is_(None))
        .order_by(AudioSession.created_at.desc())
    )
    return list(db.exec(query).all())


def soft_delete_audio_session(
    db: Session,
    session_id: str,
    *,
    user_id: str | None = None,
) -> AudioSession | None:
    audio_session = get_audio_session(db, session_id, user_id=user_id)
    if audio_session is None:
        return None
    now = utc_now()
    audio_session.status = SessionStatus.deleted
    audio_session.deleted_at = now
    audio_session.updated_at = now
    if audio_session.processing_job_id:
        job = get_processing_job(db, audio_session.processing_job_id)
        if job and job.status not in TERMINAL_JOB_STATUSES:
            job.status = JobStatus.cancelled
            job.finished_at = now
            job.updated_at = now
            db.add(job)
    delete_session_analysis(db, session_id, commit=False)
    return _commit_refresh(db, audio_session)


def create_processing_job(
    db: Session,
    session_id: str,
    job_type: str,
) -> ProcessingJob:
    job = ProcessingJob(session_id=session_id, job_type=job_type)
    db.add(job)
    audio_session = db.get(AudioSession, session_id)
    if audio_session is not None:
        audio_session.processing_job_id = job.id
        audio_session.status = SessionStatus.queued
        audio_session.updated_at = utc_now()
        db.add(audio_session)
    db.commit()
    db.refresh(job)
    return job


def get_processing_job(db: Session, job_id: str) -> ProcessingJob | None:
    return db.get(ProcessingJob, job_id)


def get_next_queued_job(db: Session, job_type: str | None = None) -> ProcessingJob | None:
    query = select(ProcessingJob).where(ProcessingJob.status == JobStatus.queued)
    if job_type is not None:
        query = query.where(ProcessingJob.job_type == job_type)
    query = query.order_by(ProcessingJob.created_at.asc())
    return db.exec(query).first()


def count_pending_jobs(db: Session, job_type: str | None = None) -> int:
    query = select(func.count(ProcessingJob.id)).where(
        ProcessingJob.status.in_([JobStatus.queued, JobStatus.processing])
    )
    if job_type is not None:
        query = query.where(ProcessingJob.job_type == job_type)
    return int(db.exec(query).one())


def count_jobs_by_status(db: Session, status: JobStatus, job_type: str | None = None) -> int:
    query = select(func.count(ProcessingJob.id)).where(ProcessingJob.status == status)
    if job_type is not None:
        query = query.where(ProcessingJob.job_type == job_type)
    return int(db.exec(query).one())


def get_queue_status(db: Session, job: ProcessingJob) -> QueueStatus:
    active_processing_count = count_jobs_by_status(db, JobStatus.processing, job.job_type)
    pending_count = count_pending_jobs(db, job.job_type)

    queued_ahead_count = 0
    people_ahead_count = 0
    queue_position: int | None = None
    if job.status == JobStatus.queued:
        queued_ahead_count = _count_queued_jobs_before(db, job)
        people_ahead_count = active_processing_count + queued_ahead_count
        queue_position = people_ahead_count + 1
    elif job.status == JobStatus.processing:
        queue_position = 0

    return QueueStatus(
        job_id=job.id,
        job_type=job.job_type,
        status=job.status,
        queued_ahead_count=queued_ahead_count,
        active_processing_count=active_processing_count,
        people_ahead_count=people_ahead_count,
        queue_position=queue_position,
        pending_count=pending_count,
    )


def queue_status_payload(status: QueueStatus) -> dict[str, Any]:
    return {
        "job_id": status.job_id,
        "job_type": status.job_type,
        "status": status.status,
        "queued_ahead_count": status.queued_ahead_count,
        "active_processing_count": status.active_processing_count,
        "people_ahead_count": status.people_ahead_count,
        "queue_position": status.queue_position,
        "pending_count": status.pending_count,
    }


def _count_queued_jobs_before(db: Session, job: ProcessingJob) -> int:
    query = (
        select(func.count(ProcessingJob.id))
        .where(ProcessingJob.job_type == job.job_type)
        .where(ProcessingJob.status == JobStatus.queued)
        .where(ProcessingJob.id != job.id)
        .where(ProcessingJob.created_at < job.created_at)
    )
    return int(db.exec(query).one())


def update_job_status(
    db: Session,
    job_id: str,
    status: JobStatus,
    *,
    error_message: str | None = None,
    increment_attempt: bool = False,
) -> ProcessingJob:
    job = db.get(ProcessingJob, job_id)
    if job is None:
        raise ValueError(f"Processing job not found: {job_id}")

    now = utc_now()
    job.status = status
    job.updated_at = now
    job.error_message = error_message
    if increment_attempt:
        job.attempt_count += 1
    if status == JobStatus.processing and job.started_at is None:
        job.started_at = now
    if status in TERMINAL_JOB_STATUSES:
        job.finished_at = now

    audio_session = db.get(AudioSession, job.session_id)
    if audio_session is not None:
        audio_session.updated_at = now
        if status == JobStatus.processing:
            audio_session.status = SessionStatus.processing
        elif status == JobStatus.completed:
            audio_session.status = SessionStatus.processed
        elif status == JobStatus.failed:
            audio_session.status = SessionStatus.failed
        db.add(audio_session)

    return _commit_refresh(db, job)


def create_reference_scoring_run(
    db: Session,
    *,
    user_id: str,
    session_id: str,
    job_id: str,
    youtube_url: str,
    reference_source: str,
    reference_storage_key: str | None = None,
    reference_original_filename: str | None = None,
    reference_content_type: str | None = None,
) -> ReferenceScoringRun:
    scoring_run = ReferenceScoringRun(
        user_id=user_id,
        session_id=session_id,
        job_id=job_id,
        youtube_url=youtube_url,
        reference_source=reference_source,
        reference_storage_key=reference_storage_key,
        reference_original_filename=reference_original_filename,
        reference_content_type=reference_content_type,
    )
    return _commit_refresh(db, scoring_run)


def get_reference_scoring_run(
    db: Session,
    scoring_run_id: str,
    *,
    user_id: str | None = None,
) -> ReferenceScoringRun | None:
    query = select(ReferenceScoringRun).where(ReferenceScoringRun.id == scoring_run_id)
    if user_id is not None:
        query = query.where(ReferenceScoringRun.user_id == user_id)
    return db.exec(query).first()


def get_reference_scoring_run_by_job(
    db: Session,
    job_id: str,
    *,
    user_id: str | None = None,
) -> ReferenceScoringRun | None:
    query = select(ReferenceScoringRun).where(ReferenceScoringRun.job_id == job_id)
    if user_id is not None:
        query = query.where(ReferenceScoringRun.user_id == user_id)
    return db.exec(query).first()


def update_reference_scoring_run(
    db: Session,
    scoring_run_id: str,
    *,
    status: str,
    scores: dict[str, Any] | None | object = _UNSET,
    reference_summary: dict[str, Any] | None | object = _UNSET,
    feedback: list[str] | None | object = _UNSET,
    warnings: list[str] | None | object = _UNSET,
    error_message: str | None | object = _UNSET,
) -> ReferenceScoringRun:
    scoring_run = db.get(ReferenceScoringRun, scoring_run_id)
    if scoring_run is None:
        raise ValueError(f"Reference scoring run not found: {scoring_run_id}")

    scoring_run.status = status
    scoring_run.updated_at = utc_now()
    if scores is not _UNSET:
        scoring_run.scores_json = _json_dumps(scores, {})
    if reference_summary is not _UNSET:
        scoring_run.reference_summary_json = _json_dumps(reference_summary, {})
    if feedback is not _UNSET:
        scoring_run.feedback_json = _json_dumps(feedback, [])
    if warnings is not _UNSET:
        scoring_run.warnings_json = _json_dumps(warnings, [])
    if error_message is not _UNSET:
        scoring_run.error_message = error_message if isinstance(error_message, str) else None

    return _commit_refresh(db, scoring_run)


def reference_scoring_run_payload(run: ReferenceScoringRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "user_id": run.user_id,
        "session_id": run.session_id,
        "job_id": run.job_id,
        "youtube_url": run.youtube_url,
        "reference_source": run.reference_source,
        "reference_original_filename": run.reference_original_filename,
        "reference_content_type": run.reference_content_type,
        "status": run.status,
        "scores": _json_loads(run.scores_json, {}),
        "reference_summary": _json_loads(run.reference_summary_json, {}),
        "feedback": _json_loads(run.feedback_json, []),
        "warnings": _json_loads(run.warnings_json, []),
        "error_message": run.error_message,
        "created_at": run.created_at,
        "updated_at": run.updated_at,
    }


def replace_session_analysis(
    db: Session,
    *,
    session_id: str,
    job_id: str | None,
    analysis_payload: dict[str, Any],
) -> SessionAnalysis:
    delete_session_analysis(db, session_id, commit=False)

    params = dict(analysis_payload.get("params") or {})
    run = FingerprintAnalysisRun(
        session_id=session_id,
        job_id=job_id,
        provider=_text(analysis_payload.get("provider") or params.get("provider") or "unknown"),
        status=_text(analysis_payload.get("status") or "completed"),
        provider_status=_text(analysis_payload.get("provider_status")),
        recording_duration_s=_optional_float(analysis_payload.get("recording_duration_s")),
        window_s=_optional_float(analysis_payload.get("window_s") or params.get("window_s")),
        hop_s=_optional_float(analysis_payload.get("hop_s") or params.get("hop_s")),
        max_windows=(
            _int(analysis_payload.get("max_windows") or params.get("max_windows"))
            if analysis_payload.get("max_windows") is not None or params.get("max_windows") is not None
            else None
        ),
        use_whole=_bool(analysis_payload.get("use_whole") or params.get("use_whole")),
        params_json=_json_dumps(params, {}),
        summary_json=_json_dumps(analysis_payload.get("summary"), {}),
        interpretations_json=_json_dumps(analysis_payload.get("interpretations"), {}),
        warnings_json=_json_dumps(_warnings(analysis_payload.get("warnings")), []),
    )
    db.add(run)

    windows = [
        _window_record(run, session_id, index, row)
        for index, row in enumerate(analysis_payload.get("windows") or [], start=1)
    ]
    intervals = [
        _interval_record(run, session_id, row)
        for row in analysis_payload.get("intervals") or []
    ]
    weak_candidates = [
        _weak_candidate_record(run, session_id, row, source="segmentation")
        for row in analysis_payload.get("weak_candidates") or []
    ]

    diagnostic_data = dict(analysis_payload.get("diagnostic") or {})
    for candidate in diagnostic_data.get("weak_candidates") or []:
        weak_candidates.append(
            _weak_candidate_record(
                run,
                session_id,
                candidate,
                source="diagnostic",
                fallback_index=len(weak_candidates) + 1,
            )
        )

    diagnostic = _diagnostic_record(run, session_id, diagnostic_data, analysis_payload)
    for record in [*windows, *intervals, *weak_candidates, diagnostic]:
        if record is not None:
            db.add(record)

    db.commit()
    db.refresh(run)
    for record in [*windows, *intervals, *weak_candidates]:
        db.refresh(record)
    if diagnostic is not None:
        db.refresh(diagnostic)

    return SessionAnalysis(
        run=run,
        windows=windows,
        intervals=intervals,
        weak_candidates=weak_candidates,
        diagnostic=diagnostic,
    )


def get_session_analysis(db: Session, session_id: str) -> SessionAnalysis | None:
    run = db.exec(
        select(FingerprintAnalysisRun)
        .where(FingerprintAnalysisRun.session_id == session_id)
        .order_by(FingerprintAnalysisRun.created_at.desc())
    ).first()
    if run is None:
        return None

    windows = list(
        db.exec(
            select(FingerprintWindowRecord)
            .where(FingerprintWindowRecord.run_id == run.id)
            .order_by(FingerprintWindowRecord.window_index.asc())
        ).all()
    )
    intervals = list(
        db.exec(
            select(SongIntervalRecord)
            .where(SongIntervalRecord.run_id == run.id)
            .order_by(SongIntervalRecord.interval_index.asc())
        ).all()
    )
    weak_candidates = list(
        db.exec(
            select(WeakCandidateRecord)
            .where(WeakCandidateRecord.run_id == run.id)
            .order_by(WeakCandidateRecord.candidate_index.asc())
        ).all()
    )
    diagnostic = db.exec(
        select(FingerprintDiagnosticRecord).where(FingerprintDiagnosticRecord.run_id == run.id)
    ).first()
    return SessionAnalysis(
        run=run,
        windows=windows,
        intervals=intervals,
        weak_candidates=weak_candidates,
        diagnostic=diagnostic,
    )


def create_session_feedback(
    db: Session,
    *,
    user_id: str,
    session_id: str,
    helped_review: str,
    rating: int,
    answer_text: str | None,
    context: str,
) -> SessionFeedback:
    feedback = SessionFeedback(
        user_id=user_id,
        session_id=session_id,
        helped_review=helped_review,
        rating=rating,
        answer_text=answer_text,
        context=context,
    )
    return _commit_refresh(db, feedback)


def delete_session_analysis(db: Session, session_id: str, *, commit: bool = True) -> int:
    deleted = 0
    for model in (
        FingerprintDiagnosticRecord,
        WeakCandidateRecord,
        SongIntervalRecord,
        FingerprintWindowRecord,
        FingerprintAnalysisRun,
    ):
        records = list(db.exec(select(model).where(model.session_id == session_id)).all())
        for record in records:
            db.delete(record)
            deleted += 1
    if commit:
        db.commit()
    return deleted


def create_report_request(
    db: Session,
    *,
    user_id: str,
    session_id: str,
    request_type: str,
    priority: ReportPriority,
    target_turnaround_hours: int,
    due_at: datetime | None,
    user_notes: str | None = None,
    actor_key: str | None = None,
) -> ReportRequest:
    existing = get_active_report_request_for_session(db, session_id)
    if existing is not None:
        raise ValueError("An active report request already exists for this session")

    report_request = ReportRequest(
        user_id=user_id,
        session_id=session_id,
        request_type=request_type,
        priority=priority,
        target_turnaround_hours=int(target_turnaround_hours),
        due_at=due_at,
        user_notes=user_notes,
    )
    db.add(report_request)
    db.flush()
    _add_report_event_no_commit(
        db,
        report_request_id=report_request.id,
        actor_type="user",
        actor_key=actor_key,
        event_type="created",
        to_status=report_request.status.value,
        message="Report request created.",
        data={"request_type": request_type, "priority": priority.value},
    )
    db.commit()
    db.refresh(report_request)
    return report_request


def get_report_request(
    db: Session,
    request_id: str,
    *,
    user_id: str | None = None,
) -> ReportRequest | None:
    query = select(ReportRequest).where(ReportRequest.id == request_id)
    if user_id is not None:
        query = query.where(ReportRequest.user_id == user_id)
    return db.exec(query).first()


def get_report_request_detail(
    db: Session,
    request_id: str,
    *,
    user_id: str | None = None,
    artifact_visibility: ReportArtifactVisibility | None = None,
) -> ReportRequestDetail | None:
    report_request = get_report_request(db, request_id, user_id=user_id)
    if report_request is None:
        return None
    return ReportRequestDetail(
        request=report_request,
        user=db.get(BetaUser, report_request.user_id),
        audio_session=db.get(AudioSession, report_request.session_id),
        artifacts=list_report_artifacts(
            db,
            report_request.id,
            visibility=artifact_visibility,
        ),
        events=list_report_events(db, report_request.id),
    )


def get_active_report_request_for_session(
    db: Session,
    session_id: str,
) -> ReportRequest | None:
    requests = list(
        db.exec(select(ReportRequest).where(ReportRequest.session_id == session_id)).all()
    )
    for report_request in requests:
        if report_request.status not in TERMINAL_REPORT_STATUSES:
            return report_request
    return None


def list_report_requests_for_user(db: Session, user_id: str) -> list[ReportRequest]:
    requests = list(
        db.exec(select(ReportRequest).where(ReportRequest.user_id == user_id)).all()
    )
    return _sort_report_requests(requests)


def list_report_queue(
    db: Session,
    *,
    status: ReportRequestStatus | None = None,
    priority: ReportPriority | None = None,
) -> list[ReportRequest]:
    query = select(ReportRequest)
    if status is not None:
        query = query.where(ReportRequest.status == status)
    if priority is not None:
        query = query.where(ReportRequest.priority == priority)
    return _sort_report_requests(list(db.exec(query).all()))


def update_report_request(
    db: Session,
    request_id: str,
    *,
    status: ReportRequestStatus | None = None,
    priority: ReportPriority | None = None,
    due_at: datetime | None | object = None,
    admin_notes: str | None | object = None,
    blocker_reason: str | None | object = None,
    actor_type: str = "admin",
    actor_key: str | None = None,
    message: str | None = None,
) -> ReportRequest:
    report_request = db.get(ReportRequest, request_id)
    if report_request is None:
        raise ValueError(f"Report request not found: {request_id}")

    now = utc_now()
    previous_status = report_request.status
    changed: dict[str, Any] = {}

    if status is not None and status != report_request.status:
        report_request.status = status
        changed["status"] = status.value
        if status == ReportRequestStatus.delivered:
            report_request.delivered_at = now
        if status == ReportRequestStatus.cancelled:
            report_request.cancelled_at = now
    if priority is not None and priority != report_request.priority:
        report_request.priority = priority
        changed["priority"] = priority.value
    if due_at is not None:
        report_request.due_at = due_at if isinstance(due_at, datetime) else None
        changed["due_at"] = report_request.due_at.isoformat() if report_request.due_at else None
    if admin_notes is not None:
        report_request.admin_notes = admin_notes if isinstance(admin_notes, str) else None
        changed["admin_notes"] = report_request.admin_notes
    if blocker_reason is not None:
        report_request.blocker_reason = (
            blocker_reason if isinstance(blocker_reason, str) else None
        )
        changed["blocker_reason"] = report_request.blocker_reason

    report_request.updated_at = now
    db.add(report_request)
    if changed:
        _add_report_event_no_commit(
            db,
            report_request_id=report_request.id,
            actor_type=actor_type,
            actor_key=actor_key,
            event_type="updated",
            from_status=previous_status.value,
            to_status=report_request.status.value,
            message=message or "Report request updated.",
            data=changed,
        )
    db.commit()
    db.refresh(report_request)
    return report_request


def add_report_event(
    db: Session,
    *,
    report_request_id: str,
    actor_type: str,
    actor_key: str | None,
    event_type: str,
    from_status: str | None = None,
    to_status: str | None = None,
    message: str | None = None,
    data: dict[str, Any] | None = None,
) -> ReportEvent:
    event = _add_report_event_no_commit(
        db,
        report_request_id=report_request_id,
        actor_type=actor_type,
        actor_key=actor_key,
        event_type=event_type,
        from_status=from_status,
        to_status=to_status,
        message=message,
        data=data,
    )
    db.commit()
    db.refresh(event)
    return event


def list_report_events(db: Session, report_request_id: str) -> list[ReportEvent]:
    query = (
        select(ReportEvent)
        .where(ReportEvent.report_request_id == report_request_id)
        .order_by(ReportEvent.created_at.asc())
    )
    return list(db.exec(query).all())


def add_report_artifact(
    db: Session,
    *,
    report_request_id: str,
    session_id: str,
    artifact_type: str,
    title: str,
    body_text: str | None = None,
    storage_key: str | None = None,
    content_type: str | None = None,
    filename: str | None = None,
    visibility: ReportArtifactVisibility = ReportArtifactVisibility.internal,
    metadata: dict[str, Any] | None = None,
    actor_key: str | None = None,
) -> ReportArtifact:
    now = utc_now()
    artifact = ReportArtifact(
        report_request_id=report_request_id,
        session_id=session_id,
        artifact_type=artifact_type,
        title=title,
        body_text=body_text,
        storage_key=storage_key,
        content_type=content_type,
        filename=filename,
        visibility=visibility,
        metadata_json=_json_dumps(metadata, {}),
        published_at=now if visibility == ReportArtifactVisibility.user_visible else None,
    )
    db.add(artifact)
    db.flush()
    _add_report_event_no_commit(
        db,
        report_request_id=report_request_id,
        actor_type="admin",
        actor_key=actor_key,
        event_type="artifact_added",
        message=f"Artifact added: {title}",
        data={
            "artifact_id": artifact.id,
            "artifact_type": artifact_type,
            "visibility": visibility.value,
        },
    )
    db.commit()
    db.refresh(artifact)
    return artifact


def update_report_artifact(
    db: Session,
    artifact_id: str,
    *,
    title: str | None = None,
    body_text: str | None | object = None,
    visibility: ReportArtifactVisibility | None = None,
    actor_key: str | None = None,
) -> ReportArtifact:
    artifact = db.get(ReportArtifact, artifact_id)
    if artifact is None:
        raise ValueError(f"Report artifact not found: {artifact_id}")

    changed: dict[str, Any] = {}
    if title is not None:
        artifact.title = title
        changed["title"] = title
    if body_text is not None:
        artifact.body_text = body_text if isinstance(body_text, str) else None
        changed["body_text"] = artifact.body_text
    if visibility is not None and visibility != artifact.visibility:
        artifact.visibility = visibility
        artifact.published_at = (
            artifact.published_at or utc_now()
            if visibility == ReportArtifactVisibility.user_visible
            else None
        )
        changed["visibility"] = visibility.value
    artifact.updated_at = utc_now()
    db.add(artifact)
    if changed:
        _add_report_event_no_commit(
            db,
            report_request_id=artifact.report_request_id,
            actor_type="admin",
            actor_key=actor_key,
            event_type="artifact_updated",
            message=f"Artifact updated: {artifact.title}",
            data={"artifact_id": artifact.id, **changed},
        )
    db.commit()
    db.refresh(artifact)
    return artifact


def get_report_artifact(db: Session, artifact_id: str) -> ReportArtifact | None:
    return db.get(ReportArtifact, artifact_id)


def list_report_artifacts(
    db: Session,
    report_request_id: str,
    *,
    visibility: ReportArtifactVisibility | None = None,
    artifact_type: str | None = None,
) -> list[ReportArtifact]:
    query = select(ReportArtifact).where(ReportArtifact.report_request_id == report_request_id)
    if visibility is not None:
        query = query.where(ReportArtifact.visibility == visibility)
    if artifact_type is not None:
        query = query.where(ReportArtifact.artifact_type == artifact_type)
    query = query.order_by(ReportArtifact.created_at.asc())
    return list(db.exec(query).all())


def _sort_report_requests(requests: list[ReportRequest]) -> list[ReportRequest]:
    return sorted(
        requests,
        key=lambda request: (
            REPORT_PRIORITY_ORDER.get(request.priority, 9),
            request.due_at or datetime.max.replace(tzinfo=timezone.utc),
            request.created_at,
        ),
    )


def _add_report_event_no_commit(
    db: Session,
    *,
    report_request_id: str,
    actor_type: str,
    actor_key: str | None,
    event_type: str,
    from_status: str | None = None,
    to_status: str | None = None,
    message: str | None = None,
    data: dict[str, Any] | None = None,
) -> ReportEvent:
    event = ReportEvent(
        report_request_id=report_request_id,
        actor_type=actor_type,
        actor_key=actor_key,
        event_type=event_type,
        from_status=from_status,
        to_status=to_status,
        message=message,
        data_json=_json_dumps(data, {}),
    )
    db.add(event)
    return event


def _window_record(
    run: FingerprintAnalysisRun,
    session_id: str,
    index: int,
    row: dict[str, Any],
) -> FingerprintWindowRecord:
    raw = dict(row.get("raw") or row.get("row") or row)
    return FingerprintWindowRecord(
        run_id=run.id,
        session_id=session_id,
        window_index=index,
        provider=_text(row.get("provider") or run.provider),
        window_start_s=_float(row.get("window_start_s") or row.get("start_s")),
        window_end_s=_float(row.get("window_end_s") or row.get("end_s")),
        status=_text(row.get("status") or "unknown"),
        recognized=_bool(row.get("recognized")),
        matched_title=_text(row.get("matched_title") or row.get("title") or row.get("song")),
        matched_artist=_text(row.get("matched_artist") or row.get("artist")),
        identity_key=_text(row.get("identity_key")),
        isrc=_text(row.get("isrc")),
        confidence=_optional_float(row.get("confidence")),
        audio_file=_text(row.get("audio_file")),
        error=_text(row.get("error")),
        raw_json=_json_dumps(raw, {}),
    )


def _interval_record(
    run: FingerprintAnalysisRun,
    session_id: str,
    row: dict[str, Any],
) -> SongIntervalRecord:
    start_s = _float(row.get("start_s"))
    end_s = _float(row.get("end_s"))
    return SongIntervalRecord(
        run_id=run.id,
        session_id=session_id,
        interval_index=_int(row.get("index") or row.get("interval_index"), 1),
        song=_text(row.get("song") or row.get("title")),
        artist=_text(row.get("artist")),
        identity_key=_text(row.get("identity_key")),
        start_s=start_s,
        end_s=end_s,
        duration_s=_float(row.get("duration_s"), max(0.0, end_s - start_s)),
        confidence_score=_float(row.get("confidence") or row.get("confidence_score")),
        confidence_level=_text(row.get("confidence_level") or "unknown"),
        recognized_window_count=_int(row.get("recognized_windows")),
        total_window_count=_int(row.get("total_windows")),
        gap_window_count=_int(row.get("gap_windows")),
        conflict_window_count=_int(row.get("conflict_windows")),
        provider_confidence=_float(row.get("provider_confidence")),
        warnings_json=_json_dumps(_warnings(row.get("warnings")), []),
        raw_json=_json_dumps(row, {}),
    )


def _weak_candidate_record(
    run: FingerprintAnalysisRun,
    session_id: str,
    row: dict[str, Any],
    *,
    source: str,
    fallback_index: int = 1,
) -> WeakCandidateRecord:
    start_s = _float(row.get("start_s"))
    end_s = _float(row.get("end_s"))
    return WeakCandidateRecord(
        run_id=run.id,
        session_id=session_id,
        candidate_index=_int(row.get("index"), fallback_index),
        source=source,
        song=_text(row.get("song") or row.get("title")),
        artist=_text(row.get("artist")),
        identity_key=_text(row.get("identity_key")),
        start_s=start_s,
        end_s=end_s,
        duration_s=_float(row.get("duration_s"), max(0.0, end_s - start_s)),
        recognized_window_count=_int(row.get("recognized_windows") or row.get("match_count")),
        total_window_count=_int(row.get("total_windows") or row.get("match_count")),
        provider_confidence=_optional_float(row.get("provider_confidence") or row.get("confidence")),
        reason=_text(row.get("reason")),
        recovery_start_s=_optional_float(row.get("recovery_start_s")),
        recovery_end_s=_optional_float(row.get("recovery_end_s")),
        warnings_json=_json_dumps(_warnings(row.get("warnings")), []),
        raw_json=_json_dumps(row, {}),
    )


def _diagnostic_record(
    run: FingerprintAnalysisRun,
    session_id: str,
    diagnostic_data: dict[str, Any],
    analysis_payload: dict[str, Any],
) -> FingerprintDiagnosticRecord | None:
    if not diagnostic_data:
        return None
    return FingerprintDiagnosticRecord(
        run_id=run.id,
        session_id=session_id,
        provider=_text(diagnostic_data.get("provider") or run.provider),
        can_segment=_bool(diagnostic_data.get("can_segment")),
        confidence_level=_text(diagnostic_data.get("confidence_level") or "failed"),
        profile_json=_json_dumps(diagnostic_data.get("profile"), {}),
        flags_json=_json_dumps(diagnostic_data.get("flags"), []),
        recommendations_json=_json_dumps(diagnostic_data.get("recommendations"), []),
        recovery_sweeps_json=_json_dumps(analysis_payload.get("recovery_sweeps"), []),
        raw_json=_json_dumps(diagnostic_data, {}),
    )
