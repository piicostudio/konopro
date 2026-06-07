from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid4())


class SessionStatus(str, Enum):
    uploaded = "uploaded"
    queued = "queued"
    processing = "processing"
    processed = "processed"
    failed = "failed"
    deleted = "deleted"


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class ReportRequestStatus(str, Enum):
    requested = "requested"
    triaged = "triaged"
    in_progress = "in_progress"
    blocked = "blocked"
    ready = "ready"
    delivered = "delivered"
    cancelled = "cancelled"
    unable_to_complete = "unable_to_complete"


class ReportPriority(str, Enum):
    low = "low"
    normal = "normal"
    high = "high"


class ReportArtifactVisibility(str, Enum):
    internal = "internal"
    user_visible = "user_visible"


class BetaUser(SQLModel, table=True):
    __tablename__ = "beta_users"

    id: str = Field(default_factory=new_id, primary_key=True)
    external_key: str = Field(index=True, unique=True)
    display_name: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class AudioSession(SQLModel, table=True):
    __tablename__ = "audio_sessions"

    id: str = Field(default_factory=new_id, primary_key=True)
    user_id: str = Field(foreign_key="beta_users.id", index=True)
    original_filename: str
    content_type: str
    storage_key: str = Field(index=True)
    sha256: str
    size_bytes: int
    duration_s: float | None = None
    client_recorded_at: datetime | None = None
    source: str | None = None
    status: SessionStatus = Field(default=SessionStatus.uploaded, index=True)
    processing_job_id: str | None = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    deleted_at: datetime | None = Field(default=None, index=True)


class ProcessingJob(SQLModel, table=True):
    __tablename__ = "processing_jobs"

    id: str = Field(default_factory=new_id, primary_key=True)
    session_id: str = Field(foreign_key="audio_sessions.id", index=True)
    job_type: str = Field(index=True)
    status: JobStatus = Field(default=JobStatus.queued, index=True)
    attempt_count: int = 0
    error_message: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class FingerprintAnalysisRun(SQLModel, table=True):
    __tablename__ = "fingerprint_analysis_runs"

    id: str = Field(default_factory=new_id, primary_key=True)
    session_id: str = Field(foreign_key="audio_sessions.id", index=True)
    job_id: str | None = Field(default=None, foreign_key="processing_jobs.id", index=True)
    provider: str = Field(index=True)
    status: str = Field(default="completed", index=True)
    provider_status: str | None = None
    recording_duration_s: float | None = None
    window_s: float | None = None
    hop_s: float | None = None
    max_windows: int | None = None
    use_whole: bool = False
    params_json: str = "{}"
    summary_json: str = "{}"
    interpretations_json: str = "{}"
    warnings_json: str = "[]"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class FingerprintWindowRecord(SQLModel, table=True):
    __tablename__ = "fingerprint_window_records"

    id: str = Field(default_factory=new_id, primary_key=True)
    run_id: str = Field(foreign_key="fingerprint_analysis_runs.id", index=True)
    session_id: str = Field(foreign_key="audio_sessions.id", index=True)
    window_index: int = Field(index=True)
    provider: str = Field(index=True)
    window_start_s: float
    window_end_s: float
    status: str = Field(index=True)
    recognized: bool = Field(default=False, index=True)
    matched_title: str = ""
    matched_artist: str = ""
    identity_key: str = ""
    isrc: str = ""
    confidence: float | None = None
    audio_file: str = ""
    error: str = ""
    raw_json: str = "{}"
    created_at: datetime = Field(default_factory=utc_now)


class SongIntervalRecord(SQLModel, table=True):
    __tablename__ = "song_interval_records"

    id: str = Field(default_factory=new_id, primary_key=True)
    run_id: str = Field(foreign_key="fingerprint_analysis_runs.id", index=True)
    session_id: str = Field(foreign_key="audio_sessions.id", index=True)
    interval_index: int = Field(index=True)
    song: str = ""
    artist: str = ""
    identity_key: str = Field(default="", index=True)
    start_s: float
    end_s: float
    duration_s: float
    confidence_score: float
    confidence_level: str = Field(index=True)
    recognized_window_count: int = 0
    total_window_count: int = 0
    gap_window_count: int = 0
    conflict_window_count: int = 0
    provider_confidence: float = 0.0
    warnings_json: str = "[]"
    raw_json: str = "{}"
    created_at: datetime = Field(default_factory=utc_now)


class WeakCandidateRecord(SQLModel, table=True):
    __tablename__ = "weak_candidate_records"

    id: str = Field(default_factory=new_id, primary_key=True)
    run_id: str = Field(foreign_key="fingerprint_analysis_runs.id", index=True)
    session_id: str = Field(foreign_key="audio_sessions.id", index=True)
    candidate_index: int = Field(index=True)
    source: str = Field(default="segmentation", index=True)
    song: str = ""
    artist: str = ""
    identity_key: str = Field(default="", index=True)
    start_s: float
    end_s: float
    duration_s: float
    recognized_window_count: int = 0
    total_window_count: int = 0
    provider_confidence: float | None = None
    reason: str = ""
    recovery_start_s: float | None = None
    recovery_end_s: float | None = None
    warnings_json: str = "[]"
    raw_json: str = "{}"
    created_at: datetime = Field(default_factory=utc_now)


class FingerprintDiagnosticRecord(SQLModel, table=True):
    __tablename__ = "fingerprint_diagnostic_records"

    id: str = Field(default_factory=new_id, primary_key=True)
    run_id: str = Field(foreign_key="fingerprint_analysis_runs.id", index=True)
    session_id: str = Field(foreign_key="audio_sessions.id", index=True)
    provider: str = Field(index=True)
    can_segment: bool = Field(default=False, index=True)
    confidence_level: str = Field(default="failed", index=True)
    profile_json: str = "{}"
    flags_json: str = "[]"
    recommendations_json: str = "[]"
    recovery_sweeps_json: str = "[]"
    raw_json: str = "{}"
    created_at: datetime = Field(default_factory=utc_now)


class ReportRequest(SQLModel, table=True):
    __tablename__ = "report_requests"

    id: str = Field(default_factory=new_id, primary_key=True)
    user_id: str = Field(foreign_key="beta_users.id", index=True)
    session_id: str = Field(foreign_key="audio_sessions.id", index=True)
    status: ReportRequestStatus = Field(default=ReportRequestStatus.requested, index=True)
    priority: ReportPriority = Field(default=ReportPriority.low, index=True)
    request_type: str = Field(default="free", index=True)
    target_turnaround_hours: int = 72
    due_at: datetime | None = Field(default=None, index=True)
    user_notes: str | None = None
    admin_notes: str | None = None
    blocker_reason: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    delivered_at: datetime | None = None
    cancelled_at: datetime | None = None


class ReportEvent(SQLModel, table=True):
    __tablename__ = "report_events"

    id: str = Field(default_factory=new_id, primary_key=True)
    report_request_id: str = Field(foreign_key="report_requests.id", index=True)
    actor_type: str = Field(default="system", index=True)
    actor_key: str | None = None
    event_type: str = Field(index=True)
    from_status: str | None = None
    to_status: str | None = None
    message: str | None = None
    data_json: str = "{}"
    created_at: datetime = Field(default_factory=utc_now)


class ReportArtifact(SQLModel, table=True):
    __tablename__ = "report_artifacts"

    id: str = Field(default_factory=new_id, primary_key=True)
    report_request_id: str = Field(foreign_key="report_requests.id", index=True)
    session_id: str = Field(foreign_key="audio_sessions.id", index=True)
    artifact_type: str = Field(default="report_markdown", index=True)
    title: str
    body_text: str | None = None
    storage_key: str | None = Field(default=None, index=True)
    content_type: str | None = None
    filename: str | None = None
    visibility: ReportArtifactVisibility = Field(
        default=ReportArtifactVisibility.internal,
        index=True,
    )
    metadata_json: str = "{}"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    published_at: datetime | None = Field(default=None, index=True)
