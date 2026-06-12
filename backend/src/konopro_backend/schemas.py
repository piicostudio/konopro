from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field as PydanticField

from konopro_backend.models import JobStatus, ReportPriority, ReportRequestStatus, SessionStatus


class ErrorDetail(BaseModel):
    detail: str


class BetaUserResponse(BaseModel):
    id: str
    external_key: str
    display_name: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ProcessingJobResponse(BaseModel):
    id: str
    session_id: str
    job_type: str
    status: JobStatus
    attempt_count: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    model_config = ConfigDict(from_attributes=True)


class AudioSessionResponse(BaseModel):
    id: str
    user_id: str
    original_filename: str
    content_type: str
    sha256: str
    size_bytes: int
    duration_s: float | None
    client_recorded_at: datetime | None
    source: str | None
    status: SessionStatus
    processing_job_id: str | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class UploadSessionResponse(BaseModel):
    session: AudioSessionResponse
    job: ProcessingJobResponse


class ReferenceScoringRunResponse(BaseModel):
    id: str
    user_id: str
    session_id: str
    job_id: str
    youtube_url: str
    reference_source: str
    reference_original_filename: str | None
    reference_content_type: str | None
    status: str
    scores: dict[str, Any]
    reference_summary: dict[str, Any]
    feedback: list[str]
    warnings: list[str]
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class ScoringJobResponse(BaseModel):
    session: AudioSessionResponse
    job: ProcessingJobResponse
    scoring_run: ReferenceScoringRunResponse


class FingerprintWindowResponse(BaseModel):
    window_index: int
    provider: str
    window_start_s: float
    window_end_s: float
    status: str
    recognized: bool
    matched_title: str
    matched_artist: str
    identity_key: str
    isrc: str
    confidence: float | None
    audio_file: str
    error: str

    model_config = ConfigDict(from_attributes=True)


class SongIntervalResponse(BaseModel):
    interval_index: int
    song: str
    artist: str
    identity_key: str
    start_s: float
    end_s: float
    duration_s: float
    confidence_score: float
    confidence_level: str
    recognized_window_count: int
    total_window_count: int
    gap_window_count: int
    conflict_window_count: int
    provider_confidence: float
    warnings: list[str]


class WeakCandidateResponse(BaseModel):
    candidate_index: int
    source: str
    song: str
    artist: str
    identity_key: str
    start_s: float
    end_s: float
    duration_s: float
    recognized_window_count: int
    total_window_count: int
    provider_confidence: float | None
    reason: str
    recovery_start_s: float | None
    recovery_end_s: float | None
    warnings: list[str]


class FingerprintDiagnosticResponse(BaseModel):
    provider: str
    can_segment: bool
    confidence_level: str
    profile: dict[str, Any]
    flags: list[dict[str, Any]]
    recommendations: list[dict[str, Any]]
    recovery_sweeps: list[dict[str, Any]]


class AnalysisResultSummary(BaseModel):
    status: str
    message: str
    confidence_level: str
    can_segment: bool
    accepted_interval_count: int
    weak_candidate_count: int
    window_count: int
    limitations: list[str]


class FingerprintAnalysisResponse(BaseModel):
    run_id: str
    session_id: str
    job_id: str | None
    provider: str
    status: str
    provider_status: str | None
    recording_duration_s: float | None
    window_s: float | None
    hop_s: float | None
    max_windows: int | None
    use_whole: bool
    summary: dict[str, Any]
    interpretations: dict[str, Any]
    warnings: list[str]
    result_summary: AnalysisResultSummary
    windows: list[FingerprintWindowResponse]
    intervals: list[SongIntervalResponse]
    weak_candidates: list[WeakCandidateResponse]
    diagnostic: FingerprintDiagnosticResponse | None


class SessionFeedbackCreate(BaseModel):
    helped_review: str = PydanticField(pattern="^(yes|not_sure|no)$")
    rating: int = PydanticField(ge=1, le=5)
    answer_text: str | None = PydanticField(default=None, max_length=500)
    context: str = PydanticField(default="post_analysis", max_length=80)


class SessionFeedbackResponse(BaseModel):
    id: str
    user_id: str
    session_id: str
    helped_review: str
    rating: int
    answer_text: str | None
    context: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ReportRequestCreate(BaseModel):
    request_type: str = "free"
    user_notes: str | None = None


class ReportArtifactResponse(BaseModel):
    id: str
    report_request_id: str
    artifact_type: str
    title: str
    body_text: str | None
    content_type: str | None
    filename: str | None
    visibility: str
    created_at: datetime
    updated_at: datetime
    published_at: datetime | None

    model_config = ConfigDict(from_attributes=True)


class ReportEventResponse(BaseModel):
    id: str
    report_request_id: str
    actor_type: str
    actor_key: str | None
    event_type: str
    from_status: str | None
    to_status: str | None
    message: str | None
    data: dict[str, Any]
    created_at: datetime


class ReportRequestResponse(BaseModel):
    id: str
    session_id: str
    user_id: str
    status: ReportRequestStatus
    priority: ReportPriority
    request_type: str
    target_turnaround_hours: int
    due_at: datetime | None
    user_notes: str | None
    admin_notes: str | None
    blocker_reason: str | None
    created_at: datetime
    updated_at: datetime
    delivered_at: datetime | None
    cancelled_at: datetime | None
    artifacts: list[ReportArtifactResponse] = []


class AdminReportRequestResponse(ReportRequestResponse):
    session: AudioSessionResponse | None = None
    user: BetaUserResponse | None = None
    events: list[ReportEventResponse] = []


class AdminReportRequestUpdate(BaseModel):
    status: ReportRequestStatus | None = None
    priority: ReportPriority | None = None
    due_at: datetime | None = None
    admin_notes: str | None = None
    blocker_reason: str | None = None
    message: str | None = None


class AdminEvidenceClipResponse(ReportArtifactResponse):
    download_url: str


class AdminReportEvidenceResponse(BaseModel):
    report_request: AdminReportRequestResponse
    analysis: FingerprintAnalysisResponse | None
    original_audio_url: str
    interval_clips: list[AdminEvidenceClipResponse]
    limitations: list[str]


class AdminReportArtifactCreate(BaseModel):
    artifact_type: str = "report_markdown"
    title: str
    body_text: str | None = None
    visibility: str = "internal"


class AdminReportArtifactUpdate(BaseModel):
    title: str | None = None
    body_text: str | None = None
    visibility: str | None = None
