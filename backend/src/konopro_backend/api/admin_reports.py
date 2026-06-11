from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlmodel import Session

from konopro_backend.api.analysis import _analysis_response
from konopro_backend.api.reports import artifact_response, event_response, report_detail_response
from konopro_backend.dependencies import get_admin_access, get_db, get_storage
from konopro_backend.models import ReportArtifactVisibility, ReportPriority, ReportRequestStatus
from konopro_backend.repositories import (
    add_report_artifact,
    get_audio_session,
    get_report_artifact,
    get_report_request_detail,
    get_session_analysis,
    list_report_queue,
    update_report_artifact,
    update_report_request,
)
from konopro_backend.schemas import (
    AdminReportArtifactCreate,
    AdminReportArtifactUpdate,
    AdminReportEvidenceResponse,
    AdminReportRequestResponse,
    AdminReportRequestUpdate,
    ReportArtifactResponse,
)
from konopro_backend.services.report_evidence import build_report_evidence_bundle
from konopro_backend.storage import LocalAudioStorage

router = APIRouter(prefix="/v1/admin", tags=["admin reports"])


@router.get("/report-requests", response_model=list[AdminReportRequestResponse])
def list_admin_report_requests(
    status_filter: ReportRequestStatus | None = Query(default=None, alias="status"),
    priority: ReportPriority | None = None,
    _admin_key: str = Depends(get_admin_access),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    return [
        admin_report_detail_response(get_report_request_detail(db, report_request.id))
        for report_request in list_report_queue(db, status=status_filter, priority=priority)
    ]


@router.get("/report-requests/{request_id}", response_model=AdminReportRequestResponse)
def get_admin_report_request(
    request_id: str,
    _admin_key: str = Depends(get_admin_access),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    detail = get_report_request_detail(db, request_id)
    if detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report request not found")
    return admin_report_detail_response(detail)


@router.patch("/report-requests/{request_id}", response_model=AdminReportRequestResponse)
def update_admin_report_request(
    request_id: str,
    body: AdminReportRequestUpdate,
    admin_key: str = Depends(get_admin_access),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    detail = get_report_request_detail(db, request_id)
    if detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report request not found")

    if body.status == ReportRequestStatus.delivered:
        visible_artifacts = [
            artifact
            for artifact in detail.artifacts
            if artifact.visibility == ReportArtifactVisibility.user_visible
        ]
        if not visible_artifacts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot mark delivered without a user-visible report artifact",
            )

    try:
        update_report_request(
            db,
            request_id,
            status=body.status,
            priority=body.priority,
            due_at=body.due_at,
            admin_notes=body.admin_notes,
            blocker_reason=body.blocker_reason,
            actor_key=admin_key,
            message=body.message,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    updated_detail = get_report_request_detail(db, request_id)
    if updated_detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report request not found")
    return admin_report_detail_response(updated_detail)


@router.post(
    "/report-requests/{request_id}/artifacts",
    response_model=ReportArtifactResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_admin_report_artifact(
    request_id: str,
    body: AdminReportArtifactCreate,
    admin_key: str = Depends(get_admin_access),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    detail = get_report_request_detail(db, request_id)
    if detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report request not found")
    visibility = _artifact_visibility(body.visibility)
    artifact = add_report_artifact(
        db,
        report_request_id=detail.request.id,
        session_id=detail.request.session_id,
        artifact_type=body.artifact_type,
        title=body.title,
        body_text=body.body_text,
        visibility=visibility,
        actor_key=admin_key,
    )
    return artifact_response(artifact)


@router.patch("/report-artifacts/{artifact_id}", response_model=ReportArtifactResponse)
def update_admin_report_artifact(
    artifact_id: str,
    body: AdminReportArtifactUpdate,
    admin_key: str = Depends(get_admin_access),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    artifact = get_report_artifact(db, artifact_id)
    if artifact is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")
    visibility = _artifact_visibility(body.visibility) if body.visibility is not None else None
    updated = update_report_artifact(
        db,
        artifact_id,
        title=body.title,
        body_text=body.body_text,
        visibility=visibility,
        actor_key=admin_key,
    )
    return artifact_response(updated)


@router.get(
    "/report-requests/{request_id}/evidence",
    response_model=AdminReportEvidenceResponse,
)
def get_admin_report_evidence(
    request_id: str,
    _admin_key: str = Depends(get_admin_access),
    db: Session = Depends(get_db),
    storage: LocalAudioStorage = Depends(get_storage),
) -> dict[str, Any]:
    detail = get_report_request_detail(db, request_id)
    if detail is None or detail.audio_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report request not found")
    analysis = get_session_analysis(db, detail.request.session_id)
    bundle = build_report_evidence_bundle(db, detail, analysis, storage)
    return {
        "report_request": admin_report_detail_response(
            get_report_request_detail(db, request_id)
        ),
        "analysis": _analysis_response(analysis) if analysis is not None else None,
        "original_audio_url": f"/v1/admin/sessions/{detail.request.session_id}/audio",
        "interval_clips": [
            {
                **artifact_response(artifact),
                "download_url": f"/v1/admin/report-artifacts/{artifact.id}/download",
            }
            for artifact in bundle["interval_clips"]
        ],
        "limitations": bundle["limitations"],
    }


@router.get("/sessions/{session_id}/audio")
def download_admin_session_audio(
    session_id: str,
    _admin_key: str = Depends(get_admin_access),
    db: Session = Depends(get_db),
    storage: LocalAudioStorage = Depends(get_storage),
) -> FileResponse:
    audio_session = get_audio_session(db, session_id, include_deleted=True)
    if audio_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    path = storage.path_for(audio_session.storage_key)
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audio file not found")
    return FileResponse(
        path,
        media_type=audio_session.content_type,
        filename=audio_session.original_filename,
    )


@router.get("/report-artifacts/{artifact_id}/download")
def download_admin_report_artifact(
    artifact_id: str,
    _admin_key: str = Depends(get_admin_access),
    db: Session = Depends(get_db),
    storage: LocalAudioStorage = Depends(get_storage),
) -> FileResponse:
    artifact = get_report_artifact(db, artifact_id)
    if artifact is None or not artifact.storage_key:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")
    path = storage.path_for(artifact.storage_key)
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact file not found")
    return FileResponse(
        path,
        media_type=artifact.content_type or "application/octet-stream",
        filename=artifact.filename or path.name,
    )


def admin_report_detail_response(detail) -> dict[str, Any]:
    payload = report_detail_response(detail)
    payload["session"] = detail.audio_session
    payload["user"] = detail.user
    payload["artifacts"] = [artifact_response(artifact) for artifact in detail.artifacts]
    payload["events"] = [event_response(event) for event in detail.events]
    return payload


def _artifact_visibility(value: str) -> ReportArtifactVisibility:
    normalized = (value or "internal").strip().casefold()
    try:
        return ReportArtifactVisibility(normalized)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="visibility must be one of: internal, user_visible",
        ) from exc
