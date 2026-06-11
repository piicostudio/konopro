from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from konopro_backend.config import BackendSettings
from konopro_backend.dependencies import get_beta_user_key, get_db, get_settings
from konopro_backend.models import ReportArtifactVisibility, ReportPriority, utc_now
from konopro_backend.repositories import (
    ReportRequestDetail,
    create_report_request,
    get_audio_session,
    get_or_create_beta_user,
    get_report_request_detail,
    list_report_requests_for_user,
)
from konopro_backend.schemas import ReportRequestCreate, ReportRequestResponse

router = APIRouter(tags=["reports"])


REQUEST_TYPE_PRIORITY = {
    "free": ReportPriority.low,
    "paid": ReportPriority.high,
    "manual_comp": ReportPriority.normal,
}


@router.post(
    "/v1/sessions/{session_id}/report-requests",
    response_model=ReportRequestResponse,
    status_code=status.HTTP_201_CREATED,
)
def request_verified_report(
    session_id: str,
    body: ReportRequestCreate,
    beta_user_key: str = Depends(get_beta_user_key),
    db: Session = Depends(get_db),
    settings: BackendSettings = Depends(get_settings),
) -> dict[str, Any]:
    user = get_or_create_beta_user(db, beta_user_key)
    audio_session = get_audio_session(db, session_id, user_id=user.id)
    if audio_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    request_type = _normalize_request_type(body.request_type)
    priority = REQUEST_TYPE_PRIORITY[request_type]
    turnaround_hours = _turnaround_hours(settings, request_type)
    due_at = utc_now() + timedelta(hours=turnaround_hours)
    try:
        report_request = create_report_request(
            db,
            user_id=user.id,
            session_id=audio_session.id,
            request_type=request_type,
            priority=priority,
            target_turnaround_hours=turnaround_hours,
            due_at=due_at,
            user_notes=body.user_notes,
            actor_key=user.external_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    detail = get_report_request_detail(
        db,
        report_request.id,
        user_id=user.id,
        artifact_visibility=ReportArtifactVisibility.user_visible,
    )
    if detail is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Report lost")
    return report_detail_response(detail)


@router.get("/v1/report-requests", response_model=list[ReportRequestResponse])
def list_report_requests(
    beta_user_key: str = Depends(get_beta_user_key),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    user = get_or_create_beta_user(db, beta_user_key)
    return [
        report_detail_response(
            get_report_request_detail(
                db,
                report_request.id,
                user_id=user.id,
                artifact_visibility=ReportArtifactVisibility.user_visible,
            )
        )
        for report_request in list_report_requests_for_user(db, user.id)
    ]


@router.get("/v1/report-requests/{request_id}", response_model=ReportRequestResponse)
def get_report_request(
    request_id: str,
    beta_user_key: str = Depends(get_beta_user_key),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    user = get_or_create_beta_user(db, beta_user_key)
    detail = get_report_request_detail(
        db,
        request_id,
        user_id=user.id,
        artifact_visibility=ReportArtifactVisibility.user_visible,
    )
    if detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report request not found")
    return report_detail_response(detail)


def report_detail_response(detail: ReportRequestDetail | None) -> dict[str, Any]:
    if detail is None:
        raise RuntimeError("Report detail is required")
    request = detail.request
    return {
        "id": request.id,
        "session_id": request.session_id,
        "user_id": request.user_id,
        "status": request.status,
        "priority": request.priority,
        "request_type": request.request_type,
        "target_turnaround_hours": request.target_turnaround_hours,
        "due_at": request.due_at,
        "user_notes": request.user_notes,
        "admin_notes": request.admin_notes,
        "blocker_reason": request.blocker_reason,
        "created_at": request.created_at,
        "updated_at": request.updated_at,
        "delivered_at": request.delivered_at,
        "cancelled_at": request.cancelled_at,
        "artifacts": [artifact_response(artifact) for artifact in detail.artifacts],
    }


def artifact_response(artifact) -> dict[str, Any]:
    return {
        "id": artifact.id,
        "report_request_id": artifact.report_request_id,
        "artifact_type": artifact.artifact_type,
        "title": artifact.title,
        "body_text": artifact.body_text,
        "content_type": artifact.content_type,
        "filename": artifact.filename,
        "visibility": artifact.visibility.value,
        "created_at": artifact.created_at,
        "updated_at": artifact.updated_at,
        "published_at": artifact.published_at,
    }


def event_response(event) -> dict[str, Any]:
    try:
        data = json.loads(event.data_json) if event.data_json else {}
    except json.JSONDecodeError:
        data = {}
    return {
        "id": event.id,
        "report_request_id": event.report_request_id,
        "actor_type": event.actor_type,
        "actor_key": event.actor_key,
        "event_type": event.event_type,
        "from_status": event.from_status,
        "to_status": event.to_status,
        "message": event.message,
        "data": data,
        "created_at": event.created_at,
    }


def _normalize_request_type(request_type: str) -> str:
    normalized = (request_type or "free").strip().casefold()
    if normalized not in REQUEST_TYPE_PRIORITY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="request_type must be one of: free, paid, manual_comp",
        )
    return normalized


def _turnaround_hours(settings: BackendSettings, request_type: str) -> int:
    if request_type == "paid":
        return int(settings.paid_report_turnaround_hours)
    if request_type == "manual_comp":
        return int(settings.manual_comp_report_turnaround_hours)
    return int(settings.free_report_turnaround_hours)
