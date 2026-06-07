from datetime import timedelta
from pathlib import Path

import pytest
from sqlmodel import Session

from konopro_backend.config import BackendSettings
from konopro_backend.db import create_db_and_tables, create_engine_from_settings
from konopro_backend.models import (
    ReportArtifactVisibility,
    ReportPriority,
    ReportRequestStatus,
    utc_now,
)
from konopro_backend.repositories import (
    add_report_artifact,
    create_audio_session,
    create_report_request,
    get_active_report_request_for_session,
    get_or_create_beta_user,
    get_report_request_detail,
    list_report_artifacts,
    list_report_queue,
    list_report_requests_for_user,
    update_report_artifact,
    update_report_request,
)


def _settings(tmp_path: Path) -> BackendSettings:
    return BackendSettings(
        database_url=f"sqlite:///{tmp_path / 'reports-test.db'}",
        storage_root=tmp_path / "storage",
        environment="test",
    )


def _session(tmp_path: Path) -> Session:
    engine = create_engine_from_settings(_settings(tmp_path))
    create_db_and_tables(engine)
    return Session(engine)


def _user_session(db: Session, *, key: str = "tester"):
    user = get_or_create_beta_user(db, key)
    audio_session = create_audio_session(
        db,
        user_id=user.id,
        original_filename="song.wav",
        content_type="audio/wav",
        storage_key=f"audio/{key}.wav",
        sha256="abc",
        size_bytes=3,
    )
    return user, audio_session


def test_create_report_request_and_prevent_duplicate_active_request(tmp_path):
    with _session(tmp_path) as db:
        user, audio_session = _user_session(db)
        due_at = utc_now() + timedelta(hours=24)

        report = create_report_request(
            db,
            user_id=user.id,
            session_id=audio_session.id,
            request_type="paid",
            priority=ReportPriority.high,
            target_turnaround_hours=24,
            due_at=due_at,
            user_notes="Please check chorus.",
            actor_key=user.external_key,
        )

        assert report.status == ReportRequestStatus.requested
        assert report.priority == ReportPriority.high
        assert get_active_report_request_for_session(db, audio_session.id).id == report.id
        with pytest.raises(ValueError, match="active report request"):
            create_report_request(
                db,
                user_id=user.id,
                session_id=audio_session.id,
                request_type="paid",
                priority=ReportPriority.high,
                target_turnaround_hours=24,
                due_at=due_at,
            )


def test_report_queue_sorts_by_priority_then_due_time(tmp_path):
    with _session(tmp_path) as db:
        user, first_session = _user_session(db, key="first")
        _other_user, second_session = _user_session(db, key="second")

        low = create_report_request(
            db,
            user_id=user.id,
            session_id=first_session.id,
            request_type="free",
            priority=ReportPriority.low,
            target_turnaround_hours=72,
            due_at=utc_now() + timedelta(hours=1),
        )
        high = create_report_request(
            db,
            user_id=user.id,
            session_id=second_session.id,
            request_type="paid",
            priority=ReportPriority.high,
            target_turnaround_hours=24,
            due_at=utc_now() + timedelta(hours=24),
        )

        queue = list_report_queue(db)
        user_reports = list_report_requests_for_user(db, user.id)

    assert [request.id for request in queue] == [high.id, low.id]
    assert [request.id for request in user_reports] == [high.id, low.id]


def test_report_status_updates_create_events(tmp_path):
    with _session(tmp_path) as db:
        user, audio_session = _user_session(db)
        report = create_report_request(
            db,
            user_id=user.id,
            session_id=audio_session.id,
            request_type="free",
            priority=ReportPriority.low,
            target_turnaround_hours=72,
            due_at=utc_now() + timedelta(hours=72),
        )

        updated = update_report_request(
            db,
            report.id,
            status=ReportRequestStatus.in_progress,
            admin_notes="Working on it.",
            actor_key="admin",
        )
        detail = get_report_request_detail(db, report.id)

    assert updated.status == ReportRequestStatus.in_progress
    assert updated.admin_notes == "Working on it."
    assert detail is not None
    assert [event.event_type for event in detail.events] == ["created", "updated"]
    assert detail.events[-1].from_status == "requested"
    assert detail.events[-1].to_status == "in_progress"


def test_report_artifacts_have_visibility_and_publish_state(tmp_path):
    with _session(tmp_path) as db:
        user, audio_session = _user_session(db)
        report = create_report_request(
            db,
            user_id=user.id,
            session_id=audio_session.id,
            request_type="free",
            priority=ReportPriority.low,
            target_turnaround_hours=72,
            due_at=utc_now() + timedelta(hours=72),
        )

        draft = add_report_artifact(
            db,
            report_request_id=report.id,
            session_id=audio_session.id,
            artifact_type="report_markdown",
            title="Draft",
            body_text="Internal notes",
            visibility=ReportArtifactVisibility.internal,
        )
        published = update_report_artifact(
            db,
            draft.id,
            title="Verified report",
            body_text="User-facing report",
            visibility=ReportArtifactVisibility.user_visible,
        )
        user_artifacts = list_report_artifacts(
            db,
            report.id,
            visibility=ReportArtifactVisibility.user_visible,
        )

    assert published.published_at is not None
    assert [artifact.id for artifact in user_artifacts] == [published.id]
