import json
from pathlib import Path

from sqlmodel import Session

from konopro_backend.config import BackendSettings
from konopro_backend.db import create_db_and_tables, create_engine_from_settings
from konopro_backend.repositories import (
    create_audio_session,
    create_processing_job,
    get_or_create_beta_user,
    get_session_analysis,
    replace_session_analysis,
)


def _settings(tmp_path: Path) -> BackendSettings:
    return BackendSettings(
        database_url=f"sqlite:///{tmp_path / 'analysis-test.db'}",
        storage_root=tmp_path / "storage",
        environment="test",
    )


def _session(tmp_path: Path) -> Session:
    engine = create_engine_from_settings(_settings(tmp_path))
    create_db_and_tables(engine)
    return Session(engine)


def _analysis_payload(title: str = "Test Song") -> dict:
    return {
        "provider": "acrcloud",
        "status": "completed",
        "provider_status": "ACRCloud fingerprinting found matches.",
        "recording_duration_s": 30.0,
        "window_s": 10.0,
        "hop_s": 5.0,
        "max_windows": 6,
        "use_whole": False,
        "params": {"provider": "acrcloud", "window_s": 10.0, "hop_s": 5.0},
        "summary": {"windows_tested": 2, "recognized_windows": 2},
        "interpretations": {"result": "recognized"},
        "warnings": ["one weak clue"],
        "windows": [
            {
                "provider": "acrcloud",
                "window_start_s": 0.0,
                "window_end_s": 10.0,
                "status": "matched",
                "recognized": True,
                "matched_title": title,
                "matched_artist": "Singer",
                "identity_key": "isrc:test",
                "isrc": "TEST",
                "confidence": 0.91,
                "audio_file": "window-1.wav",
                "raw": {"raw_provider": "row-1"},
            },
            {
                "provider": "acrcloud",
                "window_start_s": 5.0,
                "window_end_s": 15.0,
                "status": "matched",
                "recognized": True,
                "matched_title": title,
                "matched_artist": "Singer",
                "identity_key": "isrc:test",
                "confidence": 0.89,
                "audio_file": "window-2.wav",
            },
        ],
        "intervals": [
            {
                "index": 1,
                "song": title,
                "artist": "Singer",
                "identity_key": "isrc:test",
                "start_s": 0.0,
                "end_s": 15.0,
                "duration_s": 15.0,
                "confidence": 92.5,
                "confidence_level": "high",
                "recognized_windows": 2,
                "total_windows": 2,
                "gap_windows": 0,
                "conflict_windows": 0,
                "provider_confidence": 0.9,
                "warnings": [],
            }
        ],
        "weak_candidates": [
            {
                "index": 1,
                "song": "Maybe Song",
                "artist": "Maybe Singer",
                "identity_key": "title:maybe",
                "start_s": 20.0,
                "end_s": 30.0,
                "duration_s": 10.0,
                "recognized_windows": 1,
                "total_windows": 2,
                "provider_confidence": 0.4,
                "reason": "singleton_match",
                "recovery_start_s": 0.0,
                "recovery_end_s": 60.0,
                "warnings": ["needs retry"],
            }
        ],
        "diagnostic": {
            "provider": "acrcloud",
            "profile": {"tested_windows": 2, "recognized_windows": 2},
            "flags": [{"code": "singleton_match", "severity": "warning"}],
            "weak_candidates": [],
            "recommendations": [{"action": "retry_dense"}],
            "can_segment": True,
            "confidence_level": "recoverable",
        },
        "recovery_sweeps": [{"name": "Focused singleton recovery", "priority": 20}],
    }


def test_replace_and_load_session_analysis(tmp_path):
    with _session(tmp_path) as db:
        user = get_or_create_beta_user(db, "tester")
        audio_session = create_audio_session(
            db,
            user_id=user.id,
            original_filename="song.wav",
            content_type="audio/wav",
            storage_key="audio/song.wav",
            sha256="abc",
            size_bytes=3,
        )
        job = create_processing_job(db, audio_session.id, "fingerprint_segmentation")

        replace_session_analysis(
            db,
            session_id=audio_session.id,
            job_id=job.id,
            analysis_payload=_analysis_payload(),
        )
        analysis = get_session_analysis(db, audio_session.id)

    assert analysis is not None
    assert analysis.run.provider == "acrcloud"
    assert json.loads(analysis.run.summary_json)["recognized_windows"] == 2
    assert [window.window_start_s for window in analysis.windows] == [0.0, 5.0]
    assert analysis.windows[0].matched_title == "Test Song"
    assert json.loads(analysis.windows[0].raw_json)["raw_provider"] == "row-1"
    assert [interval.song for interval in analysis.intervals] == ["Test Song"]
    assert [candidate.reason for candidate in analysis.weak_candidates] == ["singleton_match"]
    assert analysis.diagnostic is not None
    assert analysis.diagnostic.confidence_level == "recoverable"
    assert json.loads(analysis.diagnostic.recovery_sweeps_json)[0]["priority"] == 20


def test_replace_session_analysis_removes_prior_rows(tmp_path):
    with _session(tmp_path) as db:
        user = get_or_create_beta_user(db, "tester")
        audio_session = create_audio_session(
            db,
            user_id=user.id,
            original_filename="song.wav",
            content_type="audio/wav",
            storage_key="audio/song.wav",
            sha256="abc",
            size_bytes=3,
        )
        job = create_processing_job(db, audio_session.id, "fingerprint_segmentation")

        replace_session_analysis(
            db,
            session_id=audio_session.id,
            job_id=job.id,
            analysis_payload=_analysis_payload("Old Song"),
        )
        replace_session_analysis(
            db,
            session_id=audio_session.id,
            job_id=job.id,
            analysis_payload=_analysis_payload("New Song"),
        )
        analysis = get_session_analysis(db, audio_session.id)

    assert analysis is not None
    assert len(analysis.windows) == 2
    assert [interval.song for interval in analysis.intervals] == ["New Song"]
    assert analysis.windows[0].matched_title == "New Song"
