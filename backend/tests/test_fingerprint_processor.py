import json
import wave
from pathlib import Path

from sqlmodel import Session

from konopro_backend.config import BackendSettings
from konopro_backend.db import create_db_and_tables, create_engine_from_settings
from konopro_backend.models import AudioSession, ProcessingJob
from konopro_backend.processing import FingerprintSegmentationProcessor
from konopro_backend.repositories import (
    create_audio_session,
    create_processing_job,
    get_or_create_beta_user,
    get_session_analysis,
)
from konopro_backend.storage import LocalAudioStorage


def _settings(tmp_path: Path) -> BackendSettings:
    return BackendSettings(
        database_url=f"sqlite:///{tmp_path / 'processor-test.db'}",
        storage_root=tmp_path / "storage",
        processing_root=tmp_path / "processing",
        environment="test",
        fingerprint_provider="acrcloud",
        fingerprint_window_s=5.0,
        fingerprint_hop_s=5.0,
        fingerprint_max_windows=4,
    )


def _write_silence(path: Path, *, duration_s: float = 20.0, sample_rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(duration_s * sample_rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frame_count)


def _queued_job(settings: BackendSettings) -> tuple[str, str]:
    storage = LocalAudioStorage(settings.storage_root)
    _write_silence(storage.path_for("audio/song.wav"))

    engine = create_engine_from_settings(settings)
    create_db_and_tables(engine)
    with Session(engine) as db:
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
        return audio_session.id, job.id


def test_processor_persists_accepted_interval_with_fake_recognizer(tmp_path):
    settings = _settings(tmp_path)
    session_id, job_id = _queued_job(settings)
    calls = {"count": 0}

    def recognizer(_window_path: Path):
        calls["count"] += 1
        if calls["count"] <= 2:
            return {
                "status": "matched",
                "title": "Karaoke Song",
                "artist": "Guide Singer",
                "isrc": "TESTISRC",
                "confidence": 0.92,
            }
        return {"status": "no_match"}

    engine = create_engine_from_settings(settings)
    with Session(engine) as db:
        processor = FingerprintSegmentationProcessor(settings)
        processor.process(
            db.get(ProcessingJob, job_id),
            db.get(AudioSession, session_id),
            recognizer=recognizer,
            db=db,
        )
        analysis = get_session_analysis(db, session_id)

    assert analysis is not None
    assert len(analysis.windows) == 4
    assert len(analysis.intervals) == 1
    assert analysis.intervals[0].song == "Karaoke Song"
    assert analysis.diagnostic is not None
    assert json.loads(analysis.diagnostic.profile_json)["recognized_windows"] == 2


def test_processor_persists_no_match_diagnostics_without_intervals(tmp_path):
    settings = _settings(tmp_path)
    session_id, job_id = _queued_job(settings)

    def recognizer(_window_path: Path):
        return {"status": "no_match"}

    engine = create_engine_from_settings(settings)
    with Session(engine) as db:
        processor = FingerprintSegmentationProcessor(settings)
        processor.process(
            db.get(ProcessingJob, job_id),
            db.get(AudioSession, session_id),
            recognizer=recognizer,
            db=db,
        )
        analysis = get_session_analysis(db, session_id)

    assert analysis is not None
    assert len(analysis.windows) == 4
    assert analysis.intervals == []
    assert analysis.diagnostic is not None
    assert analysis.diagnostic.confidence_level == "failed"
    assert analysis.diagnostic.can_segment is False
