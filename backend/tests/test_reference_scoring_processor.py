from pathlib import Path
from types import SimpleNamespace

from sqlmodel import Session

from konopro_backend.config import BackendSettings
from konopro_backend.db import create_db_and_tables, create_engine_from_settings
from konopro_backend.models import JobStatus, SessionStatus
from konopro_backend.processing.scoring import ReferenceScoringProcessor
from konopro_backend.repositories import (
    create_audio_session,
    create_processing_job,
    create_reference_scoring_run,
    get_audio_session,
    get_or_create_beta_user,
    get_processing_job,
    get_reference_scoring_run_by_job,
    reference_scoring_run_payload,
)
from konopro_backend.storage import LocalAudioStorage
from konopro_research.demo_data import ensure_demo_data


def _settings(tmp_path: Path) -> BackendSettings:
    return BackendSettings(
        database_url=f"sqlite:///{tmp_path / 'reference-scoring-test.db'}",
        storage_root=tmp_path / "storage",
        processing_root=tmp_path / "processing",
        environment="test",
    )


def _create_scoring_job_with_reference(settings: BackendSettings, tmp_path: Path) -> tuple[str, str]:
    demo = ensure_demo_data(tmp_path / "demo")
    storage = LocalAudioStorage(settings.storage_root)
    with demo["current"].open("rb") as fileobj:
        stored_take = storage.save(fileobj, "current_take.wav", "audio/wav", settings.max_upload_bytes)
    with demo["reference"].open("rb") as fileobj:
        stored_reference = storage.save(
            fileobj,
            "reference_melody.wav",
            "audio/wav",
            settings.max_upload_bytes,
        )

    engine = create_engine_from_settings(settings)
    create_db_and_tables(engine)
    with Session(engine) as db:
        user = get_or_create_beta_user(db, "tester")
        audio_session = create_audio_session(
            db,
            user_id=user.id,
            original_filename="current_take.wav",
            content_type=stored_take.content_type,
            storage_key=stored_take.storage_key,
            sha256=stored_take.sha256,
            size_bytes=stored_take.size_bytes,
            source="web_reference_scoring",
        )
        job = create_processing_job(db, audio_session.id, "reference_scoring")
        create_reference_scoring_run(
            db,
            user_id=user.id,
            session_id=audio_session.id,
            job_id=job.id,
            youtube_url="https://www.youtube.com/watch?v=demo",
            reference_source="upload",
            reference_storage_key=stored_reference.storage_key,
            reference_original_filename="reference_melody.wav",
            reference_content_type=stored_reference.content_type,
        )
        return audio_session.id, job.id


def test_reference_scoring_processor_persists_scores_with_uploaded_reference(tmp_path):
    settings = _settings(tmp_path)
    session_id, job_id = _create_scoring_job_with_reference(settings, tmp_path)

    engine = create_engine_from_settings(settings)
    with Session(engine) as db:
        audio_session = get_audio_session(db, session_id)
        job = get_processing_job(db, job_id)
        assert audio_session is not None
        assert job is not None
        processor = ReferenceScoringProcessor(settings)
        processor.process(job, audio_session, db=db)

        scoring_run = get_reference_scoring_run_by_job(db, job_id)

    assert scoring_run is not None
    payload = reference_scoring_run_payload(scoring_run)
    assert payload["status"] == "completed"
    assert payload["scores"]["overall_score"] > 0
    assert payload["scores"]["pitch_accuracy_score"] > 0
    assert payload["reference_summary"]["source"] == "upload"
    assert payload["feedback"]


def test_reference_scoring_processor_uses_finalized_scoring_configuration(tmp_path, monkeypatch):
    import konopro_backend.processing.scoring as scoring_module

    settings = _settings(tmp_path)
    session_id, job_id = _create_scoring_job_with_reference(settings, tmp_path)
    calls: dict[str, list[dict[str, object]] | dict[str, object]] = {
        "prepare": [],
        "rms": [],
    }

    class FakeSeparation:
        def __init__(self, path: Path):
            self.analysis_path = Path(path)
            self.warnings = ()

        def to_dict(self):
            return {"analysis_path": str(self.analysis_path), "warnings": []}

    class FakeLoudness:
        def __init__(self, path: Path):
            self.analysis_path = Path(path)
            self.warnings = ()

        def to_dict(self):
            return {"analysis_path": str(self.analysis_path), "warnings": []}

    def fake_prepare(path, **kwargs):
        calls["prepare"].append({"path": Path(path), **kwargs})
        return FakeSeparation(Path(path))

    def fake_rms(path, **kwargs):
        calls["rms"].append({"path": Path(path), **kwargs})
        return FakeLoudness(Path(path))

    def fake_extract(path, **kwargs):
        calls["extract"] = {"path": Path(path), **kwargs}
        return SimpleNamespace(
            contour=object(),
            audio_summary=SimpleNamespace(warnings=(), to_dict=lambda: {}),
            quality=SimpleNamespace(warnings=(), to_dict=lambda: {}),
        )

    def fake_score(take, reference, **kwargs):
        calls["score"] = {"take": Path(take), "reference": reference, **kwargs}
        return SimpleNamespace(
            overall_score=59.76,
            pitch_accuracy_score=47.54,
            timing_score=90.6,
            stability_score=56.3,
            coverage_score=74.31,
            recording_confidence_level="high",
            warnings=(),
            to_dict=lambda: {
                "overall_score": 59.76,
                "pitch_accuracy_score": 47.54,
                "timing_score": 90.6,
                "stability_score": 56.3,
                "coverage_score": 74.31,
                "recording_confidence_level": "high",
            },
        )

    monkeypatch.setattr(scoring_module, "_default_demucs_device", lambda: "cuda")
    monkeypatch.setattr(scoring_module, "prepare_vocal_analysis_audio", fake_prepare)
    monkeypatch.setattr(scoring_module, "normalize_active_rms_file", fake_rms)
    monkeypatch.setattr(scoring_module, "extract_reference_audio", fake_extract)
    monkeypatch.setattr(scoring_module, "score_take_against_reference_contour_global_offset", fake_score)

    engine = create_engine_from_settings(settings)
    with Session(engine) as db:
        audio_session = get_audio_session(db, session_id)
        job = get_processing_job(db, job_id)
        assert audio_session is not None
        assert job is not None
        processor = ReferenceScoringProcessor(settings)
        processor.process(job, audio_session, db=db)
        scoring_run = get_reference_scoring_run_by_job(db, job_id)

    assert scoring_run is not None
    payload = reference_scoring_run_payload(scoring_run)
    assert payload["scores"]["overall_score"] == 59.76
    assert payload["reference_summary"]["preprocessing"]["scoring_method"] == "global_offset_1_to_1_contour"

    prepare_calls = calls["prepare"]
    assert len(prepare_calls) == 2
    assert all(call["backend"] == "demucs" for call in prepare_calls)
    assert all(call["stem"] == "vocals" for call in prepare_calls)
    assert all(call["model"] == "htdemucs" for call in prepare_calls)
    assert all(call["device"] == "cuda" for call in prepare_calls)

    rms_calls = calls["rms"]
    assert len(rms_calls) == 2
    assert all(call["target_rms"] == 0.08 for call in rms_calls)
    assert all(call["active_percentile"] == 60.0 for call in rms_calls)

    extract_call = calls["extract"]
    assert extract_call["pitch_kwargs"] == scoring_module.PITCH_KWARGS
    assert extract_call["clean_kwargs"] == scoring_module.CLEAN_KWARGS

    score_call = calls["score"]
    assert score_call["pitch_kwargs"] == scoring_module.PITCH_KWARGS
    assert score_call["clean_kwargs"] == scoring_module.CLEAN_KWARGS
    assert score_call["stability_penalty"] == 0.20
    assert score_call["pitch_error_penalty"] == 0.70
    assert score_call["dtw_band_radius"] == 0.06
    assert score_call["max_dtw_frames"] == 2400


class FakeReferenceScoringProcessor:
    def __init__(self, _settings: BackendSettings):
        pass

    def process(self, _job, _audio_session) -> None:
        return None


def test_worker_dispatches_reference_scoring_jobs(tmp_path, monkeypatch):
    from konopro_backend.jobs import run_next_job

    settings = _settings(tmp_path)
    session_id, job_id = _create_scoring_job_with_reference(settings, tmp_path)
    monkeypatch.setattr(
        "konopro_backend.jobs.ReferenceScoringProcessor",
        FakeReferenceScoringProcessor,
    )

    processed = run_next_job(settings)

    engine = create_engine_from_settings(settings)
    with Session(engine) as db:
        job = get_processing_job(db, job_id)
        audio_session = get_audio_session(db, session_id)

    assert processed is not None
    assert job is not None
    assert job.status == JobStatus.completed
    assert audio_session is not None
    assert audio_session.status == SessionStatus.processed
