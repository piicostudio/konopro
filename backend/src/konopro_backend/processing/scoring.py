from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from sqlmodel import Session

from konopro_backend.config import BackendSettings
from konopro_backend.db import create_db_and_tables, create_engine_from_settings
from konopro_backend.models import AudioSession, ProcessingJob, ReferenceScoringRun
from konopro_backend.repositories import (
    get_reference_scoring_run_by_job,
    update_reference_scoring_run,
)
from konopro_backend.storage import LocalAudioStorage
from konopro_research.contour_scoring import score_take_against_reference_contour_global_offset
from konopro_research.loudness import ActiveRmsNormalizationResult, normalize_active_rms_file
from konopro_research.reference_audio import extract_reference_audio
from konopro_research.scoring import ScoreResult
from konopro_research.separation import SeparationResult, prepare_vocal_analysis_audio


PITCH_KWARGS = {
    "fmin_hz": 80.0,
    "fmax_hz": 1000.0,
    "frame_length": 2048,
    "hop_length": 256,
}
CLEAN_KWARGS = {
    "min_confidence": 0.25,
    "max_jump_cents": 700.0,
    "correct_octaves": True,
}
CONTOUR_SCORE_KWARGS = {
    "pitch_kwargs": PITCH_KWARGS,
    "clean_kwargs": CLEAN_KWARGS,
    "dtw_time_weight": 20.0,
    "dtw_band_radius": 0.06,
    "max_dtw_frames": 2400,
    "pitch_error_penalty": 0.70,
    "stability_penalty": 0.20,
    "timing_penalty": 90.0,
    "transposition_warning_cents": 90.0,
}


class ReferenceFetchError(RuntimeError):
    """Raised when a reference track cannot be acquired."""


@dataclass(frozen=True)
class AnalysisAudio:
    analysis_path: Path
    separation: SeparationResult
    loudness: ActiveRmsNormalizationResult | None

    @property
    def warnings(self) -> tuple[str, ...]:
        loudness_warnings = self.loudness.warnings if self.loudness is not None else ()
        return (*self.separation.warnings, *loudness_warnings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_path": str(self.analysis_path),
            "separation": self.separation.to_dict(),
            "active_rms": self.loudness.to_dict() if self.loudness is not None else None,
        }


class YoutubeReferenceFetcher:
    def __init__(self, settings: BackendSettings):
        self.settings = settings

    def fetch(self, youtube_url: str, output_dir: Path) -> Path:
        parsed = urlparse(youtube_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ReferenceFetchError("YouTube URL must be a valid http(s) URL")

        output_dir.mkdir(parents=True, exist_ok=True)
        output_template = output_dir / "reference.%(ext)s"
        command = [
            self.settings.reference_download_tool,
            "--no-playlist",
            "--extract-audio",
            "--audio-format",
            "wav",
            "--audio-quality",
            "0",
            "--output",
            str(output_template),
            youtube_url,
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=float(self.settings.reference_fetch_timeout_s),
            check=False,
        )
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "").strip()
            raise ReferenceFetchError(message or "Reference download failed")

        wav_path = output_dir / "reference.wav"
        if wav_path.exists():
            return wav_path
        matches = sorted(output_dir.glob("reference.*"))
        if not matches:
            raise ReferenceFetchError("Reference download finished without an audio file")
        return matches[0]


class ReferenceScoringProcessor:
    """Score one uploaded singing take against a reference audio source."""

    def __init__(
        self,
        settings: BackendSettings,
        storage: LocalAudioStorage | None = None,
        fetcher: YoutubeReferenceFetcher | None = None,
    ):
        self.settings = settings
        self.storage = storage or LocalAudioStorage(settings.storage_root)
        self.fetcher = fetcher or YoutubeReferenceFetcher(settings)

    def process(
        self,
        job: ProcessingJob,
        audio_session: AudioSession,
        *,
        db: Session | None = None,
    ) -> ReferenceScoringRun:
        if db is not None:
            return self._process_with_db(db, job, audio_session)

        engine = create_engine_from_settings(self.settings)
        create_db_and_tables(engine)
        with Session(engine) as owned_db:
            return self._process_with_db(owned_db, job, audio_session)

    def _process_with_db(
        self,
        db: Session,
        job: ProcessingJob,
        audio_session: AudioSession,
    ) -> ReferenceScoringRun:
        scoring_run = get_reference_scoring_run_by_job(db, job.id)
        if scoring_run is None:
            raise ValueError(f"Reference scoring run not found for job: {job.id}")

        update_reference_scoring_run(
            db,
            scoring_run.id,
            status="processing",
            error_message=None,
        )
        try:
            payload = self._score(scoring_run, audio_session)
        except Exception as exc:
            update_reference_scoring_run(
                db,
                scoring_run.id,
                status="failed",
                error_message=str(exc),
            )
            raise

        return update_reference_scoring_run(
            db,
            scoring_run.id,
            status="completed",
            scores=payload["scores"],
            reference_summary=payload["reference_summary"],
            feedback=payload["feedback"],
            warnings=payload["warnings"],
            error_message=None,
        )

    def _score(
        self,
        scoring_run: ReferenceScoringRun,
        audio_session: AudioSession,
    ) -> dict[str, object]:
        take_path = self.storage.path_for(audio_session.storage_key)
        if not take_path.exists():
            raise FileNotFoundError(f"Stored take audio file not found: {audio_session.storage_key}")

        reference_path = self._reference_path(scoring_run)
        reference_audio = self._prepare_analysis_audio(reference_path, slot="reference")
        take_audio = self._prepare_analysis_audio(take_path, slot="take")

        reference = extract_reference_audio(
            reference_audio.analysis_path,
            title="Reference song",
            pitch_kwargs=PITCH_KWARGS,
            clean_kwargs=CLEAN_KWARGS,
        )
        score = score_take_against_reference_contour_global_offset(
            take_audio.analysis_path,
            reference.contour,
            name=audio_session.original_filename,
            take_audio_path=take_audio.analysis_path,
            **CONTOUR_SCORE_KWARGS,
        )
        warnings = list(
            dict.fromkeys(
                [
                    *score.warnings,
                    *reference.audio_summary.warnings,
                    *reference.quality.warnings,
                    *reference_audio.warnings,
                    *take_audio.warnings,
                ]
            )
        )
        return {
            "scores": score.to_dict(),
            "reference_summary": {
                "source": scoring_run.reference_source,
                "audio": reference.audio_summary.to_dict(),
                "quality": reference.quality.to_dict(),
                "preprocessing": {
                    "reference": reference_audio.to_dict(),
                    "take": take_audio.to_dict(),
                    "scoring_method": "global_offset_1_to_1_contour",
                    "pitch_kwargs": PITCH_KWARGS,
                    "clean_kwargs": CLEAN_KWARGS,
                    "score_kwargs": {
                        key: value
                        for key, value in CONTOUR_SCORE_KWARGS.items()
                        if key not in {"pitch_kwargs", "clean_kwargs"}
                    },
                },
            },
            "feedback": _practice_feedback(score),
            "warnings": warnings,
        }

    def _prepare_analysis_audio(self, path: Path, *, slot: str) -> AnalysisAudio:
        analysis_cache_dir = self.settings.processing_path / "_cache" / "reference_scoring"
        separation = prepare_vocal_analysis_audio(
            path,
            cache_dir=analysis_cache_dir,
            backend="demucs" if self.settings.reference_scoring_use_demucs else "none",
            stem="vocals",
            model=self.settings.reference_scoring_demucs_model,
            device=self._demucs_device(),
            shifts=1,
            overlap=0.25,
            timeout_s=self.settings.reference_scoring_demucs_timeout_s,
        )
        analysis_path = separation.analysis_path
        loudness: ActiveRmsNormalizationResult | None = None
        if self.settings.reference_scoring_use_active_rms:
            loudness = normalize_active_rms_file(
                analysis_path,
                cache_dir=analysis_cache_dir / slot,
                target_rms=self.settings.reference_scoring_target_active_rms,
                active_percentile=self.settings.reference_scoring_active_rms_percentile,
            )
            analysis_path = loudness.analysis_path
        return AnalysisAudio(
            analysis_path=analysis_path,
            separation=separation,
            loudness=loudness,
        )

    def _demucs_device(self) -> str:
        if self.settings.reference_scoring_demucs_device:
            return self.settings.reference_scoring_demucs_device
        return _default_demucs_device()

    def _reference_path(self, scoring_run: ReferenceScoringRun) -> Path:
        if scoring_run.reference_storage_key:
            path = self.storage.path_for(scoring_run.reference_storage_key)
            if not path.exists():
                raise FileNotFoundError(
                    f"Stored reference audio file not found: {scoring_run.reference_storage_key}"
                )
            return path

        output_dir = self.settings.processing_path / scoring_run.session_id / scoring_run.job_id
        return self.fetcher.fetch(scoring_run.youtube_url, output_dir)


def _default_demucs_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _practice_feedback(score: ScoreResult) -> list[str]:
    feedback: list[str] = []
    if score.overall_score >= 80:
        feedback.append("Strong take overall. Use this as a reference point for later attempts.")
    elif score.overall_score >= 60:
        feedback.append("Usable practice take. Focus on the lowest metric first before re-recording.")
    else:
        feedback.append("Treat this as a diagnostic take. Re-record a shorter, cleaner section if needed.")

    if score.pitch_accuracy_score < 70:
        feedback.append("Pitch contour is the main gap. Practice the melody slowly before singing full tempo.")
    if score.timing_score < 70:
        feedback.append("Timing alignment is weak. Trim the take/reference to the same phrase and retry.")
    if score.stability_score < 70:
        feedback.append("Pitch stability is low. Hold longer notes steadily before adding style or vibrato.")
    if score.coverage_score < 70:
        feedback.append("Detected singing coverage is low. Sing through more of the reference phrase.")
    if score.recording_confidence_level == "low":
        feedback.append("Recording confidence is low, so use this score as rough feedback only.")

    return feedback[:5]
