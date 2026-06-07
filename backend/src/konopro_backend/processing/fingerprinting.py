from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlmodel import Session

from konopro_backend.config import BackendSettings
from konopro_backend.db import create_db_and_tables, create_engine_from_settings
from konopro_backend.models import AudioSession, ProcessingJob
from konopro_backend.repositories import SessionAnalysis, replace_session_analysis
from konopro_backend.storage import LocalAudioStorage
from konopro_research.fingerprint_diagnostics import (
    diagnose_fingerprint_rows,
    plan_recovery_sweeps,
)
from konopro_research.session_segmentation import segment_long_recording


class FingerprintSegmentationProcessor:
    """Run research fingerprint segmentation for one stored backend audio session."""

    def __init__(self, settings: BackendSettings, storage: LocalAudioStorage | None = None):
        self.settings = settings
        self.storage = storage or LocalAudioStorage(settings.storage_root)

    def process(
        self,
        job: ProcessingJob,
        audio_session: AudioSession,
        *,
        recognizer: Any | None = None,
        db: Session | None = None,
    ) -> SessionAnalysis:
        audio_path = self.storage.path_for(audio_session.storage_key)
        if not audio_path.exists():
            raise FileNotFoundError(f"Stored audio file not found: {audio_session.storage_key}")

        output_dir = self._output_dir(audio_session.id, job.id)
        result = segment_long_recording(
            audio_path,
            output_dir,
            provider=self.provider,
            window_s=float(self.settings.fingerprint_window_s),
            hop_s=float(self.settings.fingerprint_hop_s),
            max_windows=int(self.settings.fingerprint_max_windows),
            recognizer=recognizer,
            use_whole=bool(self.settings.fingerprint_use_whole),
            **self._provider_kwargs(),
        )
        payload = self._analysis_payload(result)

        if db is not None:
            return replace_session_analysis(
                db,
                session_id=audio_session.id,
                job_id=job.id,
                analysis_payload=payload,
            )

        engine = create_engine_from_settings(self.settings)
        create_db_and_tables(engine)
        with Session(engine) as owned_db:
            return replace_session_analysis(
                owned_db,
                session_id=audio_session.id,
                job_id=job.id,
                analysis_payload=payload,
            )

    @property
    def provider(self) -> str:
        return self.settings.fingerprint_provider.strip().casefold()

    def _output_dir(self, session_id: str, job_id: str) -> Path:
        path = self.settings.processing_path / session_id / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _provider_kwargs(self) -> dict[str, Any]:
        timeout = float(self.settings.fingerprint_timeout_s)
        if self.provider == "audd":
            return {"api_token": self.settings.audd_api_token, "timeout_s": timeout}
        if self.provider == "acrcloud":
            return {
                "host": self.settings.acrcloud_host,
                "access_key": self.settings.acrcloud_access_key,
                "access_secret": self.settings.acrcloud_access_secret,
                "timeout_s": timeout,
            }
        if self.provider == "shazamkit":
            return {"helper_path": self.settings.shazamkit_helper_path, "timeout_s": timeout}
        return {"timeout_s": timeout}

    def _analysis_payload(self, result: Any) -> dict[str, Any]:
        provider_result = result.provider_result
        provider_rows = list(getattr(provider_result, "rows", ()) or ())
        diagnostic = diagnose_fingerprint_rows(
            provider_rows,
            provider=self.provider,
            recording_duration_s=float(result.recording_duration_s),
            requested_window_s=float(self.settings.fingerprint_window_s),
            requested_hop_s=float(self.settings.fingerprint_hop_s),
        )
        recovery_sweeps = plan_recovery_sweeps(
            diagnostic,
            recording_duration_s=float(result.recording_duration_s),
            request_budget=int(self.settings.fingerprint_max_windows),
        )
        windows = []
        for window in result.windows:
            window_data = window.to_dict()
            if window.row is not None:
                window_data["raw"] = dict(window.row)
            windows.append(window_data)

        warnings = [
            *list(getattr(provider_result, "warnings", ()) or ()),
            *list(result.warnings or ()),
        ]
        return {
            "provider": self.provider,
            "status": "completed",
            "provider_status": getattr(provider_result, "status", None),
            "recording_duration_s": float(result.recording_duration_s),
            "window_s": float(self.settings.fingerprint_window_s),
            "hop_s": float(result.hop_s),
            "max_windows": int(self.settings.fingerprint_max_windows),
            "use_whole": bool(self.settings.fingerprint_use_whole),
            "params": {
                "provider": self.provider,
                "window_s": float(self.settings.fingerprint_window_s),
                "hop_s": float(self.settings.fingerprint_hop_s),
                "max_windows": int(self.settings.fingerprint_max_windows),
                "use_whole": bool(self.settings.fingerprint_use_whole),
            },
            "summary": dict(getattr(provider_result, "summary", {}) or {}),
            "interpretations": dict(getattr(provider_result, "interpretations", {}) or {}),
            "warnings": list(dict.fromkeys(warnings)),
            "windows": windows,
            "intervals": [interval.to_dict() for interval in result.intervals],
            "weak_candidates": [candidate.to_dict() for candidate in result.weak_candidates],
            "diagnostic": diagnostic.to_dict(),
            "recovery_sweeps": [sweep.to_dict() for sweep in recovery_sweeps],
        }
