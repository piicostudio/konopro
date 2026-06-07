from __future__ import annotations

from typing import Any

import numpy as np
from sqlmodel import Session

from konopro_backend.models import ReportArtifact, ReportArtifactVisibility
from konopro_backend.repositories import (
    ReportRequestDetail,
    SessionAnalysis,
    add_report_artifact,
    list_report_artifacts,
)
from konopro_backend.storage import LocalAudioStorage
from konopro_research.audio_io import load_audio, write_wav


def build_report_evidence_bundle(
    db: Session,
    detail: ReportRequestDetail,
    analysis: SessionAnalysis | None,
    storage: LocalAudioStorage,
) -> dict[str, Any]:
    limitations: list[str] = []
    if detail.audio_session is None:
        return {"interval_clips": [], "limitations": ["Audio session is missing."]}
    if analysis is None:
        return {
            "interval_clips": [],
            "limitations": ["No Phase 05 analysis exists for this session yet."],
        }
    if not analysis.intervals:
        return {
            "interval_clips": [],
            "limitations": ["No accepted intervals are available for clip generation."],
        }

    existing = list_report_artifacts(
        db,
        detail.request.id,
        visibility=ReportArtifactVisibility.internal,
        artifact_type="interval_clip",
    )
    if existing:
        return {"interval_clips": existing, "limitations": limitations}

    source_path = storage.path_for(detail.audio_session.storage_key)
    if not source_path.exists():
        return {"interval_clips": [], "limitations": ["Original audio file is missing."]}

    audio, sample_rate = load_audio(source_path, target_sr=44100)
    generated: list[ReportArtifact] = []
    for interval in analysis.intervals:
        start_index = max(0, int(round(interval.start_s * sample_rate)))
        end_index = min(len(audio), int(round(interval.end_s * sample_rate)))
        if end_index <= start_index:
            limitations.append(f"Interval {interval.interval_index} has no audio duration.")
            continue
        clip_audio = np.asarray(audio[start_index:end_index], dtype=np.float32)
        filename = f"interval_{interval.interval_index:02d}_{start_index}_{end_index}.wav"
        storage_key = f"reports/{detail.request.id}/clips/{filename}"
        clip_path = storage.path_for(storage_key)
        write_wav(clip_path, clip_audio, sample_rate)
        generated.append(
            add_report_artifact(
                db,
                report_request_id=detail.request.id,
                session_id=detail.request.session_id,
                artifact_type="interval_clip",
                title=f"Interval {interval.interval_index}: {interval.song or 'Unknown song'}",
                storage_key=storage_key,
                content_type="audio/wav",
                filename=filename,
                visibility=ReportArtifactVisibility.internal,
                metadata={
                    "interval_index": interval.interval_index,
                    "start_s": interval.start_s,
                    "end_s": interval.end_s,
                    "song": interval.song,
                    "artist": interval.artist,
                    "confidence_level": interval.confidence_level,
                },
            )
        )

    return {"interval_clips": generated, "limitations": limitations}
