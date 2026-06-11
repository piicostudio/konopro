from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from konopro_backend.dependencies import get_beta_user_key, get_db
from konopro_backend.repositories import (
    SessionAnalysis,
    get_audio_session,
    get_or_create_beta_user,
    get_session_analysis,
)
from konopro_backend.schemas import FingerprintAnalysisResponse

router = APIRouter(prefix="/v1/sessions", tags=["analysis"])


@router.get("/{session_id}/analysis", response_model=FingerprintAnalysisResponse)
def get_analysis(
    session_id: str,
    beta_user_key: str = Depends(get_beta_user_key),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    user = get_or_create_beta_user(db, beta_user_key)
    audio_session = get_audio_session(db, session_id, user_id=user.id)
    if audio_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    analysis = get_session_analysis(db, session_id)
    if analysis is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not ready")
    return _analysis_response(analysis)


def _analysis_response(analysis: SessionAnalysis) -> dict[str, Any]:
    run = analysis.run
    return {
        "run_id": run.id,
        "session_id": run.session_id,
        "job_id": run.job_id,
        "provider": run.provider,
        "status": run.status,
        "provider_status": run.provider_status,
        "recording_duration_s": run.recording_duration_s,
        "window_s": run.window_s,
        "hop_s": run.hop_s,
        "max_windows": run.max_windows,
        "use_whole": run.use_whole,
        "summary": _json(run.summary_json, {}),
        "interpretations": _json(run.interpretations_json, {}),
        "warnings": _json(run.warnings_json, []),
        "result_summary": _result_summary(analysis),
        "windows": [
            {
                "window_index": window.window_index,
                "provider": window.provider,
                "window_start_s": window.window_start_s,
                "window_end_s": window.window_end_s,
                "status": window.status,
                "recognized": window.recognized,
                "matched_title": window.matched_title,
                "matched_artist": window.matched_artist,
                "identity_key": window.identity_key,
                "isrc": window.isrc,
                "confidence": window.confidence,
                "audio_file": window.audio_file,
                "error": window.error,
            }
            for window in analysis.windows
        ],
        "intervals": [
            {
                "interval_index": interval.interval_index,
                "song": interval.song,
                "artist": interval.artist,
                "identity_key": interval.identity_key,
                "start_s": interval.start_s,
                "end_s": interval.end_s,
                "duration_s": interval.duration_s,
                "confidence_score": interval.confidence_score,
                "confidence_level": interval.confidence_level,
                "recognized_window_count": interval.recognized_window_count,
                "total_window_count": interval.total_window_count,
                "gap_window_count": interval.gap_window_count,
                "conflict_window_count": interval.conflict_window_count,
                "provider_confidence": interval.provider_confidence,
                "warnings": _json(interval.warnings_json, []),
            }
            for interval in analysis.intervals
        ],
        "weak_candidates": [
            {
                "candidate_index": candidate.candidate_index,
                "source": candidate.source,
                "song": candidate.song,
                "artist": candidate.artist,
                "identity_key": candidate.identity_key,
                "start_s": candidate.start_s,
                "end_s": candidate.end_s,
                "duration_s": candidate.duration_s,
                "recognized_window_count": candidate.recognized_window_count,
                "total_window_count": candidate.total_window_count,
                "provider_confidence": candidate.provider_confidence,
                "reason": candidate.reason,
                "recovery_start_s": candidate.recovery_start_s,
                "recovery_end_s": candidate.recovery_end_s,
                "warnings": _json(candidate.warnings_json, []),
            }
            for candidate in analysis.weak_candidates
        ],
        "diagnostic": _diagnostic_response(analysis),
    }


def _diagnostic_response(analysis: SessionAnalysis) -> dict[str, Any] | None:
    diagnostic = analysis.diagnostic
    if diagnostic is None:
        return None
    return {
        "provider": diagnostic.provider,
        "can_segment": diagnostic.can_segment,
        "confidence_level": diagnostic.confidence_level,
        "profile": _json(diagnostic.profile_json, {}),
        "flags": _json(diagnostic.flags_json, []),
        "recommendations": _json(diagnostic.recommendations_json, []),
        "recovery_sweeps": _json(diagnostic.recovery_sweeps_json, []),
    }


def _result_summary(analysis: SessionAnalysis) -> dict[str, Any]:
    diagnostic = analysis.diagnostic
    accepted_count = len(analysis.intervals)
    weak_count = len(analysis.weak_candidates)
    can_segment = bool(accepted_count) or bool(diagnostic and diagnostic.can_segment)
    confidence_level = diagnostic.confidence_level if diagnostic else "unknown"
    limitations: list[str] = []

    if accepted_count:
        result_status = "accepted_intervals"
        message = "Likely song intervals were found, but this is still experimental."
    elif weak_count:
        result_status = "weak_candidates"
        message = "Only weak song clues were found; do not treat them as confirmed matches."
        limitations.append("No accepted intervals passed the confidence gates.")
    else:
        result_status = "no_match"
        message = "No reliable song interval was found in this fingerprinting run."
        limitations.append("Fingerprinting produced no accepted intervals.")

    if diagnostic and diagnostic.flags_json:
        flags = _json(diagnostic.flags_json, [])
        limitations.extend(str(flag.get("message") or flag.get("code")) for flag in flags)

    return {
        "status": result_status,
        "message": message,
        "confidence_level": confidence_level,
        "can_segment": can_segment,
        "accepted_interval_count": accepted_count,
        "weak_candidate_count": weak_count,
        "window_count": len(analysis.windows),
        "limitations": list(dict.fromkeys(item for item in limitations if item)),
    }


def _json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default
