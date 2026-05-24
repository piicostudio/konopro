from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from konopro_research.pitch import PitchContour


@dataclass(frozen=True)
class RecordingConfidence:
    score: float
    level: str
    reasons: tuple[str, ...]


def estimate_recording_confidence(
    audio: np.ndarray | None,
    contour: PitchContour,
) -> RecordingConfidence:
    reasons: list[str] = []
    voiced_ratio = float(np.mean(contour.voiced_mask)) if contour.times_s.size else 0.0
    median_pitch_conf = (
        float(np.nanmedian(contour.confidence[contour.voiced_mask])) if np.any(contour.voiced_mask) else 0.0
    )

    score = 0.55 * median_pitch_conf + 0.45 * min(1.0, voiced_ratio / 0.55)

    if audio is not None and len(audio) > 0:
        audio = np.asarray(audio, dtype=float)
        rms = float(np.sqrt(np.mean(audio**2)))
        peak = float(np.max(np.abs(audio)))
        clipping = float(np.mean(np.abs(audio) > 0.98))
        if rms < 0.015:
            reasons.append("signal is very quiet")
            score -= 0.10
        if peak > 0.98 or clipping > 0.002:
            reasons.append("possible clipping")
            score -= 0.15
        if voiced_ratio < 0.20:
            reasons.append("low voiced-frame ratio")
            score -= 0.20
    else:
        if voiced_ratio < 0.20:
            reasons.append("low voiced-frame ratio")

    score = float(np.clip(score, 0.0, 1.0))
    if score >= 0.75:
        level = "high"
    elif score >= 0.45:
        level = "medium"
    else:
        level = "low"
    return RecordingConfidence(score=score, level=level, reasons=tuple(reasons))
