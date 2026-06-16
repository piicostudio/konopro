from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from konopro_research.audio_io import load_audio
from konopro_research.baseline import MelodyBaseline


@dataclass(frozen=True)
class AudioSummary:
    duration_s: float
    sample_rate: int
    rms: float
    peak: float
    clipping_ratio: float
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BaselineQuality:
    duration_s: float
    note_count: int
    voiced_coverage_ratio: float
    gap_ratio: float
    median_note_duration_s: float
    notes_per_second: float
    level: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_audio(path: str | Path) -> AudioSummary:
    """Load audio from disk and compute a summary."""
    audio, sample_rate = load_audio(path, target_sr=22050)
    return summarize_audio_from_array(audio, sample_rate)


def summarize_audio_from_array(audio: np.ndarray, sample_rate: int) -> AudioSummary:
    """Compute an audio summary from an already-loaded numpy array (avoids re-reading disk)."""
    duration = len(audio) / sample_rate if sample_rate else 0.0
    rms = float(np.sqrt(np.mean(np.asarray(audio, dtype=float) ** 2))) if len(audio) else 0.0
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    clipping_ratio = float(np.mean(np.abs(audio) > 0.98)) if len(audio) else 0.0
    warnings: list[str] = []
    if duration < 5:
        warnings.append("audio is very short; use 20-40 seconds for meaningful tests")
    if duration > 90:
        warnings.append("audio is long; trim to a matching phrase before scoring")
    if rms < 0.01:
        warnings.append("audio is very quiet")
    if clipping_ratio > 0.002:
        warnings.append("audio may be clipped")
    return AudioSummary(
        duration_s=round(duration, 3),
        sample_rate=sample_rate,
        rms=round(rms, 5),
        peak=round(peak, 5),
        clipping_ratio=round(clipping_ratio, 5),
        warnings=tuple(warnings),
    )


def analyze_baseline_quality(
    baseline: MelodyBaseline,
    *,
    source_duration_s: float | None = None,
) -> BaselineQuality:
    duration = source_duration_s or baseline.duration_s
    covered = sum(note.duration_s for note in baseline.notes)
    coverage = covered / duration if duration > 0 else 0.0
    gaps = max(0.0, duration - covered)
    gap_ratio = gaps / duration if duration > 0 else 0.0
    note_durations = [note.duration_s for note in baseline.notes]
    median_note_duration = float(np.median(note_durations)) if note_durations else 0.0
    notes_per_second = len(baseline.notes) / duration if duration > 0 else 0.0

    warnings: list[str] = []
    if len(baseline.notes) < 8:
        warnings.append("reference has very few detected notes")
    if coverage < 0.35:
        warnings.append("reference extraction is sparse")
    if gap_ratio > 0.60:
        warnings.append("reference has large unvoiced gaps")
    if median_note_duration < 0.08:
        warnings.append("reference notes are very short; pitch tracking may be unstable")
    if notes_per_second > 8:
        warnings.append("reference changes pitch unusually often; it may be tracking noise")

    if warnings:
        level = "low" if coverage < 0.35 or len(baseline.notes) < 8 else "medium"
    else:
        level = "high"

    return BaselineQuality(
        duration_s=round(duration, 3),
        note_count=len(baseline.notes),
        voiced_coverage_ratio=round(float(coverage), 3),
        gap_ratio=round(float(gap_ratio), 3),
        median_note_duration_s=round(float(median_note_duration), 3),
        notes_per_second=round(float(notes_per_second), 3),
        level=level,
        warnings=tuple(warnings),
    )


def duration_mismatch_warnings(
    baseline_duration_s: float,
    previous_duration_s: float,
    current_duration_s: float,
    *,
    tolerance_ratio: float = 0.25,
    tolerance_s: float = 2.0,
) -> tuple[str, ...]:
    warnings: list[str] = []
    durations = {
        "reference": baseline_duration_s,
        "previous": previous_duration_s,
        "current": current_duration_s,
    }
    for label, duration in durations.items():
        if duration <= 0:
            warnings.append(f"{label} duration could not be measured")

    reference = max(baseline_duration_s, 0.001)
    for label, duration in (("previous", previous_duration_s), ("current", current_duration_s)):
        diff = abs(duration - baseline_duration_s)
        if diff > tolerance_s and diff / reference > tolerance_ratio:
            warnings.append(
                f"{label} duration differs from reference by {diff:.1f}s; trim all files to the same section"
            )

    take_diff = abs(previous_duration_s - current_duration_s)
    shorter_take = max(min(previous_duration_s, current_duration_s), 0.001)
    if take_diff > tolerance_s and take_diff / shorter_take > tolerance_ratio:
        warnings.append(
            f"previous/current durations differ by {take_diff:.1f}s; progress deltas may be misleading"
        )
    return tuple(warnings)
