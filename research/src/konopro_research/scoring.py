from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from konopro_research.alignment import estimate_global_offset
from konopro_research.audio_io import load_audio
from konopro_research.baseline import MelodyBaseline, cents_difference
from konopro_research.diagnostics import estimate_recording_confidence
from konopro_research.pitch import PitchContour, clean_pitch_contour, extract_pitch


@dataclass(frozen=True)
class ScoreResult:
    overall_score: float
    song_correctness_score: float
    technical_control_score: float
    pitch_accuracy_score: float
    timing_score: float
    stability_score: float
    coverage_score: float
    mean_pitch_error_cents: float
    estimated_transposition_cents: float
    pitch_stability_cents: float
    note_coverage_pct: float
    timing_offset_s: float
    take_duration_s: float
    baseline_duration_s: float
    recording_confidence_score: float
    recording_confidence_level: str
    covered_notes: int
    total_notes: int
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonResult:
    previous: ScoreResult
    current: ScoreResult
    overall_delta: float
    pitch_accuracy_delta: float
    stability_delta: float
    coverage_delta: float
    timing_delta: float
    verdict: str
    feedback: tuple[str, ...]
    feedback_by_category: dict[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


TakeInput = str | Path | PitchContour


def score_take(take: TakeInput, baseline: MelodyBaseline, *, name: str = "take") -> ScoreResult:
    contour, audio = _as_pitch_contour(take, name=name)
    contour = clean_pitch_contour(contour)
    alignment = estimate_global_offset(contour, baseline)
    take_duration = _contour_duration(contour)

    reference_times = contour.times_s - alignment.offset_s
    expected = baseline.hz_at(reference_times)
    voiced = contour.voiced_mask
    matched = voiced & np.isfinite(expected) & (expected > 0)

    warnings: list[str] = []
    estimated_transposition = 0.0
    if np.count_nonzero(matched) < 3:
        warnings.append("not enough matched voiced frames for reliable pitch scoring")
        mean_error = 999.0
        pitch_score = 0.0
    else:
        signed_errors = cents_difference(contour.frequencies_hz[matched], expected[matched])
        estimated_transposition = float(np.nanmedian(signed_errors))
        errors = np.abs(signed_errors)
        mean_error = float(np.nanmean(errors))
        pitch_score = _clamp_score(100.0 - mean_error * 0.70)
        semitone_shift = round(estimated_transposition / 100.0)
        if abs(estimated_transposition) >= 90.0:
            warnings.append(
                f"take appears shifted by about {semitone_shift:+d} semitone(s); check key/transposition"
            )

    covered_notes, total_notes, note_stabilities = _note_coverage_and_stability(
        contour,
        baseline,
        alignment.offset_s,
    )
    coverage_pct = 100.0 * covered_notes / max(1, total_notes)
    coverage_score = _clamp_score(coverage_pct)

    if note_stabilities:
        stability_cents = float(np.nanmedian(note_stabilities))
        stability_score = _clamp_score(100.0 - stability_cents * 1.10)
    else:
        stability_cents = 999.0
        stability_score = 0.0

    timing_score = _clamp_score(100.0 - abs(alignment.offset_s) * 180.0)
    confidence = estimate_recording_confidence(audio, contour)
    warnings.extend(confidence.reasons)
    if confidence.level == "low":
        warnings.append("recording confidence is low; interpret score carefully")
    warnings.extend(_duration_warnings(take_duration, baseline.duration_s))

    song_correctness = 0.55 * pitch_score + 0.25 * coverage_score + 0.20 * timing_score
    technical_control = 0.70 * stability_score + 0.30 * confidence.score * 100.0

    overall = (
        0.45 * pitch_score
        + 0.20 * stability_score
        + 0.20 * coverage_score
        + 0.15 * timing_score
    )

    return ScoreResult(
        overall_score=round(float(overall), 2),
        song_correctness_score=round(float(song_correctness), 2),
        technical_control_score=round(float(technical_control), 2),
        pitch_accuracy_score=round(float(pitch_score), 2),
        timing_score=round(float(timing_score), 2),
        stability_score=round(float(stability_score), 2),
        coverage_score=round(float(coverage_score), 2),
        mean_pitch_error_cents=round(float(mean_error), 2),
        estimated_transposition_cents=round(float(estimated_transposition), 2),
        pitch_stability_cents=round(float(stability_cents), 2),
        note_coverage_pct=round(float(coverage_pct), 2),
        timing_offset_s=round(float(alignment.offset_s), 3),
        take_duration_s=round(float(take_duration), 3),
        baseline_duration_s=round(float(baseline.duration_s), 3),
        recording_confidence_score=round(float(confidence.score * 100.0), 2),
        recording_confidence_level=confidence.level,
        covered_notes=covered_notes,
        total_notes=total_notes,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def compare_takes(previous: TakeInput, current: TakeInput, baseline: MelodyBaseline) -> ComparisonResult:
    previous_score = score_take(previous, baseline, name="previous")
    current_score = score_take(current, baseline, name="current")

    overall_delta = round(current_score.overall_score - previous_score.overall_score, 2)
    pitch_delta = round(current_score.pitch_accuracy_score - previous_score.pitch_accuracy_score, 2)
    stability_delta = round(current_score.stability_score - previous_score.stability_score, 2)
    coverage_delta = round(current_score.coverage_score - previous_score.coverage_score, 2)
    timing_delta = round(current_score.timing_score - previous_score.timing_score, 2)

    if overall_delta >= 5:
        verdict = "improved"
    elif overall_delta <= -5:
        verdict = "declined"
    else:
        verdict = "roughly unchanged"
    feedback_by_category = _feedback_by_category(previous_score, current_score)

    return ComparisonResult(
        previous=previous_score,
        current=current_score,
        overall_delta=overall_delta,
        pitch_accuracy_delta=pitch_delta,
        stability_delta=stability_delta,
        coverage_delta=coverage_delta,
        timing_delta=timing_delta,
        verdict=verdict,
        feedback=tuple(message for messages in feedback_by_category.values() for message in messages),
        feedback_by_category=feedback_by_category,
    )


def _as_pitch_contour(take: TakeInput, *, name: str) -> tuple[PitchContour, np.ndarray | None]:
    if isinstance(take, PitchContour):
        return take, None
    audio, sample_rate = load_audio(take)
    return extract_pitch(audio, sample_rate, name=name), audio


def _note_coverage_and_stability(
    contour: PitchContour,
    baseline: MelodyBaseline,
    offset_s: float,
) -> tuple[int, int, list[float]]:
    covered = 0
    stabilities: list[float] = []
    reference_times = contour.times_s - offset_s
    for note in baseline.notes:
        in_note = (reference_times >= note.start_s) & (reference_times < note.end_s)
        voiced_in_note = in_note & contour.voiced_mask
        expected_count = max(1, np.count_nonzero(in_note))
        voiced_ratio = np.count_nonzero(voiced_in_note) / expected_count
        if voiced_ratio >= 0.35 and np.count_nonzero(voiced_in_note) >= 3:
            covered += 1
            errors = cents_difference(
                contour.frequencies_hz[voiced_in_note],
                np.full(np.count_nonzero(voiced_in_note), note.frequency_hz),
            )
            stabilities.append(float(np.nanstd(errors)))
    return covered, len(baseline.notes), stabilities


def _feedback(previous: ScoreResult, current: ScoreResult) -> tuple[str, ...]:
    return tuple(message for messages in _feedback_by_category(previous, current).values() for message in messages)


def _feedback_by_category(
    previous: ScoreResult,
    current: ScoreResult,
) -> dict[str, tuple[str, ...]]:
    messages: list[str] = []
    categories: dict[str, list[str]] = {
        "song_correctness": [],
        "technical_control": [],
        "recording_confidence": [],
    }
    if current.pitch_accuracy_score > previous.pitch_accuracy_score + 3:
        categories["song_correctness"].append("Pitch accuracy improved against the reference.")
    elif current.pitch_accuracy_score < previous.pitch_accuracy_score - 3:
        categories["song_correctness"].append("Pitch accuracy moved farther from the reference.")

    if current.stability_score > previous.stability_score + 3:
        categories["technical_control"].append("Pitch stability improved.")
    elif current.stability_score < previous.stability_score - 3:
        categories["technical_control"].append("Pitch stability became less consistent.")

    if current.coverage_score < previous.coverage_score - 5:
        categories["song_correctness"].append("The current take missed more of the expected notes.")
    elif current.coverage_score > previous.coverage_score + 5:
        categories["song_correctness"].append("The current take covered more of the melody.")

    if (
        current.stability_score > previous.stability_score + 5
        and current.pitch_accuracy_score < previous.pitch_accuracy_score - 5
    ):
        categories["song_correctness"].append(
            "The take is more stable, but less correct against the melody."
        )

    if current.recording_confidence_score < 45:
        categories["recording_confidence"].append("Recording confidence is low; interpret scores carefully.")

    compact = {key: tuple(value) for key, value in categories.items() if value}
    if compact:
        return compact

    messages.append("No major metric changed enough to call out.")
    return {"summary": tuple(messages)}


def _clamp_score(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))


def _contour_duration(contour: PitchContour) -> float:
    if contour.times_s.size == 0:
        return 0.0
    return float(np.nanmax(contour.times_s) - np.nanmin(contour.times_s))


def _duration_warnings(take_duration_s: float, baseline_duration_s: float) -> tuple[str, ...]:
    if baseline_duration_s <= 0 or take_duration_s <= 0:
        return ()
    diff = abs(take_duration_s - baseline_duration_s)
    if diff > 2.0 and diff / max(baseline_duration_s, 0.001) > 0.25:
        return (
            f"take duration differs from baseline by {diff:.1f}s; trim files to the same section",
        )
    return ()
