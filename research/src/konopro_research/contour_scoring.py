from __future__ import annotations

from pathlib import Path

import numpy as np

from konopro_research.audio_io import load_audio
from konopro_research.baseline import hz_to_midi
from konopro_research.diagnostics import estimate_recording_confidence
from konopro_research.pitch import PitchContour, clean_pitch_contour, extract_pitch
from konopro_research.scoring import ComparisonResult, ScoreResult


TakeInput = str | Path | PitchContour


def score_take_against_reference_contour(
    take: TakeInput,
    reference: PitchContour,
    *,
    name: str = "take",
    pitch_kwargs: dict[str, object] | None = None,
    clean_kwargs: dict[str, object] | None = None,
    dtw_time_weight: float = 20.0,
    dtw_band_radius: float = 0.06,
    max_dtw_frames: int = 2400,
    pitch_error_penalty: float = 0.70,
    stability_penalty: float = 1.10,
    timing_penalty: float = 90.0,
    transposition_warning_cents: float = 90.0,
) -> ScoreResult:
    """Score a take against a reference pitch contour using dynamic time warping."""
    reference = clean_pitch_contour(reference, **(clean_kwargs or {}))
    take_contour, audio = _as_pitch_contour(take, name=name, pitch_kwargs=pitch_kwargs)
    take_contour = clean_pitch_contour(take_contour, **(clean_kwargs or {}))

    warnings: list[str] = []
    alignment = _align_voiced_contours(
        reference,
        take_contour,
        time_weight=dtw_time_weight,
        band_radius=dtw_band_radius,
        max_frames=max_dtw_frames,
    )
    if alignment is None:
        confidence = estimate_recording_confidence(audio, take_contour)
        warnings.extend(confidence.reasons)
        warnings.append("not enough voiced pitch frames for reliable contour scoring")
        return _empty_score(
            take_contour,
            reference,
            confidence.score,
            confidence.level,
            tuple(dict.fromkeys(warnings)),
        )

    ref_midi = hz_to_midi(alignment.reference_hz)
    take_midi = hz_to_midi(alignment.take_hz)
    signed_errors = (take_midi - ref_midi) * 100.0
    estimated_transposition = float(np.nanmedian(signed_errors))
    abs_errors = np.abs(signed_errors)
    mean_error = float(np.nanmean(abs_errors))
    pitch_score = _clamp_score(100.0 - mean_error * pitch_error_penalty)

    residual_errors = signed_errors - estimated_transposition
    stability_cents = float(np.nanstd(residual_errors))
    stability_score = _clamp_score(100.0 - stability_cents * stability_penalty)

    reference_voiced = np.count_nonzero(reference.voiced_mask)
    take_voiced = np.count_nonzero(take_contour.voiced_mask)
    coverage_pct = 100.0 * min(1.0, take_voiced / max(1, reference_voiced))
    coverage_score = _clamp_score(coverage_pct)

    timing_deltas = alignment.take_times_s - alignment.reference_times_s
    global_offset_s = float(np.nanmedian(timing_deltas))
    local_timing_errors = timing_deltas - global_offset_s
    median_abs_timing_error = float(np.nanmedian(np.abs(local_timing_errors)))
    timing_score = _clamp_score(100.0 - median_abs_timing_error * timing_penalty)

    confidence = estimate_recording_confidence(audio, take_contour)
    warnings.extend(confidence.reasons)
    if confidence.level == "low":
        warnings.append("recording confidence is low; interpret score carefully")
    if abs(estimated_transposition) >= transposition_warning_cents:
        semitone_shift = round(estimated_transposition / 100.0)
        warnings.append(
            f"take appears shifted by about {semitone_shift:+d} semitone(s); check key/transposition"
        )
    take_duration = _contour_duration(take_contour)
    reference_duration = _contour_duration(reference)
    if reference_duration > 0 and abs(take_duration - reference_duration) > max(2.0, reference_duration * 0.25):
        warnings.append("take duration differs from reference; trim files to the same section")

    song_correctness = 0.60 * pitch_score + 0.20 * coverage_score + 0.20 * timing_score
    technical_control = 0.70 * stability_score + 0.30 * confidence.score * 100.0
    overall = 0.50 * pitch_score + 0.20 * stability_score + 0.15 * coverage_score + 0.15 * timing_score

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
        timing_offset_s=round(float(global_offset_s), 3),
        take_duration_s=round(float(take_duration), 3),
        baseline_duration_s=round(float(reference_duration), 3),
        recording_confidence_score=round(float(confidence.score * 100.0), 2),
        recording_confidence_level=confidence.level,
        covered_notes=int(min(take_voiced, reference_voiced)),
        total_notes=int(reference_voiced),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def score_take_against_reference_contour_global_offset(
    take: TakeInput,
    reference: PitchContour,
    *,
    name: str = "take",
    take_audio_path: str | Path | None = None,
    pitch_kwargs: dict[str, object] | None = None,
    clean_kwargs: dict[str, object] | None = None,
    dtw_time_weight: float = 20.0,
    dtw_band_radius: float = 0.06,
    max_dtw_frames: int = 2400,
    pitch_error_penalty: float = 0.70,
    stability_penalty: float = 1.10,
    timing_penalty: float = 90.0,
    transposition_warning_cents: float = 90.0,
) -> ScoreResult:
    """Score a take by estimating one global offset, then comparing same-time frames.

    DTW is used only to estimate the global timing offset and timing-drift diagnostic.
    Pitch accuracy, stability, and coverage are computed from 1:1 frames after applying
    that single offset.
    """
    reference = clean_pitch_contour(reference, **(clean_kwargs or {}))
    take_contour, audio = _as_pitch_contour(take, name=name, pitch_kwargs=pitch_kwargs)
    take_contour = clean_pitch_contour(take_contour, **(clean_kwargs or {}))
    if audio is None and take_audio_path is not None:
        audio, _ = load_audio(take_audio_path)

    warnings: list[str] = []
    alignment = _align_voiced_contours(
        reference,
        take_contour,
        time_weight=dtw_time_weight,
        band_radius=dtw_band_radius,
        max_frames=max_dtw_frames,
    )
    if alignment is None:
        confidence = estimate_recording_confidence(audio, take_contour)
        warnings.extend(confidence.reasons)
        warnings.append("not enough voiced pitch frames for reliable global-offset scoring")
        return _empty_score(
            take_contour,
            reference,
            confidence.score,
            confidence.level,
            tuple(dict.fromkeys(warnings)),
        )

    timing_deltas = alignment.take_times_s - alignment.reference_times_s
    global_offset_s = float(np.nanmedian(timing_deltas))
    local_timing_errors = timing_deltas - global_offset_s
    median_abs_timing_error = float(np.nanmedian(np.abs(local_timing_errors)))
    timing_score = _clamp_score(100.0 - median_abs_timing_error * timing_penalty)

    take_hz_at_reference_times, take_available = _sample_contour_nearest(
        take_contour,
        reference.times_s + global_offset_s,
    )
    reference_expected = reference.voiced_mask
    matched = reference_expected & take_available & np.isfinite(take_hz_at_reference_times)
    matched &= take_hz_at_reference_times > 0
    reference_voiced = int(np.count_nonzero(reference_expected))
    matched_count = int(np.count_nonzero(matched))
    coverage_pct = 100.0 * matched_count / max(1, reference_voiced)
    coverage_score = _clamp_score(coverage_pct)

    if matched_count < 3:
        mean_error = 999.0
        estimated_transposition = 0.0
        stability_cents = 999.0
        pitch_score = 0.0
        stability_score = 0.0
        warnings.append("not enough same-time voiced frames for reliable global-offset pitch scoring")
    else:
        ref_midi = hz_to_midi(reference.frequencies_hz[matched])
        take_midi = hz_to_midi(take_hz_at_reference_times[matched])
        signed_errors = (take_midi - ref_midi) * 100.0
        estimated_transposition = float(np.nanmedian(signed_errors))
        abs_errors = np.abs(signed_errors)
        mean_error = float(np.nanmean(abs_errors))
        pitch_score = _clamp_score(100.0 - mean_error * pitch_error_penalty)

        residual_errors = signed_errors - estimated_transposition
        stability_cents = float(np.nanstd(residual_errors))
        stability_score = _clamp_score(100.0 - stability_cents * stability_penalty)

    confidence = estimate_recording_confidence(audio, take_contour)
    warnings.extend(confidence.reasons)
    if confidence.level == "low":
        warnings.append("recording confidence is low; interpret score carefully")
    if abs(estimated_transposition) >= transposition_warning_cents:
        semitone_shift = round(estimated_transposition / 100.0)
        warnings.append(
            f"take appears shifted by about {semitone_shift:+d} semitone(s); check key/transposition"
        )
    take_duration = _contour_duration(take_contour)
    reference_duration = _contour_duration(reference)
    if reference_duration > 0 and abs(take_duration - reference_duration) > max(2.0, reference_duration * 0.25):
        warnings.append("take duration differs from reference; trim files to the same section")

    song_correctness = 0.60 * pitch_score + 0.20 * coverage_score + 0.20 * timing_score
    technical_control = 0.70 * stability_score + 0.30 * confidence.score * 100.0
    overall = 0.50 * pitch_score + 0.20 * stability_score + 0.15 * coverage_score + 0.15 * timing_score

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
        timing_offset_s=round(float(global_offset_s), 3),
        take_duration_s=round(float(take_duration), 3),
        baseline_duration_s=round(float(reference_duration), 3),
        recording_confidence_score=round(float(confidence.score * 100.0), 2),
        recording_confidence_level=confidence.level,
        covered_notes=matched_count,
        total_notes=reference_voiced,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def compare_takes_to_reference_contour(
    previous: TakeInput,
    current: TakeInput,
    reference: PitchContour,
    **score_kwargs: object,
) -> ComparisonResult:
    previous_score = score_take_against_reference_contour(previous, reference, name="previous", **score_kwargs)
    current_score = score_take_against_reference_contour(current, reference, name="current", **score_kwargs)

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


def compare_takes_to_reference_contour_global_offset(
    previous: TakeInput,
    current: TakeInput,
    reference: PitchContour,
    *,
    previous_audio_path: str | Path | None = None,
    current_audio_path: str | Path | None = None,
    **score_kwargs: object,
) -> ComparisonResult:
    previous_score = score_take_against_reference_contour_global_offset(
        previous,
        reference,
        name="previous",
        take_audio_path=previous_audio_path,
        **score_kwargs,
    )
    current_score = score_take_against_reference_contour_global_offset(
        current,
        reference,
        name="current",
        take_audio_path=current_audio_path,
        **score_kwargs,
    )

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


def contour_timing_debug(
    take: TakeInput,
    reference: PitchContour,
    *,
    name: str = "take",
    pitch_kwargs: dict[str, object] | None = None,
    clean_kwargs: dict[str, object] | None = None,
    dtw_time_weight: float = 20.0,
    dtw_band_radius: float = 0.06,
    max_dtw_frames: int = 2400,
    timing_penalty: float = 90.0,
) -> dict[str, object]:
    reference = clean_pitch_contour(reference, **(clean_kwargs or {}))
    take_contour, _ = _as_pitch_contour(take, name=name, pitch_kwargs=pitch_kwargs)
    take_contour = clean_pitch_contour(take_contour, **(clean_kwargs or {}))
    alignment = _align_voiced_contours(
        reference,
        take_contour,
        time_weight=dtw_time_weight,
        band_radius=dtw_band_radius,
        max_frames=max_dtw_frames,
    )
    if alignment is None:
        return {
            "matched_pairs_count": 0,
            "interpretation": "Not enough matched voiced frames to compute timing diagnostics.",
            "score_currently_used_by_final_result": "absolute timing score",
        }

    raw_delta = alignment.take_times_s - alignment.reference_times_s
    global_offset = float(np.nanmedian(raw_delta))
    local_error = raw_delta - global_offset
    median_abs_raw_delta = float(np.nanmedian(np.abs(raw_delta)))
    median_abs_local_error = float(np.nanmedian(np.abs(local_error)))
    old_score = _clamp_score(100.0 - median_abs_raw_delta * timing_penalty)
    corrected_score = _clamp_score(100.0 - median_abs_local_error * timing_penalty)
    spread = float(np.nanmedian(np.abs(raw_delta - global_offset)))
    if median_abs_raw_delta > 1.0 and median_abs_local_error < 0.5:
        interpretation = (
            "Raw timing offset is large but local timing error is small, "
            "suggesting recording start offset rather than rhythmic drift."
        )
    elif median_abs_local_error >= 0.5:
        interpretation = (
            "Local timing error is still large after offset correction, suggesting DTW mismatch, "
            "sparse contour, drift, or bad extraction."
        )
    else:
        interpretation = "Raw and local timing errors are both small."

    return {
        "matched_pairs_count": int(raw_delta.size),
        "raw_delta_s_sample": [round(float(value), 3) for value in raw_delta[:12]],
        "raw_delta_s_median": round(global_offset, 3),
        "raw_delta_s_mad_or_std": round(spread, 3),
        "global_offset_s": round(global_offset, 3),
        "local_error_s_sample": [round(float(value), 3) for value in local_error[:12]],
        "median_abs_raw_delta_s": round(median_abs_raw_delta, 3),
        "median_abs_local_error_s": round(median_abs_local_error, 3),
        "old_timing_score_absolute": round(float(old_score), 2),
        "new_timing_score_offset_corrected": round(float(corrected_score), 2),
        "score_currently_used_by_final_result": "new_timing_score_offset_corrected",
        "interpretation": interpretation,
    }


class _ContourAlignment:
    def __init__(
        self,
        reference_times_s: np.ndarray,
        reference_hz: np.ndarray,
        take_times_s: np.ndarray,
        take_hz: np.ndarray,
    ) -> None:
        self.reference_times_s = reference_times_s
        self.reference_hz = reference_hz
        self.take_times_s = take_times_s
        self.take_hz = take_hz


def _align_voiced_contours(
    reference: PitchContour,
    take: PitchContour,
    *,
    time_weight: float,
    band_radius: float,
    max_frames: int,
) -> _ContourAlignment | None:
    ref_times, ref_hz = _voiced_sequence(reference)
    take_times, take_hz = _voiced_sequence(take)
    if len(ref_hz) < 3 or len(take_hz) < 3:
        return None

    ref_times, ref_hz = _thin_sequence(ref_times, ref_hz, max_frames=max_frames)
    take_times, take_hz = _thin_sequence(take_times, take_hz, max_frames=max_frames)
    ref_midi = hz_to_midi(ref_hz)
    take_midi = hz_to_midi(take_hz)
    ref_features = np.vstack([ref_midi, _normalized_time_feature(ref_times, weight=time_weight)])
    take_features = np.vstack([take_midi, _normalized_time_feature(take_times, weight=time_weight)])

    try:
        import librosa

        _, path = librosa.sequence.dtw(
            X=ref_features,
            Y=take_features,
            metric="euclidean",
            global_constraints=True,
            band_rad=band_radius,
        )
        pairs = path[::-1]
        ref_indices = pairs[:, 0]
        take_indices = pairs[:, 1]
    except Exception:
        length = min(len(ref_midi), len(take_midi))
        ref_indices = np.linspace(0, len(ref_midi) - 1, num=length).round().astype(int)
        take_indices = np.linspace(0, len(take_midi) - 1, num=length).round().astype(int)

    return _ContourAlignment(
        reference_times_s=ref_times[ref_indices],
        reference_hz=ref_hz[ref_indices],
        take_times_s=take_times[take_indices],
        take_hz=take_hz[take_indices],
    )


def _as_pitch_contour(
    take: TakeInput,
    *,
    name: str,
    pitch_kwargs: dict[str, object] | None = None,
) -> tuple[PitchContour, np.ndarray | None]:
    if isinstance(take, PitchContour):
        return take, None
    audio, sample_rate = load_audio(take)
    return extract_pitch(audio, sample_rate, name=name, **(pitch_kwargs or {})), audio


def _voiced_sequence(contour: PitchContour) -> tuple[np.ndarray, np.ndarray]:
    mask = contour.voiced_mask
    return contour.times_s[mask], contour.frequencies_hz[mask]


def _sample_contour_nearest(
    contour: PitchContour,
    target_times_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target_times = np.asarray(target_times_s, dtype=float)
    sampled = np.full(target_times.shape, np.nan, dtype=float)
    available = np.zeros(target_times.shape, dtype=bool)
    if contour.times_s.size == 0 or target_times.size == 0:
        return sampled, available

    indices = np.searchsorted(contour.times_s, target_times)
    right = np.clip(indices, 0, contour.times_s.size - 1)
    left = np.clip(indices - 1, 0, contour.times_s.size - 1)
    left_distance = np.abs(target_times - contour.times_s[left])
    right_distance = np.abs(target_times - contour.times_s[right])
    nearest = np.where(left_distance <= right_distance, left, right)
    nearest_distance = np.minimum(left_distance, right_distance)

    frame_step = _median_frame_step(contour)
    max_distance = max(frame_step * 1.5, 1e-6)
    available = nearest_distance <= max_distance
    sampled[available] = contour.frequencies_hz[nearest[available]]
    return sampled, available


def _median_frame_step(contour: PitchContour) -> float:
    if contour.times_s.size < 2:
        return 0.02
    diffs = np.diff(np.sort(contour.times_s))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 0.02
    return float(np.nanmedian(diffs))


def _thin_sequence(
    times_s: np.ndarray,
    hz: np.ndarray,
    *,
    max_frames: int = 2400,
) -> tuple[np.ndarray, np.ndarray]:
    if len(hz) <= max_frames:
        return times_s, hz
    stride = int(np.ceil(len(hz) / max_frames))
    return times_s[::stride], hz[::stride]


def _normalized_time_feature(times_s: np.ndarray, *, weight: float) -> np.ndarray:
    if len(times_s) == 0:
        return times_s
    start = float(np.nanmin(times_s))
    stop = float(np.nanmax(times_s))
    duration = max(stop - start, 0.001)
    return ((times_s - start) / duration) * weight


def _empty_score(
    take: PitchContour,
    reference: PitchContour,
    confidence_score: float,
    confidence_level: str,
    warnings: tuple[str, ...],
) -> ScoreResult:
    return ScoreResult(
        overall_score=0.0,
        song_correctness_score=0.0,
        technical_control_score=round(float(confidence_score * 30.0), 2),
        pitch_accuracy_score=0.0,
        timing_score=0.0,
        stability_score=0.0,
        coverage_score=0.0,
        mean_pitch_error_cents=999.0,
        estimated_transposition_cents=0.0,
        pitch_stability_cents=999.0,
        note_coverage_pct=0.0,
        timing_offset_s=0.0,
        take_duration_s=round(float(_contour_duration(take)), 3),
        baseline_duration_s=round(float(_contour_duration(reference)), 3),
        recording_confidence_score=round(float(confidence_score * 100.0), 2),
        recording_confidence_level=confidence_level,
        covered_notes=0,
        total_notes=int(np.count_nonzero(reference.voiced_mask)),
        warnings=warnings,
    )


def _feedback_by_category(
    previous: ScoreResult,
    current: ScoreResult,
) -> dict[str, tuple[str, ...]]:
    categories: dict[str, list[str]] = {
        "song_correctness": [],
        "technical_control": [],
        "recording_confidence": [],
    }
    if current.pitch_accuracy_score > previous.pitch_accuracy_score + 3:
        categories["song_correctness"].append("Pitch contour matched the reference more closely.")
    elif current.pitch_accuracy_score < previous.pitch_accuracy_score - 3:
        categories["song_correctness"].append("Pitch contour moved farther from the reference.")

    if current.stability_score > previous.stability_score + 3:
        categories["technical_control"].append("Pitch deviations became more consistent.")
    elif current.stability_score < previous.stability_score - 3:
        categories["technical_control"].append("Pitch deviations became less consistent.")

    if current.coverage_score < previous.coverage_score - 5:
        categories["song_correctness"].append("The current take has less detected singing coverage.")
    elif current.coverage_score > previous.coverage_score + 5:
        categories["song_correctness"].append("The current take has more detected singing coverage.")

    if current.recording_confidence_score < 45:
        categories["recording_confidence"].append("Recording confidence is low; interpret scores carefully.")

    compact = {key: tuple(value) for key, value in categories.items() if value}
    return compact or {"summary": ("No major metric changed enough to call out.",)}


def _contour_duration(contour: PitchContour) -> float:
    if contour.times_s.size == 0:
        return 0.0
    return float(np.nanmax(contour.times_s) - np.nanmin(contour.times_s))


def _clamp_score(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))
