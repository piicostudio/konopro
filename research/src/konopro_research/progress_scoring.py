from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from konopro_research.audio_io import load_audio
from konopro_research.contour_scoring import compare_takes_to_reference_contour
from konopro_research.matching import (
    SectionMatch,
    SongSection,
    crop_contour,
    match_query_to_sections,
)
from konopro_research.pitch import PitchContour, clean_pitch_contour, extract_pitch
from konopro_research.scoring import ComparisonResult


TakeInput = str | Path | PitchContour


METRIC_LABELS = {
    "overall": "Overall diagnostic",
    "pitch_correctness": "Pitch correctness",
    "technical_control": "Technical control",
    "timing": "Timing",
    "stability": "Stability",
    "coverage": "Coverage",
    "confidence": "Recording confidence",
}


@dataclass(frozen=True)
class MatchedTakeWindow:
    role: str
    start_s: float
    end_s: float
    duration_s: float
    voiced_frame_count: int
    coverage_score: float
    match_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "duration_s": round(self.duration_s, 3),
            "voiced_frame_count": self.voiced_frame_count,
            "coverage_score": round(self.coverage_score, 2),
            "match_score": round(float(self.match_score), 2) if self.match_score is not None else None,
        }


@dataclass(frozen=True)
class ProgressConfidence:
    can_score: bool
    confidence_level: str
    score: float
    match_score: float
    match_margin: float | None
    coverage_score: float
    recording_confidence_score: float
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "can_score": self.can_score,
            "confidence_level": self.confidence_level,
            "score": round(self.score, 2),
            "match_score": round(self.match_score, 2),
            "match_margin": round(self.match_margin, 2) if self.match_margin is not None else None,
            "coverage_score": round(self.coverage_score, 2),
            "recording_confidence_score": round(self.recording_confidence_score, 2),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class MatchedProgressScore:
    section: SongSection
    current_match: SectionMatch
    previous_match: SectionMatch
    reference_window: MatchedTakeWindow
    previous_window: MatchedTakeWindow
    current_window: MatchedTakeWindow
    comparison: ComparisonResult
    confidence: ProgressConfidence
    verdict: str
    feedback: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    previous_crop: PitchContour | None = None
    current_crop: PitchContour | None = None
    reference_crop: PitchContour | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "section": self.section.to_dict(),
            "current_match": self.current_match.to_dict(),
            "previous_match": self.previous_match.to_dict(),
            "windows": {
                "reference": self.reference_window.to_dict(),
                "previous": self.previous_window.to_dict(),
                "current": self.current_window.to_dict(),
            },
            "scores": {
                "previous": self.comparison.previous.to_dict(),
                "current": self.comparison.current.to_dict(),
            },
            "deltas": {
                "overall": self.comparison.overall_delta,
                "pitch_correctness": self.comparison.pitch_accuracy_delta,
                "stability": self.comparison.stability_delta,
                "coverage": self.comparison.coverage_delta,
                "timing": self.comparison.timing_delta,
            },
            "metric_labels": dict(METRIC_LABELS),
            "confidence": self.confidence.to_dict(),
            "verdict": self.verdict,
            "feedback": list(self.feedback),
            "warnings": list(self.warnings),
        }


def score_matched_section_progress(
    previous: TakeInput,
    current: TakeInput,
    sections: tuple[SongSection, ...] | list[SongSection],
    *,
    pitch_kwargs: dict[str, object] | None = None,
    clean_kwargs: dict[str, object] | None = None,
    top_k: int = 5,
    query_hop_s: float = 0.5,
    max_query_windows: int = 120,
    min_match_score: float = 70.0,
    min_match_margin: float = 8.0,
    min_coverage_score: float = 45.0,
    low_recording_confidence_score: float = 45.0,
    score_kwargs: dict[str, object] | None = None,
) -> MatchedProgressScore:
    if not sections:
        raise ValueError("at least one reference section is required")

    match_clean_kwargs = {"min_confidence": 0.0}
    if clean_kwargs:
        match_clean_kwargs.update(clean_kwargs)
    previous_contour = _as_contour(previous, name="previous", pitch_kwargs=pitch_kwargs, clean_kwargs=match_clean_kwargs)
    current_contour = _as_contour(current, name="current", pitch_kwargs=pitch_kwargs, clean_kwargs=match_clean_kwargs)

    current_result = match_query_to_sections(
        current_contour,
        sections,
        top_k=max(2, int(top_k)),
        query_hop_s=query_hop_s,
        max_query_windows=max_query_windows,
    )
    if current_result.best is None:
        raise ValueError("current take could not be matched to any reference section")

    selected_section = current_result.best.section
    previous_result = match_query_to_sections(
        previous_contour,
        [selected_section],
        top_k=1,
        query_hop_s=query_hop_s,
        max_query_windows=max_query_windows,
    )
    if previous_result.best is None:
        raise ValueError("previous take could not be matched to the selected reference section")

    current_match = current_result.best
    previous_match = previous_result.best
    current_crop = crop_contour(
        current_contour,
        current_match.query_start_s,
        current_match.query_end_s,
        name="current matched section",
    )
    previous_crop = crop_contour(
        previous_contour,
        previous_match.query_start_s,
        previous_match.query_end_s,
        name="previous matched section",
    )
    reference_crop = crop_contour(
        selected_section.contour,
        0.0,
        selected_section.duration_s,
        name="matched reference section",
    )

    comparison = compare_takes_to_reference_contour(
        previous_crop,
        current_crop,
        reference_crop,
        **(score_kwargs or {}),
    )

    match_margin = _match_margin(current_result.candidates)
    confidence = _progress_confidence(
        comparison=comparison,
        current_match=current_match,
        match_margin=match_margin,
        min_match_score=min_match_score,
        min_match_margin=min_match_margin,
        min_coverage_score=min_coverage_score,
        low_recording_confidence_score=low_recording_confidence_score,
    )
    verdict = "insufficient confidence" if not confidence.can_score else comparison.verdict
    feedback = _progress_feedback(comparison)
    warnings = tuple(
        dict.fromkeys(
            [
                *current_result.warnings,
                *current_match.warnings,
                *previous_match.warnings,
                *comparison.previous.warnings,
                *comparison.current.warnings,
                *confidence.reasons,
            ]
        )
    )

    return MatchedProgressScore(
        section=selected_section,
        current_match=current_match,
        previous_match=previous_match,
        reference_window=_window_from_contour("reference", reference_crop, 100.0, selected_section.duration_s),
        previous_window=_window_from_match("previous", previous_crop, previous_match, comparison.previous.coverage_score),
        current_window=_window_from_match("current", current_crop, current_match, comparison.current.coverage_score),
        comparison=comparison,
        confidence=confidence,
        verdict=verdict,
        feedback=feedback,
        warnings=warnings,
        previous_crop=previous_crop,
        current_crop=current_crop,
        reference_crop=reference_crop,
    )


def _as_contour(
    take: TakeInput,
    *,
    name: str,
    pitch_kwargs: dict[str, object] | None,
    clean_kwargs: dict[str, object],
) -> PitchContour:
    if isinstance(take, PitchContour):
        contour = take
    else:
        audio, sample_rate = load_audio(take)
        contour = extract_pitch(audio, sample_rate, name=name, **(pitch_kwargs or {}))
    return clean_pitch_contour(contour, **clean_kwargs)


def _match_margin(candidates: tuple[SectionMatch, ...]) -> float | None:
    if len(candidates) < 2:
        return None
    return round(float(candidates[0].score - candidates[1].score), 2)


def _progress_confidence(
    *,
    comparison: ComparisonResult,
    current_match: SectionMatch,
    match_margin: float | None,
    min_match_score: float,
    min_match_margin: float,
    min_coverage_score: float,
    low_recording_confidence_score: float,
) -> ProgressConfidence:
    reasons: list[str] = []
    coverage_score = min(comparison.previous.coverage_score, comparison.current.coverage_score)
    recording_confidence = min(
        comparison.previous.recording_confidence_score,
        comparison.current.recording_confidence_score,
    )

    if current_match.score < min_match_score:
        reasons.append(f"match score below threshold ({current_match.score:.1f} < {min_match_score:.1f})")
    if match_margin is not None and match_margin < min_match_margin:
        reasons.append(f"ambiguous section match; top-two margin is {match_margin:.1f}")
    if coverage_score < min_coverage_score:
        reasons.append(f"coverage below threshold ({coverage_score:.1f} < {min_coverage_score:.1f})")
    if recording_confidence < low_recording_confidence_score:
        reasons.append(
            f"recording confidence below threshold ({recording_confidence:.1f} < {low_recording_confidence_score:.1f})"
        )

    can_score = not reasons
    score_parts = [
        current_match.score,
        100.0 if match_margin is None else min(100.0, max(0.0, match_margin / max(min_match_margin, 0.001) * 100.0)),
        coverage_score,
        recording_confidence,
    ]
    score = float(np.nanmean(score_parts))
    if can_score and score >= 80:
        level = "high"
    elif can_score and score >= 60:
        level = "medium"
    else:
        level = "low" if not can_score else "medium"
    return ProgressConfidence(
        can_score=can_score,
        confidence_level=level,
        score=round(score, 2),
        match_score=float(current_match.score),
        match_margin=match_margin,
        coverage_score=float(coverage_score),
        recording_confidence_score=float(recording_confidence),
        reasons=tuple(reasons),
    )


def _progress_feedback(comparison: ComparisonResult) -> tuple[str, ...]:
    messages = list(comparison.feedback)
    if comparison.current.stability_score > comparison.previous.stability_score + 5 and (
        comparison.current.pitch_accuracy_score < comparison.previous.pitch_accuracy_score - 5
    ):
        messages.append("The current take is more stable, but less correct against the matched reference section.")
    return tuple(dict.fromkeys(messages))


def _window_from_match(
    role: str,
    contour: PitchContour,
    match: SectionMatch,
    coverage_score: float,
) -> MatchedTakeWindow:
    return MatchedTakeWindow(
        role=role,
        start_s=float(match.query_start_s),
        end_s=float(match.query_end_s),
        duration_s=float(max(0.0, match.query_end_s - match.query_start_s)),
        voiced_frame_count=int(np.count_nonzero(contour.voiced_mask)),
        coverage_score=float(coverage_score),
        match_score=float(match.score),
    )


def _window_from_contour(
    role: str,
    contour: PitchContour,
    coverage_score: float,
    duration_s: float,
) -> MatchedTakeWindow:
    return MatchedTakeWindow(
        role=role,
        start_s=0.0,
        end_s=float(duration_s),
        duration_s=float(duration_s),
        voiced_frame_count=int(np.count_nonzero(contour.voiced_mask)),
        coverage_score=float(coverage_score),
    )
