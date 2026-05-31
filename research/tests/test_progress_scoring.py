from __future__ import annotations

import numpy as np

from konopro_research.baseline import MelodyBaseline, MelodyNote, demo_baseline, midi_to_hz
from konopro_research.matching import build_demo_section_catalog, sections_from_baseline
from konopro_research.pitch import PitchContour
from konopro_research.progress_scoring import score_matched_section_progress


def test_matched_progress_scores_same_reference_section_with_cropped_windows() -> None:
    catalog = build_demo_section_catalog()
    baseline = demo_baseline()
    previous = with_padding(contour_from_baseline(baseline, cents_error=80, vibrato_cents=28), before_s=1.6, after_s=1.2)
    current = with_padding(contour_from_baseline(baseline, cents_error=15, vibrato_cents=7), before_s=2.2, after_s=1.0)

    result = score_matched_section_progress(previous, current, catalog)

    assert result.section.section_label == "Chorus"
    assert result.current_window.start_s > 1.5
    assert result.previous_window.start_s >= 1.0
    assert result.current_window.end_s < contour_duration(current)
    assert result.previous_window.end_s < contour_duration(previous)
    assert result.comparison.verdict == "improved"
    assert result.comparison.pitch_accuracy_delta > 30
    assert result.confidence.can_score is True


def test_matched_progress_result_dictionary_has_clear_labels() -> None:
    catalog = build_demo_section_catalog()
    baseline = demo_baseline()
    previous = contour_from_baseline(baseline, cents_error=70, vibrato_cents=20)
    current = contour_from_baseline(baseline, cents_error=15, vibrato_cents=7)

    details = score_matched_section_progress(previous, current, catalog).to_dict()

    assert details["section"]["section"] == "Chorus"
    assert details["verdict"] == "improved"
    assert "current" in details["scores"]
    assert "previous" in details["scores"]
    assert "deltas" in details
    assert "song" not in details["metric_labels"]
    assert details["metric_labels"]["overall"] == "Overall diagnostic"
    assert details["metric_labels"]["pitch_correctness"] == "Pitch correctness"
    assert details["metric_labels"]["technical_control"] == "Technical control"


def test_stable_but_wrong_is_not_reported_as_matched_progress() -> None:
    catalog = build_demo_section_catalog()
    baseline = demo_baseline()
    previous = contour_from_baseline(baseline, cents_error=35, vibrato_cents=70)
    stable_wrong = contour_from_baseline(baseline, cents_error=100, vibrato_cents=2)

    result = score_matched_section_progress(previous, stable_wrong, catalog)

    assert result.comparison.current.stability_score > result.comparison.previous.stability_score
    assert result.comparison.current.pitch_accuracy_score < result.comparison.previous.pitch_accuracy_score
    assert result.verdict != "improved"
    assert any("stable" in message.lower() for message in result.feedback)


def test_low_match_score_blocks_strong_progress_claim() -> None:
    sections = sections_from_baseline(
        MelodyBaseline(
            tuple(MelodyNote(index * 0.6, index * 0.6 + 0.5, midi) for index, midi in enumerate([60, 72, 61, 73, 62, 74])),
            title="Mismatch",
        )
    )
    baseline = demo_baseline()
    previous = contour_from_baseline(baseline, cents_error=80, vibrato_cents=20)
    current = contour_from_baseline(baseline, cents_error=10, vibrato_cents=5)

    result = score_matched_section_progress(
        previous,
        current,
        sections,
        min_match_score=95.0,
    )

    assert result.confidence.can_score is False
    assert result.verdict == "insufficient confidence"
    assert any("match score" in reason for reason in result.confidence.reasons)


def test_ambiguous_section_match_blocks_strong_progress_claim() -> None:
    baseline = demo_baseline()
    duplicate_sections = sections_from_baseline(baseline, song_id="a", section_label="A") + sections_from_baseline(
        baseline,
        song_id="b",
        section_label="B",
    )
    previous = contour_from_baseline(baseline, cents_error=70, vibrato_cents=20)
    current = contour_from_baseline(baseline, cents_error=10, vibrato_cents=5)

    result = score_matched_section_progress(
        previous,
        current,
        duplicate_sections,
        min_match_margin=8.0,
    )

    assert result.confidence.can_score is False
    assert result.confidence.match_margin < 1.0
    assert result.verdict == "insufficient confidence"
    assert any("ambiguous" in reason for reason in result.confidence.reasons)


def test_low_coverage_blocks_strong_progress_claim() -> None:
    catalog = build_demo_section_catalog()
    baseline = demo_baseline()
    previous = contour_from_baseline(baseline, cents_error=80, vibrato_cents=20)
    current = contour_from_baseline(baseline, cents_error=10, vibrato_cents=5, mute_note_indices={1, 2, 3, 4, 5, 6, 7})

    result = score_matched_section_progress(
        previous,
        current,
        catalog,
        min_coverage_score=80.0,
    )

    assert result.confidence.can_score is False
    assert result.verdict == "insufficient confidence"
    assert any("coverage" in reason for reason in result.confidence.reasons)


def test_low_recording_confidence_warns_without_hiding_reason() -> None:
    catalog = build_demo_section_catalog()
    baseline = demo_baseline()
    previous = contour_from_baseline(baseline, cents_error=80, vibrato_cents=20)
    current_base = contour_from_baseline(baseline, cents_error=10, vibrato_cents=5)
    current = PitchContour(
        current_base.times_s,
        current_base.frequencies_hz,
        np.full_like(current_base.times_s, 0.2),
        name="quiet_current",
    )

    result = score_matched_section_progress(previous, current, catalog)

    assert any("confidence" in reason for reason in result.confidence.reasons)
    assert result.confidence.confidence_level in {"medium", "low"}


def contour_from_baseline(
    baseline: MelodyBaseline,
    *,
    cents_error: float,
    vibrato_cents: float,
    mute_note_indices: set[int] | None = None,
) -> PitchContour:
    mute_note_indices = mute_note_indices or set()
    times: list[float] = []
    hz_values: list[float] = []
    confidence: list[float] = []
    for index, note in enumerate(baseline.notes):
        note_times = np.arange(note.start_s + 0.03, note.end_s - 0.03, 0.025)
        if index in mute_note_indices:
            times.extend(note_times.tolist())
            hz_values.extend([np.nan] * len(note_times))
            confidence.extend([0.0] * len(note_times))
            continue
        local_t = note_times - note.start_s
        vibrato = vibrato_cents * np.sin(2.0 * np.pi * 5.2 * local_t)
        midi = note.midi + (cents_error + vibrato) / 100.0
        times.extend(note_times.tolist())
        hz_values.extend(midi_to_hz(midi).tolist())
        confidence.extend([0.95] * len(note_times))
    return PitchContour(np.asarray(times), np.asarray(hz_values), np.asarray(confidence), name="synthetic")


def with_padding(contour: PitchContour, *, before_s: float, after_s: float) -> PitchContour:
    before_times = np.arange(0.0, before_s, 0.025)
    shifted_times = contour.times_s + before_s
    after_start = float(shifted_times.max()) + 0.025 if shifted_times.size else before_s
    after_times = np.arange(after_start, after_start + after_s, 0.025)
    return PitchContour(
        np.concatenate([before_times, shifted_times, after_times]),
        np.concatenate(
            [
                np.full(before_times.shape, np.nan),
                contour.frequencies_hz,
                np.full(after_times.shape, np.nan),
            ]
        ),
        np.concatenate(
            [
                np.zeros(before_times.shape),
                contour.confidence,
                np.zeros(after_times.shape),
            ]
        ),
        name=contour.name,
    )


def contour_duration(contour: PitchContour) -> float:
    return float(np.nanmax(contour.times_s) - np.nanmin(contour.times_s))
