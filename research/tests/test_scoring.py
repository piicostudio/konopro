from __future__ import annotations

import numpy as np

from konopro_research.baseline import MelodyBaseline, cents_difference, demo_baseline, midi_to_hz
from konopro_research.contour_scoring import compare_takes_to_reference_contour
from konopro_research.demo_data import synthesize_take
from konopro_research.pitch import PitchContour
from konopro_research.quality import analyze_baseline_quality, duration_mismatch_warnings
from konopro_research.reference_audio import baseline_from_reference_audio
from konopro_research.scoring import compare_takes, score_take
from konopro_research.audio_io import write_wav


def test_stable_but_wrong_is_not_treated_as_song_improvement() -> None:
    baseline = demo_baseline()
    previous = contour_from_baseline(baseline, cents_error=35, vibrato_cents=30)
    stable_wrong = contour_from_baseline(baseline, cents_error=180, vibrato_cents=2)

    comparison = compare_takes(previous, stable_wrong, baseline)

    assert comparison.current.stability_score > comparison.previous.stability_score
    assert comparison.current.pitch_accuracy_score < comparison.previous.pitch_accuracy_score
    assert comparison.overall_delta < 0
    assert any("more stable, but less correct" in msg for msg in comparison.feedback)


def test_current_take_can_improve_against_reference() -> None:
    baseline = demo_baseline()
    previous = contour_from_baseline(baseline, cents_error=80, vibrato_cents=30)
    current = contour_from_baseline(baseline, cents_error=15, vibrato_cents=8)

    comparison = compare_takes(previous, current, baseline)

    assert comparison.verdict == "improved"
    assert comparison.overall_delta > 10
    assert comparison.pitch_accuracy_delta > 0


def test_contour_scoring_rewards_closer_dynamic_melody() -> None:
    reference = dynamic_contour()
    previous = dynamic_contour(cents_error=120, vibrato_cents=30)
    current = dynamic_contour(cents_error=20, vibrato_cents=8)

    comparison = compare_takes_to_reference_contour(previous, current, reference)

    assert comparison.verdict == "improved"
    assert comparison.current.pitch_accuracy_score > 80
    assert comparison.pitch_accuracy_delta > 40


def test_contour_scoring_penalizes_same_shape_wrong_key() -> None:
    reference = dynamic_contour()
    previous = dynamic_contour(cents_error=25, vibrato_cents=10)
    shifted = dynamic_contour(cents_error=210, vibrato_cents=8)

    comparison = compare_takes_to_reference_contour(previous, shifted, reference)

    assert comparison.current.pitch_accuracy_score < comparison.previous.pitch_accuracy_score
    assert any("semitone" in warning for warning in comparison.current.warnings)


def test_missing_notes_reduce_coverage() -> None:
    baseline = demo_baseline()
    missing = contour_from_baseline(baseline, cents_error=10, vibrato_cents=5, mute_note_indices={1, 5})

    result = score_take(missing, baseline)

    assert result.covered_notes == len(baseline.notes) - 2
    assert result.note_coverage_pct < 80
    assert result.coverage_score < 80


def test_transposition_warning_is_reported() -> None:
    baseline = demo_baseline()
    shifted = contour_from_baseline(baseline, cents_error=110, vibrato_cents=4)

    result = score_take(shifted, baseline)

    assert result.estimated_transposition_cents > 90
    assert any("semitone" in warning for warning in result.warnings)


def test_reference_audio_can_be_exported_to_baseline(tmp_path) -> None:
    baseline = demo_baseline()
    audio = synthesize_take(baseline, cents_error=0, vibrato_cents=2)
    reference_path = tmp_path / "reference.wav"
    write_wav(reference_path, audio, 22050)

    extracted = baseline_from_reference_audio(reference_path)
    quality = analyze_baseline_quality(extracted)

    assert extracted.notes
    assert quality.note_count >= 8
    assert quality.voiced_coverage_ratio > 0.50


def test_duration_mismatch_warning_mentions_trimming() -> None:
    warnings = duration_mismatch_warnings(30.0, 4.0, 31.0)

    assert any("trim" in warning for warning in warnings)


def test_low_confidence_recording_adds_warning() -> None:
    baseline = demo_baseline()
    times = np.arange(0.0, baseline.duration_s, 0.05)
    silent = PitchContour(times, np.full(times.shape, np.nan), np.zeros(times.shape))

    result = score_take(silent, baseline)

    assert result.recording_confidence_level == "low"
    assert any("confidence is low" in warning for warning in result.warnings)


def test_cents_difference_sign_and_scale() -> None:
    a4 = midi_to_hz(69)
    one_semitone_up = midi_to_hz(70)
    cents = cents_difference(np.asarray([one_semitone_up]), np.asarray([a4]))

    assert abs(float(cents[0]) - 100.0) < 0.001


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
        note_times = np.arange(note.start_s + 0.03, note.end_s - 0.03, 0.02)
        if index in mute_note_indices:
            times.extend(note_times.tolist())
            hz_values.extend([np.nan] * len(note_times))
            confidence.extend([0.0] * len(note_times))
            continue
        local_t = note_times - note.start_s
        vibrato = vibrato_cents * np.sin(2.0 * np.pi * 5.0 * local_t)
        midi = note.midi + (cents_error + vibrato) / 100.0
        times.extend(note_times.tolist())
        hz_values.extend(midi_to_hz(midi).tolist())
        confidence.extend([0.95] * len(note_times))

    return PitchContour(np.asarray(times), np.asarray(hz_values), np.asarray(confidence))


def dynamic_contour(
    *,
    cents_error: float = 0.0,
    vibrato_cents: float = 0.0,
) -> PitchContour:
    times = np.arange(0.0, 8.0, 0.025)
    melodic_motion = 2.8 * np.sin(2.0 * np.pi * times / 4.0) + 0.9 * np.sin(2.0 * np.pi * times / 1.7)
    vibrato = (vibrato_cents / 100.0) * np.sin(2.0 * np.pi * 5.2 * times)
    midi = 62.0 + melodic_motion + cents_error / 100.0 + vibrato
    confidence = np.full(times.shape, 0.95)
    return PitchContour(times, midi_to_hz(midi), confidence)
