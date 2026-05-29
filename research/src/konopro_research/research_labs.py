from __future__ import annotations

import importlib.util
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from konopro_research.audio_io import load_audio, write_wav
from konopro_research.baseline import (
    MelodyBaseline,
    demo_baseline,
    hz_to_midi,
    midi_to_hz,
)
from konopro_research.contour_scoring import (
    contour_timing_debug,
)
from konopro_research.demo_data import synthesize_take
from konopro_research.matching import (
    build_demo_section_catalog,
    extract_matching_query,
    match_query_to_sections,
    split_contour_into_sections,
)
from konopro_research.pitch import PitchContour, clean_pitch_contour, extract_pitch
from konopro_research.quality import analyze_baseline_quality, summarize_audio
from konopro_research.reference_audio import extract_reference_audio
from konopro_research.scoring import score_take


DEFAULT_PITCH_KWARGS = {
    "fmin_hz": 80.0,
    "fmax_hz": 1000.0,
    "frame_length": 2048,
    "hop_length": 256,
}
DEFAULT_CLEAN_KWARGS = {
    "min_confidence": 0.25,
    "max_jump_cents": 700.0,
    "correct_octaves": True,
}


def crop_audio_to_file(
    path: str | Path,
    output_dir: str | Path,
    *,
    start_s: float = 0.0,
    duration_s: float = 30.0,
    label: str = "excerpt",
) -> tuple[str, float, float]:
    audio, sample_rate = load_audio(path)
    total_duration = len(audio) / sample_rate if sample_rate else 0.0
    start_s = min(max(0.0, float(start_s)), max(0.0, total_duration))
    end_s = min(total_duration, start_s + max(0.1, float(duration_s)))
    start_index = int(round(start_s * sample_rate))
    end_index = max(start_index + 1, int(round(end_s * sample_rate)))
    output_path = Path(output_dir) / f"{label}_{start_s:.2f}_{end_s:.2f}.wav"
    write_wav(output_path, audio[start_index:end_index], sample_rate)
    return str(output_path), start_s, end_s


def extract_clean_contour(
    path: str | Path,
    *,
    name: str,
    pitch_kwargs: dict[str, Any] | None = None,
    clean_kwargs: dict[str, Any] | None = None,
) -> PitchContour:
    audio, sample_rate = load_audio(path)
    return clean_pitch_contour(
        extract_pitch(audio, sample_rate, name=name, **(pitch_kwargs or DEFAULT_PITCH_KWARGS)),
        **(clean_kwargs or DEFAULT_CLEAN_KWARGS),
    )


def contour_summary(contour: PitchContour) -> dict[str, Any]:
    mask = contour.voiced_mask
    total = int(contour.times_s.size)
    voiced = int(np.count_nonzero(mask))
    confidence = contour.confidence[mask]
    midi = hz_to_midi(contour.frequencies_hz[mask]) if voiced else np.asarray([])
    jumps = np.abs(np.diff(midi)) * 100.0 if midi.size > 1 else np.asarray([])
    return {
        "frames": total,
        "voiced_frames": voiced,
        "voiced_ratio_pct": round(100.0 * voiced / total, 2) if total else 0.0,
        "median_confidence": round(float(np.nanmedian(confidence)), 3) if confidence.size else 0.0,
        "median_midi": round(float(np.nanmedian(midi)), 2) if midi.size else np.nan,
        "pitch_range_semitones": round(float(np.nanpercentile(midi, 95) - np.nanpercentile(midi, 5)), 2)
        if midi.size >= 3
        else np.nan,
        "median_jump_cents": round(float(np.nanmedian(jumps)), 2) if jumps.size else np.nan,
        "octave_jump_count": int(np.count_nonzero(jumps > 700.0)) if jumps.size else 0,
        "duration_s": round(float(np.nanmax(contour.times_s) - np.nanmin(contour.times_s)), 3)
        if total
        else 0.0,
    }


def run_pitch_extractor_lab(
    audio_path: str | None,
    output_dir: str | Path,
    *,
    methods: list[str],
    start_s: float,
    duration_s: float,
    min_confidence: float,
    max_jump_cents: float,
) -> tuple[str, pd.DataFrame, str | None, dict[str, PitchContour], dict[str, str]]:
    if not audio_path:
        return "No audio selected.", pd.DataFrame(), None, {}, {}

    started = time.perf_counter()
    excerpt_path, actual_start, actual_end = crop_audio_to_file(
        audio_path,
        output_dir,
        start_s=start_s,
        duration_s=duration_s,
        label="pitch_lab_source",
    )
    rows: list[dict[str, Any]] = []
    contours: dict[str, PitchContour] = {}
    notes: dict[str, str] = {}
    clean_kwargs = {
        **DEFAULT_CLEAN_KWARGS,
        "min_confidence": float(min_confidence),
        "max_jump_cents": float(max_jump_cents),
    }

    for method in methods:
        method_started = time.perf_counter()
        if method == "pYIN":
            contour = extract_clean_contour(excerpt_path, name="pYIN", clean_kwargs=clean_kwargs)
            row = {"method": method, "status": "ready", **contour_summary(contour)}
            contours[method] = contour
            notes[method] = "librosa pYIN baseline."
        elif method == "CREPE / torchcrepe":
            if importlib.util.find_spec("torchcrepe") is None:
                row = _unavailable_method_row(method, "Install `torchcrepe` to run this extractor.")
            else:
                row = _unavailable_method_row(method, "torchcrepe is installed, but adapter is not wired yet.")
        elif method == "RMVPE":
            if importlib.util.find_spec("rmvpe") is None and importlib.util.find_spec("torchfcpe") is None:
                row = _unavailable_method_row(method, "Install an RMVPE/FCPE package to run this extractor.")
            else:
                row = _unavailable_method_row(method, "RMVPE package detected, but adapter is not wired yet.")
        elif method == "Basic Pitch":
            if importlib.util.find_spec("basic_pitch") is None:
                row = _unavailable_method_row(method, "Install `basic-pitch` to run audio-to-MIDI extraction.")
            else:
                row = _unavailable_method_row(method, "Basic Pitch is installed, but adapter is not wired yet.")
        else:
            row = _unavailable_method_row(method, "Unknown method.")
        row["elapsed_s"] = round(time.perf_counter() - method_started, 2)
        rows.append(row)

    status = (
        f"Pitch extractor lab complete in {time.perf_counter() - started:.2f}s. "
        f"Excerpt {actual_start:.2f}s-{actual_end:.2f}s."
    )
    return status, pd.DataFrame(rows), excerpt_path, contours, notes


def _unavailable_method_row(method: str, note: str) -> dict[str, Any]:
    return {
        "method": method,
        "status": "unavailable",
        "note": note,
        "frames": np.nan,
        "voiced_frames": np.nan,
        "voiced_ratio_pct": np.nan,
        "median_confidence": np.nan,
        "median_midi": np.nan,
        "pitch_range_semitones": np.nan,
        "median_jump_cents": np.nan,
        "octave_jump_count": np.nan,
        "duration_s": np.nan,
    }


def run_reference_builder_lab(
    reference_audio: str | None,
    *,
    window_s: float,
) -> tuple[str, pd.DataFrame, MelodyBaseline | None, PitchContour | None, dict[str, Any]]:
    if not reference_audio:
        baseline = demo_baseline()
        quality = analyze_baseline_quality(baseline)
        return (
            "Using demo symbolic baseline.",
            pd.DataFrame([quality.to_dict()]),
            baseline,
            None,
            {"baseline": {"title": baseline.title, "notes": len(baseline.notes)}},
        )

    started = time.perf_counter()
    extraction = extract_reference_audio(
        reference_audio,
        title=Path(reference_audio).name,
        window_s=float(window_s),
        pitch_kwargs=DEFAULT_PITCH_KWARGS,
        clean_kwargs=DEFAULT_CLEAN_KWARGS,
    )
    quality = analyze_baseline_quality(
        extraction.baseline,
        source_duration_s=summarize_audio(reference_audio).duration_s,
    )
    contour_quality = contour_summary(extraction.contour)
    table = pd.DataFrame(
        [
            {"artifact": "baseline", **quality.to_dict()},
            {"artifact": "reference_contour", **contour_quality},
        ]
    )
    status = f"Reference built in {time.perf_counter() - started:.2f}s."
    return (
        status,
        table,
        extraction.baseline,
        extraction.contour,
        {
            "baseline": {
                "title": extraction.baseline.title,
                "notes": len(extraction.baseline.notes),
                "duration_s": extraction.baseline.duration_s,
            },
            "audio_summary": extraction.audio_summary.to_dict(),
            "quality": extraction.quality.to_dict(),
            "contour": contour_quality,
        },
    )


def run_song_identification_lab(
    reference_audio: str | None,
    query_audio: str | None,
    *,
    catalog_source: str,
    top_k: int,
    window_s: float,
    hop_s: float,
) -> tuple[str, pd.DataFrame, dict[str, Any]]:
    if not query_audio:
        return "No query/current take selected.", pd.DataFrame(), {}

    started = time.perf_counter()
    query = extract_matching_query(query_audio, pitch_kwargs=DEFAULT_PITCH_KWARGS, clean_kwargs=DEFAULT_CLEAN_KWARGS)
    if catalog_source == "Uploaded reference sections" and reference_audio:
        extraction = extract_reference_audio(reference_audio, pitch_kwargs=DEFAULT_PITCH_KWARGS, clean_kwargs=DEFAULT_CLEAN_KWARGS)
        sections = split_contour_into_sections(
            extraction.contour,
            song_id="uploaded_reference",
            song_title=Path(reference_audio).stem,
            window_s=float(window_s),
            hop_s=float(hop_s),
        )
    else:
        sections = build_demo_section_catalog()
    result = match_query_to_sections(query, sections, top_k=int(top_k))
    table = pd.DataFrame([candidate.to_dict() for candidate in result.candidates])
    best = result.best.section.display_name if result.best else "none"
    status = f"Song identification complete in {time.perf_counter() - started:.2f}s. Best: {best}."
    return status, table, result.to_dict()


def run_timing_lab(
    reference_audio: str | None,
    current_audio: str | None,
    *,
    timing_penalty: float,
) -> tuple[str, pd.DataFrame, dict[str, Any]]:
    if not reference_audio or not current_audio:
        return "Reference audio and current take are required.", pd.DataFrame(), {}

    started = time.perf_counter()
    reference = extract_reference_audio(
        reference_audio,
        pitch_kwargs=DEFAULT_PITCH_KWARGS,
        clean_kwargs=DEFAULT_CLEAN_KWARGS,
    )
    debug = contour_timing_debug(
        current_audio,
        reference.contour,
        pitch_kwargs=DEFAULT_PITCH_KWARGS,
        clean_kwargs=DEFAULT_CLEAN_KWARGS,
        timing_penalty=float(timing_penalty),
    )
    table = pd.DataFrame(
        [
            {"metric": key, "value": value}
            for key, value in debug.items()
            if not isinstance(value, list)
        ]
    )
    status = f"Timing analysis complete in {time.perf_counter() - started:.2f}s."
    return status, table, debug


def run_scoring_calibration_lab(
    scenario: str,
    *,
    pitch_weight: float,
    stability_weight: float,
    coverage_weight: float,
    timing_weight: float,
) -> tuple[str, pd.DataFrame, dict[str, Any]]:
    baseline = demo_baseline()
    contour = _scenario_contour(baseline, scenario)
    score = score_take(contour, baseline)
    total_weight = max(0.001, pitch_weight + stability_weight + coverage_weight + timing_weight)
    calibrated = (
        score.pitch_accuracy_score * pitch_weight
        + score.stability_score * stability_weight
        + score.coverage_score * coverage_weight
        + score.timing_score * timing_weight
    ) / total_weight
    rows = [
        {"metric": "pitch_accuracy", "raw": score.pitch_accuracy_score, "weight": pitch_weight},
        {"metric": "stability", "raw": score.stability_score, "weight": stability_weight},
        {"metric": "coverage", "raw": score.coverage_score, "weight": coverage_weight},
        {"metric": "timing", "raw": score.timing_score, "weight": timing_weight},
        {"metric": "overall_original", "raw": score.overall_score, "weight": np.nan},
        {"metric": "overall_calibrated", "raw": round(float(calibrated), 2), "weight": total_weight},
    ]
    details = score.to_dict()
    details["calibrated_overall"] = round(float(calibrated), 2)
    return f"Calibration scenario `{scenario}` scored.", pd.DataFrame(rows), details


def run_stress_test_lab(
    *,
    scenarios: list[str],
) -> tuple[str, pd.DataFrame, dict[str, Any]]:
    baseline = demo_baseline()
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        contour = _scenario_contour(baseline, scenario)
        score = score_take(contour, baseline)
        rows.append(
            {
                "scenario": scenario,
                "overall": score.overall_score,
                "pitch_accuracy": score.pitch_accuracy_score,
                "stability": score.stability_score,
                "coverage": score.coverage_score,
                "timing": score.timing_score,
                "confidence": score.recording_confidence_score,
                "warnings": "; ".join(score.warnings),
            }
        )
    table = pd.DataFrame(rows)
    return f"Stress suite complete for {len(rows)} scenario(s).", table, {"scenarios": rows}


def _scenario_contour(baseline: MelodyBaseline, scenario: str) -> PitchContour:
    if scenario == "accurate":
        return _contour_from_baseline(baseline, cents_error=12, vibrato_cents=5)
    if scenario == "stable but wrong":
        return _contour_from_baseline(baseline, cents_error=180, vibrato_cents=2)
    if scenario == "unstable but close":
        return _contour_from_baseline(baseline, cents_error=20, vibrato_cents=70)
    if scenario == "missing notes":
        return _contour_from_baseline(baseline, cents_error=15, vibrato_cents=8, mute_note_indices={1, 4, 7})
    if scenario == "late start":
        contour = _contour_from_baseline(baseline, cents_error=12, vibrato_cents=5)
        return contour.shifted(2.5)
    if scenario == "low confidence":
        contour = _contour_from_baseline(baseline, cents_error=12, vibrato_cents=5)
        return PitchContour(contour.times_s, contour.frequencies_hz, np.full(contour.times_s.shape, 0.18), name=scenario)
    return _contour_from_baseline(baseline, cents_error=60, vibrato_cents=35)


def _contour_from_baseline(
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
            hz_values.extend([math.nan] * len(note_times))
            confidence.extend([0.0] * len(note_times))
            continue
        local_t = note_times - note.start_s
        vibrato = vibrato_cents * np.sin(2.0 * np.pi * 5.2 * local_t)
        midi = note.midi + (cents_error + vibrato) / 100.0
        times.extend(note_times.tolist())
        hz_values.extend(midi_to_hz(midi).tolist())
        confidence.extend([0.95] * len(note_times))
    return PitchContour(np.asarray(times), np.asarray(hz_values), np.asarray(confidence), name="scenario")


def write_synthetic_scenario_audio(
    scenario: str,
    output_dir: str | Path,
) -> str:
    baseline = demo_baseline()
    if scenario == "accurate":
        audio = synthesize_take(baseline, cents_error=12, vibrato_cents=5)
    elif scenario == "stable but wrong":
        audio = synthesize_take(baseline, cents_error=180, vibrato_cents=2)
    elif scenario == "unstable but close":
        audio = synthesize_take(baseline, cents_error=20, vibrato_cents=70)
    elif scenario == "missing notes":
        audio = synthesize_take(baseline, cents_error=15, vibrato_cents=8, mute_note_indices={1, 4, 7})
    elif scenario == "late start":
        prefix = np.zeros(int(2.5 * 22050), dtype=np.float32)
        audio = np.concatenate([prefix, synthesize_take(baseline, cents_error=12, vibrato_cents=5)])
    elif scenario == "low confidence":
        audio = 0.12 * synthesize_take(baseline, cents_error=12, vibrato_cents=5)
    else:
        audio = synthesize_take(baseline, cents_error=60, vibrato_cents=35)
    path = Path(output_dir) / f"stress_{scenario.replace(' ', '_')}.wav"
    write_wav(path, audio, 22050)
    return str(path)
