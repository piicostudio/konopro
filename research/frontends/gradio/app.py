from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd

RESEARCH_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = RESEARCH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from konopro_research.audio_io import load_audio, write_wav  # noqa: E402
from konopro_research.baseline import demo_baseline, hz_to_midi, midi_to_hz  # noqa: E402
from konopro_research.contour_scoring import (  # noqa: E402
    compare_takes_to_reference_contour,
    score_take_against_reference_contour,
)
from konopro_research.demo_data import ensure_demo_data  # noqa: E402
from konopro_research.matching import (  # noqa: E402
    build_demo_section_catalog,
    crop_contour,
    extract_matching_query,
    match_query_to_sections,
    split_contour_into_sections,
)
from konopro_research.pitch import PitchContour, clean_pitch_contour, extract_pitch  # noqa: E402
from konopro_research.plots import (  # noqa: E402
    plot_contour_comparison,
    plot_contour_voiced_coverage,
    plot_section_match,
    plot_take_comparison,
    plot_voiced_coverage,
)
from konopro_research.reference_audio import extract_reference_audio  # noqa: E402
from konopro_research.scoring import compare_takes, score_take  # noqa: E402
from konopro_research.separation import prepare_vocal_analysis_audio  # noqa: E402


DEMO_PATHS = ensure_demo_data(RESEARCH_ROOT / "data" / "demo")
STEM_CACHE_DIR = RESEARCH_ROOT / ".cache" / "stems"
OUTPUT_DIR = RESEARCH_ROOT / ".cache" / "gradio_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


PITCH_KWARGS = {
    "fmin_hz": 80.0,
    "fmax_hz": 1000.0,
    "frame_length": 2048,
    "hop_length": 256,
}
CLEAN_KWARGS = {
    "min_confidence": 0.25,
    "max_jump_cents": 700.0,
    "correct_octaves": True,
}
CONTOUR_SCORE_KWARGS = {
    "pitch_kwargs": PITCH_KWARGS,
    "clean_kwargs": CLEAN_KWARGS,
    "dtw_time_weight": 20.0,
    "dtw_band_radius": 0.06,
    "max_dtw_frames": 2400,
    "pitch_error_penalty": 0.70,
    "stability_penalty": 1.10,
    "timing_penalty": 90.0,
    "transposition_warning_cents": 90.0,
}
SYMBOLIC_SCORE_KWARGS = {
    "pitch_kwargs": PITCH_KWARGS,
    "clean_kwargs": CLEAN_KWARGS,
    "alignment_kwargs": {
        "search_radius_s": 0.5,
        "step_s": 0.02,
    },
    "note_coverage_min_ratio": 0.35,
    "pitch_error_penalty": 0.70,
    "stability_penalty": 1.10,
    "timing_offset_penalty": 180.0,
    "transposition_warning_cents": 90.0,
}


def _human_slot_name(label: str) -> str:
    if label == "reference":
        return "Reference"
    if label == "current":
        return "Current"
    return "Previous"


def _to_path(value: str | None) -> str | None:
    return value or None


def _table_bool(row: dict[str, Any], key: str) -> bool:
    value = row.get(key, False)
    if pd.isna(value):
        return False
    return bool(value)


def _table_int(row: dict[str, Any], key: str) -> int:
    value = row.get(key, 0)
    if pd.isna(value):
        return 0
    return int(value)


def _format_status(
    label: str,
    mode: str,
    elapsed_s: float,
    warning_count: int,
    used_cache: bool,
    used_original: bool,
) -> str:
    parts = [
        f"{_human_slot_name(label)}: {mode}",
        f"{elapsed_s:.2f}s",
    ]
    if used_cache:
        parts.append("cached")
    if used_original:
        parts.append("fallback")
    if warning_count:
        parts.append(f"{warning_count} warning(s)")
    return " • ".join(parts)


def _prepare_slot(
    path: str | None,
    label: str,
    should_prepare: bool,
    *,
    model: str,
    device: str,
    shifts: int,
    overlap: float,
) -> tuple[str, str, dict[str, Any]]:
    if path is None:
        return "", "", {
            "slot": _human_slot_name(label),
            "input": "",
            "analysis_mode": "missing",
            "analysis_file": "",
            "timing_seconds": 0.0,
            "notes": "Missing file",
            "used_cache": False,
            "used_original": False,
            "warnings": 0,
        }

    source = path
    if not should_prepare:
        return source, source, {
            "slot": _human_slot_name(label),
            "input": Path(source).name,
            "analysis_mode": "original audio",
            "analysis_file": Path(source).name,
            "timing_seconds": 0.0,
            "notes": "No separation requested",
            "used_cache": False,
            "used_original": True,
            "warnings": 0,
        }

    started = time.perf_counter()
    result = prepare_vocal_analysis_audio(
        path,
        cache_dir=STEM_CACHE_DIR,
        backend="demucs",
        model=model,
        device=device,
        shifts=int(shifts),
        overlap=float(overlap),
    )
    elapsed = time.perf_counter() - started
    if result.used_cache and result.used_original:
        mode = "cached fallback to original"
    elif result.used_cache:
        mode = "cached vocals stem"
    elif result.used_original:
        mode = "fallback to original"
    else:
        mode = "vocal stem (vocals)"

    notes = "\n".join(result.warnings) if result.warnings else "Ready"
    return (
        source,
        str(result.analysis_path),
        {
            "slot": _human_slot_name(label),
            "input": Path(source).name,
            "analysis_mode": mode,
            "analysis_file": Path(result.analysis_path).name,
            "timing_seconds": elapsed,
            "notes": notes,
            "used_cache": result.used_cache,
            "used_original": result.used_original,
            "warnings": len(result.warnings),
            "cache_key": result.cache_key,
            "cache_status": result.cache_status,
            "cache_path": result.cache_path,
        },
    )


def _ready_status(prep_elapsed: float, use_demucs: bool, table: pd.DataFrame) -> str:
    if table.empty:
        return f"Prepared analysis audio in {prep_elapsed:.2f}s."
    lines = [
        _format_status(
            row["slot"].lower(),
            row["analysis_mode"],
            float(row.get("timing_seconds", 0.0)),
            _table_int(row, "warnings"),
            _table_bool(row, "used_cache"),
            _table_bool(row, "used_original"),
        )
        for row in table.to_dict("records")
        if row["slot"] != "Previous" or row["input"]
    ]
    status = f"Prepared audio in {prep_elapsed:.2f}s.\n" + "\n".join(lines)
    if use_demucs:
        status = f"Demucs requested. {status}"
    else:
        status = f"Using original audio. {status}"
    return status


def _contour_duration_s(contour: Any) -> float:
    if contour.times_s.size == 0:
        return 0.0
    return float(contour.times_s.max() - contour.times_s.min())


def _voiced_frame_count(contour: Any) -> int:
    return int(contour.voiced_mask.sum())


def _matching_steps_table(
    *,
    catalog_source: str,
    query_path: str,
    reference_path: str | None,
    query: Any,
    section_count: int,
    result: Any,
    handoff_score: Any | None,
) -> pd.DataFrame:
    best = result.best
    best_section = best.section.display_name if best else "none"
    best_reference = (
        f"{best.section.start_s:.2f}s-{best.section.end_s:.2f}s" if best else "none"
    )
    best_query = f"{best.query_start_s:.2f}s-{best.query_end_s:.2f}s" if best else "none"
    handoff = f"{handoff_score.overall_score:.1f}" if handoff_score is not None else "not run"

    rows = [
        {
            "step": 1,
            "stage": "Choose analysis audio",
            "what happens": "Use the prepared current take as the query and either uploaded reference sections or the demo catalog as the search space.",
            "latest value": f"query={Path(query_path).name}; catalog={catalog_source}",
        },
        {
            "step": 2,
            "stage": "Extract query pitch",
            "what happens": "Convert the current take into a cleaned pitch contour using pYIN-style pitch tracking.",
            "latest value": f"{_voiced_frame_count(query)} voiced frames over {_contour_duration_s(query):.2f}s",
        },
        {
            "step": 3,
            "stage": "Build reference sections",
            "what happens": "Split the reference contour into overlapping phrase windows, or use the small demo section catalog.",
            "latest value": f"{section_count} sections from {Path(reference_path).name if reference_path else 'demo catalog'}",
        },
        {
            "step": 4,
            "stage": "Create query windows",
            "what happens": "Try phrase-sized windows inside the current take so a late start or partial song can still match.",
            "latest value": best_query,
        },
        {
            "step": 5,
            "stage": "Normalize pitch shape",
            "what happens": "Convert each window to relative MIDI pitch so the search focuses on melody shape instead of absolute key.",
            "latest value": "median pitch removed; 96 samples per curve",
        },
        {
            "step": 6,
            "stage": "Run DTW comparison",
            "what happens": "Use dynamic time warping to compare reference and query shapes even when timing is slightly stretched.",
            "latest value": f"best={best_section}; reference={best_reference}",
        },
        {
            "step": 7,
            "stage": "Rank candidates",
            "what happens": "Combine shape similarity, voiced coverage, and duration fit into the final match score.",
            "latest value": (
                f"score={best.score:.1f}; shape={best.shape_score:.1f}; "
                f"coverage={best.coverage_score:.1f}; duration_fit={best.duration_score:.1f}"
                if best
                else "no candidates"
            ),
        },
        {
            "step": 8,
            "stage": "Handoff to scoring",
            "what happens": "Crop the matched query window and score it against the matched reference section.",
            "latest value": f"matched-section evaluation overall={handoff}",
        },
    ]
    return pd.DataFrame(rows)


def _empty_matching_response(
    message: str,
    *,
    candidates: pd.DataFrame | None = None,
    steps: pd.DataFrame | None = None,
    details: dict[str, Any] | None = None,
) -> tuple[Any, ...]:
    return (
        message,
        candidates if candidates is not None else pd.DataFrame(),
        None,
        pd.DataFrame(),
        steps if steps is not None else pd.DataFrame(),
        pd.DataFrame(),
        None,
        None,
        None,
        None,
        pd.DataFrame(),
        None,
        None,
        None,
        pd.DataFrame(),
        None,
        None,
        None,
        pd.DataFrame(),
        None,
        None,
        None,
        None,
        None,
        None,
        details or {},
        message,
    )


def _empty_evaluation_response(message: str) -> tuple[Any, ...]:
    return (
        message,
        pd.DataFrame(),
        None,
        None,
        pd.DataFrame(),
        pd.DataFrame(),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        pd.DataFrame(),
        {},
        message,
    )


def _evaluation_steps_table(
    *,
    reference_mode: str,
    reference_path: str | None,
    current_path: str,
    previous_path: str | None,
    current_contour: PitchContour,
    previous_contour: PitchContour | None,
    score: Any,
) -> pd.DataFrame:
    previous_value = (
        f"{Path(previous_path).name}; {_voiced_frame_count(previous_contour)} voiced frames"
        if previous_path and previous_contour is not None
        else "not provided"
    )
    return pd.DataFrame(
        [
            {
                "step": 1,
                "stage": "Prepared analysis audio",
                "what happens": "The evaluator uses the audio selected in Step 2: original audio, cached vocal stem, or fallback audio.",
                "why it matters": "Bad source audio or a failed stem affects every later score.",
                "latest value": f"reference={Path(reference_path).name if reference_path else reference_mode}; current={Path(current_path).name}",
            },
            {
                "step": 2,
                "stage": "Build reference representation",
                "what happens": "Uploaded reference mode extracts a cleaned pitch contour from the reference audio. Demo mode uses the synthetic symbolic baseline.",
                "why it matters": "If the reference contour is sparse or wrong, the score is not meaningful.",
                "latest value": reference_mode,
            },
            {
                "step": 3,
                "stage": "Extract current pitch",
                "what happens": "The current take is converted into pitch frames with pYIN, then cleaned by confidence and jump thresholds.",
                "why it matters": "This is where silence, accompaniment, echo, or octave mistakes can corrupt the singer contour.",
                "latest value": f"{_voiced_frame_count(current_contour)} voiced frames over {_contour_duration_s(current_contour):.2f}s",
            },
            {
                "step": 4,
                "stage": "Optional previous pitch",
                "what happens": "If a previous take exists, it goes through the same pitch extraction and cleaning path.",
                "why it matters": "Previous/current deltas only mean something if both takes have comparable extracted contours.",
                "latest value": previous_value,
            },
            {
                "step": 5,
                "stage": "Full-track alignment",
                "what happens": "Uploaded-reference scoring aligns the full reference contour against the full current contour. Demo-baseline scoring estimates timing offset against symbolic notes.",
                "why it matters": "A late start or extra intro can dominate full-track timing, even when a short section matches well.",
                "latest value": f"timing_offset_s={score.timing_offset_s}",
            },
            {
                "step": 6,
                "stage": "Metric calculation",
                "what happens": "Pitch accuracy, stability, coverage, and timing are computed from the aligned contours.",
                "why it matters": "Each metric answers a different question; one bad metric can explain a low overall score.",
                "latest value": f"pitch={score.pitch_accuracy_score}; stability={score.stability_score}; timing={score.timing_score}; coverage={score.coverage_score}",
            },
            {
                "step": 7,
                "stage": "Confidence and warnings",
                "what happens": "The recording is checked for voiced-frame ratio, signal quality, and duration mismatch warnings.",
                "why it matters": "High musical scores are not reliable if pitch extraction confidence is low.",
                "latest value": f"{score.recording_confidence_level} ({score.recording_confidence_score})",
            },
            {
                "step": 8,
                "stage": "Final score",
                "what happens": "The evaluator combines pitch, stability, coverage, and timing into overall/song/control scores.",
                "why it matters": "This is the final full-track diagnostic score, not the matched-section score.",
                "latest value": f"overall={score.overall_score}; song={score.song_correctness_score}; control={score.technical_control_score}",
            },
        ]
    )


def _score_breakdown_table(score: Any) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric": "pitch_accuracy",
                "value": score.pitch_accuracy_score,
                "input signal": f"mean pitch error {score.mean_pitch_error_cents} cents",
                "interpretation": "How close the aligned pitch frames are to the reference.",
            },
            {
                "metric": "stability",
                "value": score.stability_score,
                "input signal": f"pitch stability {score.pitch_stability_cents} cents",
                "interpretation": "How consistent the pitch errors are after removing estimated transposition.",
            },
            {
                "metric": "coverage",
                "value": score.coverage_score,
                "input signal": f"{score.covered_notes}/{score.total_notes} voiced frames",
                "interpretation": "Whether enough singing was detected where comparison expected pitch.",
            },
            {
                "metric": "timing",
                "value": score.timing_score,
                "input signal": f"timing offset {score.timing_offset_s}s",
                "interpretation": "How well the full-track timing aligns. This currently punishes late starts.",
            },
            {
                "metric": "overall",
                "value": score.overall_score,
                "input signal": "weighted pitch + stability + coverage + timing",
                "interpretation": "Full-track diagnostic score.",
            },
        ]
    )


def _build_evaluation_diagnostics(
    *,
    reference_mode: str,
    reference_path: str | None,
    current_path: str,
    previous_path: str | None,
    current_contour: PitchContour,
    previous_contour: PitchContour | None,
    reference_contour: PitchContour | None,
    score: Any,
) -> tuple[Any, ...]:
    steps = _evaluation_steps_table(
        reference_mode=reference_mode,
        reference_path=reference_path,
        current_path=current_path,
        previous_path=previous_path,
        current_contour=current_contour,
        previous_contour=previous_contour,
        score=score,
    )
    audio_table = pd.DataFrame(
        [
            {
                "artifact": "reference analysis audio",
                "file": Path(reference_path).name if reference_path else reference_mode,
                "role": "full-track reference",
            },
            {
                "artifact": "current analysis audio",
                "file": Path(current_path).name,
                "role": "full-track current take",
            },
            {
                "artifact": "previous analysis audio",
                "file": Path(previous_path).name if previous_path else "not provided",
                "role": "optional comparison take",
            },
        ]
    )
    reference_preview = _crop_audio_file(reference_path, 0.0, 30.0, "evaluation_reference_preview.wav")
    current_preview = _crop_audio_file(current_path, 0.0, 30.0, "evaluation_current_preview.wav")
    previous_preview = _crop_audio_file(previous_path, 0.0, 30.0, "evaluation_previous_preview.wav")
    reference_raw_plot = _raw_clean_plot_for_path(
        reference_path,
        "evaluation reference raw vs cleaned pitch",
        "evaluation_reference_raw_clean.png",
    )
    current_raw_plot = _raw_clean_plot_for_path(
        current_path,
        "evaluation current raw vs cleaned pitch",
        "evaluation_current_raw_clean.png",
    )
    previous_raw_plot = _raw_clean_plot_for_path(
        previous_path,
        "evaluation previous raw vs cleaned pitch",
        "evaluation_previous_raw_clean.png",
    )
    alignment_plot = (
        _save_dtw_alignment_plot(reference_contour, current_contour, "evaluation_full_track_dtw.png")
        if reference_contour is not None
        else None
    )
    normalization_plot = (
        _save_normalization_plot(reference_contour, current_contour, "evaluation_full_track_normalization.png")
        if reference_contour is not None
        else None
    )
    return (
        steps,
        audio_table,
        reference_path,
        current_path,
        previous_path,
        reference_preview,
        current_preview,
        previous_preview,
        reference_raw_plot,
        current_raw_plot,
        previous_raw_plot,
        alignment_plot,
        normalization_plot,
        _score_breakdown_table(score),
    )


def _build_matching_diagnostics(
    *,
    catalog_source: str,
    query_path: str,
    reference_path: str | None,
    query: PitchContour,
    sections: tuple[Any, ...] | list[Any],
    result: Any,
    handoff_score: Any | None,
) -> tuple[Any, ...]:
    best = result.best
    audio_table = pd.DataFrame(
        [
            {
                "artifact": "reference analysis audio",
                "file": Path(reference_path).name if reference_path else "demo catalog",
                "role": "search source",
            },
            {
                "artifact": "current analysis audio",
                "file": Path(query_path).name,
                "role": "query source",
            },
        ]
    )

    raw_reference_plot = _raw_clean_plot_for_path(
        reference_path,
        "reference raw vs cleaned pitch",
        "matcher_reference_raw_clean.png",
    )
    raw_query_plot = _raw_clean_plot_for_path(
        query_path,
        "query raw vs cleaned pitch",
        "matcher_query_raw_clean.png",
    )

    top_matches = list(result.candidates[:3])
    reference_sections_table = pd.DataFrame(
        [
            {
                "rank": rank,
                "section": match.section.section_label,
                "reference_start_s": match.section.start_s,
                "reference_end_s": match.section.end_s,
                "score": match.score,
            }
            for rank, match in enumerate(top_matches, start=1)
        ]
    )
    query_windows_table = pd.DataFrame(
        [
            {
                "rank": rank,
                "query_start_s": match.query_start_s,
                "query_end_s": match.query_end_s,
                "matched_reference": match.section.section_label,
                "score": match.score,
            }
            for rank, match in enumerate(top_matches, start=1)
        ]
    )

    ref_section_audio = [
        _crop_audio_file(
            reference_path,
            match.section.start_s,
            match.section.end_s,
            f"matcher_reference_section_{rank}.wav",
        )
        for rank, match in enumerate(top_matches, start=1)
    ]
    query_window_audio = [
        _crop_audio_file(
            query_path,
            match.query_start_s,
            match.query_end_s,
            f"matcher_query_window_{rank}.wav",
        )
        for rank, match in enumerate(top_matches, start=1)
    ]
    ref_section_audio = _pad_paths(ref_section_audio, 3)
    query_window_audio = _pad_paths(query_window_audio, 3)

    if best is None:
        median_table = pd.DataFrame()
        normalization_plot = None
        median_reference_tone = None
        median_query_tone = None
        dtw_plot = None
        handoff_reference_audio = None
        handoff_query_audio = None
    else:
        query_window = crop_contour(
            query,
            best.query_start_s,
            best.query_end_s,
            name="matched query window",
            shift_to_zero=True,
        )
        reference_window = best.section.contour
        reference_median = _median_midi(reference_window)
        query_median = _median_midi(query_window)
        median_table = pd.DataFrame(
            [
                {
                    "contour": "matched reference section",
                    "median_midi": reference_median,
                    "median_hz": midi_to_hz(reference_median) if np.isfinite(reference_median) else np.nan,
                },
                {
                    "contour": "matched query window",
                    "median_midi": query_median,
                    "median_hz": midi_to_hz(query_median) if np.isfinite(query_median) else np.nan,
                },
            ]
        )
        normalization_plot = _save_normalization_plot(
            reference_window,
            query_window,
            "matcher_pitch_normalization.png",
        )
        median_reference_tone = _write_median_tone(reference_median, "matcher_reference_median_tone.wav")
        median_query_tone = _write_median_tone(query_median, "matcher_query_median_tone.wav")
        dtw_plot = _save_dtw_alignment_plot(
            reference_window,
            query_window,
            "matcher_dtw_alignment.png",
        )
        handoff_reference_audio = _crop_audio_file(
            reference_path,
            best.section.start_s,
            best.section.end_s,
            "matcher_handoff_reference.wav",
        )
        handoff_query_audio = _crop_audio_file(
            query_path,
            best.query_start_s,
            best.query_end_s,
            "matcher_handoff_query.wav",
        )

    return (
        audio_table,
        reference_path,
        query_path,
        raw_reference_plot,
        raw_query_plot,
        reference_sections_table,
        ref_section_audio[0],
        ref_section_audio[1],
        ref_section_audio[2],
        query_windows_table,
        query_window_audio[0],
        query_window_audio[1],
        query_window_audio[2],
        median_table,
        normalization_plot,
        median_reference_tone,
        median_query_tone,
        dtw_plot,
        handoff_reference_audio,
        handoff_query_audio,
    )


def _pad_paths(paths: list[str | None], count: int) -> list[str | None]:
    return [*paths, *([None] * count)][:count]


def _crop_audio_file(
    path: str | None,
    start_s: float,
    end_s: float,
    filename: str,
) -> str | None:
    if not path:
        return None
    audio, sample_rate = load_audio(path)
    start_index = max(0, int(round(start_s * sample_rate)))
    end_index = min(len(audio), int(round(end_s * sample_rate)))
    if end_index <= start_index:
        return None
    output_path = OUTPUT_DIR / filename
    write_wav(output_path, audio[start_index:end_index], sample_rate)
    return str(output_path)


def _raw_clean_plot_for_path(path: str | None, title: str, filename: str) -> str | None:
    if not path:
        return None
    audio, sample_rate = load_audio(path)
    raw = extract_pitch(audio, sample_rate, name=title, **PITCH_KWARGS)
    cleaned = clean_pitch_contour(raw, **CLEAN_KWARGS)
    return _save_raw_clean_plot(raw, cleaned, title, filename)


def _save_raw_clean_plot(raw: PitchContour, cleaned: PitchContour, title: str, filename: str) -> str:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.2))
    _plot_pitch_contour(ax, raw, "raw", "#9ca3af", linewidth=1.0, alpha=0.55)
    _plot_pitch_contour(ax, cleaned, "cleaned", "#2563eb", linewidth=1.4, alpha=0.95)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch (MIDI note)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return _save_plot(fig, filename)


def _plot_pitch_contour(
    ax: Any,
    contour: PitchContour,
    label: str,
    color: str,
    *,
    linewidth: float,
    alpha: float,
) -> None:
    mask = contour.voiced_mask
    if not np.any(mask):
        return
    ax.plot(
        contour.times_s[mask],
        hz_to_midi(contour.frequencies_hz[mask]),
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
    )


def _median_midi(contour: PitchContour) -> float:
    mask = contour.voiced_mask
    if not np.any(mask):
        return float("nan")
    return float(np.nanmedian(hz_to_midi(contour.frequencies_hz[mask])))


def _write_median_tone(midi: float, filename: str) -> str | None:
    if not np.isfinite(midi):
        return None
    sample_rate = 22050
    duration_s = 1.5
    times = np.linspace(0.0, duration_s, int(sample_rate * duration_s), endpoint=False)
    envelope = np.minimum(1.0, np.linspace(0.0, 8.0, times.size))
    envelope *= np.minimum(1.0, np.linspace(8.0, 0.0, times.size)[::-1])
    audio = 0.22 * envelope * np.sin(2.0 * np.pi * midi_to_hz(midi) * times)
    output_path = OUTPUT_DIR / filename
    write_wav(output_path, audio.astype(np.float32), sample_rate)
    return str(output_path)


def _relative_curve(contour: PitchContour, sample_count: int = 96) -> tuple[np.ndarray, np.ndarray] | None:
    mask = contour.voiced_mask
    if np.count_nonzero(mask) < 3:
        return None
    times = contour.times_s[mask]
    midi = hz_to_midi(contour.frequencies_hz[mask])
    order = np.argsort(times)
    times = times[order]
    midi = midi[order]
    unique_times, unique_indices = np.unique(times, return_index=True)
    midi = midi[unique_indices]
    if unique_times.size < 3:
        return None
    duration = max(float(unique_times[-1] - unique_times[0]), 0.001)
    normalized_times = (unique_times - unique_times[0]) / duration
    target = np.linspace(0.0, 1.0, sample_count)
    absolute_curve = np.interp(target, normalized_times, midi)
    relative_curve = absolute_curve - float(np.nanmedian(absolute_curve))
    return target, relative_curve


def _absolute_curve(contour: PitchContour, sample_count: int = 96) -> tuple[np.ndarray, np.ndarray] | None:
    mask = contour.voiced_mask
    if np.count_nonzero(mask) < 3:
        return None
    times = contour.times_s[mask]
    midi = hz_to_midi(contour.frequencies_hz[mask])
    order = np.argsort(times)
    times = times[order]
    midi = midi[order]
    duration = max(float(times[-1] - times[0]), 0.001)
    normalized_times = (times - times[0]) / duration
    target = np.linspace(0.0, 1.0, sample_count)
    return target, np.interp(target, normalized_times, midi)


def _save_normalization_plot(reference: PitchContour, query: PitchContour, filename: str) -> str | None:
    import matplotlib.pyplot as plt

    ref_abs = _absolute_curve(reference)
    query_abs = _absolute_curve(query)
    ref_rel = _relative_curve(reference)
    query_rel = _relative_curve(query)
    if ref_abs is None or query_abs is None or ref_rel is None or query_rel is None:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.0), sharex=True)
    axes[0].plot(ref_abs[0], ref_abs[1], color="#111827", label="reference absolute")
    axes[0].plot(query_abs[0], query_abs[1], color="#2563eb", label="query absolute")
    axes[0].set_ylabel("MIDI note")
    axes[0].set_title("Before pitch normalization")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    axes[1].plot(ref_rel[0], ref_rel[1], color="#111827", label="reference relative")
    axes[1].plot(query_rel[0], query_rel[1], color="#2563eb", label="query relative")
    axes[1].set_xlabel("Normalized phrase time")
    axes[1].set_ylabel("Relative semitones")
    axes[1].set_title("After median pitch removal")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")
    fig.tight_layout()
    return _save_plot(fig, filename)


def _save_dtw_alignment_plot(reference: PitchContour, query: PitchContour, filename: str) -> str | None:
    import matplotlib.pyplot as plt

    ref_curve = _relative_curve(reference)
    query_curve = _relative_curve(query)
    if ref_curve is None or query_curve is None:
        return None

    try:
        import librosa

        _, path = librosa.sequence.dtw(
            X=ref_curve[1].reshape(1, -1),
            Y=query_curve[1].reshape(1, -1),
            metric="euclidean",
        )
        pairs = path[::-1]
    except Exception:
        length = min(len(ref_curve[0]), len(query_curve[0]))
        pairs = np.column_stack([np.arange(length), np.arange(length)])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    query_y_offset = 5.0
    ax.plot(ref_curve[0], ref_curve[1], color="#111827", label="reference relative")
    ax.plot(query_curve[0], query_curve[1] + query_y_offset, color="#2563eb", label="query relative + offset")
    stride = max(1, len(pairs) // 32)
    for ref_i, query_i in pairs[::stride]:
        ax.plot(
            [ref_curve[0][ref_i], query_curve[0][query_i]],
            [ref_curve[1][ref_i], query_curve[1][query_i] + query_y_offset],
            color="#94a3b8",
            linewidth=0.6,
            alpha=0.45,
        )
    ax.set_title("DTW alignment links")
    ax.set_xlabel("Normalized phrase time")
    ax.set_ylabel("Relative pitch, query shifted up for display")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return _save_plot(fig, filename)


def load_demo_files() -> tuple[str, str, str, str, str]:
    return (
        str(DEMO_PATHS["reference"]),
        str(DEMO_PATHS["current"]),
        str(DEMO_PATHS["previous"]),
        "Loaded synthetic demo set from `data/demo/`.",
        "Step 1 complete: files loaded. Next: prepare analysis audio.",
    )


def prepare_audio(
    reference_audio: str | None,
    current_take: str | None,
    previous_take: str | None,
    use_demucs: bool,
    apply_to_takes: bool,
    model: str,
    device: str,
    shifts: int,
    overlap: float,
    progress: gr.Progress | None = None,
) -> tuple[
    dict[str, Any],
    str,
    pd.DataFrame,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str,
]:
    if current_take is None:
        message = "Upload or load a current take before preparing audio."
        return (
            {},
            message,
            pd.DataFrame(),
            None,
            None,
            None,
            None,
            None,
            None,
            message,
        )

    if progress is not None:
        progress(0.00, desc="Preparing pipeline")

    start_time = time.perf_counter()
    rows: list[dict[str, Any]] = []

    if progress is not None:
        progress(0.10, desc="Reference: preparing source")
    reference_source, reference_analysis, reference_row = _prepare_slot(
        reference_audio,
        label="reference",
        should_prepare=use_demucs and reference_audio is not None,
        model=model,
        device=device,
        shifts=shifts,
        overlap=overlap,
    )
    rows.append(reference_row)

    if progress is not None:
        progress(0.40, desc="Current: preparing source")
    current_source, current_analysis, current_row = _prepare_slot(
        current_take,
        label="current",
        should_prepare=use_demucs and apply_to_takes,
        model=model,
        device=device,
        shifts=shifts,
        overlap=overlap,
    )
    rows.append(current_row)

    if progress is not None:
        progress(0.70, desc="Previous: preparing source")
    previous_source, previous_analysis, previous_row = _prepare_slot(
        previous_take,
        label="previous",
        should_prepare=use_demucs and apply_to_takes,
        model=model,
        device=device,
        shifts=shifts,
        overlap=overlap,
    )
    rows.append(previous_row)

    elapsed = time.perf_counter() - start_time
    table = pd.DataFrame(rows)
    config = {
        "backend": "demucs" if use_demucs else "none",
        "model": model,
        "device": device,
        "shifts": int(shifts),
        "overlap": float(overlap),
        "apply_to_takes": bool(apply_to_takes),
    }
    state = {
        "reference_analysis": reference_analysis,
        "current_analysis": current_analysis,
        "previous_analysis": previous_analysis,
        "reference_source": reference_source,
        "current_source": current_source,
        "previous_source": previous_source,
        "config": config,
        "elapsed_s": elapsed,
    }

    status = _ready_status(elapsed, use_demucs, table)
    if progress is not None:
        progress(1.0, desc="Done")

    workflow = (
        "Step 2 complete: we now have analysis inputs. "
        "Next: run Evaluation (Step 3), then matching (Step 4)."
    )
    return (
        state,
        status,
        table,
        _to_path(reference_source),
        _to_path(reference_analysis),
        _to_path(current_source),
        _to_path(current_analysis),
        _to_path(previous_source),
        _to_path(previous_analysis),
        workflow,
    )


def run_evaluation(
    reference_mode: str,
    reference_audio: str | None,
    current_take: str | None,
    previous_take: str | None,
    prepared_state: dict[str, Any] | None,
    progress: gr.Progress | None = None,
) -> tuple[Any, ...]:
    if progress is not None:
        progress(0.00, desc="Loading files and baseline")
    started = time.perf_counter()
    prepared_state = prepared_state or {}
    reference_path = prepared_state.get("reference_analysis") or reference_audio
    current_path = prepared_state.get("current_analysis") or current_take
    previous_path = prepared_state.get("previous_analysis") or previous_take
    if current_path is None:
        message = "Upload or load a current take before evaluation."
        return _empty_evaluation_response(message)

    try:
        if progress is not None:
            progress(0.20, desc="Extracting baseline")
        reference_extraction = None
        if reference_mode == "Uploaded reference audio":
            if reference_path is None:
                message = "Upload or load reference audio for this mode."
                return _empty_evaluation_response(message)
            reference_extraction = extract_reference_audio(
                reference_path,
                title=Path(reference_path).name,
                window_s=0.20,
                pitch_kwargs=PITCH_KWARGS,
                clean_kwargs=CLEAN_KWARGS,
            )
            baseline = reference_extraction.baseline
        else:
            baseline = demo_baseline()

        if progress is not None:
            progress(0.45, desc="Running scoring")
        if previous_path:
            if reference_extraction is not None:
                comparison = compare_takes_to_reference_contour(
                    previous_path,
                    current_path,
                    reference_extraction.contour,
                    **CONTOUR_SCORE_KWARGS,
                )
            else:
                comparison = compare_takes(
                    previous_path,
                    current_path,
                    baseline,
                    **SYMBOLIC_SCORE_KWARGS,
                )
            score = comparison.current
            details = comparison.to_dict()
            verdict = comparison.verdict
        else:
            if reference_extraction is not None:
                score = score_take_against_reference_contour(
                    current_path,
                    reference_extraction.contour,
                    name="current",
                    **CONTOUR_SCORE_KWARGS,
                )
            else:
                score = score_take(current_path, baseline, name="current", **SYMBOLIC_SCORE_KWARGS)
            details = score.to_dict()
            verdict = "single take"

        if progress is not None:
            progress(0.75, desc="Building plots")
        current_contour = _extract_clean_contour(current_path, "current")
        previous_contour = _extract_clean_contour(previous_path, "previous") if previous_path else None
        if reference_extraction is not None:
            pitch_plot = _save_plot(
                plot_contour_comparison(
                    reference_extraction.contour,
                    previous_contour,
                    current_contour,
                ),
                "gradio_evaluation_pitch.png",
            )
            coverage_plot = _save_plot(
                plot_contour_voiced_coverage(
                    reference_extraction.contour,
                    previous_contour,
                    current_contour,
                ),
                "gradio_evaluation_coverage.png",
            )
        else:
            pitch_plot = _save_plot(
                plot_take_comparison(baseline, previous_contour, current_contour),
                "gradio_evaluation_pitch.png",
            )
            coverage_plot = _save_plot(
                plot_voiced_coverage(baseline, previous_contour, current_contour),
                "gradio_evaluation_coverage.png",
            )

        elapsed = time.perf_counter() - started
        metrics = pd.DataFrame(
            [
                {"metric": "overall", "value": score.overall_score},
                {"metric": "pitch_accuracy", "value": score.pitch_accuracy_score},
                {"metric": "stability", "value": score.stability_score},
                {"metric": "coverage", "value": score.coverage_score},
                {"metric": "timing", "value": score.timing_score},
            ]
        )
        status = f"Evaluation complete in {elapsed:.2f}s. Verdict: {verdict}."
        warnings = "\n".join(score.warnings)
        if warnings:
            status = f"{status}\n\nWarnings:\n{warnings}"
        if progress is not None:
            progress(1.0, desc="Done")
        workflow = "Evaluation complete. Next: optional step 4 (song/section matching)."
        diagnostics = _build_evaluation_diagnostics(
            reference_mode=reference_mode,
            reference_path=reference_path,
            current_path=current_path,
            previous_path=previous_path,
            current_contour=current_contour,
            previous_contour=previous_contour,
            reference_contour=reference_extraction.contour if reference_extraction is not None else None,
            score=score,
        )
        return (
            status,
            metrics,
            pitch_plot,
            coverage_plot,
            *diagnostics,
            details,
            workflow,
        )
    except Exception as exc:
        status = f"Evaluation failed after {time.perf_counter() - started:.2f}s: {exc}"
        return _empty_evaluation_response(status)


def run_matching(
    catalog_source: str,
    reference_audio: str | None,
    current_take: str | None,
    prepared_state: dict[str, Any] | None,
    progress: gr.Progress | None = None,
) -> tuple[Any, ...]:
    if progress is not None:
        progress(0.00, desc="Preparing matching inputs")
    started = time.perf_counter()
    prepared_state = prepared_state or {}
    query_path = prepared_state.get("current_analysis") or current_take
    reference_path = prepared_state.get("reference_analysis") or reference_audio
    if query_path is None:
        message = "Upload or load a current take before matching."
        return _empty_matching_response(message)

    try:
        if progress is not None:
            progress(0.20, desc="Extracting query contour")
        query = extract_matching_query(
            query_path,
            name="matching query",
            pitch_kwargs=PITCH_KWARGS,
            clean_kwargs=CLEAN_KWARGS,
        )
        if catalog_source == "Uploaded reference sections":
            if reference_path is None:
                message = "Upload or load reference audio for uploaded-reference matching."
                return _empty_matching_response(message)
            if progress is not None:
                progress(0.40, desc="Building reference sections")
            reference = extract_reference_audio(
                reference_path,
                title=Path(reference_path).name,
                pitch_kwargs=PITCH_KWARGS,
                clean_kwargs=CLEAN_KWARGS,
            )
            sections = split_contour_into_sections(
                reference.contour,
                song_id="uploaded_reference",
                song_title=Path(reference_path).stem,
                window_s=20.0,
                hop_s=10.0,
            )
        else:
            sections = build_demo_section_catalog()

        if progress is not None:
            progress(0.70, desc="Matching query to sections")
        result = match_query_to_sections(query, sections, top_k=5)
        candidates = pd.DataFrame([candidate.to_dict() for candidate in result.candidates])
        if result.best is None:
            elapsed = time.perf_counter() - started
            message = f"No match found in {elapsed:.2f}s."
            steps = _matching_steps_table(
                catalog_source=catalog_source,
                query_path=query_path,
                reference_path=reference_path,
                query=query,
                section_count=len(sections),
                result=result,
                handoff_score=None,
            )
            diagnostics = _build_matching_diagnostics(
                catalog_source=catalog_source,
                query_path=query_path,
                reference_path=reference_path,
                query=query,
                sections=sections,
                result=result,
                handoff_score=None,
            )
            return (
                message,
                candidates,
                None,
                pd.DataFrame(),
                steps,
                *diagnostics,
                result.to_dict(),
                message,
            )

        if progress is not None:
            progress(0.85, desc="Scoring best match")
        plot_path = _save_plot(
            plot_section_match(
                result.best.section.contour,
                result.query,
                query_start_s=result.best.query_start_s,
                query_end_s=result.best.query_end_s,
            ),
            "gradio_section_match.png",
        )
        query_window = crop_contour(
            result.query,
            result.best.query_start_s,
            result.best.query_end_s,
            name="matched query window",
            shift_to_zero=True,
        )
        score = score_take_against_reference_contour(
            query_window,
            result.best.section.contour,
            name="matched query",
            **CONTOUR_SCORE_KWARGS,
        )
        handoff = pd.DataFrame(
            [
                {"metric": "overall", "value": score.overall_score},
                {"metric": "pitch_accuracy", "value": score.pitch_accuracy_score},
                {"metric": "stability", "value": score.stability_score},
                {"metric": "timing", "value": score.timing_score},
            ]
        )
        elapsed = time.perf_counter() - started
        status = (
            f"Matching complete in {elapsed:.2f}s. "
            f"Best match: {result.best.section.display_name} ({result.best.score:.1f})."
        )
        warnings = "\n".join((*result.warnings, *score.warnings))
        if warnings:
            status = f"{status}\n\nWarnings:\n{warnings}"
        if progress is not None:
            progress(1.0, desc="Done")
        steps = _matching_steps_table(
            catalog_source=catalog_source,
            query_path=query_path,
            reference_path=reference_path,
            query=query,
            section_count=len(sections),
            result=result,
            handoff_score=score,
        )
        diagnostics = _build_matching_diagnostics(
            catalog_source=catalog_source,
            query_path=query_path,
            reference_path=reference_path,
            query=query,
            sections=sections,
            result=result,
            handoff_score=score,
        )
        return (
            status,
            candidates,
            plot_path,
            handoff,
            steps,
            *diagnostics,
            result.to_dict(),
            "Matching complete. Workflow finished.",
        )
    except Exception as exc:
        message = f"Matching failed after {time.perf_counter() - started:.2f}s: {exc}"
        return _empty_matching_response(message)


def _extract_clean_contour(path: str, name: str):
    from konopro_research.audio_io import load_audio

    audio, sample_rate = load_audio(path)
    return clean_pitch_contour(
        extract_pitch(audio, sample_rate, name=name, **PITCH_KWARGS),
        **CLEAN_KWARGS,
    )


def _save_plot(fig: Any, filename: str) -> str:
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=160, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass
    return str(path)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Konopro Gradio Research Demo") as demo:
        gr.Markdown(
            """
# Konopro Gradio Research Demo

This is the same research backend as Streamlit, but with explicit stages and
playback checkpoints for local research testing.
"""
        )

        prepared_state = gr.State({})

        workflow_status = gr.Markdown("### Workflow: Step 1 not started.")
        with gr.Accordion("Step 1 - Import & Preview Files", open=True):
            with gr.Row():
                reference_audio = gr.Audio(label="Reference audio", type="filepath")
                current_take = gr.Audio(label="Current take", type="filepath")
                previous_take = gr.Audio(label="Previous take (optional)", type="filepath")
            step1_status = gr.Markdown()
            load_demo = gr.Button("Load Synthetic Demo Files", variant="primary")
            load_demo.click(
                load_demo_files,
                outputs=[reference_audio, current_take, previous_take, step1_status, workflow_status],
            )

        with gr.Accordion("Step 2 - Prepare Analysis Audio", open=True):
            use_demucs = gr.Checkbox(label="Use Demucs vocal stem", value=False)
            gr.Markdown("We currently support the **vocals** stem only. No other stems are used for scoring.")
            stem_selector = gr.Radio(
                ["vocals"],
                value="vocals",
                label="Stem used for analysis",
                interactive=False,
            )
            apply_to_takes = gr.Checkbox(label="Apply Demucs to current/previous takes too", value=False)
            with gr.Row():
                model = gr.Dropdown(
                    ["htdemucs", "htdemucs_ft", "mdx_extra", "mdx_q"],
                    value="htdemucs",
                    label="Demucs model",
                )
                device = gr.Dropdown(["cpu", "mps", "cuda"], value="cpu", label="Device")
                shifts = gr.Slider(1, 4, value=1, step=1, label="Shifts")
                overlap = gr.Slider(0.10, 0.50, value=0.25, step=0.05, label="Overlap")
            prepare_button = gr.Button("Prepare Analysis Audio", variant="primary")
            step2_status = gr.Markdown()
            prepare_table = gr.Dataframe(label="Preparation Results")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Reference")
                    step2_reference_source = gr.Audio(
                        label="Original reference",
                        type="filepath",
                        interactive=False,
                    )
                    step2_reference_analysis = gr.Audio(
                        label="Reference analysis audio",
                        type="filepath",
                        interactive=False,
                    )
                with gr.Column():
                    gr.Markdown("### Current")
                    step2_current_source = gr.Audio(
                        label="Original current",
                        type="filepath",
                        interactive=False,
                    )
                    step2_current_analysis = gr.Audio(
                        label="Current analysis audio",
                        type="filepath",
                        interactive=False,
                    )
                with gr.Column():
                    gr.Markdown("### Previous (optional)")
                    step2_previous_source = gr.Audio(
                        label="Original previous",
                        type="filepath",
                        interactive=False,
                    )
                    step2_previous_analysis = gr.Audio(
                        label="Previous analysis audio",
                        type="filepath",
                        interactive=False,
                    )

            prepare_button.click(
                prepare_audio,
                inputs=[
                    reference_audio,
                    current_take,
                    previous_take,
                    use_demucs,
                    apply_to_takes,
                    model,
                    device,
                    shifts,
                    overlap,
                ],
                outputs=[
                    prepared_state,
                    step2_status,
                    prepare_table,
                    step2_reference_source,
                    step2_reference_analysis,
                    step2_current_source,
                    step2_current_analysis,
                    step2_previous_source,
                    step2_previous_analysis,
                    workflow_status,
                ],
            )

        with gr.Accordion("Step 3 - Evaluate Singing", open=False):
            step3_status = gr.Markdown()
            reference_mode = gr.Radio(
                ["Demo symbolic baseline", "Uploaded reference audio"],
                value="Demo symbolic baseline",
                label="Reference mode",
            )
            evaluation_button = gr.Button("Run Evaluation", variant="primary")
            with gr.Tabs():
                with gr.Tab("Results"):
                    evaluation_metrics = gr.Dataframe(label="Evaluation Metrics")
                    with gr.Row():
                        pitch_plot = gr.Image(label="Pitch plot", type="filepath")
                        coverage_plot = gr.Image(label="Coverage plot", type="filepath")
                    evaluation_json = gr.JSON(label="Evaluation Details")
                with gr.Tab("Inside the evaluator"):
                    evaluation_steps = gr.Dataframe(label="Eight internal steps", wrap=True)
                    with gr.Accordion("1. Prepared analysis audio", open=True):
                        gr.Markdown(
                            "**?** This shows the exact audio Step 3 scored. If Step 2 used Demucs, these are the prepared/stem files, not necessarily the original uploads."
                        )
                        evaluation_audio_table = gr.Dataframe(label="Audio selected for evaluation")
                        with gr.Row():
                            eval_reference_audio = gr.Audio(
                                label="Reference analysis audio",
                                type="filepath",
                                interactive=False,
                            )
                            eval_current_audio = gr.Audio(
                                label="Current analysis audio",
                                type="filepath",
                                interactive=False,
                            )
                            eval_previous_audio = gr.Audio(
                                label="Previous analysis audio",
                                type="filepath",
                                interactive=False,
                            )
                        with gr.Row():
                            eval_reference_preview = gr.Audio(
                                label="Reference first 30s",
                                type="filepath",
                                interactive=False,
                            )
                            eval_current_preview = gr.Audio(
                                label="Current first 30s",
                                type="filepath",
                                interactive=False,
                            )
                            eval_previous_preview = gr.Audio(
                                label="Previous first 30s",
                                type="filepath",
                                interactive=False,
                            )
                    with gr.Accordion("2-4. Pitch extraction and cleaning", open=True):
                        gr.Markdown(
                            "**?** Gray is raw pYIN output; blue is after confidence filtering, octave correction, and jump cleaning. This is where sparse or broken contours usually appear."
                        )
                        with gr.Row():
                            eval_reference_raw_plot = gr.Image(
                                label="Reference raw vs cleaned pitch",
                                type="filepath",
                            )
                            eval_current_raw_plot = gr.Image(
                                label="Current raw vs cleaned pitch",
                                type="filepath",
                            )
                            eval_previous_raw_plot = gr.Image(
                                label="Previous raw vs cleaned pitch",
                                type="filepath",
                            )
                    with gr.Accordion("5. Full-track alignment", open=True):
                        gr.Markdown(
                            "**?** Step 3 currently compares the full reference against the full current take. This can punish late starts or extra silence more than matched-section scoring."
                        )
                        eval_alignment_plot = gr.Image(
                            label="Full-track DTW alignment",
                            type="filepath",
                        )
                        eval_normalization_plot = gr.Image(
                            label="Full-track pitch normalization",
                            type="filepath",
                        )
                    with gr.Accordion("6-8. Metrics and final score", open=True):
                        gr.Markdown(
                            "**?** This table maps each score back to the signal that created it, so low timing/stability/pitch values can be debugged directly."
                        )
                        evaluation_score_breakdown = gr.Dataframe(label="Score breakdown", wrap=True)
            evaluation_button.click(
                run_evaluation,
                inputs=[
                    reference_mode,
                    reference_audio,
                    current_take,
                    previous_take,
                    prepared_state,
                ],
                outputs=[
                    step3_status,
                    evaluation_metrics,
                    pitch_plot,
                    coverage_plot,
                    evaluation_steps,
                    evaluation_audio_table,
                    eval_reference_audio,
                    eval_current_audio,
                    eval_previous_audio,
                    eval_reference_preview,
                    eval_current_preview,
                    eval_previous_preview,
                    eval_reference_raw_plot,
                    eval_current_raw_plot,
                    eval_previous_raw_plot,
                    eval_alignment_plot,
                    eval_normalization_plot,
                    evaluation_score_breakdown,
                    evaluation_json,
                    workflow_status,
                ],
            )

        with gr.Accordion("Step 4 - Match Song / Section", open=False):
            step4_status = gr.Markdown()
            catalog_source = gr.Radio(
                ["Demo catalog", "Uploaded reference sections"],
                value="Demo catalog",
                label="Catalog source",
            )
            matching_button = gr.Button("Run Song/Section Matching", variant="primary")
            with gr.Tabs():
                with gr.Tab("Results"):
                    match_table = gr.Dataframe(label="Candidate Matches")
                    match_plot = gr.Image(label="Best match pitch-shape plot", type="filepath")
                    handoff_metrics = gr.Dataframe(label="Handoff Score")
                    matching_json = gr.JSON(label="Matching Details")
                with gr.Tab("Inside the matcher"):
                    matching_steps = gr.Dataframe(label="Eight internal steps", wrap=True)
                    with gr.Accordion("1. Analysis audio", open=True):
                        diagnostic_audio_table = gr.Dataframe(label="Audio selected for matching")
                        with gr.Row():
                            diag_reference_audio = gr.Audio(
                                label="Reference analysis audio",
                                type="filepath",
                                interactive=False,
                            )
                            diag_query_audio = gr.Audio(
                                label="Current/query analysis audio",
                                type="filepath",
                                interactive=False,
                            )
                    with gr.Accordion("2-3. Raw pitch and cleaned pitch", open=True):
                        with gr.Row():
                            raw_reference_plot = gr.Image(
                                label="Reference raw vs cleaned contour",
                                type="filepath",
                            )
                            raw_query_plot = gr.Image(
                                label="Query raw vs cleaned contour",
                                type="filepath",
                            )
                    with gr.Accordion("4. Reference section windows", open=True):
                        reference_sections_table = gr.Dataframe(label="Top reference sections")
                        with gr.Row():
                            ref_section_audio_1 = gr.Audio(
                                label="Reference section 1",
                                type="filepath",
                                interactive=False,
                            )
                            ref_section_audio_2 = gr.Audio(
                                label="Reference section 2",
                                type="filepath",
                                interactive=False,
                            )
                            ref_section_audio_3 = gr.Audio(
                                label="Reference section 3",
                                type="filepath",
                                interactive=False,
                            )
                    with gr.Accordion("5. Query phrase windows", open=True):
                        query_windows_table = gr.Dataframe(label="Top query windows")
                        with gr.Row():
                            query_window_audio_1 = gr.Audio(
                                label="Query window 1",
                                type="filepath",
                                interactive=False,
                            )
                            query_window_audio_2 = gr.Audio(
                                label="Query window 2",
                                type="filepath",
                                interactive=False,
                            )
                            query_window_audio_3 = gr.Audio(
                                label="Query window 3",
                                type="filepath",
                                interactive=False,
                            )
                    with gr.Accordion("6. Pitch normalization", open=True):
                        median_table = gr.Dataframe(label="Median pitch values")
                        normalization_plot = gr.Image(
                            label="Before and after median pitch removal",
                            type="filepath",
                        )
                        with gr.Row():
                            median_reference_tone = gr.Audio(
                                label="Reference median pitch tone",
                                type="filepath",
                                interactive=False,
                            )
                            median_query_tone = gr.Audio(
                                label="Query median pitch tone",
                                type="filepath",
                                interactive=False,
                            )
                    with gr.Accordion("7. DTW alignment", open=True):
                        dtw_plot = gr.Image(label="DTW alignment plot", type="filepath")
                    with gr.Accordion("8. Scoring handoff", open=True):
                        with gr.Row():
                            handoff_reference_audio = gr.Audio(
                                label="Matched reference audio",
                                type="filepath",
                                interactive=False,
                            )
                            handoff_query_audio = gr.Audio(
                                label="Matched query audio",
                                type="filepath",
                                interactive=False,
                            )
            matching_button.click(
                run_matching,
                inputs=[catalog_source, reference_audio, current_take, prepared_state],
                outputs=[
                    step4_status,
                    match_table,
                    match_plot,
                    handoff_metrics,
                    matching_steps,
                    diagnostic_audio_table,
                    diag_reference_audio,
                    diag_query_audio,
                    raw_reference_plot,
                    raw_query_plot,
                    reference_sections_table,
                    ref_section_audio_1,
                    ref_section_audio_2,
                    ref_section_audio_3,
                    query_windows_table,
                    query_window_audio_1,
                    query_window_audio_2,
                    query_window_audio_3,
                    median_table,
                    normalization_plot,
                    median_reference_tone,
                    median_query_tone,
                    dtw_plot,
                    handoff_reference_audio,
                    handoff_query_audio,
                    matching_json,
                    workflow_status,
                ],
            )
    return demo


if __name__ == "__main__":
    port = int(os.environ["GRADIO_SERVER_PORT"]) if os.environ.get("GRADIO_SERVER_PORT") else None
    build_app().queue().launch(server_name="127.0.0.1", server_port=port)
