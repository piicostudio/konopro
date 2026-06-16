from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

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
    compare_takes_to_reference_contour_global_offset,
    compare_takes_to_reference_contour,
    contour_timing_debug,
    score_take_against_reference_contour_global_offset,
    score_take_against_reference_contour,
)
from konopro_research.demo_data import ensure_demo_data  # noqa: E402
from konopro_research.env import load_research_env  # noqa: E402
from konopro_research.fingerprinting import (  # noqa: E402
    prepare_fingerprint_windows,
    run_acrcloud_fingerprinting,
    run_audd_fingerprinting,
    run_shazam_fingerprinting,
)
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
    plot_reference_extraction,
    plot_section_match,
    plot_take_comparison,
    plot_voiced_coverage,
)
from konopro_research.reference_audio import extract_reference_audio  # noqa: E402
from konopro_research.research_labs import (  # noqa: E402
    run_fingerprint_diagnostics_lab as lab_run_fingerprint_diagnostics,
    run_long_session_segmentation_lab as lab_run_long_session_segmentation,
    run_matched_progress_lab as lab_run_matched_progress,
    run_pitch_extractor_lab as lab_run_pitch_extractors,
    run_reference_builder_lab as lab_run_reference_builder,
    run_scoring_calibration_lab as lab_run_scoring_calibration,
    run_song_identification_lab as lab_run_song_identification,
    run_stress_test_lab as lab_run_stress_test,
    run_timing_lab as lab_run_timing,
)
from konopro_research.scoring import compare_takes, score_take  # noqa: E402
from konopro_research.separation import prepare_vocal_analysis_audio  # noqa: E402


load_research_env(RESEARCH_ROOT / ".env")
DEMO_PATHS = ensure_demo_data(RESEARCH_ROOT / "data" / "demo")
STEM_CACHE_DIR = RESEARCH_ROOT / ".cache" / "stems"
OUTPUT_DIR = RESEARCH_ROOT / ".cache" / "gradio_outputs"
DEREVERB_CACHE_DIR = RESEARCH_ROOT / ".cache" / "dereverb"
PITCH_CONTOUR_CACHE_DIR = RESEARCH_ROOT / ".cache" / "pitch_contours"
ACTIVE_RMS_CACHE_DIR = RESEARCH_ROOT / ".cache" / "active_rms"
ALIGNMENT_CHECK_CACHE_DIR = RESEARCH_ROOT / ".cache" / "alignment_checks"
PITCH_CONTOUR_CACHE_VERSION = 1
ACTIVE_RMS_CACHE_VERSION = 1
ALIGNMENT_CHECK_CACHE_VERSION = 1
MAX_REQUEST_WINDOW_PREVIEWS = 40
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEREVERB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
PITCH_CONTOUR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
ACTIVE_RMS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
ALIGNMENT_CHECK_CACHE_DIR.mkdir(parents=True, exist_ok=True)


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
    "stability_penalty": 0.20,
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
    "stability_penalty": 0.20,
    "timing_offset_penalty": 180.0,
    "transposition_warning_cents": 90.0,
}


def _default_demucs_device() -> str:
    if importlib.util.find_spec("torch") is None:
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEFAULT_DEMUCS_DEVICE = _default_demucs_device()


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


def _apply_active_rms_to_slot(
    analysis_path: str,
    row: dict[str, Any],
    *,
    enabled: bool,
    target_rms: float,
    active_percentile: float,
) -> tuple[str, dict[str, Any]]:
    if not analysis_path or not enabled:
        row = {**row}
        row.update(
            {
                "active_rms_enabled": bool(enabled),
                "active_rms_before": np.nan,
                "active_rms_after": np.nan,
                "active_rms_gain": np.nan,
                "active_rms_status": "not applied" if not enabled else "missing file",
            }
        )
        return analysis_path, row

    started = time.perf_counter()
    normalized_path, metrics = _active_rms_normalized_file(
        analysis_path,
        target_rms=target_rms,
        active_percentile=active_percentile,
    )
    row = {**row}
    row["analysis_file"] = Path(normalized_path).name
    row["analysis_mode"] = f"{row.get('analysis_mode', 'analysis audio')} + active RMS"
    row["timing_seconds"] = float(row.get("timing_seconds", 0.0) or 0.0) + (time.perf_counter() - started)
    row["active_rms_enabled"] = True
    row["active_rms_before"] = metrics["active_rms_before"]
    row["active_rms_after"] = metrics["active_rms_after"]
    row["active_rms_gain"] = metrics["gain"]
    row["active_rms_status"] = metrics["status"]
    row["active_rms_cache_key"] = metrics["cache_key"]
    existing_notes = str(row.get("notes") or "")
    rms_note = (
        f"Active RMS {metrics['status']}: {metrics['active_rms_before']:.5f} -> "
        f"{metrics['active_rms_after']:.5f}, gain {metrics['gain']:.2f}x"
    )
    row["notes"] = f"{existing_notes}\n{rms_note}" if existing_notes and existing_notes != "Ready" else rms_note
    return normalized_path, row


def _active_rms_normalized_file(
    source_path: str,
    *,
    target_rms: float,
    active_percentile: float,
) -> tuple[str, dict[str, Any]]:
    cache_path, cache_key = _active_rms_cache_path(
        source_path,
        target_rms=target_rms,
        active_percentile=active_percentile,
    )
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.exists() and metadata_path.exists():
        try:
            metrics = json.loads(metadata_path.read_text(encoding="utf-8"))
            metrics["status"] = "cache hit"
            return str(cache_path), metrics
        except Exception:
            pass

    audio, sample_rate = load_audio(source_path)
    normalized, metrics = _normalize_active_rms(
        audio,
        sample_rate=sample_rate,
        target_rms=target_rms,
        active_percentile=active_percentile,
    )
    write_wav(cache_path, normalized, sample_rate)
    metrics = {
        **metrics,
        "cache_key": cache_key,
        "source_path": str(source_path),
        "target_rms": float(target_rms),
        "active_percentile": float(active_percentile),
        "status": "computed",
    }
    metadata_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return str(cache_path), metrics


def _active_rms_cache_path(
    source_path: str,
    *,
    target_rms: float,
    active_percentile: float,
) -> tuple[Path, str]:
    payload = {
        "version": ACTIVE_RMS_CACHE_VERSION,
        "source_hash": _sha256_file(source_path),
        "target_rms": round(float(target_rms), 5),
        "active_percentile": round(float(active_percentile), 3),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    cache_key = hashlib.sha256(encoded).hexdigest()[:24]
    return ACTIVE_RMS_CACHE_DIR / f"{cache_key}.wav", cache_key


def _normalize_active_rms(
    audio: np.ndarray,
    *,
    sample_rate: int,
    target_rms: float,
    active_percentile: float,
) -> tuple[np.ndarray, dict[str, float]]:
    source = np.nan_to_num(np.asarray(audio, dtype=np.float32))
    if source.size == 0:
        return source, {
            "active_rms_before": 0.0,
            "active_rms_after": 0.0,
            "gain": 1.0,
            "active_threshold": 0.0,
            "active_ratio": 0.0,
            "peak_after": 0.0,
        }

    frame_length = min(max(256, int(round(0.05 * sample_rate))), max(1, source.size))
    hop_length = max(1, frame_length // 2)
    frame_rms = _frame_rms(source, frame_length=frame_length, hop_length=hop_length)
    if frame_rms.size == 0:
        before = _rms(source)
        active_threshold = 0.0
        active_ratio = 1.0
    else:
        percentile = float(np.clip(active_percentile, 0.0, 95.0))
        active_threshold = float(np.percentile(frame_rms, percentile))
        active_frames = frame_rms >= max(active_threshold, 1e-8)
        if np.count_nonzero(active_frames) < max(1, int(0.05 * frame_rms.size)):
            active_frames = frame_rms >= 1e-8
        before = float(np.sqrt(np.mean(np.square(frame_rms[active_frames])))) if np.any(active_frames) else _rms(source)
        active_ratio = float(np.mean(active_frames)) if frame_rms.size else 1.0

    if before <= 1e-8:
        gain = 1.0
    else:
        gain = float(target_rms) / before
    normalized = source * gain
    peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    if peak > 0.98:
        normalized = normalized * (0.98 / peak)
        gain *= 0.98 / peak
        peak = 0.98
    after = _rms(normalized[np.abs(normalized) >= 0]) if normalized.size else 0.0
    after_active = before * gain
    return normalized.astype(np.float32), {
        "active_rms_before": round(float(before), 6),
        "active_rms_after": round(float(after_active), 6),
        "full_rms_after": round(float(after), 6),
        "gain": round(float(gain), 4),
        "active_threshold": round(float(active_threshold), 6),
        "active_ratio": round(float(active_ratio), 4),
        "peak_after": round(float(peak), 6),
    }


def _frame_rms(audio: np.ndarray, *, frame_length: int, hop_length: int) -> np.ndarray:
    if audio.size == 0:
        return np.asarray([], dtype=np.float32)
    values: list[float] = []
    for start in range(0, max(1, audio.size - frame_length + 1), hop_length):
        frame = audio[start : start + frame_length]
        if frame.size:
            values.append(_rms(frame))
    if not values:
        values.append(_rms(audio))
    return np.asarray(values, dtype=np.float32)


def _rms(audio: np.ndarray) -> float:
    values = np.asarray(audio, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0


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
        None,
        pd.DataFrame(),
        {},
        pd.DataFrame(),
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
        None,
        None,
        None,
        None,
        pd.DataFrame(),
        {},
        {},
        message,
    )


def _empty_contour_state() -> dict[str, Any]:
    return {
        "reference_mode": None,
        "reference_path": None,
        "current_path": None,
        "previous_path": None,
        "reference_contour": None,
        "current_contour": None,
        "previous_contour": None,
        "raw_plots": {},
        "profile_rows": [],
    }


def _analysis_paths(
    reference_audio: str | None,
    current_take: str | None,
    previous_take: str | None,
    prepared_state: dict[str, Any] | None,
) -> tuple[str | None, str | None, str | None]:
    prepared_state = prepared_state or {}
    return (
        prepared_state.get("reference_analysis") or reference_audio,
        prepared_state.get("current_analysis") or current_take,
        prepared_state.get("previous_analysis") or previous_take,
    )


def _contour_state_matches(
    contour_state: dict[str, Any] | None,
    *,
    reference_mode: str,
    reference_path: str | None,
    current_path: str | None,
    previous_path: str | None,
) -> bool:
    if not contour_state:
        return False
    return (
        contour_state.get("reference_mode") == reference_mode
        and contour_state.get("reference_path") == reference_path
        and contour_state.get("current_path") == current_path
        and contour_state.get("previous_path") == previous_path
        and contour_state.get("current_contour") is not None
    )


def _profile_table(rows: list[dict[str, Any]] | None) -> pd.DataFrame:
    columns = ["stage", "elapsed_s", "notes"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def _profile_row(stage: str, started: float, notes: str = "") -> dict[str, Any]:
    return {
        "stage": stage,
        "elapsed_s": round(time.perf_counter() - started, 2),
        "notes": notes,
    }


def _empty_alignment_check_response(
    contour_state: dict[str, Any] | None,
    message: str,
) -> tuple[Any, ...]:
    return (
        contour_state or _empty_contour_state(),
        message,
        pd.DataFrame(),
        None,
        None,
        None,
        None,
        None,
        None,
        {},
        message,
    )


def _alignment_check_cache_dir(
    reference_path: str,
    current_path: str,
    *,
    max_duration_s: float = 24.0,
) -> tuple[Path, str]:
    payload = {
        "version": ALIGNMENT_CHECK_CACHE_VERSION,
        "reference_hash": _sha256_file(reference_path),
        "current_hash": _sha256_file(current_path),
        "pitch_kwargs": PITCH_KWARGS,
        "clean_kwargs": CLEAN_KWARGS,
        "alignment_kwargs": {
            "dtw_time_weight": CONTOUR_SCORE_KWARGS["dtw_time_weight"],
            "dtw_band_radius": CONTOUR_SCORE_KWARGS["dtw_band_radius"],
            "max_dtw_frames": CONTOUR_SCORE_KWARGS["max_dtw_frames"],
            "timing_penalty": CONTOUR_SCORE_KWARGS["timing_penalty"],
        },
        "preview": {
            "max_duration_s": round(float(max_duration_s), 3),
            "target_peak": 0.85,
            "ab_silence_s": 0.75,
        },
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    cache_key = hashlib.sha256(encoded).hexdigest()[:24]
    return ALIGNMENT_CHECK_CACHE_DIR / cache_key, cache_key


def _load_alignment_check_cache(
    reference_path: str,
    current_path: str,
    *,
    max_duration_s: float = 24.0,
) -> dict[str, Any] | None:
    cache_dir, cache_key = _alignment_check_cache_dir(
        reference_path,
        current_path,
        max_duration_s=max_duration_s,
    )
    metadata_path = cache_dir / "alignment_check.json"
    if not metadata_path.exists():
        return None

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    files = metadata.get("files") if isinstance(metadata, dict) else None
    if not isinstance(files, dict):
        return None

    def cached_file(name: str) -> str | None:
        value = files.get(name)
        if not value:
            return None
        path = Path(str(value))
        return str(path) if path.exists() else None

    alignment_plot = cached_file("alignment_plot")
    normalization_plot = cached_file("normalization_plot")
    timing_debug_payload = metadata.get("timing_debug_payload")
    if not alignment_plot or not normalization_plot or not isinstance(timing_debug_payload, dict):
        return None

    result = {
        "cache_key": cache_key,
        "alignment_plot": alignment_plot,
        "normalization_plot": normalization_plot,
        "aligned_reference_audio": cached_file("aligned_reference_audio"),
        "aligned_current_audio": cached_file("aligned_current_audio"),
        "aligned_ab_audio": cached_file("aligned_ab_audio"),
        "aligned_mix_audio": cached_file("aligned_mix_audio"),
        "timing_debug_payload": timing_debug_payload,
    }
    audible = timing_debug_payload.get("audible_alignment_check")
    if isinstance(audible, dict) and audible.get("available"):
        required_audio = [
            result["aligned_reference_audio"],
            result["aligned_current_audio"],
            result["aligned_ab_audio"],
            result["aligned_mix_audio"],
        ]
        if not all(required_audio):
            return None
    return result


def _write_alignment_check_cache(
    *,
    cache_dir: Path,
    cache_key: str,
    reference_path: str,
    current_path: str,
    alignment_plot: str | None,
    normalization_plot: str | None,
    aligned_reference_audio: str | None,
    aligned_current_audio: str | None,
    aligned_ab_audio: str | None,
    aligned_mix_audio: str | None,
    timing_debug_payload: dict[str, Any],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "cache_key": cache_key,
        "version": ALIGNMENT_CHECK_CACHE_VERSION,
        "created_at": time.time(),
        "reference_path": str(reference_path),
        "current_path": str(current_path),
        "files": {
            "alignment_plot": alignment_plot,
            "normalization_plot": normalization_plot,
            "aligned_reference_audio": aligned_reference_audio,
            "aligned_current_audio": aligned_current_audio,
            "aligned_ab_audio": aligned_ab_audio,
            "aligned_mix_audio": aligned_mix_audio,
        },
        "timing_debug_payload": _json_safe(timing_debug_payload),
    }
    (cache_dir / "alignment_check.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


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
                "what happens": "Uploaded-reference scoring uses DTW only to estimate one global offset. The score then compares same-time frames after that offset. Demo-baseline scoring estimates timing offset against symbolic notes.",
                "why it matters": "The full-song score no longer lets DTW stretch or compress local timing for pitch/stability scoring.",
                "latest value": f"timing_offset_s={score.timing_offset_s}",
            },
            {
                "step": 6,
                "stage": "Metric calculation",
                "what happens": "Pitch accuracy, stability, and coverage are computed from 1:1 same-time contour frames after the global offset.",
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


def _empty_explanation_response(message: str) -> tuple[Any, ...]:
    return (
        {},
        message,
        message,
        pd.DataFrame(),
        {"available": False, "reason": message},
    )


def _empty_problem_moments_response(message: str, explanation_state: dict[str, Any] | None = None) -> tuple[Any, ...]:
    payload = {
        "available": False,
        "reason": message,
        "explanation": explanation_state or {},
        "moments": [],
    }
    return (
        message,
        pd.DataFrame(),
        message,
        None,
        None,
        payload,
        payload,
        message,
    )


def _current_score_dict(evaluation_details: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(evaluation_details, dict):
        return {}
    current = evaluation_details.get("current")
    if isinstance(current, dict):
        return current
    return evaluation_details


def _score_float(score: dict[str, Any], key: str, default: float = np.nan) -> float:
    value = score.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _score_warnings(score: dict[str, Any]) -> list[str]:
    warnings = score.get("warnings", [])
    if isinstance(warnings, tuple):
        return [str(item) for item in warnings]
    if isinstance(warnings, list):
        return [str(item) for item in warnings]
    if warnings:
        return [str(warnings)]
    return []


def _format_cents(value: float) -> str:
    return f"{abs(float(value)):.0f} cents"


def _format_ms(seconds: float) -> str:
    return f"{abs(float(seconds) * 1000.0):.0f} ms"


def _pitch_direction(value_cents: float) -> str:
    if value_cents > 20.0:
        return "above"
    if value_cents < -20.0:
        return "below"
    return "away from"


def _primary_coach_issue(
    score: dict[str, Any],
    timing_debug: dict[str, Any] | None,
    stability_debug: dict[str, Any] | None,
) -> tuple[str, str]:
    pitch = _score_float(score, "pitch_accuracy_score", 100.0)
    stability = _score_float(score, "stability_score", 100.0)
    coverage = _score_float(score, "coverage_score", 100.0)
    timing = _score_float(score, "timing_score", 100.0)
    confidence = _score_float(score, "recording_confidence_score", 100.0)
    stability_cents = _score_float(score, "pitch_stability_cents", np.nan)
    mean_error = _score_float(score, "mean_pitch_error_cents", np.nan)
    global_offset = _score_float(score, "timing_offset_s", 0.0)
    local_timing = (
        _score_float(timing_debug or {}, "median_abs_local_error_s", 0.0)
        if isinstance(timing_debug, dict)
        else 0.0
    )
    p95_residual = (
        _score_float(stability_debug or {}, "p95_abs_residual_cents", np.nan)
        if isinstance(stability_debug, dict)
        else np.nan
    )

    if confidence < 55 or coverage < 45:
        return (
            "recording_confidence",
            f"The app detected only about {coverage:.0f}% of the expected vocal frames. Record a clearer take before judging the score.",
        )
    if timing < 65 or local_timing > 0.18:
        timing_text = (
            f"Your timing differs from the reference by about {_format_ms(local_timing)} on average."
            if local_timing > 0
            else f"Your take starts about {_format_ms(global_offset)} from the reference."
        )
        return (
            "timing",
            timing_text,
        )
    if pitch < 65 and np.isfinite(mean_error):
        return (
            "pitch_accuracy",
            f"The main practice target is pitch accuracy. The average pitch error is about {mean_error:.0f} cents.",
        )
    if stability < 65 and np.isfinite(stability_cents):
        if np.isfinite(p95_residual) and p95_residual > 500:
            return (
                "analysis_check",
                "A few very large pitch spikes are affecting the score. Listen to the highlighted A/B clips first.",
            )
        return (
            "pitch_consistency",
            f"The main practice target is pitch consistency. Pitch errors vary by about {stability_cents:.0f} cents.",
        )
    return (
        "polish",
        "The score is mostly healthy. Use the problem moments to find the most useful phrase-level practice targets.",
    )


def _practice_tip_for_issue(issue: str) -> str:
    tips = {
        "recording_confidence": "Record again with a clearer vocal, then re-run the score.",
        "timing": "Practice the highlighted moments where your timing is farthest from the reference.",
        "pitch_accuracy": "Practice the highlighted moments where your notes are farthest from the reference melody.",
        "pitch_consistency": "Practice the highlighted moments where your pitch moves too much within the phrase.",
        "analysis_check": "Listen to the A/B clips first; these spikes may come from matching or pitch tracking.",
        "polish": "Replay the highlighted moments and repeat only those phrases.",
    }
    return tips.get(issue, tips["polish"])


def _build_explanation_payload(
    evaluation_details: dict[str, Any] | None,
    timing_debug: dict[str, Any] | None,
    stability_debug: dict[str, Any] | None,
) -> dict[str, Any]:
    score = _current_score_dict(evaluation_details)
    if not score:
        return {"available": False, "reason": "Run Step 4 full-song evaluation first."}

    issue, summary = _primary_coach_issue(score, timing_debug, stability_debug)
    overall = _score_float(score, "overall_score", np.nan)
    pitch = _score_float(score, "pitch_accuracy_score", np.nan)
    timing = _score_float(score, "timing_score", np.nan)
    stability = _score_float(score, "stability_score", np.nan)
    coverage = _score_float(score, "coverage_score", np.nan)
    confidence = _score_float(score, "recording_confidence_score", np.nan)
    warnings = _score_warnings(score)

    confidence_label = "high"
    if confidence < 55 or coverage < 50:
        confidence_label = "low"
    elif confidence < 75 or coverage < 75 or warnings:
        confidence_label = "medium"

    issue_rows = [
        {"signal": "pitch_accuracy", "score": round(float(pitch), 2) if np.isfinite(pitch) else np.nan},
        {"signal": "timing", "score": round(float(timing), 2) if np.isfinite(timing) else np.nan},
        {"signal": "stability", "score": round(float(stability), 2) if np.isfinite(stability) else np.nan},
        {"signal": "coverage", "score": round(float(coverage), 2) if np.isfinite(coverage) else np.nan},
    ]
    issue_rows = sorted(
        issue_rows,
        key=lambda row: row["score"] if np.isfinite(row["score"]) else 999.0,
    )
    weakest = [row["signal"] for row in issue_rows[:2]]
    payload = {
        "available": True,
        "summary": summary,
        "primary_issue": issue,
        "analysis_confidence": confidence_label,
        "overall_score": round(float(overall), 2) if np.isfinite(overall) else None,
        "weakest_signals": weakest,
        "practice_tip": _practice_tip_for_issue(issue),
        "warnings": warnings,
        "score_snapshot": {
            "pitch_accuracy": pitch,
            "timing": timing,
            "stability": stability,
            "coverage": coverage,
            "recording_confidence": confidence,
        },
    }
    return _json_safe(payload)


def _explanation_markdown(payload: dict[str, Any]) -> str:
    if not payload.get("available"):
        return str(payload.get("reason") or "Run Step 4 first.")
    weakest = ", ".join(str(item) for item in payload.get("weakest_signals", []))
    warnings = payload.get("warnings") or []
    warning_text = f"\n\nWarnings: {'; '.join(warnings[:3])}" if warnings else ""
    return (
        f"### Coach explanation\n\n"
        f"{payload['summary']}\n\n"
        f"- Primary issue: `{payload['primary_issue']}`\n"
        f"- Analysis confidence: `{payload['analysis_confidence']}`\n"
        f"- Weakest signals: {weakest or 'none'}\n"
        f"- Practice tip: {payload['practice_tip']}"
        f"{warning_text}"
    )


def _explanation_table(payload: dict[str, Any]) -> pd.DataFrame:
    if not payload.get("available"):
        return pd.DataFrame()
    snapshot = payload.get("score_snapshot") or {}
    rows = [
        {"field": "overall_score", "value": payload.get("overall_score")},
        {"field": "primary_issue", "value": payload.get("primary_issue")},
        {"field": "analysis_confidence", "value": payload.get("analysis_confidence")},
    ]
    rows.extend({"field": key, "value": value} for key, value in snapshot.items())
    return pd.DataFrame(rows)


def _build_stability_diagnostics(
    *,
    reference_contour: PitchContour | None,
    current_contour: PitchContour | None,
    score: Any,
    stability_penalty: float,
) -> tuple[str | None, pd.DataFrame, dict[str, Any]]:
    calibration = _stability_calibration_table(score, stability_penalty)
    if reference_contour is None or current_contour is None:
        return (
            None,
            calibration,
            {
                "available": False,
                "reason": "stability residual plot needs uploaded-reference contours",
                "selected_stability_penalty": stability_penalty,
                "pitch_stability_cents": getattr(score, "pitch_stability_cents", None),
            },
        )

    errors = _same_time_pitch_errors_cents(
        reference_contour,
        current_contour,
        offset_s=float(getattr(score, "timing_offset_s", 0.0) or 0.0),
    )
    if errors.empty:
        return (
            None,
            calibration,
            {
                "available": False,
                "reason": "not enough same-time voiced frames for stability diagnostics",
                "selected_stability_penalty": stability_penalty,
                "pitch_stability_cents": getattr(score, "pitch_stability_cents", None),
            },
        )

    signed_errors = errors["signed_error_cents"].to_numpy(dtype=float)
    transposition = float(np.nanmedian(signed_errors))
    residuals = signed_errors - transposition
    errors["residual_error_cents"] = residuals
    stability_cents = float(np.nanstd(residuals))
    abs_residuals = np.abs(residuals)
    plot_path = _save_stability_residual_plot(
        errors,
        stability_cents=stability_cents,
        stability_penalty=stability_penalty,
        filename="evaluation_stability_residuals.png",
    )
    diagnostics = {
        "available": True,
        "method": "global-offset 1:1 residual pitch errors",
        "matched_same_time_frames": int(len(errors)),
        "timing_offset_s": round(float(getattr(score, "timing_offset_s", 0.0) or 0.0), 3),
        "estimated_transposition_cents": round(transposition, 2),
        "pitch_stability_cents": round(stability_cents, 2),
        "selected_stability_penalty": stability_penalty,
        "zero_score_threshold_cents": round(100.0 / stability_penalty, 2) if stability_penalty > 0 else None,
        "median_abs_residual_cents": round(float(np.nanmedian(abs_residuals)), 2),
        "p90_abs_residual_cents": round(float(np.nanpercentile(abs_residuals, 90)), 2),
        "p95_abs_residual_cents": round(float(np.nanpercentile(abs_residuals, 95)), 2),
        "max_abs_residual_cents": round(float(np.nanmax(abs_residuals)), 2),
    }
    return plot_path, calibration, diagnostics


def _same_time_pitch_errors_cents(
    reference: PitchContour,
    current: PitchContour,
    *,
    offset_s: float,
) -> pd.DataFrame:
    current_hz, available = _sample_contour_nearest_for_times(current, reference.times_s + offset_s)
    matched = reference.voiced_mask & available & np.isfinite(current_hz)
    matched &= np.isfinite(reference.frequencies_hz) & (reference.frequencies_hz > 0) & (current_hz > 0)
    if np.count_nonzero(matched) < 3:
        return pd.DataFrame(columns=["reference_time_s", "current_time_s", "signed_error_cents"])
    signed_errors = (hz_to_midi(current_hz[matched]) - hz_to_midi(reference.frequencies_hz[matched])) * 100.0
    return pd.DataFrame(
        {
            "reference_time_s": reference.times_s[matched],
            "current_time_s": reference.times_s[matched] + offset_s,
            "signed_error_cents": signed_errors,
        }
    )


def _problem_moments_from_contours(
    reference: PitchContour,
    current: PitchContour,
    *,
    score: dict[str, Any],
    timing_debug: dict[str, Any] | None,
    max_moments: int = 2,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    offset_s = _score_float(score, "timing_offset_s", np.nan)
    if not np.isfinite(offset_s) and isinstance(timing_debug, dict):
        offset_s = _score_float(timing_debug, "global_offset_s", 0.0)
    if not np.isfinite(offset_s):
        offset_s = 0.0

    errors = _same_time_pitch_errors_cents(reference, current, offset_s=float(offset_s))
    if errors.empty:
        return errors, []

    signed_errors = errors["signed_error_cents"].to_numpy(dtype=float)
    transposition = float(np.nanmedian(signed_errors))
    residuals = signed_errors - transposition
    abs_residuals = np.abs(residuals)
    errors = errors.copy()
    errors["residual_error_cents"] = residuals
    errors["abs_residual_cents"] = abs_residuals
    errors["abs_pitch_error_cents"] = np.abs(signed_errors)

    threshold = float(max(120.0, np.nanpercentile(abs_residuals, 90)))
    bad_indices = np.flatnonzero(abs_residuals >= threshold)
    clusters = _cluster_problem_indices(errors, bad_indices)
    if not clusters:
        clusters = _fallback_peak_clusters(errors, max_moments=max_moments)

    moments: list[dict[str, Any]] = []
    for cluster in clusters:
        window = errors.iloc[cluster]
        if window.empty:
            continue
        start_s = float(window["reference_time_s"].min())
        end_s = float(window["reference_time_s"].max())
        if end_s <= start_s:
            end_s = start_s + max(_median_contour_step_s(reference), 0.20)
        median_signed = float(np.nanmedian(window["signed_error_cents"]))
        median_residual = float(np.nanmedian(window["residual_error_cents"]))
        median_abs = float(np.nanmedian(window["abs_residual_cents"]))
        max_abs = float(np.nanmax(window["abs_residual_cents"]))
        mean_abs_pitch = float(np.nanmean(window["abs_pitch_error_cents"]))
        timing_error_ms = _score_float(timing_debug or {}, "median_abs_local_error_s", 0.0) * 1000.0
        likely_issue = _classify_problem_moment(
            median_abs_residual=median_abs,
            max_abs_residual=max_abs,
            mean_abs_pitch_error=mean_abs_pitch,
            score=score,
            timing_debug=timing_debug,
        )
        moments.append(
            {
                "start_s": round(start_s, 2),
                "end_s": round(end_s, 2),
                "duration_s": round(end_s - start_s, 2),
                "likely_issue": likely_issue,
                "median_signed_error_cents": round(median_signed, 2),
                "median_residual_error_cents": round(median_residual, 2),
                "median_abs_residual_cents": round(median_abs, 2),
                "max_abs_residual_cents": round(max_abs, 2),
                "mean_abs_pitch_error_cents": round(mean_abs_pitch, 2),
                "timing_error_ms": round(float(timing_error_ms), 0),
                "frames": int(len(window)),
                "practice_tip": _moment_practice_tip(
                    likely_issue,
                    median_signed_error=median_signed,
                    median_abs_residual=median_abs,
                    max_abs_residual=max_abs,
                    timing_error_ms=timing_error_ms,
                ),
                "evidence": _moment_evidence(
                    likely_issue,
                    median_signed,
                    median_abs,
                    max_abs,
                    mean_abs_pitch,
                    timing_error_ms,
                ),
            }
        )

    moments = sorted(
        moments,
        key=lambda item: (
            float(item["median_abs_residual_cents"]) + 0.25 * float(item["max_abs_residual_cents"]),
            float(item["duration_s"]),
        ),
        reverse=True,
    )
    for rank, moment in enumerate(moments[:max_moments], start=1):
        moment["rank"] = rank
        moment["timing_offset_s"] = round(float(offset_s), 3)
    return errors, moments[:max_moments]


def _cluster_problem_indices(errors: pd.DataFrame, bad_indices: np.ndarray) -> list[list[int]]:
    if bad_indices.size == 0:
        return []
    times = errors["reference_time_s"].to_numpy(dtype=float)
    clusters: list[list[int]] = []
    current: list[int] = [int(bad_indices[0])]
    for index in bad_indices[1:]:
        index = int(index)
        prev = current[-1]
        if times[index] - times[prev] <= 0.75:
            current.append(index)
        else:
            if len(current) >= 3:
                clusters.append(current)
            current = [index]
    if len(current) >= 3:
        clusters.append(current)
    return clusters


def _fallback_peak_clusters(errors: pd.DataFrame, *, max_moments: int) -> list[list[int]]:
    if errors.empty:
        return []
    times = errors["reference_time_s"].to_numpy(dtype=float)
    residuals = errors["abs_residual_cents"].to_numpy(dtype=float)
    ranked = np.argsort(residuals)[::-1]
    clusters: list[list[int]] = []
    used_centers: list[float] = []
    for index in ranked:
        center = float(times[index])
        if any(abs(center - used) < 4.0 for used in used_centers):
            continue
        mask = np.flatnonzero(np.abs(times - center) <= 1.25)
        if mask.size:
            clusters.append([int(item) for item in mask])
            used_centers.append(center)
        if len(clusters) >= max_moments:
            break
    return clusters


def _classify_problem_moment(
    *,
    median_abs_residual: float,
    max_abs_residual: float,
    mean_abs_pitch_error: float,
    score: dict[str, Any],
    timing_debug: dict[str, Any] | None,
) -> str:
    timing_score = _score_float(score, "timing_score", 100.0)
    local_timing = _score_float(timing_debug or {}, "median_abs_local_error_s", 0.0)
    timing_spread = _score_float(timing_debug or {}, "raw_delta_s_mad_or_std", 0.0)
    if max_abs_residual >= 700.0:
        return "analysis_check"
    if timing_score < 70.0 or local_timing > 0.18 or timing_spread > 0.18:
        return "timing"
    if mean_abs_pitch_error >= 150.0:
        return "pitch_accuracy"
    if median_abs_residual >= 100.0:
        return "pitch_consistency"
    return "polish"


def _moment_practice_tip(
    issue: str,
    *,
    median_signed_error: float,
    median_abs_residual: float,
    max_abs_residual: float,
    timing_error_ms: float,
) -> str:
    direction = _pitch_direction(median_signed_error)
    cents = _format_cents(median_signed_error)
    if issue == "analysis_check":
        return (
            f"Listen to the A/B clip first. This moment has a {max_abs_residual:.0f}-cent spike, "
            "which may come from matching or pitch tracking."
        )
    if issue == "timing":
        return f"Your timing is off by about {timing_error_ms:.0f} ms here. Practice starting this phrase with the reference."
    if issue == "pitch_accuracy":
        if direction == "above":
            return f"Your notes are about {cents} above the reference here. Practice singing this phrase lower by about {cents}."
        if direction == "below":
            return f"Your notes are about {cents} below the reference here. Practice singing this phrase higher by about {cents}."
        return f"Your notes are about {cents} from the reference here. Practice this phrase again."
    if issue == "pitch_consistency":
        return f"Your pitch moves by about {_format_cents(median_abs_residual)} here. Practice keeping these notes steadier."
    return f"The largest pitch difference here is about {_format_cents(max_abs_residual)}. Replay and retry this phrase."


def _moment_evidence(
    issue: str,
    median_signed: float,
    median_abs: float,
    max_abs: float,
    mean_abs_pitch: float,
    timing_error_ms: float,
) -> str:
    return (
        f"{issue}; median pitch error {median_signed:.0f} cents, "
        f"median residual {median_abs:.0f} cents, max residual {max_abs:.0f} cents, "
        f"mean pitch error {mean_abs_pitch:.0f} cents, timing error {timing_error_ms:.0f} ms"
    )


def _write_problem_moment_ab_audio(
    *,
    reference_path: str,
    current_path: str,
    moment: dict[str, Any],
    index: int,
    context_s: float = 1.25,
    max_duration_s: float = 10.0,
) -> str | None:
    offset_s = float(moment.get("timing_offset_s", 0.0) or 0.0)
    start_s = max(0.0, float(moment["start_s"]) - context_s)
    end_s = float(moment["end_s"]) + context_s
    duration_s = min(max_duration_s, max(2.5, end_s - start_s))

    reference_audio, reference_sr = load_audio(reference_path)
    current_audio, current_sr = load_audio(current_path, target_sr=reference_sr)
    reference_clip = _slice_audio(reference_audio, reference_sr, start_s, duration_s)
    current_clip = _slice_audio(current_audio, current_sr, start_s + offset_s, duration_s)
    clip_length = min(reference_clip.size, current_clip.size)
    if clip_length <= 1:
        return None
    reference_clip = _normalize_preview_audio(reference_clip[:clip_length])
    current_clip = _normalize_preview_audio(current_clip[:clip_length])
    silence = np.zeros(int(round(reference_sr * 0.65)), dtype=np.float32)
    ab_audio = np.concatenate([reference_clip, silence, current_clip]).astype(np.float32)
    output_path = OUTPUT_DIR / f"coach_problem_moment_{index}_ab.wav"
    write_wav(output_path, ab_audio, reference_sr)
    return str(output_path)


def _coach_feedback_markdown(
    explanation: dict[str, Any],
    moments: list[dict[str, Any]],
) -> str:
    if not explanation.get("available"):
        return str(explanation.get("reason") or "Run Step 6 explanation first.")
    lines = [
        "### Coach feedback",
        "",
        f"Main issue: **{explanation.get('primary_issue', 'unknown')}**",
        str(explanation.get("summary", "")),
    ]
    if moments:
        lines.extend(["", "Practice these moments:"])
        for moment in moments:
            lines.append(
                f"- #{moment['rank']} {moment['start_s']:.2f}s-{moment['end_s']:.2f}s: "
                f"{moment['practice_tip']}"
            )
    else:
        lines.extend(["", "No strong problem moments were found from the current residual trace."])
    return "\n".join(lines)


def _sample_contour_nearest_for_times(
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
    max_distance = max(_median_contour_step_s(contour) * 1.5, 1e-6)
    available = nearest_distance <= max_distance
    sampled[available] = contour.frequencies_hz[nearest[available]]
    return sampled, available


def _median_contour_step_s(contour: PitchContour) -> float:
    if contour.times_s.size < 2:
        return 0.02
    diffs = np.diff(np.sort(contour.times_s))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 0.02
    return float(np.nanmedian(diffs))


def _stability_calibration_table(score: Any, selected_penalty: float) -> pd.DataFrame:
    stability_cents = float(getattr(score, "pitch_stability_cents", np.nan))
    pitch = float(getattr(score, "pitch_accuracy_score", 0.0) or 0.0)
    coverage = float(getattr(score, "coverage_score", 0.0) or 0.0)
    timing = float(getattr(score, "timing_score", 0.0) or 0.0)
    penalties = sorted({0.25, 0.35, 0.50, 0.70, 0.90, 1.10, 1.30, round(float(selected_penalty), 2)})
    rows: list[dict[str, Any]] = []
    for penalty in penalties:
        if np.isfinite(stability_cents):
            stability = float(np.clip(100.0 - stability_cents * penalty, 0.0, 100.0))
            overall = 0.50 * pitch + 0.20 * stability + 0.15 * coverage + 0.15 * timing
            zero_threshold = 100.0 / penalty if penalty > 0 else np.nan
        else:
            stability = np.nan
            overall = np.nan
            zero_threshold = np.nan
        rows.append(
            {
                "selected": penalty == round(float(selected_penalty), 2),
                "stability_penalty": penalty,
                "zero_if_stability_cents_above": round(float(zero_threshold), 2)
                if np.isfinite(zero_threshold)
                else np.nan,
                "stability_score": round(float(stability), 2) if np.isfinite(stability) else np.nan,
                "overall_if_only_stability_changes": round(float(overall), 2) if np.isfinite(overall) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _save_stability_residual_plot(
    errors: pd.DataFrame,
    *,
    stability_cents: float,
    stability_penalty: float,
    filename: str,
) -> str:
    import matplotlib.pyplot as plt

    times = errors["reference_time_s"].to_numpy(dtype=float)
    residuals = errors["residual_error_cents"].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2))
    axes[0].plot(times, residuals, color="#2563eb", linewidth=0.9, alpha=0.85)
    axes[0].axhline(0.0, color="#111827", linewidth=0.9, alpha=0.75)
    axes[0].axhline(stability_cents, color="#ef4444", linewidth=0.8, linestyle="--", alpha=0.65)
    axes[0].axhline(-stability_cents, color="#ef4444", linewidth=0.8, linestyle="--", alpha=0.65)
    axes[0].set_title("Residual pitch error after transposition removal")
    axes[0].set_xlabel("Reference time (s)")
    axes[0].set_ylabel("Residual error (cents)")
    axes[0].grid(True, alpha=0.25)

    axes[1].hist(residuals, bins=48, color="#2563eb", alpha=0.78)
    axes[1].axvline(0.0, color="#111827", linewidth=0.9, alpha=0.75)
    axes[1].axvline(stability_cents, color="#ef4444", linewidth=0.8, linestyle="--", alpha=0.65)
    axes[1].axvline(-stability_cents, color="#ef4444", linewidth=0.8, linestyle="--", alpha=0.65)
    axes[1].set_title(
        f"Distribution, std={stability_cents:.1f} cents, penalty={stability_penalty:.2f}"
    )
    axes[1].set_xlabel("Residual error (cents)")
    axes[1].set_ylabel("Frames")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    return _save_plot(fig, filename)


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
    timing_debug: dict[str, object] | None,
    raw_plot_paths: dict[str, str | None] | None = None,
) -> tuple[Any, ...]:
    raw_plot_paths = raw_plot_paths or {}
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
    reference_raw_plot = raw_plot_paths.get("reference") or _raw_clean_plot_for_path(
        reference_path,
        "evaluation reference raw vs cleaned pitch",
        "evaluation_reference_raw_clean.png",
    )
    current_raw_plot = raw_plot_paths.get("current") or _raw_clean_plot_for_path(
        current_path,
        "evaluation current raw vs cleaned pitch",
        "evaluation_current_raw_clean.png",
    )
    previous_raw_plot = raw_plot_paths.get("previous") or _raw_clean_plot_for_path(
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
    (
        aligned_reference_audio,
        aligned_current_audio,
        aligned_ab_audio,
        aligned_mix_audio,
        audible_alignment_details,
    ) = _build_offset_alignment_audio(
        reference_path=reference_path,
        current_path=current_path,
        timing_debug=timing_debug,
        score=score,
        reference_contour=reference_contour,
        current_contour=current_contour,
    )
    timing_debug_payload = dict(timing_debug or {})
    timing_debug_payload["audible_alignment_check"] = audible_alignment_details
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
        aligned_reference_audio,
        aligned_current_audio,
        aligned_ab_audio,
        aligned_mix_audio,
        _score_breakdown_table(score),
        timing_debug_payload,
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


def _build_offset_alignment_audio(
    *,
    reference_path: str | None,
    current_path: str | None,
    timing_debug: dict[str, object] | None,
    score: Any,
    reference_contour: PitchContour | None,
    current_contour: PitchContour | None,
    max_duration_s: float = 24.0,
    output_dir: Path | None = None,
    filename_prefix: str = "evaluation_offset",
) -> tuple[str | None, str | None, str | None, str | None, dict[str, object]]:
    if not reference_path or not current_path:
        return (None, None, None, None, {"available": False, "reason": "reference/current audio missing"})

    offset_s = _alignment_offset_s(timing_debug, score)
    if offset_s is None:
        return (None, None, None, None, {"available": False, "reason": "timing offset unavailable"})

    reference_audio, reference_sr = load_audio(reference_path)
    current_audio, current_sr = load_audio(current_path, target_sr=reference_sr)
    reference_duration_s = len(reference_audio) / float(reference_sr) if reference_sr else 0.0
    current_duration_s = len(current_audio) / float(current_sr) if current_sr else 0.0

    # The scorer reports offset as: current/take time - reference time.
    overlap_start_s = max(0.0, -offset_s)
    overlap_end_s = min(reference_duration_s, current_duration_s - offset_s)
    if overlap_end_s <= overlap_start_s:
        return (
            None,
            None,
            None,
            None,
            {
                "available": False,
                "reason": "no overlapping audio after applying timing offset",
                "global_offset_s": round(float(offset_s), 3),
            },
        )

    audition_start_s = _audition_start_s(
        reference_contour=reference_contour,
        current_contour=current_contour,
        offset_s=offset_s,
        overlap_start_s=overlap_start_s,
        overlap_end_s=overlap_end_s,
        max_duration_s=max_duration_s,
    )
    audition_duration_s = min(float(max_duration_s), overlap_end_s - audition_start_s)
    if audition_duration_s <= 0.1:
        return (
            None,
            None,
            None,
            None,
            {
                "available": False,
                "reason": "overlap is too short for an audible preview",
                "global_offset_s": round(float(offset_s), 3),
            },
        )

    reference_start_s = audition_start_s
    current_start_s = audition_start_s + offset_s
    reference_clip = _slice_audio(reference_audio, reference_sr, reference_start_s, audition_duration_s)
    current_clip = _slice_audio(current_audio, current_sr, current_start_s, audition_duration_s)
    clip_length = min(reference_clip.size, current_clip.size)
    if clip_length <= 1:
        return (
            None,
            None,
            None,
            None,
            {
                "available": False,
                "reason": "aligned preview clips are empty",
                "global_offset_s": round(float(offset_s), 3),
            },
        )

    reference_clip = reference_clip[:clip_length]
    current_clip = current_clip[:clip_length]
    reference_preview = _normalize_preview_audio(reference_clip)
    current_preview = _normalize_preview_audio(current_clip)
    silence = np.zeros(int(round(reference_sr * 0.75)), dtype=np.float32)
    ab_audio = np.concatenate([reference_preview, silence, current_preview]).astype(np.float32)
    mix_audio = _normalize_preview_audio(0.5 * reference_preview + 0.5 * current_preview)

    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_out = output_dir / f"{filename_prefix}_reference.wav"
    current_out = output_dir / f"{filename_prefix}_current.wav"
    ab_out = output_dir / f"{filename_prefix}_ab.wav"
    mix_out = output_dir / f"{filename_prefix}_overlay.wav"
    write_wav(reference_out, reference_preview, reference_sr)
    write_wav(current_out, current_preview, reference_sr)
    write_wav(ab_out, ab_audio, reference_sr)
    write_wav(mix_out, mix_audio, reference_sr)

    details = {
        "available": True,
        "method": "global-offset crop only; no DTW time-warping",
        "global_offset_s": round(float(offset_s), 3),
        "reference_start_s": round(float(reference_start_s), 3),
        "current_start_s": round(float(current_start_s), 3),
        "duration_s": round(float(clip_length / reference_sr), 3),
        "interpretation": (
            "If these excerpts do not sound like the same phrase, the full-track scorer is "
            "probably aligning the wrong region."
        ),
    }
    return (str(reference_out), str(current_out), str(ab_out), str(mix_out), details)


def _alignment_offset_s(timing_debug: dict[str, object] | None, score: Any) -> float | None:
    if timing_debug:
        for key in ("global_offset_s", "raw_delta_s_median"):
            value = timing_debug.get(key)
            if isinstance(value, (int, float)) and np.isfinite(value):
                return float(value)
    value = getattr(score, "timing_offset_s", None)
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    return None


def _audition_start_s(
    *,
    reference_contour: PitchContour | None,
    current_contour: PitchContour | None,
    offset_s: float,
    overlap_start_s: float,
    overlap_end_s: float,
    max_duration_s: float,
) -> float:
    anchors: list[float] = []
    if reference_contour is not None:
        reference_times = reference_contour.times_s[reference_contour.voiced_mask]
        reference_times = reference_times[
            (reference_times >= overlap_start_s) & (reference_times <= overlap_end_s)
        ]
        if reference_times.size:
            anchors.append(float(reference_times[0]))
    if current_contour is not None:
        current_times = current_contour.times_s[current_contour.voiced_mask] - offset_s
        current_times = current_times[(current_times >= overlap_start_s) & (current_times <= overlap_end_s)]
        if current_times.size:
            anchors.append(float(current_times[0]))

    if anchors:
        start_s = max(overlap_start_s, max(anchors) - 1.0)
    else:
        start_s = overlap_start_s

    return min(start_s, overlap_end_s)



def _slice_audio(audio: np.ndarray, sample_rate: int, start_s: float, duration_s: float) -> np.ndarray:
    start_index = max(0, int(round(start_s * sample_rate)))
    end_index = min(len(audio), start_index + max(1, int(round(duration_s * sample_rate))))
    return np.asarray(audio[start_index:end_index], dtype=np.float32)


def _normalize_preview_audio(audio: np.ndarray, *, target_peak: float = 0.85) -> np.ndarray:
    preview = np.nan_to_num(np.asarray(audio, dtype=np.float32))
    peak = float(np.max(np.abs(preview))) if preview.size else 0.0
    if peak > 1e-8:
        preview = preview * min(1.0, float(target_peak) / peak)
    return np.clip(preview, -1.0, 1.0).astype(np.float32)


def _raw_clean_plot_for_path(path: str | None, title: str, filename: str) -> str | None:
    if not path:
        return None
    raw, cleaned, _ = _load_or_extract_pitch_contours(path, name=title)
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


def _extract_clean_contour_with_plot(
    path: str,
    *,
    name: str,
    title: str,
    filename: str,
    stage: str,
) -> tuple[PitchContour, str, dict[str, Any]]:
    started = time.perf_counter()
    raw, cleaned, cache_status = _load_or_extract_pitch_contours(path, name=name)
    plot_path = _save_raw_clean_plot(raw, cleaned, title, filename)
    notes = (
        f"{Path(path).name}; "
        f"{_voiced_frame_count(cleaned)} voiced frames over {_contour_duration_s(cleaned):.2f}s; "
        f"{cache_status}"
    )
    return cleaned, plot_path, _profile_row(stage, started, notes)


def _load_or_extract_pitch_contours(path: str, *, name: str) -> tuple[PitchContour, PitchContour, str]:
    cache_path, cache_key = _pitch_contour_cache_path(path)
    cached = _load_cached_pitch_contours(cache_path, name=name)
    if cached is not None:
        raw, cleaned = cached
        return raw, cleaned, f"cache hit {cache_key}"

    audio, sample_rate = load_audio(path)
    raw = extract_pitch(audio, sample_rate, name=name, **PITCH_KWARGS)
    cleaned = clean_pitch_contour(raw, **CLEAN_KWARGS)
    _write_cached_pitch_contours(cache_path, raw, cleaned, source_path=path, cache_key=cache_key)
    return raw, cleaned, f"computed and cached {cache_key}"


def _pitch_contour_cache_path(path: str) -> tuple[Path, str]:
    payload = {
        "version": PITCH_CONTOUR_CACHE_VERSION,
        "source_hash": _sha256_file(path),
        "pitch_kwargs": PITCH_KWARGS,
        "clean_kwargs": CLEAN_KWARGS,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    cache_key = hashlib.sha256(encoded).hexdigest()[:24]
    return PITCH_CONTOUR_CACHE_DIR / f"{cache_key}.npz", cache_key


def _load_cached_pitch_contours(
    cache_path: Path,
    *,
    name: str,
) -> tuple[PitchContour, PitchContour] | None:
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            raw = PitchContour(
                data["raw_times_s"],
                data["raw_frequencies_hz"],
                data["raw_confidence"],
                name=name,
            )
            cleaned = PitchContour(
                data["cleaned_times_s"],
                data["cleaned_frequencies_hz"],
                data["cleaned_confidence"],
                name=name,
            )
        return raw, cleaned
    except Exception:
        return None


def _write_cached_pitch_contours(
    cache_path: Path,
    raw: PitchContour,
    cleaned: PitchContour,
    *,
    source_path: str,
    cache_key: str,
) -> None:
    metadata = {
        "cache_key": cache_key,
        "version": PITCH_CONTOUR_CACHE_VERSION,
        "source_path": str(source_path),
        "pitch_kwargs": PITCH_KWARGS,
        "clean_kwargs": CLEAN_KWARGS,
        "created_at": time.time(),
    }
    tmp_path = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp_path,
        metadata=json.dumps(metadata, sort_keys=True),
        raw_times_s=raw.times_s,
        raw_frequencies_hz=raw.frequencies_hz,
        raw_confidence=raw.confidence,
        cleaned_times_s=cleaned.times_s,
        cleaned_frequencies_hz=cleaned.frequencies_hz,
        cleaned_confidence=cleaned.confidence,
    )
    tmp_path.replace(cache_path)


def _extract_evaluation_contours(
    *,
    reference_mode: str,
    reference_path: str | None,
    current_path: str | None,
    previous_path: str | None,
    progress: gr.Progress | None = None,
) -> tuple[dict[str, Any], str, pd.DataFrame]:
    if current_path is None:
        message = "Upload or load a current take before extracting pitch contours."
        return _empty_contour_state(), message, pd.DataFrame()
    if reference_mode == "Uploaded reference audio" and reference_path is None:
        message = "Upload or load reference audio for uploaded-reference contour extraction."
        return _empty_contour_state(), message, pd.DataFrame()

    profile_rows: list[dict[str, Any]] = []
    raw_plots: dict[str, str | None] = {"reference": None, "current": None, "previous": None}
    reference_contour: PitchContour | None = None
    previous_contour: PitchContour | None = None

    if reference_mode == "Uploaded reference audio" and reference_path is not None:
        if progress is not None:
            progress(0.15, desc="Extracting reference pitch contour")
        reference_contour, raw_plots["reference"], row = _extract_clean_contour_with_plot(
            reference_path,
            name="reference",
            title="evaluation reference raw vs cleaned pitch",
            filename="evaluation_reference_raw_clean.png",
            stage="Reference pYIN + cleaning",
        )
        profile_rows.append(row)

    if progress is not None:
        progress(0.45, desc="Extracting current pitch contour")
    current_contour, raw_plots["current"], row = _extract_clean_contour_with_plot(
        current_path,
        name="current",
        title="evaluation current raw vs cleaned pitch",
        filename="evaluation_current_raw_clean.png",
        stage="Current pYIN + cleaning",
    )
    profile_rows.append(row)

    if previous_path:
        if progress is not None:
            progress(0.70, desc="Extracting previous pitch contour")
        previous_contour, raw_plots["previous"], row = _extract_clean_contour_with_plot(
            previous_path,
            name="previous",
            title="evaluation previous raw vs cleaned pitch",
            filename="evaluation_previous_raw_clean.png",
            stage="Previous pYIN + cleaning",
        )
        profile_rows.append(row)

    state = {
        "reference_mode": reference_mode,
        "reference_path": reference_path,
        "current_path": current_path,
        "previous_path": previous_path,
        "reference_contour": reference_contour,
        "current_contour": current_contour,
        "previous_contour": previous_contour,
        "raw_plots": raw_plots,
        "profile_rows": profile_rows,
    }
    status = (
        "Pitch contours ready. "
        f"Current has {_voiced_frame_count(current_contour)} voiced frames over "
        f"{_contour_duration_s(current_contour):.2f}s."
    )
    if progress is not None:
        progress(1.0, desc="Contours ready")
    return state, status, _profile_table(profile_rows)


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


DEREVERB_METHODS = {
    "Baseline - no cleanup": "baseline",
    "Built-in light spectral cleanup": "builtin_light",
    "Built-in strong spectral cleanup": "builtin_strong",
    "Built-in cleanup + loudness normalization": "builtin_normalized",
    "noisereduce": "noisereduce",
    "DeepFilterNet": "deepfilternet",
    "WPE": "wpe",
    "External enhancement model": "external_enhancer",
}


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dereverb_cache_key(source_path: str, method_id: str, params: dict[str, Any]) -> str:
    payload = {
        "source_hash": _sha256_file(source_path),
        "method_id": method_id,
        "params": params,
        "version": 2,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _dereverb_method_params(method_id: str, params: dict[str, Any]) -> dict[str, Any]:
    if method_id == "baseline":
        return {}
    if method_id == "builtin_light":
        return {
            "light_gate_percentile": params["light_gate_percentile"],
            "tail_attenuation": params["tail_attenuation"],
        }
    if method_id == "builtin_strong":
        return {
            "strong_gate_percentile": params["strong_gate_percentile"],
            "tail_attenuation": params["tail_attenuation"],
        }
    if method_id == "builtin_normalized":
        return {
            "strong_gate_percentile": params["strong_gate_percentile"],
            "tail_attenuation": params["tail_attenuation"],
            "target_rms": params["target_rms"],
        }
    if method_id == "noisereduce":
        return {"noisereduce_strength": params["noisereduce_strength"]}
    if method_id == "deepfilternet":
        return {"deepfilter_atten_lim_db": params["deepfilter_atten_lim_db"]}
    if method_id == "external_enhancer":
        return {"command": os.environ.get("KONOPRO_ENHANCER_CMD", "").strip()}
    return {}


def _select_dereverb_source(
    source_slot: str,
    reference_audio: str | None,
    current_take: str | None,
    previous_take: str | None,
    prepared_state: dict[str, Any] | None,
) -> tuple[str | None, str]:
    prepared_state = prepared_state or {}
    if source_slot == "Reference audio":
        return prepared_state.get("reference_analysis") or reference_audio, "reference"
    if source_slot == "Previous take":
        return prepared_state.get("previous_analysis") or previous_take, "previous"
    return prepared_state.get("current_analysis") or current_take, "current"


def _crop_audio_array(
    audio: np.ndarray,
    sample_rate: int,
    start_s: float,
    duration_s: float,
) -> tuple[np.ndarray, float, float]:
    duration = len(audio) / float(sample_rate) if sample_rate else 0.0
    start_s = min(max(0.0, float(start_s)), max(0.0, duration))
    end_s = min(duration, start_s + max(0.1, float(duration_s)))
    start_index = int(round(start_s * sample_rate))
    end_index = max(start_index + 1, int(round(end_s * sample_rate)))
    return audio[start_index:end_index], start_s, end_s


def _normalize_rms(audio: np.ndarray, target_rms: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    if rms <= 1e-8:
        return audio
    normalized = audio * (float(target_rms) / rms)
    peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    if peak > 0.98:
        normalized = normalized * (0.98 / peak)
    return normalized.astype(np.float32)


def _spectral_cleanup(
    audio: np.ndarray,
    sample_rate: int,
    *,
    gate_percentile: float,
    tail_attenuation: float,
    normalize: bool,
    target_rms: float,
) -> np.ndarray:
    """Small built-in cleanup experiment. This is not a production dereverb model."""
    if audio.size == 0:
        return audio.astype(np.float32)
    try:
        import librosa

        y = np.nan_to_num(np.asarray(audio, dtype=np.float32))
        y = y - float(np.mean(y))
        n_fft = 2048
        hop_length = 512
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.exp(1j * np.angle(stft))
        floor = np.percentile(magnitude, float(gate_percentile), axis=1, keepdims=True)
        threshold = np.maximum(floor * 1.35, 1e-7)
        softness = np.maximum(threshold * 0.40, 1e-7)
        mask = 1.0 / (1.0 + np.exp(-(magnitude - threshold) / softness))
        mask = float(tail_attenuation) + (1.0 - float(tail_attenuation)) * mask
        cleaned = librosa.istft(magnitude * mask * phase, hop_length=hop_length, length=len(y))
    except Exception:
        cleaned = np.asarray(audio, dtype=np.float32).copy()

    cleaned = np.nan_to_num(cleaned).astype(np.float32)
    if normalize:
        cleaned = _normalize_rms(cleaned, target_rms)
    return np.clip(cleaned, -1.0, 1.0).astype(np.float32)


def _apply_noisereduce(audio: np.ndarray, sample_rate: int, strength: float) -> tuple[np.ndarray | None, str]:
    if importlib.util.find_spec("noisereduce") is None:
        return None, "Package `noisereduce` is not installed."
    try:
        import noisereduce as nr

        processed = nr.reduce_noise(
            y=np.asarray(audio, dtype=np.float32),
            sr=sample_rate,
            stationary=False,
            prop_decrease=float(strength),
        )
        return np.asarray(processed, dtype=np.float32), "Processed with noisereduce."
    except Exception as exc:
        return None, f"noisereduce failed: {exc}"


def _run_deepfilternet(
    input_path: Path,
    output_dir: Path,
    atten_lim_db: float,
) -> tuple[Path | None, str]:
    executable = shutil.which("deepFilter")
    if executable is None:
        return None, "DeepFilterNet CLI `deepFilter` is not available on PATH."
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [executable, str(input_path), "--output-dir", str(output_dir)]
    if atten_lim_db > 0:
        command.extend(["--atten-lim", str(int(atten_lim_db))])
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except Exception as exc:
        return None, f"DeepFilterNet failed to start: {exc}"

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        if "torchaudio.backend" in stderr:
            stderr = (
                f"{stderr[-500:]}\n\n"
                "DeepFilterNet is installed, but this version expects an older TorchAudio backend API. "
                "Try TorchAudio 2.8.x or use another cleanup method."
            )
        return None, f"DeepFilterNet failed: {stderr[-500:]}"
    candidates = sorted(output_dir.rglob("*.wav"), key=lambda candidate: candidate.stat().st_mtime)
    if not candidates:
        return None, "DeepFilterNet finished but no WAV output was found."
    limit_note = f" with atten-lim={int(atten_lim_db)} dB" if atten_lim_db > 0 else " with no attenuation limit"
    return candidates[-1], f"Processed with DeepFilterNet{limit_note}."


def _run_external_enhancer(input_path: Path, output_path: Path) -> tuple[Path | None, str]:
    template = os.environ.get("KONOPRO_ENHANCER_CMD", "").strip()
    if not template:
        return None, "Set KONOPRO_ENHANCER_CMD with `{input}` and `{output}` to use an external model."
    command_text = template.format(input=str(input_path), output=str(output_path))
    try:
        completed = subprocess.run(
            shlex.split(command_text),
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
    except Exception as exc:
        return None, f"External enhancement model failed to start: {exc}"
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        return None, f"External enhancement model failed: {stderr[-500:]}"
    if not output_path.exists():
        return None, "External enhancement command finished but did not create the expected output file."
    return output_path, "Processed with external enhancement model."


def _run_wpe(input_path: Path) -> tuple[Path | None, str]:
    if importlib.util.find_spec("nara_wpe") is None:
        return None, "Package `nara_wpe` is not installed."
    return (
        None,
        "WPE support is detected but not run here yet because this prototype loads mono audio. "
        "WPE is mainly useful with multi-channel room recordings.",
    )


def _audio_summary(path: str | Path) -> dict[str, float]:
    audio, sample_rate = load_audio(path)
    if audio.size == 0:
        return {
            "duration_s": 0.0,
            "sample_rate": float(sample_rate),
            "rms": 0.0,
            "peak": 0.0,
            "clipping_ratio": 0.0,
        }
    abs_audio = np.abs(audio)
    return {
        "duration_s": round(len(audio) / float(sample_rate), 3),
        "sample_rate": float(sample_rate),
        "rms": round(float(np.sqrt(np.mean(np.square(audio)))), 5),
        "peak": round(float(np.max(abs_audio)), 5),
        "clipping_ratio": round(float(np.mean(abs_audio >= 0.98)), 6),
    }


def _contour_quality_metrics(path: str | Path, method_label: str) -> tuple[dict[str, Any], str]:
    audio, sample_rate = load_audio(path)
    raw = extract_pitch(audio, sample_rate, name=f"{method_label} raw", **PITCH_KWARGS)
    cleaned = clean_pitch_contour(raw, **CLEAN_KWARGS)
    raw_voiced = int(raw.voiced_mask.sum())
    cleaned_voiced = int(cleaned.voiced_mask.sum())
    total_frames = int(cleaned.times_s.size)
    voiced_ratio = (cleaned_voiced / total_frames * 100.0) if total_frames else 0.0
    voiced_confidence = cleaned.confidence[cleaned.voiced_mask]
    median_confidence = float(np.nanmedian(voiced_confidence)) if voiced_confidence.size else 0.0
    midi = hz_to_midi(cleaned.frequencies_hz[cleaned.voiced_mask])
    if midi.size >= 3:
        frame_jitter_cents = float(np.nanmedian(np.abs(np.diff(midi))) * 100.0)
        pitch_range_semitones = float(np.nanpercentile(midi, 95) - np.nanpercentile(midi, 5))
    else:
        frame_jitter_cents = float("nan")
        pitch_range_semitones = float("nan")
    plot_path = _save_raw_clean_plot(
        raw,
        cleaned,
        f"{method_label}: raw vs cleaned pitch",
        f"dereverb_{method_label.lower().replace(' ', '_').replace('/', '_')}_raw_clean.png",
    )
    metrics = {
        "raw_voiced_frames": raw_voiced,
        "cleaned_voiced_frames": cleaned_voiced,
        "voiced_ratio_pct": round(voiced_ratio, 2),
        "median_confidence": round(median_confidence, 3),
        "frame_jitter_cents": round(frame_jitter_cents, 2) if np.isfinite(frame_jitter_cents) else np.nan,
        "pitch_range_semitones": round(pitch_range_semitones, 2)
        if np.isfinite(pitch_range_semitones)
        else np.nan,
    }
    return metrics, plot_path


def _median_abs_pitch_delta_cents(candidate_path: str | Path, baseline_path: str | Path) -> float:
    candidate = _extract_clean_contour(str(candidate_path), "candidate")
    baseline = _extract_clean_contour(str(baseline_path), "baseline")
    candidate_mask = candidate.voiced_mask
    baseline_mask = baseline.voiced_mask
    if np.count_nonzero(candidate_mask) < 3 or np.count_nonzero(baseline_mask) < 3:
        return float("nan")
    overlap_start = max(float(candidate.times_s[candidate_mask][0]), float(baseline.times_s[baseline_mask][0]))
    overlap_end = min(float(candidate.times_s[candidate_mask][-1]), float(baseline.times_s[baseline_mask][-1]))
    if overlap_end <= overlap_start:
        return float("nan")
    sample_times = np.linspace(overlap_start, overlap_end, 300)
    candidate_midi = np.interp(
        sample_times,
        candidate.times_s[candidate_mask],
        hz_to_midi(candidate.frequencies_hz[candidate_mask]),
    )
    baseline_midi = np.interp(
        sample_times,
        baseline.times_s[baseline_mask],
        hz_to_midi(baseline.frequencies_hz[baseline_mask]),
    )
    return float(np.nanmedian(np.abs(candidate_midi - baseline_midi)) * 100.0)


def _dereverb_source_excerpt(
    source_path: str,
    source_label: str,
    start_s: float,
    duration_s: float,
) -> tuple[str, float, float]:
    audio, sample_rate = load_audio(source_path)
    excerpt, actual_start_s, actual_end_s = _crop_audio_array(audio, sample_rate, start_s, duration_s)
    output_path = DEREVERB_CACHE_DIR / f"source_{source_label}_{actual_start_s:.2f}_{actual_end_s:.2f}.wav"
    write_wav(output_path, excerpt, sample_rate)
    return str(output_path), actual_start_s, actual_end_s


def _process_dereverb_method(
    *,
    method_label: str,
    source_excerpt_path: str,
    light_gate_percentile: float,
    strong_gate_percentile: float,
    tail_attenuation: float,
    target_rms: float,
    noisereduce_strength: float,
    deepfilter_atten_lim_db: float,
) -> tuple[dict[str, Any], pd.DataFrame, str | None, str | None]:
    method_id = DEREVERB_METHODS[method_label]
    params = {
        "light_gate_percentile": float(light_gate_percentile),
        "strong_gate_percentile": float(strong_gate_percentile),
        "tail_attenuation": float(tail_attenuation),
        "target_rms": float(target_rms),
        "noisereduce_strength": float(noisereduce_strength),
        "deepfilter_atten_lim_db": float(deepfilter_atten_lim_db),
    }
    method_params = _dereverb_method_params(method_id, params)
    started = time.perf_counter()
    key = _dereverb_cache_key(source_excerpt_path, method_id, method_params)
    method_dir = DEREVERB_CACHE_DIR / key
    method_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(source_excerpt_path)
    output_path = method_dir / f"{method_id}.wav"
    metadata_path = method_dir / "metadata.json"
    steps: list[dict[str, Any]] = [
        {
            "method": method_label,
            "step": 1,
            "stage": "Select source excerpt",
            "status": "ready",
            "artifact": input_path.name,
        }
    ]

    cached = output_path.exists() and metadata_path.exists()
    status = "processed"
    note = ""
    if cached:
        status = "cached"
        note = "Reused cached processed audio."
        steps.append(
            {
                "method": method_label,
                "step": 2,
                "stage": "Apply cleanup method",
                "status": "cached",
                "artifact": output_path.name,
            }
        )
    elif method_id == "baseline":
        shutil.copyfile(input_path, output_path)
        note = "No cleanup applied."
    elif method_id.startswith("builtin"):
        audio, sample_rate = load_audio(input_path)
        gate_percentile = strong_gate_percentile if method_id != "builtin_light" else light_gate_percentile
        normalize = method_id == "builtin_normalized"
        cleaned = _spectral_cleanup(
            audio,
            sample_rate,
            gate_percentile=gate_percentile,
            tail_attenuation=tail_attenuation,
            normalize=normalize,
            target_rms=target_rms,
        )
        write_wav(output_path, cleaned, sample_rate)
        note = "Processed with built-in spectral cleanup."
    elif method_id == "noisereduce":
        audio, sample_rate = load_audio(input_path)
        processed, note = _apply_noisereduce(audio, sample_rate, noisereduce_strength)
        if processed is None:
            return _unavailable_dereverb_result(method_label, note, started, steps)
        write_wav(output_path, processed, sample_rate)
    elif method_id == "deepfilternet":
        produced, note = _run_deepfilternet(
            input_path,
            method_dir / "deepfilternet_out",
            atten_lim_db=float(deepfilter_atten_lim_db),
        )
        if produced is None:
            return _unavailable_dereverb_result(method_label, note, started, steps)
        shutil.copyfile(produced, output_path)
    elif method_id == "wpe":
        produced, note = _run_wpe(input_path)
        if produced is None:
            return _unavailable_dereverb_result(method_label, note, started, steps)
        shutil.copyfile(produced, output_path)
    elif method_id == "external_enhancer":
        produced, note = _run_external_enhancer(input_path, output_path)
        if produced is None:
            return _unavailable_dereverb_result(method_label, note, started, steps)
        output_path = produced
    else:
        return _unavailable_dereverb_result(method_label, f"Unknown method `{method_id}`.", started, steps)

    if not cached:
        metadata_path.write_text(
            json.dumps({"method": method_id, "params": method_params, "note": note}, indent=2),
            encoding="utf-8",
        )
        steps.append(
            {
                "method": method_label,
                "step": 2,
                "stage": "Apply cleanup method",
                "status": "processed",
                "artifact": output_path.name,
            }
        )

    audio_metrics = _audio_summary(output_path)
    steps.append(
        {
            "method": method_label,
            "step": 3,
            "stage": "Measure audio levels",
            "status": "done",
            "artifact": f"rms={audio_metrics['rms']}; peak={audio_metrics['peak']}",
        }
    )
    contour_metrics, plot_path = _contour_quality_metrics(output_path, method_label)
    steps.append(
        {
            "method": method_label,
            "step": 4,
            "stage": "Extract raw pitch",
            "status": "done",
            "artifact": f"{contour_metrics['raw_voiced_frames']} raw voiced frames",
        }
    )
    steps.append(
        {
            "method": method_label,
            "step": 5,
            "stage": "Clean pitch contour",
            "status": "done",
            "artifact": f"{contour_metrics['cleaned_voiced_frames']} cleaned voiced frames",
        }
    )
    elapsed = time.perf_counter() - started
    row = {
        "method": method_label,
        "status": status,
        "elapsed_s": round(elapsed, 2),
        "note": note,
        **audio_metrics,
        **contour_metrics,
        "audio_path": str(output_path),
        "plot_path": plot_path,
    }
    return row, pd.DataFrame(steps), str(output_path), plot_path


def _unavailable_dereverb_result(
    method_label: str,
    note: str,
    started: float,
    steps: list[dict[str, Any]],
) -> tuple[dict[str, Any], pd.DataFrame, None, None]:
    steps.append(
        {
            "method": method_label,
            "step": 2,
            "stage": "Apply cleanup method",
            "status": "unavailable",
            "artifact": note,
        }
    )
    return (
        {
            "method": method_label,
            "status": "unavailable",
            "elapsed_s": round(time.perf_counter() - started, 2),
            "note": note,
            "duration_s": np.nan,
            "sample_rate": np.nan,
            "rms": np.nan,
            "peak": np.nan,
            "clipping_ratio": np.nan,
            "raw_voiced_frames": np.nan,
            "cleaned_voiced_frames": np.nan,
            "voiced_ratio_pct": np.nan,
            "median_confidence": np.nan,
            "frame_jitter_cents": np.nan,
            "pitch_range_semitones": np.nan,
            "audio_path": "",
            "plot_path": "",
        },
        pd.DataFrame(steps),
        None,
        None,
    )


def _recommend_dereverb_method(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "No cleanup results yet."
    available = summary[summary["status"].isin(["processed", "cached"])].copy()
    if available.empty:
        return "No selected cleanup method produced audio. Check the unavailable/error notes in the table."

    baseline = available[available["method"] == "Baseline - no cleanup"]
    baseline_voiced = float(baseline["voiced_ratio_pct"].iloc[0]) if not baseline.empty else 0.0
    baseline_jitter = float(baseline["frame_jitter_cents"].iloc[0]) if not baseline.empty else np.inf

    def experiment_score(row: pd.Series) -> float:
        voiced = float(row.get("voiced_ratio_pct", 0.0) or 0.0)
        confidence = float(row.get("median_confidence", 0.0) or 0.0)
        jitter = float(row.get("frame_jitter_cents", np.inf) or np.inf)
        if not np.isfinite(jitter):
            jitter = 999.0
        coverage_penalty = max(0.0, baseline_voiced - voiced - 8.0) * 2.5
        change_penalty = max(0.0, float(row.get("pitch_delta_vs_baseline_cents", 0.0) or 0.0) - 35.0) * 0.5
        return (100.0 - min(jitter, 100.0)) + 20.0 * confidence - coverage_penalty - change_penalty

    available["experiment_score"] = available.apply(experiment_score, axis=1)
    best = available.sort_values("experiment_score", ascending=False).iloc[0]
    if best["method"] == "Baseline - no cleanup":
        return (
            "Recommendation: baseline is currently strongest. Cleanup did not clearly reduce pitch jitter "
            "without risking coverage or pitch changes."
        )
    jitter_delta = baseline_jitter - float(best["frame_jitter_cents"])
    return (
        f"Recommendation: try `{best['method']}` first. It reduced the reference-free jitter proxy by "
        f"{jitter_delta:.1f} cents versus baseline while preserving {best['voiced_ratio_pct']}% voiced coverage."
    )


def run_dereverb_experiment(
    source_slot: str,
    reference_audio: str | None,
    current_take: str | None,
    previous_take: str | None,
    prepared_state: dict[str, Any] | None,
    selected_methods: list[str] | None,
    excerpt_start_s: float,
    excerpt_duration_s: float,
    light_gate_percentile: float,
    strong_gate_percentile: float,
    tail_attenuation: float,
    target_rms: float,
    noisereduce_strength: float,
    deepfilter_atten_lim_db: float,
    progress: gr.Progress | None = None,
) -> tuple[Any, ...]:
    started = time.perf_counter()
    selected_methods = selected_methods or []
    ordered_methods = [label for label in DEREVERB_METHODS if label in selected_methods]
    if not ordered_methods:
        return _empty_dereverb_response("Select at least one dereverb/cleanup method.")

    source_path, source_label = _select_dereverb_source(
        source_slot,
        reference_audio,
        current_take,
        previous_take,
        prepared_state,
    )
    if not source_path:
        return _empty_dereverb_response(f"No audio is available for `{source_slot}`.")

    if progress is not None:
        progress(0.02, desc="Creating source excerpt")
    excerpt_path, actual_start_s, actual_end_s = _dereverb_source_excerpt(
        source_path,
        source_label,
        excerpt_start_s,
        excerpt_duration_s,
    )

    rows: list[dict[str, Any]] = []
    step_tables: list[pd.DataFrame] = [
        pd.DataFrame(
            [
                {
                    "method": "All",
                    "step": 0,
                    "stage": "Source excerpt",
                    "status": "ready",
                    "artifact": (
                        f"{Path(source_path).name}; excerpt {actual_start_s:.2f}s-{actual_end_s:.2f}s"
                    ),
                }
            ]
        )
    ]
    audio_outputs: dict[str, str | None] = {method_id: None for method_id in DEREVERB_METHODS.values()}
    plot_outputs: dict[str, str | None] = {method_id: None for method_id in DEREVERB_METHODS.values()}

    for index, method_label in enumerate(ordered_methods, start=1):
        if progress is not None:
            progress(index / max(1, len(ordered_methods) + 1), desc=f"Running {method_label}")
        row, steps, audio_path, plot_path = _process_dereverb_method(
            method_label=method_label,
            source_excerpt_path=excerpt_path,
            light_gate_percentile=light_gate_percentile,
            strong_gate_percentile=strong_gate_percentile,
            tail_attenuation=tail_attenuation,
            target_rms=target_rms,
            noisereduce_strength=noisereduce_strength,
            deepfilter_atten_lim_db=deepfilter_atten_lim_db,
        )
        rows.append(row)
        step_tables.append(steps)
        method_id = DEREVERB_METHODS[method_label]
        audio_outputs[method_id] = audio_path
        plot_outputs[method_id] = plot_path

    summary = pd.DataFrame(rows)
    baseline_rows = summary[summary["method"] == "Baseline - no cleanup"]
    if not baseline_rows.empty and baseline_rows.iloc[0].get("audio_path"):
        baseline_path = baseline_rows.iloc[0]["audio_path"]
        deltas = []
        for row in summary.to_dict("records"):
            if row.get("audio_path"):
                deltas.append(round(_median_abs_pitch_delta_cents(row["audio_path"], baseline_path), 2))
            else:
                deltas.append(np.nan)
        summary["pitch_delta_vs_baseline_cents"] = deltas
    else:
        summary["pitch_delta_vs_baseline_cents"] = np.nan

    recommendation = _recommend_dereverb_method(summary)
    elapsed = time.perf_counter() - started
    status = (
        f"Dereverb experiment complete in {elapsed:.2f}s. "
        f"Source: {source_slot}, excerpt {actual_start_s:.2f}s-{actual_end_s:.2f}s.\n\n"
        f"{recommendation}"
    )
    if progress is not None:
        progress(1.0, desc="Done")
    steps_df = pd.concat(step_tables, ignore_index=True)
    return (
        status,
        summary,
        steps_df,
        excerpt_path,
        audio_outputs["baseline"],
        plot_outputs["baseline"],
        audio_outputs["builtin_light"],
        plot_outputs["builtin_light"],
        audio_outputs["builtin_strong"],
        plot_outputs["builtin_strong"],
        audio_outputs["builtin_normalized"],
        plot_outputs["builtin_normalized"],
        audio_outputs["noisereduce"],
        plot_outputs["noisereduce"],
        audio_outputs["deepfilternet"],
        plot_outputs["deepfilternet"],
        audio_outputs["wpe"],
        plot_outputs["wpe"],
        audio_outputs["external_enhancer"],
        plot_outputs["external_enhancer"],
    )


def _empty_dereverb_response(message: str) -> tuple[Any, ...]:
    return (
        message,
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
        None,
        None,
        None,
        None,
        None,
        None,
    )


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
    use_active_rms: bool,
    target_active_rms: float,
    active_rms_percentile: float,
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
    reference_analysis, reference_row = _apply_active_rms_to_slot(
        reference_analysis,
        reference_row,
        enabled=use_active_rms and bool(reference_analysis),
        target_rms=target_active_rms,
        active_percentile=active_rms_percentile,
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
    current_analysis, current_row = _apply_active_rms_to_slot(
        current_analysis,
        current_row,
        enabled=use_active_rms and bool(current_analysis),
        target_rms=target_active_rms,
        active_percentile=active_rms_percentile,
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
    previous_analysis, previous_row = _apply_active_rms_to_slot(
        previous_analysis,
        previous_row,
        enabled=use_active_rms and bool(previous_analysis),
        target_rms=target_active_rms,
        active_percentile=active_rms_percentile,
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
        "active_rms": {
            "enabled": bool(use_active_rms),
            "target_rms": float(target_active_rms),
            "active_percentile": float(active_rms_percentile),
        },
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
        "Next: inspect pitch/alignment (Step 3), then run full-song evaluation (Step 4)."
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


def run_contour_extraction(
    reference_mode: str,
    reference_audio: str | None,
    current_take: str | None,
    previous_take: str | None,
    prepared_state: dict[str, Any] | None,
    progress: gr.Progress | None = None,
) -> tuple[Any, ...]:
    reference_path, current_path, previous_path = _analysis_paths(
        reference_audio,
        current_take,
        previous_take,
        prepared_state,
    )
    started = time.perf_counter()
    try:
        state, status, profile = _extract_evaluation_contours(
            reference_mode=reference_mode,
            reference_path=reference_path,
            current_path=current_path,
            previous_path=previous_path,
            progress=progress,
        )
        raw_plots = state.get("raw_plots", {})
        if state.get("current_contour") is not None:
            status = f"{status} Extraction finished in {time.perf_counter() - started:.2f}s."
            workflow = "Pitch contours ready. Next: run the audible alignment check or Step 4 full-song evaluation."
        else:
            workflow = status
        return (
            state,
            status,
            profile,
            raw_plots.get("reference"),
            raw_plots.get("current"),
            raw_plots.get("previous"),
            workflow,
        )
    except Exception as exc:
        message = f"Pitch contour extraction failed after {time.perf_counter() - started:.2f}s: {exc}"
        return (
            _empty_contour_state(),
            message,
            pd.DataFrame(),
            None,
            None,
            None,
            message,
        )


def run_alignment_check_only(
    reference_mode: str,
    reference_audio: str | None,
    current_take: str | None,
    previous_take: str | None,
    prepared_state: dict[str, Any] | None,
    contour_state: dict[str, Any] | None,
    progress: gr.Progress | None = None,
) -> tuple[Any, ...]:
    reference_path, current_path, previous_path = _analysis_paths(
        reference_audio,
        current_take,
        previous_take,
        prepared_state,
    )
    if reference_mode != "Uploaded reference audio":
        message = "Audible alignment check needs Uploaded reference audio mode."
        return _empty_alignment_check_response(contour_state, message)
    if current_path is None:
        message = "Upload or load a current take before running the alignment check."
        return _empty_alignment_check_response(contour_state, message)
    if reference_path is None:
        message = "Upload or load reference audio before running the alignment check."
        return _empty_alignment_check_response(contour_state, message)

    started = time.perf_counter()
    profile_rows: list[dict[str, Any]] = []
    try:
        cache_started = time.perf_counter()
        cached_alignment = _load_alignment_check_cache(reference_path, current_path)
        if cached_alignment is not None:
            timing_debug_payload = dict(cached_alignment["timing_debug_payload"])
            timing_debug_payload["alignment_check_cache"] = {
                "status": "hit",
                "cache_key": cached_alignment["cache_key"],
            }
            profile_rows.append(
                _profile_row(
                    "Alignment check cache",
                    cache_started,
                    f"cache hit {cached_alignment['cache_key']}",
                )
            )
            elapsed = time.perf_counter() - started
            profile_rows.append(
                {
                    "stage": "Alignment check total",
                    "elapsed_s": round(elapsed, 2),
                    "notes": "Loaded cached plots, timing JSON, and audio clips.",
                }
            )
            status = f"Audible alignment check loaded from cache in {elapsed:.2f}s."
            if progress is not None:
                progress(1.0, desc="Alignment check cache hit")
            workflow = "Alignment check loaded from cache. Listen to A/B and overlay before running full evaluation."
            cached_state = (
                contour_state
                if _contour_state_matches(
                    contour_state,
                    reference_mode=reference_mode,
                    reference_path=reference_path,
                    current_path=current_path,
                    previous_path=previous_path,
                )
                else _empty_contour_state()
            )
            return (
                cached_state,
                status,
                _profile_table(profile_rows),
                cached_alignment["alignment_plot"],
                cached_alignment["normalization_plot"],
                cached_alignment["aligned_reference_audio"],
                cached_alignment["aligned_current_audio"],
                cached_alignment["aligned_ab_audio"],
                cached_alignment["aligned_mix_audio"],
                timing_debug_payload,
                workflow,
            )

        if _contour_state_matches(
            contour_state,
            reference_mode=reference_mode,
            reference_path=reference_path,
            current_path=current_path,
            previous_path=previous_path,
        ):
            state = contour_state or _empty_contour_state()
            cache_started = time.perf_counter()
            profile_rows.append(
                _profile_row(
                    "Contour cache",
                    cache_started,
                    "Reused previously extracted reference/current contours.",
                )
            )
        else:
            if progress is not None:
                progress(0.05, desc="Extracting contours")
            state, _, contour_profile = _extract_evaluation_contours(
                reference_mode=reference_mode,
                reference_path=reference_path,
                current_path=current_path,
                previous_path=previous_path,
                progress=None,
            )
            profile_rows.extend(contour_profile.to_dict("records"))

        reference_contour = state.get("reference_contour")
        current_contour = state.get("current_contour")
        if reference_contour is None or current_contour is None:
            message = "Alignment check needs both reference and current pitch contours."
            return _empty_alignment_check_response(state, message)

        if progress is not None:
            progress(0.65, desc="Running timing alignment")
        timing_started = time.perf_counter()
        timing_debug = contour_timing_debug(
            current_contour,
            reference_contour,
            name="current",
            pitch_kwargs=PITCH_KWARGS,
            clean_kwargs=CLEAN_KWARGS,
            dtw_time_weight=CONTOUR_SCORE_KWARGS["dtw_time_weight"],
            dtw_band_radius=CONTOUR_SCORE_KWARGS["dtw_band_radius"],
            max_dtw_frames=CONTOUR_SCORE_KWARGS["max_dtw_frames"],
            timing_penalty=CONTOUR_SCORE_KWARGS["timing_penalty"],
        )
        profile_rows.append(
            _profile_row(
                "Timing DTW",
                timing_started,
                f"{timing_debug.get('matched_pairs_count', 0)} matched pairs.",
            )
        )

        alignment_cache_dir, alignment_cache_key = _alignment_check_cache_dir(reference_path, current_path)
        if progress is not None:
            progress(0.78, desc="Building alignment plots")
        plot_started = time.perf_counter()
        alignment_plot = _save_dtw_alignment_plot(
            reference_contour,
            current_contour,
            str(alignment_cache_dir / "evaluation_full_track_dtw.png"),
        )
        normalization_plot = _save_normalization_plot(
            reference_contour,
            current_contour,
            str(alignment_cache_dir / "evaluation_full_track_normalization.png"),
        )
        profile_rows.append(_profile_row("Alignment plots", plot_started, "DTW links and pitch normalization."))

        if progress is not None:
            progress(0.88, desc="Building audio audition clips")
        audio_started = time.perf_counter()
        (
            aligned_reference_audio,
            aligned_current_audio,
            aligned_ab_audio,
            aligned_mix_audio,
            audible_alignment_details,
        ) = _build_offset_alignment_audio(
            reference_path=reference_path,
            current_path=current_path,
            timing_debug=timing_debug,
            score=None,
            reference_contour=reference_contour,
            current_contour=current_contour,
            output_dir=alignment_cache_dir,
            filename_prefix="evaluation_offset",
        )
        profile_rows.append(_profile_row("Alignment audio clips", audio_started, "Global-offset excerpts only."))

        timing_debug_payload = dict(timing_debug or {})
        timing_debug_payload["audible_alignment_check"] = audible_alignment_details
        timing_debug_payload["alignment_check_cache"] = {
            "status": "stored",
            "cache_key": alignment_cache_key,
        }
        _write_alignment_check_cache(
            cache_dir=alignment_cache_dir,
            cache_key=alignment_cache_key,
            reference_path=reference_path,
            current_path=current_path,
            alignment_plot=alignment_plot,
            normalization_plot=normalization_plot,
            aligned_reference_audio=aligned_reference_audio,
            aligned_current_audio=aligned_current_audio,
            aligned_ab_audio=aligned_ab_audio,
            aligned_mix_audio=aligned_mix_audio,
            timing_debug_payload=timing_debug_payload,
        )
        elapsed = time.perf_counter() - started
        profile_rows.append(
            {
                "stage": "Alignment check total",
                "elapsed_s": round(elapsed, 2),
                "notes": f"Contours + DTW + cached clips ({alignment_cache_key}).",
            }
        )
        status = f"Audible alignment check complete in {elapsed:.2f}s."
        if progress is not None:
            progress(1.0, desc="Alignment check ready")
        workflow = "Alignment check complete. Listen to A/B and overlay before running full evaluation."
        return (
            state,
            status,
            _profile_table(profile_rows),
            alignment_plot,
            normalization_plot,
            aligned_reference_audio,
            aligned_current_audio,
            aligned_ab_audio,
            aligned_mix_audio,
            timing_debug_payload,
            workflow,
        )
    except Exception as exc:
        message = f"Alignment check failed after {time.perf_counter() - started:.2f}s: {exc}"
        return _empty_alignment_check_response(contour_state, message)


def run_evaluation(
    reference_mode: str,
    reference_audio: str | None,
    current_take: str | None,
    previous_take: str | None,
    prepared_state: dict[str, Any] | None,
    contour_state: dict[str, Any] | None,
    stability_penalty: float | None = None,
    progress: gr.Progress | None = None,
) -> tuple[Any, ...]:
    if progress is not None:
        progress(0.00, desc="Loading files")
    started = time.perf_counter()
    profile_rows: list[dict[str, Any]] = []
    reference_path, current_path, previous_path = _analysis_paths(
        reference_audio,
        current_take,
        previous_take,
        prepared_state,
    )
    if current_path is None:
        message = "Upload or load a current take before evaluation."
        return _empty_evaluation_response(message)
    if reference_mode == "Uploaded reference audio" and reference_path is None:
        message = "Upload or load reference audio for this mode."
        return _empty_evaluation_response(message)

    try:
        active_stability_penalty = (
            float(stability_penalty)
            if stability_penalty is not None
            else float(CONTOUR_SCORE_KWARGS["stability_penalty"])
        )
        contour_score_kwargs = {
            **CONTOUR_SCORE_KWARGS,
            "stability_penalty": active_stability_penalty,
        }
        symbolic_score_kwargs = {
            **SYMBOLIC_SCORE_KWARGS,
            "stability_penalty": active_stability_penalty,
        }
        if _contour_state_matches(
            contour_state,
            reference_mode=reference_mode,
            reference_path=reference_path,
            current_path=current_path,
            previous_path=previous_path,
        ):
            active_contour_state = contour_state or _empty_contour_state()
            cache_started = time.perf_counter()
            profile_rows.append(
                _profile_row(
                    "Contour cache",
                    cache_started,
                    "Reused previously extracted contours for plots and timing debug.",
                )
            )
        else:
            if progress is not None:
                progress(0.12, desc="Extracting diagnostic contours")
            active_contour_state, _, contour_profile = _extract_evaluation_contours(
                reference_mode=reference_mode,
                reference_path=reference_path,
                current_path=current_path,
                previous_path=previous_path,
                progress=None,
            )
            profile_rows.extend(contour_profile.to_dict("records"))

        reference_contour = active_contour_state.get("reference_contour")
        current_contour = active_contour_state.get("current_contour")
        previous_contour = active_contour_state.get("previous_contour")
        raw_plot_paths = active_contour_state.get("raw_plots", {})
        if current_contour is None:
            message = "Evaluation needs a current pitch contour, but extraction produced none."
            return _empty_evaluation_response(message)
        if reference_mode == "Uploaded reference audio" and reference_contour is None:
            message = "Uploaded-reference evaluation needs a reference pitch contour."
            return _empty_evaluation_response(message)

        if reference_mode == "Demo symbolic baseline":
            baseline = demo_baseline()
        else:
            baseline = None

        if progress is not None:
            progress(0.45, desc="Running full scoring")
        scoring_started = time.perf_counter()
        if previous_path:
            if reference_contour is not None:
                comparison = compare_takes_to_reference_contour_global_offset(
                    previous_contour,
                    current_contour,
                    reference_contour,
                    previous_audio_path=previous_path,
                    current_audio_path=current_path,
                    **contour_score_kwargs,
                )
            else:
                comparison = compare_takes(
                    previous_path,
                    current_path,
                    baseline,
                    **symbolic_score_kwargs,
                )
            score = comparison.current
            details = comparison.to_dict()
            verdict = comparison.verdict
        else:
            if reference_contour is not None:
                score = score_take_against_reference_contour_global_offset(
                    current_contour,
                    reference_contour,
                    name="current",
                    take_audio_path=current_path,
                    **contour_score_kwargs,
                )
            else:
                score = score_take(current_path, baseline, name="current", **symbolic_score_kwargs)
            details = score.to_dict()
            verdict = "single take"
        if reference_contour is not None:
            details["scoring_method"] = "global_offset_1_to_1_contour"
        profile_rows.append(
            _profile_row(
                "Full scoring",
                scoring_started,
                (
                    "Global-offset 1:1 contour scorer; DTW estimates offset only."
                    if reference_contour is not None
                    else "Symbolic baseline scorer."
                ),
            )
        )

        if progress is not None:
            progress(0.72, desc="Running timing debug")
        timing_debug = None
        if reference_contour is not None:
            timing_started = time.perf_counter()
            timing_debug = contour_timing_debug(
                current_contour,
                reference_contour,
                name="current",
                pitch_kwargs=PITCH_KWARGS,
                clean_kwargs=CLEAN_KWARGS,
                dtw_time_weight=CONTOUR_SCORE_KWARGS["dtw_time_weight"],
                dtw_band_radius=CONTOUR_SCORE_KWARGS["dtw_band_radius"],
                max_dtw_frames=CONTOUR_SCORE_KWARGS["max_dtw_frames"],
                timing_penalty=CONTOUR_SCORE_KWARGS["timing_penalty"],
            )
            profile_rows.append(
                _profile_row(
                    "Timing debug",
                    timing_started,
                    f"{timing_debug.get('matched_pairs_count', 0)} matched pairs.",
                )
            )

        if progress is not None:
            progress(0.82, desc="Building summary plots")
        plots_started = time.perf_counter()
        if reference_contour is not None:
            pitch_plot = _save_plot(
                plot_contour_comparison(
                    reference_contour,
                    previous_contour,
                    current_contour,
                ),
                "gradio_evaluation_pitch.png",
            )
            coverage_plot = _save_plot(
                plot_contour_voiced_coverage(
                    reference_contour,
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
        profile_rows.append(_profile_row("Summary plots", plots_started, "Pitch and coverage plots."))

        metrics = pd.DataFrame(
            [
                {"metric": "overall", "value": score.overall_score},
                {"metric": "pitch_accuracy", "value": score.pitch_accuracy_score},
                {"metric": "stability", "value": score.stability_score},
                {"metric": "coverage", "value": score.coverage_score},
                {"metric": "timing", "value": score.timing_score},
            ]
        )
        stability_plot, stability_table, stability_debug = _build_stability_diagnostics(
            reference_contour=reference_contour,
            current_contour=current_contour,
            score=score,
            stability_penalty=active_stability_penalty,
        )
        if progress is not None:
            progress(0.90, desc="Building diagnostics")
        diagnostics_started = time.perf_counter()
        diagnostics = _build_evaluation_diagnostics(
            reference_mode=reference_mode,
            reference_path=reference_path,
            current_path=current_path,
            previous_path=previous_path,
            current_contour=current_contour,
            previous_contour=previous_contour,
            reference_contour=reference_contour,
            score=score,
            timing_debug=timing_debug,
            raw_plot_paths=raw_plot_paths,
        )
        profile_rows.append(
            _profile_row(
                "Diagnostics",
                diagnostics_started,
                "Tables, previews, cached raw/clean plots, and alignment audio.",
            )
        )
        elapsed = time.perf_counter() - started
        profile_rows.append(
            {"stage": "Evaluation total", "elapsed_s": round(elapsed, 2), "notes": f"Verdict: {verdict}."}
        )

        status = f"Evaluation complete in {elapsed:.2f}s. Verdict: {verdict}."
        warnings = "\n".join(score.warnings)
        if warnings:
            status = f"{status}\n\nWarnings:\n{warnings}"
        if progress is not None:
            progress(1.0, desc="Done")
        workflow = "Full-song evaluation complete. Next: Step 6 explanation, or optional Step 5 song/section matching."
        return (
            status,
            metrics,
            pitch_plot,
            coverage_plot,
            stability_plot,
            stability_table,
            stability_debug,
            _profile_table(profile_rows),
            *diagnostics,
            details,
            workflow,
        )
    except Exception as exc:
        status = f"Evaluation failed after {time.perf_counter() - started:.2f}s: {exc}"
        return _empty_evaluation_response(status)


def run_explanation_generator(
    evaluation_details: dict[str, Any] | None,
    timing_debug: dict[str, Any] | None,
    stability_debug: dict[str, Any] | None,
) -> tuple[Any, ...]:
    payload = _build_explanation_payload(evaluation_details, timing_debug, stability_debug)
    if not payload.get("available"):
        return _empty_explanation_response(str(payload.get("reason") or "Run Step 4 first."))
    status = "Step 6 explanation generated from the latest full-song evaluation."
    return (
        payload,
        status,
        _explanation_markdown(payload),
        _explanation_table(payload),
        payload,
    )


def run_problem_moment_detection(
    reference_mode: str,
    reference_audio: str | None,
    current_take: str | None,
    previous_take: str | None,
    prepared_state: dict[str, Any] | None,
    contour_state: dict[str, Any] | None,
    evaluation_details: dict[str, Any] | None,
    timing_debug: dict[str, Any] | None,
    explanation_state: dict[str, Any] | None,
    progress: gr.Progress | None = None,
) -> tuple[Any, ...]:
    explanation = explanation_state if isinstance(explanation_state, dict) else {}
    if not explanation.get("available"):
        explanation = _build_explanation_payload(evaluation_details, timing_debug, None)
    if not explanation.get("available"):
        message = str(explanation.get("reason") or "Run Step 4 full-song evaluation first.")
        return _empty_problem_moments_response(message, explanation)
    if reference_mode != "Uploaded reference audio":
        message = "Problem moments need Uploaded reference audio mode for 1:1 residual analysis."
        return _empty_problem_moments_response(message, explanation)

    reference_path, current_path, previous_path = _analysis_paths(
        reference_audio,
        current_take,
        previous_take,
        prepared_state,
    )
    if not reference_path or not current_path:
        message = "Problem moments need both reference and current analysis audio."
        return _empty_problem_moments_response(message, explanation)

    started = time.perf_counter()
    try:
        if _contour_state_matches(
            contour_state,
            reference_mode=reference_mode,
            reference_path=reference_path,
            current_path=current_path,
            previous_path=previous_path,
        ):
            active_contour_state = contour_state or _empty_contour_state()
        else:
            if progress is not None:
                progress(0.15, desc="Loading cached contours")
            active_contour_state, _, _ = _extract_evaluation_contours(
                reference_mode=reference_mode,
                reference_path=reference_path,
                current_path=current_path,
                previous_path=previous_path,
                progress=None,
            )
        reference_contour = active_contour_state.get("reference_contour")
        current_contour = active_contour_state.get("current_contour")
        if reference_contour is None or current_contour is None:
            message = "Problem moments need both reference and current contours."
            return _empty_problem_moments_response(message, explanation)

        if progress is not None:
            progress(0.55, desc="Finding residual spikes")
        score = _current_score_dict(evaluation_details)
        _, moments = _problem_moments_from_contours(
            reference_contour,
            current_contour,
            score=score,
            timing_debug=timing_debug if isinstance(timing_debug, dict) else None,
            max_moments=2,
        )
        if progress is not None:
            progress(0.75, desc="Writing coach clips")
        for index, moment in enumerate(moments, start=1):
            moment["ab_audio"] = _write_problem_moment_ab_audio(
                reference_path=reference_path,
                current_path=current_path,
                moment=moment,
                index=index,
            )

        moments_table = pd.DataFrame(moments)
        coach_markdown = _coach_feedback_markdown(explanation, moments)
        payload = {
            "available": True,
            "elapsed_s": round(time.perf_counter() - started, 2),
            "explanation": explanation,
            "moments": moments,
            "moment_count": len(moments),
        }
        status = f"Step 7 detected {len(moments)} problem moment(s) in {payload['elapsed_s']:.2f}s."
        if progress is not None:
            progress(1.0, desc="Coach feedback ready")
        workflow = "Coach feedback ready. Listen to the timestamped A/B clips and inspect the payload."
        return (
            status,
            moments_table,
            coach_markdown,
            moments[0].get("ab_audio") if len(moments) >= 1 else None,
            moments[1].get("ab_audio") if len(moments) >= 2 else None,
            {"moments": moments},
            _json_safe(payload),
            workflow,
        )
    except Exception as exc:
        message = f"Problem moment detection failed after {time.perf_counter() - started:.2f}s: {exc}"
        return _empty_problem_moments_response(message, explanation)


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
            "Step 5 matching complete. Matched-section handoff score is shown here for now.",
        )
    except Exception as exc:
        message = f"Matching failed after {time.perf_counter() - started:.2f}s: {exc}"
        return _empty_matching_response(message)


def _extract_clean_contour(path: str, name: str):
    _, cleaned, _ = _load_or_extract_pitch_contours(path, name=name)
    return cleaned


def _save_plot(fig: Any, filename: str) -> str:
    path = Path(filename)
    if not path.is_absolute():
        path = OUTPUT_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass
    return str(path)


def _save_single_contour_plot(contour: PitchContour | None, title: str, filename: str) -> str | None:
    if contour is None:
        return None
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.2))
    _plot_pitch_contour(ax, contour, contour.name or "contour", "#2563eb", linewidth=1.5, alpha=0.95)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch (MIDI note)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return _save_plot(fig, filename)


def run_pitch_extractor_lab_ui(
    source_slot: str,
    reference_audio: str | None,
    current_take: str | None,
    previous_take: str | None,
    prepared_state: dict[str, Any] | None,
    methods: list[str] | None,
    excerpt_start_s: float,
    excerpt_duration_s: float,
    min_confidence: float,
    max_jump_cents: float,
) -> tuple[Any, ...]:
    source_path, _ = _select_dereverb_source(
        source_slot,
        reference_audio,
        current_take,
        previous_take,
        prepared_state,
    )
    status, table, excerpt_path, contours, notes = lab_run_pitch_extractors(
        source_path,
        OUTPUT_DIR,
        methods=methods or ["pYIN"],
        start_s=excerpt_start_s,
        duration_s=excerpt_duration_s,
        min_confidence=min_confidence,
        max_jump_cents=max_jump_cents,
    )
    pyin_plot = _save_single_contour_plot(
        contours.get("pYIN"),
        "pYIN cleaned pitch contour",
        "lab_pitch_pyin.png",
    )
    return status, table, excerpt_path, pyin_plot, notes


def run_reference_builder_lab_ui(
    reference_audio: str | None,
    window_s: float,
) -> tuple[Any, ...]:
    status, table, baseline, contour, details = lab_run_reference_builder(
        reference_audio,
        window_s=window_s,
    )
    plot_path = None
    if baseline is not None:
        plot_path = _save_plot(
            plot_reference_extraction(baseline, contour),
            "lab_reference_builder.png",
        )
    return status, table, plot_path, details


def run_song_identification_lab_ui(
    reference_audio: str | None,
    current_take: str | None,
    prepared_state: dict[str, Any] | None,
    catalog_source: str,
    top_k: int,
    window_s: float,
    hop_s: float,
) -> tuple[Any, ...]:
    prepared_state = prepared_state or {}
    query_path = prepared_state.get("current_analysis") or current_take
    reference_path = prepared_state.get("reference_analysis") or reference_audio
    return lab_run_song_identification(
        reference_path,
        query_path,
        catalog_source=catalog_source,
        top_k=int(top_k),
        window_s=float(window_s),
        hop_s=float(hop_s),
    )


def run_section_matching_lab_ui(
    reference_audio: str | None,
    current_take: str | None,
    prepared_state: dict[str, Any] | None,
    catalog_source: str,
    top_k: int,
    window_s: float,
    hop_s: float,
) -> tuple[Any, ...]:
    status, table, details = run_song_identification_lab_ui(
        reference_audio,
        current_take,
        prepared_state,
        catalog_source,
        top_k,
        window_s,
        hop_s,
    )
    candidates = details.get("candidates", []) if isinstance(details, dict) else []
    best = candidates[0] if candidates else {}
    prepared_state = prepared_state or {}
    query_path = prepared_state.get("current_analysis") or current_take
    reference_path = prepared_state.get("reference_analysis") or reference_audio
    reference_snippet = (
        _crop_audio_file(
            reference_path,
            float(best.get("reference_start_s", 0.0)),
            float(best.get("reference_end_s", 0.0)),
            "lab_section_reference_best.wav",
        )
        if reference_path and best
        else None
    )
    query_snippet = (
        _crop_audio_file(
            query_path,
            float(best.get("query_start_s", 0.0)),
            float(best.get("query_end_s", 0.0)),
            "lab_section_query_best.wav",
        )
        if query_path and best
        else None
    )
    return status, table, reference_snippet, query_snippet, details


def run_timing_lab_ui(
    reference_audio: str | None,
    current_take: str | None,
    prepared_state: dict[str, Any] | None,
    timing_penalty: float,
) -> tuple[Any, ...]:
    prepared_state = prepared_state or {}
    reference_path = prepared_state.get("reference_analysis") or reference_audio
    current_path = prepared_state.get("current_analysis") or current_take
    return lab_run_timing(
        reference_path,
        current_path,
        timing_penalty=float(timing_penalty),
    )


def run_scoring_calibration_lab_ui(
    scenario: str,
    pitch_weight: float,
    stability_weight: float,
    coverage_weight: float,
    timing_weight: float,
) -> tuple[Any, ...]:
    return lab_run_scoring_calibration(
        scenario,
        pitch_weight=float(pitch_weight),
        stability_weight=float(stability_weight),
        coverage_weight=float(coverage_weight),
        timing_weight=float(timing_weight),
    )


def run_stress_test_lab_ui(scenarios: list[str] | None) -> tuple[Any, ...]:
    return lab_run_stress_test(scenarios=scenarios or [])


def run_matched_progress_lab_ui(
    reference_audio: str | None,
    previous_take: str | None,
    current_take: str | None,
    prepared_state: dict[str, Any] | None,
    use_prepared_audio: bool,
    catalog_source: str,
    window_s: float,
    hop_s: float,
    min_match_score: float,
    min_match_margin: float,
    min_coverage_score: float,
) -> tuple[Any, ...]:
    prepared_state = prepared_state or {}
    if use_prepared_audio:
        reference_path = prepared_state.get("reference_analysis") or reference_audio
        previous_path = prepared_state.get("previous_analysis") or previous_take
        current_path = prepared_state.get("current_analysis") or current_take
    else:
        reference_path = reference_audio
        previous_path = previous_take
        current_path = current_take
    result = lab_run_matched_progress(
        reference_path,
        previous_path,
        current_path,
        OUTPUT_DIR,
        catalog_source=catalog_source,
        window_s=float(window_s),
        hop_s=float(hop_s),
        min_match_score=float(min_match_score),
        min_match_margin=float(min_match_margin),
        min_coverage_score=float(min_coverage_score),
    )
    (
        status,
        summary,
        section,
        metrics,
        gates,
        pitch_plot,
        coverage_plot,
        reference_clip,
        previous_clip,
        current_clip,
        details,
    ) = result
    copy_report = _matched_progress_copy_report(
        status,
        summary,
        section,
        metrics,
        gates,
        reference_clip=reference_clip,
        previous_clip=previous_clip,
        current_clip=current_clip,
        details=details,
    )
    return (
        status,
        summary,
        section,
        metrics,
        gates,
        pitch_plot,
        coverage_plot,
        reference_clip,
        previous_clip,
        current_clip,
        details,
        copy_report,
    )


def _matched_progress_copy_report(
    status: str,
    summary: pd.DataFrame,
    section: pd.DataFrame,
    metrics: pd.DataFrame,
    gates: pd.DataFrame,
    *,
    reference_clip: str | None,
    previous_clip: str | None,
    current_clip: str | None,
    details: dict[str, Any],
) -> str:
    lines = [
        "# Matched Progress Scoring Report",
        "",
        "## Status",
        str(status or ""),
        "",
        "## Interpretation Context",
        "Rows named previous/current follow the UI inputs. If you set previous=cover and current=original/reference, deltas mean original/reference minus cover.",
        "",
        "## Progress Summary",
        _dataframe_csv_text(summary),
        "",
        "## Selected Reference Section",
        _dataframe_csv_text(section),
        "",
        "## Detailed Metrics",
        _dataframe_csv_text(metrics),
        "",
        "## Confidence Gates",
        _dataframe_csv_text(gates),
        "",
        "## Clips",
        f"reference_clip,{reference_clip or ''}",
        f"previous_clip,{previous_clip or ''}",
        f"current_clip,{current_clip or ''}",
        "",
        "## Verdict Details",
        json.dumps(details or {}, indent=2, sort_keys=True),
    ]
    return "\n".join(lines)


def _dataframe_csv_text(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return "(empty)"
    return frame.to_csv(index=False).strip()


def run_fingerprinting_lab_ui(
    audio_preview_state: dict[str, Any] | None,
    expected_title: str,
    expected_artist: str,
    mode: str,
    window_s: float,
    hop_s: float,
    max_windows: int,
    start_offset_s: float,
    window_strategy: str,
) -> tuple[Any, ...]:
    original_audio_path, audio_path, preprocessing_notes = _fingerprinting_paths_from_preview_state(audio_preview_state)

    result = run_shazam_fingerprinting(
        audio_path,
        OUTPUT_DIR,
        expected_title=expected_title or "",
        expected_artist=expected_artist or "",
        mode=mode,
        window_s=float(window_s),
        hop_s=float(hop_s),
        max_windows=int(max_windows),
        start_offset_s=float(start_offset_s),
        window_strategy=window_strategy,
    )
    return _fingerprinting_lab_outputs(
        result,
        original_audio_path=original_audio_path,
        recognition_audio_path=audio_path,
        preprocessing_notes=preprocessing_notes,
    )


def run_audd_fingerprinting_lab_ui(
    audio_preview_state: dict[str, Any] | None,
    expected_title: str,
    expected_artist: str,
    mode: str,
    window_s: float,
    hop_s: float,
    max_windows: int,
    start_offset_s: float,
    window_strategy: str,
) -> tuple[Any, ...]:
    original_audio_path, audio_path, preprocessing_notes = _fingerprinting_paths_from_preview_state(audio_preview_state)

    result = run_audd_fingerprinting(
        audio_path,
        OUTPUT_DIR,
        expected_title=expected_title or "",
        expected_artist=expected_artist or "",
        mode=mode,
        window_s=float(window_s),
        hop_s=float(hop_s),
        max_windows=int(max_windows),
        start_offset_s=float(start_offset_s),
        window_strategy=window_strategy,
    )
    return _fingerprinting_lab_outputs(
        result,
        original_audio_path=original_audio_path,
        recognition_audio_path=audio_path,
        preprocessing_notes=preprocessing_notes,
    )


def run_acrcloud_fingerprinting_lab_ui(
    audio_preview_state: dict[str, Any] | None,
    expected_title: str,
    expected_artist: str,
    mode: str,
    window_s: float,
    hop_s: float,
    max_windows: int,
    start_offset_s: float,
    window_strategy: str,
) -> tuple[Any, ...]:
    original_audio_path, audio_path, preprocessing_notes = _fingerprinting_paths_from_preview_state(audio_preview_state)

    result = run_acrcloud_fingerprinting(
        audio_path,
        OUTPUT_DIR,
        expected_title=expected_title or "",
        expected_artist=expected_artist or "",
        mode=mode,
        window_s=float(window_s),
        hop_s=float(hop_s),
        max_windows=int(max_windows),
        start_offset_s=float(start_offset_s),
        window_strategy=window_strategy,
    )
    return _fingerprinting_lab_outputs(
        result,
        original_audio_path=original_audio_path,
        recognition_audio_path=audio_path,
        preprocessing_notes=preprocessing_notes,
    )


def run_long_session_segmentation_ui(
    source_choice: str,
    uploaded_recording: str | None,
    current_take: str | None,
    prepared_state: dict[str, Any] | None,
    use_demucs_instrumental: bool,
    demucs_model: str,
    demucs_device: str,
    provider: str,
    window_s: float,
    hop_s: float,
    max_windows: int,
    rms_frame_s: float,
    rms_hop_s: float,
    tempo_window_s: float,
    tempo_hop_s: float,
) -> tuple[Any, ...]:
    original_audio_path = _fingerprinting_source_path(
        source_choice,
        uploaded_recording,
        current_take,
        prepared_state,
    )
    recognition_audio_path, preprocessing_notes = _prepare_fingerprinting_audio(
        original_audio_path,
        use_demucs_instrumental=bool(use_demucs_instrumental),
        demucs_model=demucs_model,
        demucs_device=demucs_device,
    )
    provider_key = {
        "No recognition (plots only)": "none",
        "ShazamKit": "shazamkit",
        "AudD": "audd",
        "ACRCloud": "acrcloud",
    }.get(provider, str(provider).casefold())
    status, intervals, windows, details = lab_run_long_session_segmentation(
        recognition_audio_path,
        OUTPUT_DIR,
        provider=provider_key,
        window_s=float(window_s),
        hop_s=float(hop_s),
        max_windows=int(max_windows),
        rms_frame_s=float(rms_frame_s),
        rms_hop_s=float(rms_hop_s),
        tempo_window_s=float(tempo_window_s),
        tempo_hop_s=float(tempo_hop_s),
    )
    if preprocessing_notes:
        status = f"{status}\n\nPreprocessing:\n" + "\n".join(preprocessing_notes[:4])
    details = dict(details or {})
    details["original_audio_path"] = original_audio_path
    details["recognition_audio_path"] = recognition_audio_path
    details["preprocessing_notes"] = list(preprocessing_notes)
    clip_paths = [
        str(clip.get("clip_path"))
        for clip in details.get("interval_clips", [])
        if clip.get("clip_path")
    ]
    weak_candidates = pd.DataFrame(details.get("weak_candidates", []))
    return (
        status,
        intervals,
        weak_candidates,
        windows,
        details.get("timeline_plot_path"),
        clip_paths,
        details.get("best_interval_clip_path"),
        details,
    )


def run_fingerprint_diagnostics_ui(
    provider: str,
    csv_upload: str | None,
    csv_text: str,
    original_recording: str | None,
    recording_duration_s: float,
    request_budget: int,
) -> tuple[Any, ...]:
    provider_key = {
        "ShazamKit": "shazamkit",
        "AudD": "audd",
        "ACRCloud": "acrcloud",
    }.get(provider, str(provider).casefold())
    csv_source = csv_upload or csv_text
    duration = float(recording_duration_s) if recording_duration_s and recording_duration_s > 0 else None
    status, summary, weak, sweeps, recommendations, preview_files, first_preview, details = (
        lab_run_fingerprint_diagnostics(
            csv_source,
            OUTPUT_DIR,
            provider=provider_key,
            recording_duration_s=duration,
            original_audio_path=original_recording,
            request_budget=int(request_budget),
        )
    )
    return (
        status,
        summary,
        weak,
        sweeps,
        recommendations,
        preview_files,
        first_preview,
        details,
    )


def prepare_fingerprinting_audio_preview_ui(
    source_choice: str,
    uploaded_recording: str | None,
    current_take: str | None,
    prepared_state: dict[str, Any] | None,
    use_demucs_instrumental: bool,
    demucs_model: str,
    demucs_device: str,
    mode: str,
    window_s: float,
    hop_s: float,
    max_windows: int,
    start_offset_s: float,
    window_strategy: str,
) -> tuple[Any, ...]:
    original_audio_path = _fingerprinting_source_path(
        source_choice,
        uploaded_recording,
        current_take,
        prepared_state,
    )
    recognition_audio_path, preprocessing_notes = _prepare_fingerprinting_audio(
        original_audio_path,
        use_demucs_instrumental=bool(use_demucs_instrumental),
        demucs_model=demucs_model,
        demucs_device=demucs_device,
    )
    if not original_audio_path:
        status = "No recording selected."
    elif preprocessing_notes:
        status = "Preview audio prepared.\n\nPreprocessing:\n" + "\n".join(preprocessing_notes[:4])
    else:
        status = "Preview audio prepared using the original selected audio."
    state = {
        "original_audio_path": original_audio_path,
        "recognition_audio_path": recognition_audio_path,
        "preprocessing_notes": list(preprocessing_notes),
    }
    request_files, original_windows, recognition_windows, request_notes = _fingerprinting_request_preview_outputs(
        original_audio_path,
        recognition_audio_path,
        mode=mode,
        window_s=float(window_s),
        hop_s=float(hop_s),
        max_windows=int(max_windows),
        start_offset_s=float(start_offset_s),
        window_strategy=window_strategy,
    )
    if request_notes:
        status = f"{status}\n\nRequest-window preview:\n" + "\n".join(request_notes[:4])
    return (
        status,
        original_audio_path,
        recognition_audio_path,
        request_files,
        state,
        *_fingerprinting_request_window_updates(original_windows, recognition_windows),
    )


def _fingerprinting_request_preview_outputs(
    original_audio_path: str | None,
    recognition_audio_path: str | None,
    *,
    mode: str,
    window_s: float,
    hop_s: float,
    max_windows: int,
    start_offset_s: float,
    window_strategy: str,
) -> tuple[list[str], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], tuple[str, ...]]:
    if not original_audio_path:
        return [], (), (), ()

    recognition_audio_path = recognition_audio_path or original_audio_path
    notes: list[str] = []
    try:
        original_windows = prepare_fingerprint_windows(
            original_audio_path,
            OUTPUT_DIR,
            mode=mode,
            window_s=window_s,
            hop_s=hop_s,
            max_windows=max_windows,
            start_offset_s=start_offset_s,
            window_strategy=window_strategy,
            namespace="preflight_original",
        )
        if Path(recognition_audio_path) == Path(original_audio_path):
            recognition_windows = original_windows
        else:
            recognition_windows = prepare_fingerprint_windows(
                recognition_audio_path,
                OUTPUT_DIR,
                mode=mode,
                window_s=window_s,
                hop_s=hop_s,
                max_windows=max_windows,
                start_offset_s=start_offset_s,
                window_strategy=window_strategy,
                namespace="preflight_recognition",
            )
    except Exception as exc:
        return [], (), (), (str(exc),)

    if not original_windows or not recognition_windows:
        return [], (), (), (
            "No request windows were generated.",
        )

    pair_count = min(len(original_windows), len(recognition_windows))
    if len(original_windows) != len(recognition_windows):
        notes.append(
            f"Original and recognition sources produced different window counts "
            f"({len(original_windows)} vs {len(recognition_windows)}); showing {pair_count} pair(s)."
        )

    files: list[str] = []
    for original_window, recognition_window in zip(original_windows, recognition_windows):
        for path in (original_window["audio_path"], recognition_window["audio_path"]):
            if path not in files:
                files.append(path)

    return files, original_windows, recognition_windows, tuple(notes)


def _fingerprinting_request_window_updates(
    original_windows: tuple[dict[str, Any], ...],
    recognition_windows: tuple[dict[str, Any], ...],
) -> list[Any]:
    updates: list[Any] = []
    window_pairs = list(zip(original_windows, recognition_windows))
    for index in range(MAX_REQUEST_WINDOW_PREVIEWS):
        if index >= len(window_pairs):
            updates.extend(
                [
                    gr.update(visible=False, open=False),
                    "",
                    None,
                    None,
                ]
            )
            continue

        original_window, recognition_window = window_pairs[index]
        mode_label = "window" if original_window.get("mode") == "window" else "whole"
        start_s = float(original_window.get("start_s") or 0.0)
        end_s = float(original_window.get("end_s") or 0.0)
        heading = f"Window {index + 1}: {mode_label} {start_s:.1f}s-{end_s:.1f}s"
        updates.extend(
            [
                gr.update(label=heading, visible=True, open=index == 0),
                f"**{heading}**",
                str(original_window.get("audio_path") or ""),
                str(recognition_window.get("audio_path") or ""),
            ]
        )
    return updates


def _fingerprinting_paths_from_preview_state(
    audio_preview_state: dict[str, Any] | None,
) -> tuple[str | None, str | None, tuple[str, ...]]:
    audio_preview_state = audio_preview_state or {}
    original_audio_path = audio_preview_state.get("original_audio_path")
    recognition_audio_path = audio_preview_state.get("recognition_audio_path") or original_audio_path
    preprocessing_notes = tuple(audio_preview_state.get("preprocessing_notes") or ())
    if not original_audio_path and not recognition_audio_path:
        return None, None, preprocessing_notes
    return original_audio_path, recognition_audio_path, preprocessing_notes


def _prepare_fingerprinting_audio(
    audio_path: str | None,
    *,
    use_demucs_instrumental: bool,
    demucs_model: str,
    demucs_device: str,
) -> tuple[str | None, tuple[str, ...]]:
    if not audio_path or not use_demucs_instrumental:
        return audio_path, ()
    result = prepare_vocal_analysis_audio(
        audio_path,
        cache_dir=STEM_CACHE_DIR,
        backend="demucs",
        stem="no_vocals",
        model=demucs_model,
        device=demucs_device,
        shifts=1,
        overlap=0.25,
    )
    if result.used_original:
        notes = ("Demucs accompaniment requested but fallback used original audio.",)
    elif result.used_cache:
        notes = (f"Using cached Demucs accompaniment stem: {Path(result.analysis_path).name}",)
    else:
        notes = (f"Generated Demucs accompaniment stem: {Path(result.analysis_path).name}",)
    return str(result.analysis_path), notes + tuple(result.warnings)


def _fingerprinting_source_path(
    source_choice: str,
    uploaded_recording: str | None,
    current_take: str | None,
    prepared_state: dict[str, Any] | None,
) -> str | None:
    prepared_state = prepared_state or {}
    if source_choice == "Use prepared current audio":
        return prepared_state.get("current_analysis") or current_take
    if source_choice == "Upload separate recording":
        return uploaded_recording
    return current_take


def _fingerprinting_lab_outputs(
    result: Any,
    *,
    original_audio_path: str | None = None,
    recognition_audio_path: str | None = None,
    preprocessing_notes: tuple[str, ...] = (),
) -> tuple[Any, ...]:
    summary = pd.DataFrame([result.summary]) if result.summary else pd.DataFrame()
    rows = pd.DataFrame(result.rows)
    status = result.status
    if preprocessing_notes:
        status = f"{status}\n\nPreprocessing:\n" + "\n".join(preprocessing_notes[:4])
    if result.warnings:
        status = f"{status}\n\nWarnings:\n" + "\n".join(result.warnings[:3])
    preview_path, preview_label = _fingerprinting_preview_audio(result.rows)
    preview_choices = _fingerprinting_preview_choices(result.rows)
    preview_files = _fingerprinting_preview_paths(result.rows)
    preview_gallery = _fingerprinting_preview_gallery_html(result.rows)
    return (
        status,
        summary,
        rows,
        original_audio_path,
        recognition_audio_path,
        preview_files,
        preview_gallery,
        gr.update(choices=preview_choices, value=preview_path),
        preview_path,
        preview_label,
        _fingerprinting_interpretation_markdown(result.interpretations),
        result.to_dict(),
    )


def _fingerprinting_preview_audio(rows: tuple[dict[str, Any], ...]) -> tuple[str | None, str]:
    if not rows:
        return None, "Run fingerprinting to generate a processed-window preview."
    preview_row = next((row for row in rows if row.get("mode") == "window"), rows[0])
    audio_path = str(preview_row.get("audio_path") or "")
    if not audio_path:
        return None, "No preview audio was generated."
    return audio_path, _fingerprinting_preview_label(preview_row)


def _fingerprinting_preview_choices(rows: tuple[dict[str, Any], ...]) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for row in rows:
        audio_path = str(row.get("audio_path") or "")
        if not audio_path:
            continue
        choices.append((_fingerprinting_preview_label(row), audio_path))
    return choices


def _fingerprinting_preview_paths(rows: tuple[dict[str, Any], ...]) -> list[str]:
    paths: list[str] = []
    for row in rows:
        audio_path = str(row.get("audio_path") or "")
        if audio_path:
            paths.append(audio_path)
    return paths


def _fingerprinting_preview_label(row: dict[str, Any]) -> str:
    label_mode = "processed sliding window" if row.get("mode") == "window" else "processed whole clip"
    start_s = float(row.get("window_start_s") or 0.0)
    end_s = float(row.get("window_end_s") or 0.0)
    return f"Previewing {label_mode}: {start_s:.1f}s to {end_s:.1f}s"


def _fingerprinting_preview_gallery_html(rows: tuple[dict[str, Any], ...]) -> str:
    clip_rows = [row for row in rows if row.get("audio_path")]
    if not clip_rows:
        return "<p>Run fingerprinting to generate processed clip previews.</p>"

    items: list[str] = []
    for index, row in enumerate(clip_rows, start=1):
        audio_path = str(row.get("audio_path") or "")
        audio_url = "/file=" + quote(audio_path, safe="/:")
        mode_label = "window" if row.get("mode") == "window" else "whole"
        start_s = float(row.get("window_start_s") or 0.0)
        end_s = float(row.get("window_end_s") or 0.0)
        status = str(row.get("status") or "unknown")
        title = str(row.get("matched_title") or row.get("title") or "")
        match_text = f" - {title}" if title else ""
        label = f"{index}. {mode_label} {start_s:.1f}s-{end_s:.1f}s - {status}{match_text}"
        items.append(
            "<li style='margin: 0 0 12px 0;'>"
            f"<div style='font-size: 0.9rem; margin-bottom: 4px;'>{_html_escape(label)}</div>"
            f"<audio controls preload='none' src='{audio_url}' style='width: 100%;'></audio>"
            "</li>"
        )
    return (
        "<div>"
        "<div style='font-weight: 600; margin-bottom: 8px;'>Generated processed clips</div>"
        "<ol style='padding-left: 20px; margin: 0;'>"
        + "".join(items)
        + "</ol></div>"
    )


def _html_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def select_fingerprinting_preview(preview_path: str | None, details: dict[str, Any] | None) -> tuple[str | None, str]:
    if not preview_path:
        return None, "Choose a generated clip to preview."
    rows = (details or {}).get("rows") or []
    for row in rows:
        if str(row.get("audio_path") or "") == str(preview_path):
            return preview_path, _fingerprinting_preview_label(row)
    return preview_path, f"Previewing generated clip: {Path(preview_path).name}"


def _fingerprinting_interpretation_markdown(interpretations: dict[str, str]) -> str:
    if not interpretations:
        return "Run fingerprinting to generate interpretation."
    labels = {
        "does_full_file_recognition_work": "Does full-file recognition work?",
        "do_short_windows_work_better": "Do short windows work better?",
        "how_much_audio_is_needed": "How much audio is needed?",
        "does_echo_or_noise_cause_failure": "Does echo/noise cause failure?",
        "is_backing_track_bleed_enough": "Is backing-track bleed enough?",
    }
    lines = ["### Interpretation"]
    for key, label in labels.items():
        if key in interpretations:
            lines.append(f"- **{label}** {interpretations[key]}")
    return "\n".join(lines)


def _request_window_preview_components() -> list[Any]:
    outputs: list[Any] = []
    gr.Markdown("### Generated request-window pairs before API call")
    for index in range(MAX_REQUEST_WINDOW_PREVIEWS):
        with gr.Accordion(f"Window {index + 1}", open=False, visible=False) as window_card:
            window_label = gr.Markdown()
            with gr.Row():
                original_window_audio = gr.Audio(
                    label="Original window",
                    type="filepath",
                    interactive=False,
                )
                recognition_window_audio = gr.Audio(
                    label="Recognition-source window",
                    type="filepath",
                    interactive=False,
                )
        outputs.extend(
            [
                window_card,
                window_label,
                original_window_audio,
                recognition_window_audio,
            ]
        )
    return outputs


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Konopro Gradio Research Demo") as demo:
        gr.Markdown(
            """
# Konopro Gradio Research Demo

This is the same research backend as Streamlit, but with explicit stages and
playback checkpoints for local research testing.
"""
        )

        with gr.Tabs():
            with gr.Tab("Scoring Model"):
                prepared_state = gr.State({})
                contour_state = gr.State(_empty_contour_state())
                explanation_state = gr.State({})
        
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
                    use_demucs = gr.Checkbox(label="Use Demucs vocal stem", value=True)
                    gr.Markdown("We currently support the **vocals** stem only. No other stems are used for scoring.")
                    gr.Radio(
                        ["vocals"],
                        value="vocals",
                        label="Stem used for analysis",
                        interactive=False,
                    )
                    apply_to_takes = gr.Checkbox(label="Apply Demucs to current/previous takes too", value=True)
                    with gr.Accordion("Loudness normalization", open=True):
                        use_active_rms = gr.Checkbox(
                            label="Normalize active RMS after Demucs",
                            value=True,
                        )
                        with gr.Row():
                            target_active_rms = gr.Slider(
                                0.03,
                                0.14,
                                value=0.08,
                                step=0.01,
                                label="Target active RMS",
                            )
                            active_rms_percentile = gr.Slider(
                                20,
                                90,
                                value=60,
                                step=5,
                                label="Active-frame percentile",
                            )
                    with gr.Row():
                        model = gr.Dropdown(
                            ["htdemucs", "htdemucs_ft", "mdx_extra", "mdx_q"],
                            value="htdemucs",
                            label="Demucs model",
                        )
                        device = gr.Dropdown(["cpu", "mps", "cuda"], value=DEFAULT_DEMUCS_DEVICE, label="Device")
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
                            use_active_rms,
                            target_active_rms,
                            active_rms_percentile,
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
        
                with gr.Accordion("Step 3 - Pitch & Alignment Debug", open=False):
                    gr.Markdown(
                        "Use this step to inspect pitch extraction and global-offset alignment before running the full-song score."
                    )
                    reference_mode = gr.Radio(
                        ["Demo symbolic baseline", "Uploaded reference audio"],
                        value="Demo symbolic baseline",
                        label="Reference mode",
                    )
                    with gr.Row():
                        contour_button = gr.Button("Extract Pitch Contours")
                        alignment_only_button = gr.Button("Run Audible Alignment Check Only")
                    contour_status = gr.Markdown()
                    alignment_only_status = gr.Markdown()
                    with gr.Tabs():
                        with gr.Tab("Debug artifacts"):
                            evaluation_profile_table = gr.Dataframe(label="Latest action timing / profile", wrap=True)
                            evaluation_steps = gr.Dataframe(label="Eight internal steps", wrap=True)
                            with gr.Accordion("1. Prepared analysis audio", open=True):
                                gr.Markdown(
                                    "**?** This shows the exact audio Step 4 full-song evaluation scores. If Step 2 used Demucs, these are the prepared/stem files, not necessarily the original uploads."
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
                                    "**?** Step 4 compares the full reference against the full current take. This can punish late starts or extra silence more than matched-section scoring."
                                )
                                eval_alignment_plot = gr.Image(
                                    label="Full-track DTW alignment",
                                    type="filepath",
                                )
                                eval_normalization_plot = gr.Image(
                                    label="Full-track pitch normalization",
                                    type="filepath",
                                )
                                gr.Markdown(
                                    "**Audible alignment check.** These clips use the scorer's global timing offset only. They are not DTW time-warped, so a bad-sounding pair is a useful signal that the full-track scorer is comparing the wrong region."
                                )
                                with gr.Row():
                                    eval_aligned_reference_audio = gr.Audio(
                                        label="Offset-aligned reference excerpt",
                                        type="filepath",
                                        interactive=False,
                                    )
                                    eval_aligned_current_audio = gr.Audio(
                                        label="Offset-aligned current excerpt",
                                        type="filepath",
                                        interactive=False,
                                    )
                                with gr.Row():
                                    eval_alignment_ab_audio = gr.Audio(
                                        label="A/B: reference then current",
                                        type="filepath",
                                        interactive=False,
                                    )
                                    eval_alignment_mix_audio = gr.Audio(
                                        label="Overlay mix",
                                        type="filepath",
                                        interactive=False,
                                    )
                            with gr.Accordion("6-8. Full-song metrics and final score", open=True):
                                gr.Markdown(
                                    "**?** This table maps each score back to the signal that created it, so low timing/stability/pitch values can be debugged directly."
                                )
                                evaluation_timing_debug = gr.JSON(label="Timing debug")
                                evaluation_score_breakdown = gr.Dataframe(label="Score breakdown", wrap=True)
                    contour_button.click(
                        run_contour_extraction,
                        inputs=[
                            reference_mode,
                            reference_audio,
                            current_take,
                            previous_take,
                            prepared_state,
                        ],
                        outputs=[
                            contour_state,
                            contour_status,
                            evaluation_profile_table,
                            eval_reference_raw_plot,
                            eval_current_raw_plot,
                            eval_previous_raw_plot,
                            workflow_status,
                        ],
                    )
                    alignment_only_button.click(
                        run_alignment_check_only,
                        inputs=[
                            reference_mode,
                            reference_audio,
                            current_take,
                            previous_take,
                            prepared_state,
                            contour_state,
                        ],
                        outputs=[
                            contour_state,
                            alignment_only_status,
                            evaluation_profile_table,
                            eval_alignment_plot,
                            eval_normalization_plot,
                            eval_aligned_reference_audio,
                            eval_aligned_current_audio,
                            eval_alignment_ab_audio,
                            eval_alignment_mix_audio,
                            evaluation_timing_debug,
                            workflow_status,
                        ],
                    )

                with gr.Accordion("Step 4 - Full Song Evaluation", open=False):
                    full_eval_status = gr.Markdown()
                    with gr.Accordion("Scoring parameters", open=False):
                        full_eval_stability_penalty = gr.Slider(
                            0.10,
                            1.50,
                            value=CONTOUR_SCORE_KWARGS["stability_penalty"],
                            step=0.05,
                            label="Stability penalty",
                        )
                    evaluation_button = gr.Button("Run Full Song Evaluation", variant="primary")
                    with gr.Tabs():
                        with gr.Tab("Results"):
                            evaluation_metrics = gr.Dataframe(label="Full-song evaluation metrics")
                            with gr.Row():
                                pitch_plot = gr.Image(label="Pitch plot", type="filepath")
                                coverage_plot = gr.Image(label="Coverage plot", type="filepath")
                            evaluation_json = gr.JSON(label="Evaluation details")
                        with gr.Tab("Stability diagnostics"):
                            stability_residual_plot = gr.Image(
                                label="Residual pitch error after global offset",
                                type="filepath",
                            )
                            stability_calibration_table = gr.Dataframe(
                                label="Stability penalty calibration",
                                wrap=True,
                            )
                            stability_diagnostics_json = gr.JSON(label="Stability diagnostics")
                    evaluation_button.click(
                        run_evaluation,
                        inputs=[
                            reference_mode,
                            reference_audio,
                            current_take,
                            previous_take,
                            prepared_state,
                            contour_state,
                            full_eval_stability_penalty,
                        ],
                        outputs=[
                            full_eval_status,
                            evaluation_metrics,
                            pitch_plot,
                            coverage_plot,
                            stability_residual_plot,
                            stability_calibration_table,
                            stability_diagnostics_json,
                            evaluation_profile_table,
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
                            eval_aligned_reference_audio,
                            eval_aligned_current_audio,
                            eval_alignment_ab_audio,
                            eval_alignment_mix_audio,
                            evaluation_score_breakdown,
                            evaluation_timing_debug,
                            evaluation_json,
                            workflow_status,
                        ],
                    )

                with gr.Accordion("Step 5 - Match Song / Section", open=False):
                    step5_status = gr.Markdown()
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
                            step5_status,
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

                with gr.Accordion("Step 6 - Explanation Generator", open=False):
                    gr.Markdown(
                        "Turn the latest Step 4 score into a structured coaching explanation. This is deterministic and uses the score JSON, timing debug, and stability diagnostics."
                    )
                    explanation_button = gr.Button("Generate Explanation", variant="primary")
                    explanation_status = gr.Markdown()
                    coach_explanation = gr.Markdown()
                    explanation_table = gr.Dataframe(label="Explanation signals", wrap=True)
                    explanation_json = gr.JSON(label="Explanation payload")
                    explanation_button.click(
                        run_explanation_generator,
                        inputs=[
                            evaluation_json,
                            evaluation_timing_debug,
                            stability_diagnostics_json,
                        ],
                        outputs=[
                            explanation_state,
                            explanation_status,
                            coach_explanation,
                            explanation_table,
                            explanation_json,
                        ],
                    )

                with gr.Accordion("Step 7 - Problem Moments / Coach Feedback", open=False):
                    gr.Markdown(
                        "Find the top two timestamped problem moments from residual pitch errors, then combine them with Step 6 into coach feedback."
                    )
                    problem_moments_button = gr.Button("Detect Problem Moments", variant="primary")
                    problem_moments_status = gr.Markdown()
                    coach_feedback = gr.Markdown()
                    problem_moments_table = gr.Dataframe(label="Top problem moments", wrap=True)
                    with gr.Row():
                        problem_moment_audio_1 = gr.Audio(
                            label="Problem moment 1 A/B",
                            type="filepath",
                            interactive=False,
                        )
                        problem_moment_audio_2 = gr.Audio(
                            label="Problem moment 2 A/B",
                            type="filepath",
                            interactive=False,
                        )
                    with gr.Row():
                        problem_moments_json = gr.JSON(label="Problem moments payload")
                        coach_feedback_json = gr.JSON(label="Coach feedback payload")
                    problem_moments_button.click(
                        run_problem_moment_detection,
                        inputs=[
                            reference_mode,
                            reference_audio,
                            current_take,
                            previous_take,
                            prepared_state,
                            contour_state,
                            evaluation_json,
                            evaluation_timing_debug,
                            explanation_state,
                        ],
                        outputs=[
                            problem_moments_status,
                            problem_moments_table,
                            coach_feedback,
                            problem_moment_audio_1,
                            problem_moment_audio_2,
                            problem_moments_json,
                            coach_feedback_json,
                            workflow_status,
                        ],
                    )
            with gr.Tab("Preprocessing"):
                with gr.Tabs():
                    with gr.Tab("Dereverb Methods"):
                        gr.Markdown(
                            """
## Dereverb Methods

This tab runs cleanup experiments before pitch extraction. It does not change the scoring workflow yet; it helps identify whether a cleanup method improves pitch extraction enough to promote into Step 2.

Pipeline under test:

```text
selected source audio
-> optional cleanup / enhancement method
-> raw pitch extraction
-> cleaned pitch contour
-> reference-free quality comparison
```
        """
                        )
                        with gr.Row():
                            dereverb_source_slot = gr.Radio(
                                ["Current take", "Reference audio", "Previous take"],
                                value="Current take",
                                label="Experiment source",
                            )
                            dereverb_excerpt_start = gr.Number(
                                value=0.0,
                                label="Excerpt start (seconds)",
                                precision=2,
                            )
                            dereverb_excerpt_duration = gr.Slider(
                                5,
                                90,
                                value=30,
                                step=5,
                                label="Excerpt duration (seconds)",
                            )
                        dereverb_methods = gr.CheckboxGroup(
                            choices=list(DEREVERB_METHODS.keys()),
                            value=[
                                "Baseline - no cleanup",
                                "Built-in light spectral cleanup",
                                "Built-in strong spectral cleanup",
                                "Built-in cleanup + loudness normalization",
                            ],
                            label="Cleanup methods to compare",
                        )
                        with gr.Accordion("Hyperparameters", open=True):
                            gr.Markdown(
                                """
These parameters affect only the cleanup experiment. Use short excerpts first; full-song pYIN comparisons can be slow.

- **Light/strong gate percentile**: higher values suppress more low-energy spectral bins.
- **Tail attenuation**: lower values suppress quiet tails more aggressively.
- **Target RMS**: output loudness target for the normalization variant.
- **noisereduce strength**: higher values remove more estimated noise if `noisereduce` is installed.
- **DeepFilterNet attenuation limit**: `0` means no limit/aggressive; `3-12 dB` keeps more of the original signal.
        """
                            )
                            with gr.Row():
                                light_gate_percentile = gr.Slider(
                                    5,
                                    70,
                                    value=25,
                                    step=5,
                                    label="Light gate percentile",
                                )
                                strong_gate_percentile = gr.Slider(
                                    10,
                                    90,
                                    value=50,
                                    step=5,
                                    label="Strong gate percentile",
                                )
                                tail_attenuation = gr.Slider(
                                    0.02,
                                    0.60,
                                    value=0.20,
                                    step=0.02,
                                    label="Tail attenuation",
                                )
                            with gr.Row():
                                target_rms = gr.Slider(
                                    0.02,
                                    0.20,
                                    value=0.08,
                                    step=0.01,
                                    label="Target RMS",
                                )
                                noisereduce_strength = gr.Slider(
                                    0.20,
                                    1.00,
                                    value=0.70,
                                    step=0.05,
                                    label="noisereduce strength",
                                )
                                deepfilter_atten_lim_db = gr.Slider(
                                    0,
                                    30,
                                    value=6,
                                    step=1,
                                    label="DeepFilterNet attenuation limit (dB, 0 = off/aggressive)",
                                )
                        run_dereverb_button = gr.Button("Run Dereverb Experiment", variant="primary")
                        dereverb_status = gr.Markdown("Run an experiment to compare cleanup methods.")
                        dereverb_summary = gr.Dataframe(label="Cleanup summary", wrap=True)
                        dereverb_steps = gr.Dataframe(label="Intermediate steps", wrap=True)
                        dereverb_source_excerpt_audio = gr.Audio(
                            label="Source excerpt used for this experiment",
                            type="filepath",
                            interactive=False,
                        )
                        with gr.Tabs():
                            with gr.Tab("Baseline"):
                                with gr.Row():
                                    dereverb_baseline_audio = gr.Audio(
                                        label="Baseline audio",
                                        type="filepath",
                                        interactive=False,
                                    )
                                    dereverb_baseline_plot = gr.Image(
                                        label="Baseline raw vs cleaned pitch",
                                        type="filepath",
                                    )
                            with gr.Tab("Built-in Light"):
                                with gr.Row():
                                    dereverb_light_audio = gr.Audio(
                                        label="Built-in light cleanup audio",
                                        type="filepath",
                                        interactive=False,
                                    )
                                    dereverb_light_plot = gr.Image(
                                        label="Built-in light raw vs cleaned pitch",
                                        type="filepath",
                                    )
                            with gr.Tab("Built-in Strong"):
                                with gr.Row():
                                    dereverb_strong_audio = gr.Audio(
                                        label="Built-in strong cleanup audio",
                                        type="filepath",
                                        interactive=False,
                                    )
                                    dereverb_strong_plot = gr.Image(
                                        label="Built-in strong raw vs cleaned pitch",
                                        type="filepath",
                                    )
                            with gr.Tab("Normalize"):
                                with gr.Row():
                                    dereverb_normalized_audio = gr.Audio(
                                        label="Built-in cleanup + normalization audio",
                                        type="filepath",
                                        interactive=False,
                                    )
                                    dereverb_normalized_plot = gr.Image(
                                        label="Built-in cleanup + normalization pitch",
                                        type="filepath",
                                    )
                            with gr.Tab("noisereduce"):
                                with gr.Row():
                                    dereverb_noisereduce_audio = gr.Audio(
                                        label="noisereduce audio",
                                        type="filepath",
                                        interactive=False,
                                    )
                                    dereverb_noisereduce_plot = gr.Image(
                                        label="noisereduce raw vs cleaned pitch",
                                        type="filepath",
                                    )
                            with gr.Tab("DeepFilterNet"):
                                with gr.Row():
                                    dereverb_deepfilter_audio = gr.Audio(
                                        label="DeepFilterNet audio",
                                        type="filepath",
                                        interactive=False,
                                    )
                                    dereverb_deepfilter_plot = gr.Image(
                                        label="DeepFilterNet raw vs cleaned pitch",
                                        type="filepath",
                                    )
                            with gr.Tab("WPE"):
                                with gr.Row():
                                    dereverb_wpe_audio = gr.Audio(
                                        label="WPE audio",
                                        type="filepath",
                                        interactive=False,
                                    )
                                    dereverb_wpe_plot = gr.Image(
                                        label="WPE raw vs cleaned pitch",
                                        type="filepath",
                                    )
                            with gr.Tab("External Model"):
                                gr.Markdown(
                                    "Set `KONOPRO_ENHANCER_CMD` to a command template containing `{input}` and `{output}` before launching Gradio."
                                )
                                with gr.Row():
                                    dereverb_external_audio = gr.Audio(
                                        label="External enhancement audio",
                                        type="filepath",
                                        interactive=False,
                                    )
                                    dereverb_external_plot = gr.Image(
                                        label="External enhancement raw vs cleaned pitch",
                                        type="filepath",
                                    )
                        gr.Markdown(
                            """
### How to read this

        - `frame_jitter_cents` is a reference-free proxy: lower usually means smoother pitch extraction, but too low can mean over-smoothing.
        - `voiced_ratio_pct` should not collapse; a cleanup method that removes singing is not useful.
        - `pitch_delta_vs_baseline_cents` should stay modest; large values mean the method changed the detected melody.
        - Unavailable methods stay selectable so dependency/setup problems are visible instead of silent.
        """
                        )

                        run_dereverb_button.click(
                            run_dereverb_experiment,
                            inputs=[
                                dereverb_source_slot,
                                reference_audio,
                                current_take,
                                previous_take,
                                prepared_state,
                                dereverb_methods,
                                dereverb_excerpt_start,
                                dereverb_excerpt_duration,
                                light_gate_percentile,
                                strong_gate_percentile,
                                tail_attenuation,
                                target_rms,
                                noisereduce_strength,
                                deepfilter_atten_lim_db,
                            ],
                            outputs=[
                                dereverb_status,
                                dereverb_summary,
                                dereverb_steps,
                                dereverb_source_excerpt_audio,
                                dereverb_baseline_audio,
                                dereverb_baseline_plot,
                                dereverb_light_audio,
                                dereverb_light_plot,
                                dereverb_strong_audio,
                                dereverb_strong_plot,
                                dereverb_normalized_audio,
                                dereverb_normalized_plot,
                                dereverb_noisereduce_audio,
                                dereverb_noisereduce_plot,
                                dereverb_deepfilter_audio,
                                dereverb_deepfilter_plot,
                                dereverb_wpe_audio,
                                dereverb_wpe_plot,
                                dereverb_external_audio,
                                dereverb_external_plot,
                            ],
                        )
                    with gr.Tab("Audio Preparation Lab"):
                        gr.Markdown(
                            """
## Audio Preparation Lab

Compare original audio against prepared analysis audio. This tab reuses the same Step 2 backend and writes the selected output back into the shared prepared-audio state.
        """
                        )
                        with gr.Row():
                            lab_use_demucs = gr.Checkbox(label="Use Demucs vocal stem", value=True)
                            lab_apply_to_takes = gr.Checkbox(label="Apply Demucs to current/previous takes", value=True)
                        with gr.Accordion("Loudness normalization", open=True):
                            lab_use_active_rms = gr.Checkbox(
                                label="Normalize active RMS after Demucs",
                                value=True,
                            )
                            with gr.Row():
                                lab_target_active_rms = gr.Slider(
                                    0.03,
                                    0.14,
                                    value=0.08,
                                    step=0.01,
                                    label="Target active RMS",
                                )
                                lab_active_rms_percentile = gr.Slider(
                                    20,
                                    90,
                                    value=60,
                                    step=5,
                                    label="Active-frame percentile",
                                )
                        with gr.Row():
                            lab_model = gr.Dropdown(
                                ["htdemucs", "htdemucs_ft", "mdx_extra", "mdx_q"],
                                value="htdemucs",
                                label="Demucs model",
                            )
                            lab_device = gr.Dropdown(["cpu", "mps", "cuda"], value=DEFAULT_DEMUCS_DEVICE, label="Device")
                            lab_shifts = gr.Slider(1, 4, value=1, step=1, label="Shifts")
                            lab_overlap = gr.Slider(0.10, 0.50, value=0.25, step=0.05, label="Overlap")
                        lab_prepare_button = gr.Button("Run Audio Preparation Lab", variant="primary")
                        lab_prepare_status = gr.Markdown()
                        lab_prepare_table = gr.Dataframe(label="Preparation comparison", wrap=True)
                        with gr.Row():
                            lab_reference_original = gr.Audio(label="Reference original", type="filepath", interactive=False)
                            lab_reference_analysis = gr.Audio(label="Reference analysis", type="filepath", interactive=False)
                        with gr.Row():
                            lab_current_original = gr.Audio(label="Current original", type="filepath", interactive=False)
                            lab_current_analysis = gr.Audio(label="Current analysis", type="filepath", interactive=False)
                        with gr.Row():
                            lab_previous_original = gr.Audio(label="Previous original", type="filepath", interactive=False)
                            lab_previous_analysis = gr.Audio(label="Previous analysis", type="filepath", interactive=False)
                        lab_prepare_button.click(
                            prepare_audio,
                            inputs=[
                                reference_audio,
                                current_take,
                                previous_take,
                                lab_use_demucs,
                                lab_apply_to_takes,
                                lab_use_active_rms,
                                lab_target_active_rms,
                                lab_active_rms_percentile,
                                lab_model,
                                lab_device,
                                lab_shifts,
                                lab_overlap,
                            ],
                            outputs=[
                                prepared_state,
                                lab_prepare_status,
                                lab_prepare_table,
                                lab_reference_original,
                                lab_reference_analysis,
                                lab_current_original,
                                lab_current_analysis,
                                lab_previous_original,
                                lab_previous_analysis,
                                workflow_status,
                            ],
                        )

                    with gr.Tab("Pitch Extractor Lab"):
                        gr.Markdown(
                            """
## Pitch Extractor Lab

Compare pitch extraction methods on the same excerpt. Methods that are not installed or not wired return an explicit unavailable row instead of failing the run.
        """
                        )
                        with gr.Row():
                            pitch_lab_source = gr.Radio(
                                ["Current take", "Reference audio", "Previous take"],
                                value="Current take",
                                label="Audio source",
                            )
                            pitch_lab_start = gr.Number(value=0.0, label="Excerpt start (seconds)", precision=2)
                            pitch_lab_duration = gr.Slider(2, 90, value=20, step=2, label="Excerpt duration")
                        pitch_lab_methods = gr.CheckboxGroup(
                            ["pYIN", "CREPE / torchcrepe", "RMVPE", "Basic Pitch"],
                            value=["pYIN"],
                            label="Pitch extractors",
                        )
                        with gr.Row():
                            pitch_lab_min_confidence = gr.Slider(0.0, 0.95, value=0.25, step=0.05, label="Min confidence")
                            pitch_lab_max_jump = gr.Slider(100, 1600, value=700, step=50, label="Max jump cents")
                        pitch_lab_button = gr.Button("Run Pitch Extractor Lab", variant="primary")
                        pitch_lab_status = gr.Markdown()
                        pitch_lab_table = gr.Dataframe(label="Extractor comparison", wrap=True)
                        pitch_lab_audio = gr.Audio(label="Analyzed excerpt", type="filepath", interactive=False)
                        pitch_lab_plot = gr.Image(label="pYIN contour plot", type="filepath")
                        pitch_lab_json = gr.JSON(label="Extractor notes")
                        pitch_lab_button.click(
                            run_pitch_extractor_lab_ui,
                            inputs=[
                                pitch_lab_source,
                                reference_audio,
                                current_take,
                                previous_take,
                                prepared_state,
                                pitch_lab_methods,
                                pitch_lab_start,
                                pitch_lab_duration,
                                pitch_lab_min_confidence,
                                pitch_lab_max_jump,
                            ],
                            outputs=[pitch_lab_status, pitch_lab_table, pitch_lab_audio, pitch_lab_plot, pitch_lab_json],
                        )

                    with gr.Tab("Phrase / Section Matching Lab"):
                        gr.Markdown(
                            """
## Phrase / Section Matching Lab

Search for the comparable phrase or section before scoring. This is the lab for late starts, chorus-only recordings, and partial takes.
        """
                        )
                        with gr.Row():
                            section_catalog_source = gr.Radio(
                                ["Demo catalog", "Uploaded reference sections"],
                                value="Uploaded reference sections",
                                label="Catalog source",
                            )
                            section_top_k = gr.Slider(1, 10, value=5, step=1, label="Top K")
                            section_window = gr.Slider(5, 45, value=20, step=5, label="Reference window seconds")
                            section_hop = gr.Slider(1, 20, value=10, step=1, label="Reference hop seconds")
                        section_match_button = gr.Button("Run Phrase / Section Matching Lab", variant="primary")
                        section_status = gr.Markdown()
                        section_table = gr.Dataframe(label="Candidate sections", wrap=True)
                        with gr.Row():
                            section_reference_audio = gr.Audio(
                                label="Best matched reference section",
                                type="filepath",
                                interactive=False,
                            )
                            section_query_audio = gr.Audio(
                                label="Best matched query window",
                                type="filepath",
                                interactive=False,
                            )
                        section_json = gr.JSON(label="Match details")
                        section_match_button.click(
                            run_section_matching_lab_ui,
                            inputs=[
                                reference_audio,
                                current_take,
                                prepared_state,
                                section_catalog_source,
                                section_top_k,
                                section_window,
                                section_hop,
                            ],
                            outputs=[
                                section_status,
                                section_table,
                                section_reference_audio,
                                section_query_audio,
                                section_json,
                            ],
                        )

                    with gr.Tab("Matched Progress Scoring Lab"):
                        gr.Markdown(
                            """
## Matched Progress Scoring

Compare previous and current takes only after selecting a comparable reference section. This is the defensible progress path:

```text
reference + previous + current
-> matched section
-> cropped previous/current windows
-> reference-relative score deltas
```
        """
                        )
                        with gr.Row():
                            progress_catalog_source = gr.Radio(
                                ["Uploaded reference sections", "Demo catalog"],
                                value="Uploaded reference sections",
                                label="Reference section source",
                            )
                            progress_use_prepared = gr.Checkbox(
                                label="Use prepared analysis audio",
                                value=True,
                            )
                        with gr.Row():
                            progress_window = gr.Slider(
                                5,
                                45,
                                value=20,
                                step=5,
                                label="Reference section window seconds",
                            )
                            progress_hop = gr.Slider(
                                1,
                                20,
                                value=10,
                                step=1,
                                label="Reference section hop seconds",
                            )
                        with gr.Accordion("Confidence gates", open=False):
                            with gr.Row():
                                progress_min_match_score = gr.Slider(
                                    0,
                                    100,
                                    value=70,
                                    step=1,
                                    label="Minimum match score",
                                )
                                progress_min_match_margin = gr.Slider(
                                    0,
                                    30,
                                    value=8,
                                    step=1,
                                    label="Minimum top-two margin",
                                )
                                progress_min_coverage = gr.Slider(
                                    0,
                                    100,
                                    value=45,
                                    step=1,
                                    label="Minimum coverage",
                                )
                        progress_button = gr.Button("Run Matched Progress Scoring", variant="primary")
                        progress_status = gr.Markdown()
                        progress_summary = gr.Dataframe(label="Progress summary", wrap=True)
                        progress_section = gr.Dataframe(label="Selected reference section", wrap=True)
                        progress_metrics = gr.Dataframe(label="Detailed metrics", wrap=True)
                        progress_gates = gr.Dataframe(label="Confidence gates", wrap=True)
                        with gr.Row():
                            progress_pitch_plot = gr.Image(
                                label="Reference vs previous vs current pitch",
                                type="filepath",
                            )
                            progress_coverage_plot = gr.Image(
                                label="Matched-section voiced coverage",
                                type="filepath",
                            )
                        with gr.Row():
                            progress_reference_clip = gr.Audio(
                                label="Matched reference clip",
                                type="filepath",
                                interactive=False,
                            )
                            progress_previous_clip = gr.Audio(
                                label="Matched previous clip",
                                type="filepath",
                                interactive=False,
                            )
                            progress_current_clip = gr.Audio(
                                label="Matched current clip",
                                type="filepath",
                                interactive=False,
                            )
                        progress_json = gr.JSON(label="Matched progress details")
                        progress_copy_report = gr.Textbox(
                            label="Copyable matched-progress report",
                            lines=12,
                            interactive=False,
                        )
                        progress_copy_button = gr.Button("Copy Matched Progress Report")
                        progress_button.click(
                            run_matched_progress_lab_ui,
                            inputs=[
                                reference_audio,
                                previous_take,
                                current_take,
                                prepared_state,
                                progress_use_prepared,
                                progress_catalog_source,
                                progress_window,
                                progress_hop,
                                progress_min_match_score,
                                progress_min_match_margin,
                                progress_min_coverage,
                            ],
                            outputs=[
                                progress_status,
                                progress_summary,
                                progress_section,
                                progress_metrics,
                                progress_gates,
                                progress_pitch_plot,
                                progress_coverage_plot,
                                progress_reference_clip,
                                progress_previous_clip,
                                progress_current_clip,
                                progress_json,
                                progress_copy_report,
                            ],
                        )
                        progress_copy_button.click(
                            fn=None,
                            inputs=[progress_copy_report],
                            outputs=[],
                            js="(text) => navigator.clipboard.writeText(text || '')",
                        )

                    with gr.Tab("Song Identification Lab"):
                        gr.Markdown(
                            """
## Song Identification Lab

Rank likely songs and sections from the current take. For now this uses the same matching engine as the section lab, with a catalog-oriented result table.
        """
                        )
                        with gr.Row():
                            song_catalog_source = gr.Radio(
                                ["Demo catalog", "Uploaded reference sections"],
                                value="Demo catalog",
                                label="Catalog source",
                            )
                            song_top_k = gr.Slider(1, 10, value=5, step=1, label="Top K")
                            song_window = gr.Slider(5, 45, value=20, step=5, label="Section window seconds")
                            song_hop = gr.Slider(1, 20, value=10, step=1, label="Section hop seconds")
                        song_id_button = gr.Button("Run Song Identification Lab", variant="primary")
                        song_id_status = gr.Markdown()
                        song_id_table = gr.Dataframe(label="Candidate songs/sections", wrap=True)
                        song_id_json = gr.JSON(label="Identification details")
                        song_id_button.click(
                            run_song_identification_lab_ui,
                            inputs=[
                                reference_audio,
                                current_take,
                                prepared_state,
                                song_catalog_source,
                                song_top_k,
                                song_window,
                                song_hop,
                            ],
                            outputs=[song_id_status, song_id_table, song_id_json],
                        )

                    with gr.Tab("Timing / Rhythm Lab"):
                        gr.Markdown(
                            """
## Timing / Rhythm Lab

Inspect whether timing failures come from recording start offset or local rhythmic mismatch.
        """
                        )
                        timing_penalty_lab = gr.Slider(10, 200, value=90, step=5, label="Timing penalty")
                        timing_lab_button = gr.Button("Run Timing / Rhythm Lab", variant="primary")
                        timing_lab_status = gr.Markdown()
                        timing_lab_table = gr.Dataframe(label="Timing chain", wrap=True)
                        timing_lab_json = gr.JSON(label="Timing debug")
                        timing_lab_button.click(
                            run_timing_lab_ui,
                            inputs=[reference_audio, current_take, prepared_state, timing_penalty_lab],
                            outputs=[timing_lab_status, timing_lab_table, timing_lab_json],
                        )

                    with gr.Tab("Reference Builder Lab"):
                        gr.Markdown(
                            """
## Reference Builder Lab

Build and inspect the reference baseline. If no reference audio is uploaded, this falls back to the demo symbolic baseline.
        """
                        )
                        reference_builder_window = gr.Slider(0.05, 0.80, value=0.20, step=0.05, label="Reference extraction window seconds")
                        reference_builder_button = gr.Button("Run Reference Builder Lab", variant="primary")
                        reference_builder_status = gr.Markdown()
                        reference_builder_table = gr.Dataframe(label="Reference quality", wrap=True)
                        reference_builder_plot = gr.Image(label="Extracted baseline", type="filepath")
                        reference_builder_json = gr.JSON(label="Reference details")
                        reference_builder_button.click(
                            run_reference_builder_lab_ui,
                            inputs=[reference_audio, reference_builder_window],
                            outputs=[
                                reference_builder_status,
                                reference_builder_table,
                                reference_builder_plot,
                                reference_builder_json,
                            ],
                        )

                    with gr.Tab("Scoring Calibration Lab"):
                        gr.Markdown(
                            """
## Scoring Calibration Lab

Tune score weights against known synthetic failure cases. This is where we verify that stable-but-wrong is not treated as improvement.
        """
                        )
                        calibration_scenario = gr.Radio(
                            ["accurate", "stable but wrong", "unstable but close", "missing notes", "late start", "low confidence"],
                            value="accurate",
                            label="Scenario",
                        )
                        with gr.Row():
                            calibration_pitch_weight = gr.Slider(0, 1, value=0.45, step=0.05, label="Pitch weight")
                            calibration_stability_weight = gr.Slider(0, 1, value=0.25, step=0.05, label="Stability weight")
                            calibration_coverage_weight = gr.Slider(0, 1, value=0.20, step=0.05, label="Coverage weight")
                            calibration_timing_weight = gr.Slider(0, 1, value=0.10, step=0.05, label="Timing weight")
                        calibration_button = gr.Button("Run Scoring Calibration Lab", variant="primary")
                        calibration_status = gr.Markdown()
                        calibration_table = gr.Dataframe(label="Score contribution", wrap=True)
                        calibration_json = gr.JSON(label="Score details")
                        calibration_button.click(
                            run_scoring_calibration_lab_ui,
                            inputs=[
                                calibration_scenario,
                                calibration_pitch_weight,
                                calibration_stability_weight,
                                calibration_coverage_weight,
                                calibration_timing_weight,
                            ],
                            outputs=[calibration_status, calibration_table, calibration_json],
                        )

                    with gr.Tab("Stress Test Lab"):
                        gr.Markdown(
                            """
## Stress Test Lab

Run repeatable synthetic stress cases against the current scoring logic.
        """
                        )
                        stress_scenarios = gr.CheckboxGroup(
                            ["accurate", "stable but wrong", "unstable but close", "missing notes", "late start", "low confidence"],
                            value=["accurate", "stable but wrong", "missing notes", "late start"],
                            label="Stress scenarios",
                        )
                        stress_button = gr.Button("Run Stress Test Lab", variant="primary")
                        stress_status = gr.Markdown()
                        stress_table = gr.Dataframe(label="Stress results", wrap=True)
                        stress_json = gr.JSON(label="Stress details")
                        stress_button.click(
                            run_stress_test_lab_ui,
                            inputs=[stress_scenarios],
                            outputs=[stress_status, stress_table, stress_json],
                        )

            with gr.Tab("Fingerprinting"):
                gr.Markdown(
                    """
## Fingerprinting

Test whether a karaoke phone recording contains enough backing-track signal for
commercial audio fingerprinting services. Start with your 5-case dataset and
compare short valid recognition windows across providers.
"""
                )
                with gr.Tabs():
                    with gr.Tab("Long Session Segmentation"):
                        gr.Markdown(
                            """
## Long Session Segmentation

Find song-sized intervals inside a long karaoke recording. This identifies the backing-track recording and produces handoff clips; it does not score singing quality by itself.
Use 10-12s windows and a 5s hop before treating a no-match result as meaningful.
"""
                        )
                        with gr.Row():
                            session_source = gr.Radio(
                                ["Use current take", "Use prepared current audio", "Upload separate recording"],
                                value="Use current take",
                                label="Long recording source",
                            )
                            session_upload = gr.Audio(
                                label="Long karaoke recording",
                                type="filepath",
                            )
                            session_provider = gr.Radio(
                                ["No recognition (plots only)", "ShazamKit", "AudD", "ACRCloud"],
                                value="No recognition (plots only)",
                                label="Recognition provider",
                            )
                        with gr.Row():
                            session_window_s = gr.Slider(
                                3,
                                12,
                                value=10,
                                step=1,
                                label="Window length seconds",
                            )
                            session_hop_s = gr.Slider(
                                2,
                                30,
                                value=5,
                                step=1,
                                label="Window hop seconds",
                            )
                            session_max_windows = gr.Slider(
                                1,
                                500,
                                value=120,
                                step=1,
                                label="Max windows / API calls",
                            )
                        with gr.Row():
                            session_use_demucs = gr.Checkbox(
                                label="Use Demucs accompaniment stem",
                                value=False,
                            )
                            session_demucs_model = gr.Dropdown(
                                ["htdemucs", "htdemucs_ft", "mdx_extra", "mdx_q"],
                                value="htdemucs",
                                label="Demucs model",
                            )
                            session_demucs_device = gr.Dropdown(
                                ["cpu", "mps", "cuda"],
                                value="cpu",
                                label="Device",
                            )
                        with gr.Tabs():
                            with gr.Tab("RMS/BPM Hyperparameters"):
                                with gr.Row():
                                    session_rms_frame_s = gr.Slider(
                                        0.05,
                                        2.0,
                                        value=0.25,
                                        step=0.05,
                                        label="RMS frame seconds",
                                    )
                                    session_rms_hop_s = gr.Slider(
                                        0.05,
                                        1.0,
                                        value=0.1,
                                        step=0.05,
                                        label="RMS hop seconds",
                                    )
                                with gr.Row():
                                    session_tempo_window_s = gr.Slider(
                                        3,
                                        30,
                                        value=5,
                                        step=1,
                                        label="BPM window seconds",
                                    )
                                    session_tempo_hop_s = gr.Slider(
                                        1,
                                        30,
                                        value=5,
                                        step=1,
                                        label="BPM hop seconds",
                                    )
                        session_button = gr.Button("Run Long Session Segmentation", variant="primary")
                        session_status = gr.Markdown()
                        session_timeline = gr.Image(
                            label="Detected song intervals + RMS energy + tempo",
                            type="filepath",
                        )
                        session_intervals = gr.Dataframe(label="Detected song intervals", wrap=True)
                        session_weak_candidates = gr.Dataframe(label="Rejected weak candidates", wrap=True)
                        session_windows = gr.Dataframe(label="Raw recognition windows", wrap=True)
                        session_clip_files = gr.Files(
                            label="Detected interval clips",
                            interactive=False,
                        )
                        session_best_clip = gr.Audio(
                            label="Best interval clip for matching/scoring handoff",
                            type="filepath",
                            interactive=False,
                        )
                        session_details = gr.JSON(label="Segmentation details")
                        gr.Markdown(
                            """
Use the best interval clip as the next input to Song Identification, Phrase / Section Matching, or Scoring. Trust the handoff only when the timeline and clip audio match the song you expected.
"""
                        )
                        session_button.click(
                            run_long_session_segmentation_ui,
                            inputs=[
                                session_source,
                                session_upload,
                                current_take,
                                prepared_state,
                                session_use_demucs,
                                session_demucs_model,
                                session_demucs_device,
                                session_provider,
                                session_window_s,
                                session_hop_s,
                                session_max_windows,
                                session_rms_frame_s,
                                session_rms_hop_s,
                                session_tempo_window_s,
                                session_tempo_hop_s,
                            ],
                            outputs=[
                                session_status,
                                session_intervals,
                                session_weak_candidates,
                                session_windows,
                                session_timeline,
                                session_clip_files,
                                session_best_clip,
                                session_details,
                            ],
                        )

                    with gr.Tab("Diagnostics / Recovery"):
                        gr.Markdown(
                            """
## Diagnostics / Recovery

Paste or upload saved fingerprint rows to explain failed scans without rerunning a provider. Weak candidates are clues for focused rescans, not accepted song intervals.
"""
                        )
                        with gr.Row():
                            diag_provider = gr.Radio(
                                ["ShazamKit", "AudD", "ACRCloud"],
                                value="ACRCloud",
                                label="Provider",
                            )
                            diag_duration = gr.Number(
                                value=0,
                                label="Recording duration seconds",
                                precision=1,
                            )
                            diag_budget = gr.Slider(
                                1,
                                500,
                                value=120,
                                step=1,
                                label="Recovery request budget",
                            )
                        diag_csv_upload = gr.File(
                            label="Fingerprint CSV",
                            type="filepath",
                        )
                        diag_csv_text = gr.Textbox(
                            label="Paste fingerprint rows",
                            lines=8,
                            placeholder="acrcloud,480,485,matched,true,Title,Artist,isrc:...,0.34,/tmp/window.wav,window.wav,",
                        )
                        diag_original_audio = gr.Audio(
                            label="Original long recording for preview clips",
                            type="filepath",
                        )
                        diag_button = gr.Button("Run Diagnostics", variant="primary")
                        diag_status = gr.Markdown()
                        diag_summary = gr.Dataframe(label="Diagnostic summary", wrap=True)
                        diag_weak = gr.Dataframe(label="Weak candidates", wrap=True)
                        diag_sweeps = gr.Dataframe(label="Recovery sweeps", wrap=True)
                        diag_recommendations = gr.Dataframe(label="Recommendations", wrap=True)
                        diag_preview_files = gr.Files(
                            label="Preview recovery windows",
                            interactive=False,
                        )
                        diag_first_preview = gr.Audio(
                            label="First recovery preview",
                            type="filepath",
                            interactive=False,
                        )
                        diag_details = gr.JSON(label="Diagnostic details")
                        diag_button.click(
                            run_fingerprint_diagnostics_ui,
                            inputs=[
                                diag_provider,
                                diag_csv_upload,
                                diag_csv_text,
                                diag_original_audio,
                                diag_duration,
                                diag_budget,
                            ],
                            outputs=[
                                diag_status,
                                diag_summary,
                                diag_weak,
                                diag_sweeps,
                                diag_recommendations,
                                diag_preview_files,
                                diag_first_preview,
                                diag_details,
                            ],
                        )

                    with gr.Tab("ShazamKit"):
                        gr.Markdown(
                            """
Catalog matching requires a signed ShazamKit helper whose App ID has the
ShazamKit App Service enabled.
"""
                        )
                        with gr.Row():
                            fingerprint_source = gr.Radio(
                                ["Use current take", "Use prepared current audio", "Upload separate recording"],
                                value="Use current take",
                                label="Recording source",
                            )
                            fingerprint_upload = gr.Audio(
                                label="Karaoke phone recording",
                                type="filepath",
                            )
                        with gr.Row():
                            fingerprint_expected_title = gr.Textbox(
                                label="Expected song title",
                                placeholder="Optional ground-truth title",
                            )
                            fingerprint_expected_artist = gr.Textbox(
                                label="Expected artist",
                                placeholder="Optional ground-truth artist",
                            )
                        with gr.Row():
                            fingerprint_mode = gr.Radio(
                                ["Whole recording", "Sliding windows", "Whole + sliding windows"],
                                value="Whole + sliding windows",
                                label="Recognition mode",
                            )
                            fingerprint_window_s = gr.Slider(
                                3,
                                12,
                                value=10,
                                step=1,
                                label="Window length seconds (ShazamKit accepts 3-12)",
                            )
                            fingerprint_hop_s = gr.Slider(
                                2,
                                20,
                                value=5,
                                step=1,
                                label="Window hop seconds",
                                info=(
                                    "Starts are generated with this hop, then capped by Max windows."
                                ),
                            )
                            fingerprint_max_windows = gr.Slider(
                                1,
                                40,
                                value=12,
                                step=1,
                                label="Max windows / API calls",
                                info="Caps API calls by keeping the first hop-spaced windows.",
                            )
                        with gr.Row():
                            fingerprint_start_offset_s = gr.Slider(
                                0,
                                240,
                                value=0,
                                step=1,
                                label="Start offset seconds",
                                info="For From offset, first sliding window starts here.",
                            )
                            fingerprint_window_strategy = gr.Radio(
                                ["From offset", "Center-out"],
                                value="From offset",
                                label="Window strategy",
                                info="Center-out tests the middle first, then alternates earlier/later by hop.",
                            )
                        with gr.Row():
                            fingerprint_use_demucs = gr.Checkbox(
                                label="Use Demucs accompaniment stem",
                                value=False,
                            )
                            fingerprint_demucs_model = gr.Dropdown(
                                ["htdemucs", "htdemucs_ft", "mdx_extra", "mdx_q"],
                                value="htdemucs",
                                label="Demucs model",
                            )
                            fingerprint_demucs_device = gr.Dropdown(
                                ["cpu", "mps", "cuda"],
                                value="cpu",
                                label="Device",
                            )
                        fingerprint_audio_state = gr.State({})
                        fingerprint_prepare_button = gr.Button("Prepare preview audio")
                        fingerprint_prepare_status = gr.Markdown("Prepare preview audio before sending an API request.")
                        with gr.Row():
                            fingerprint_original_preview = gr.Audio(
                                label="Original selected audio",
                                type="filepath",
                            )
                            fingerprint_recognition_preview = gr.Audio(
                                label="Recognition source audio after preprocessing",
                                type="filepath",
                            )
                        fingerprint_request_preview_files = gr.Files(
                            label="Generated request-window clips before API call",
                            interactive=False,
                        )
                        fingerprint_request_window_outputs = _request_window_preview_components()
                        fingerprint_button = gr.Button("Run ShazamKit Fingerprinting", variant="primary")
                        fingerprint_status = gr.Markdown()
                        fingerprint_summary = gr.Dataframe(label="Summary metrics", wrap=True)
                        fingerprint_results = gr.Dataframe(label="Window results", wrap=True)
                        fingerprint_preview_files = gr.Files(
                            label="All generated processed clips",
                            interactive=False,
                        )
                        fingerprint_preview_gallery = gr.HTML(
                            "<p>Run fingerprinting to generate processed clip previews.</p>"
                        )
                        fingerprint_preview_choice = gr.Dropdown(
                            label="Focused processed clip preview",
                            choices=[],
                            interactive=True,
                        )
                        fingerprint_preview = gr.Audio(label="Processed audio sent to ShazamKit", type="filepath")
                        fingerprint_preview_status = gr.Markdown("Run fingerprinting to generate a processed-window preview.")
                        fingerprint_interpretation = gr.Markdown("Run fingerprinting to generate interpretation.")
                        fingerprint_details = gr.JSON(label="Fingerprinting details")
                        fingerprint_prepare_button.click(
                            prepare_fingerprinting_audio_preview_ui,
                            inputs=[
                                fingerprint_source,
                                fingerprint_upload,
                                current_take,
                                prepared_state,
                                fingerprint_use_demucs,
                                fingerprint_demucs_model,
                                fingerprint_demucs_device,
                                fingerprint_mode,
                                fingerprint_window_s,
                                fingerprint_hop_s,
                                fingerprint_max_windows,
                                fingerprint_start_offset_s,
                                fingerprint_window_strategy,
                            ],
                            outputs=[
                                fingerprint_prepare_status,
                                fingerprint_original_preview,
                                fingerprint_recognition_preview,
                                fingerprint_request_preview_files,
                                fingerprint_audio_state,
                                *fingerprint_request_window_outputs,
                            ],
                        )
                        fingerprint_button.click(
                            run_fingerprinting_lab_ui,
                            inputs=[
                                fingerprint_audio_state,
                                fingerprint_expected_title,
                                fingerprint_expected_artist,
                                fingerprint_mode,
                                fingerprint_window_s,
                                fingerprint_hop_s,
                                fingerprint_max_windows,
                                fingerprint_start_offset_s,
                                fingerprint_window_strategy,
                            ],
                            outputs=[
                                fingerprint_status,
                                fingerprint_summary,
                                fingerprint_results,
                                fingerprint_original_preview,
                                fingerprint_recognition_preview,
                                fingerprint_preview_files,
                                fingerprint_preview_gallery,
                                fingerprint_preview_choice,
                                fingerprint_preview,
                                fingerprint_preview_status,
                                fingerprint_interpretation,
                                fingerprint_details,
                            ],
                        )
                        fingerprint_preview_choice.change(
                            select_fingerprinting_preview,
                            inputs=[fingerprint_preview_choice, fingerprint_details],
                            outputs=[fingerprint_preview, fingerprint_preview_status],
                        )

                    with gr.Tab("AudD"):
                        gr.Markdown(
                            """
Uses AudD's standard music recognition endpoint. Set `AUDD_API_TOKEN` before
starting the dashboard. Results are cached by generated clip hash to avoid
spending trial requests on identical windows.
"""
                        )
                        with gr.Row():
                            audd_source = gr.Radio(
                                ["Use current take", "Use prepared current audio", "Upload separate recording"],
                                value="Use current take",
                                label="Recording source",
                            )
                            audd_upload = gr.Audio(
                                label="Karaoke phone recording",
                                type="filepath",
                            )
                        with gr.Row():
                            audd_expected_title = gr.Textbox(
                                label="Expected song title",
                                placeholder="Optional ground-truth title",
                            )
                            audd_expected_artist = gr.Textbox(
                                label="Expected artist",
                                placeholder="Optional ground-truth artist",
                            )
                        with gr.Row():
                            audd_mode = gr.Radio(
                                ["Whole recording", "Sliding windows", "Whole + sliding windows"],
                                value="Sliding windows",
                                label="Recognition mode",
                            )
                            audd_window_s = gr.Slider(
                                3,
                                12,
                                value=10,
                                step=1,
                                label="Window length seconds",
                            )
                            audd_hop_s = gr.Slider(
                                2,
                                20,
                                value=5,
                                step=1,
                                label="Window hop seconds",
                                info=(
                                    "Starts are generated with this hop, then capped by Max windows."
                                ),
                            )
                            audd_max_windows = gr.Slider(
                                1,
                                40,
                                value=12,
                                step=1,
                                label="Max windows / API calls",
                                info="Caps API calls by keeping the first hop-spaced windows.",
                            )
                        with gr.Row():
                            audd_start_offset_s = gr.Slider(
                                0,
                                240,
                                value=0,
                                step=1,
                                label="Start offset seconds",
                                info="For From offset, first sliding window starts here.",
                            )
                            audd_window_strategy = gr.Radio(
                                ["From offset", "Center-out"],
                                value="From offset",
                                label="Window strategy",
                                info="Center-out tests the middle first, then alternates earlier/later by hop.",
                            )
                        with gr.Row():
                            audd_use_demucs = gr.Checkbox(
                                label="Use Demucs accompaniment stem",
                                value=False,
                            )
                            audd_demucs_model = gr.Dropdown(
                                ["htdemucs", "htdemucs_ft", "mdx_extra", "mdx_q"],
                                value="htdemucs",
                                label="Demucs model",
                            )
                            audd_demucs_device = gr.Dropdown(
                                ["cpu", "mps", "cuda"],
                                value="cpu",
                                label="Device",
                            )
                        audd_audio_state = gr.State({})
                        audd_prepare_button = gr.Button("Prepare preview audio")
                        audd_prepare_status = gr.Markdown("Prepare preview audio before sending an API request.")
                        with gr.Row():
                            audd_original_preview = gr.Audio(
                                label="Original selected audio",
                                type="filepath",
                            )
                            audd_recognition_preview = gr.Audio(
                                label="Recognition source audio after preprocessing",
                                type="filepath",
                            )
                        audd_request_preview_files = gr.Files(
                            label="Generated request-window clips before API call",
                            interactive=False,
                        )
                        audd_request_window_outputs = _request_window_preview_components()
                        audd_button = gr.Button("Run AudD Fingerprinting", variant="primary")
                        audd_status = gr.Markdown()
                        audd_summary = gr.Dataframe(label="Summary metrics", wrap=True)
                        audd_results = gr.Dataframe(label="Window results", wrap=True)
                        audd_preview_files = gr.Files(
                            label="All generated processed clips",
                            interactive=False,
                        )
                        audd_preview_gallery = gr.HTML(
                            "<p>Run AudD to generate processed clip previews.</p>"
                        )
                        audd_preview_choice = gr.Dropdown(
                            label="Focused processed clip preview",
                            choices=[],
                            interactive=True,
                        )
                        audd_preview = gr.Audio(label="Processed audio sent to AudD", type="filepath")
                        audd_preview_status = gr.Markdown("Run AudD to generate a processed-window preview.")
                        audd_interpretation = gr.Markdown("Run AudD to generate interpretation.")
                        audd_details = gr.JSON(label="AudD details")
                        audd_prepare_button.click(
                            prepare_fingerprinting_audio_preview_ui,
                            inputs=[
                                audd_source,
                                audd_upload,
                                current_take,
                                prepared_state,
                                audd_use_demucs,
                                audd_demucs_model,
                                audd_demucs_device,
                                audd_mode,
                                audd_window_s,
                                audd_hop_s,
                                audd_max_windows,
                                audd_start_offset_s,
                                audd_window_strategy,
                            ],
                            outputs=[
                                audd_prepare_status,
                                audd_original_preview,
                                audd_recognition_preview,
                                audd_request_preview_files,
                                audd_audio_state,
                                *audd_request_window_outputs,
                            ],
                        )
                        audd_button.click(
                            run_audd_fingerprinting_lab_ui,
                            inputs=[
                                audd_audio_state,
                                audd_expected_title,
                                audd_expected_artist,
                                audd_mode,
                                audd_window_s,
                                audd_hop_s,
                                audd_max_windows,
                                audd_start_offset_s,
                                audd_window_strategy,
                            ],
                            outputs=[
                                audd_status,
                                audd_summary,
                                audd_results,
                                audd_original_preview,
                                audd_recognition_preview,
                                audd_preview_files,
                                audd_preview_gallery,
                                audd_preview_choice,
                                audd_preview,
                                audd_preview_status,
                                audd_interpretation,
                                audd_details,
                            ],
                        )
                        audd_preview_choice.change(
                            select_fingerprinting_preview,
                            inputs=[audd_preview_choice, audd_details],
                            outputs=[audd_preview, audd_preview_status],
                        )

                    with gr.Tab("ACRCloud"):
                        gr.Markdown(
                            """
Uses ACRCloud's `/v1/identify` endpoint. Set `ACRCLOUD_HOST`,
`ACRCLOUD_ACCESS_KEY`, and `ACRCLOUD_ACCESS_SECRET` before starting the
dashboard. Results are cached by generated clip hash and project host.
"""
                        )
                        with gr.Row():
                            acr_source = gr.Radio(
                                ["Use current take", "Use prepared current audio", "Upload separate recording"],
                                value="Use current take",
                                label="Recording source",
                            )
                            acr_upload = gr.Audio(
                                label="Karaoke phone recording",
                                type="filepath",
                            )
                        with gr.Row():
                            acr_expected_title = gr.Textbox(
                                label="Expected song title",
                                placeholder="Optional ground-truth title",
                            )
                            acr_expected_artist = gr.Textbox(
                                label="Expected artist",
                                placeholder="Optional ground-truth artist",
                            )
                        with gr.Row():
                            acr_mode = gr.Radio(
                                ["Whole recording", "Sliding windows", "Whole + sliding windows"],
                                value="Sliding windows",
                                label="Recognition mode",
                            )
                            acr_window_s = gr.Slider(
                                3,
                                12,
                                value=10,
                                step=1,
                                label="Window length seconds",
                            )
                            acr_hop_s = gr.Slider(
                                2,
                                20,
                                value=5,
                                step=1,
                                label="Window hop seconds",
                                info=(
                                    "Starts are generated with this hop, then capped by Max windows."
                                ),
                            )
                            acr_max_windows = gr.Slider(
                                1,
                                40,
                                value=12,
                                step=1,
                                label="Max windows / API calls",
                                info="Caps API calls by keeping the first hop-spaced windows.",
                            )
                        with gr.Row():
                            acr_start_offset_s = gr.Slider(
                                0,
                                240,
                                value=0,
                                step=1,
                                label="Start offset seconds",
                                info="For From offset, first sliding window starts here.",
                            )
                            acr_window_strategy = gr.Radio(
                                ["From offset", "Center-out"],
                                value="From offset",
                                label="Window strategy",
                                info="Center-out tests the middle first, then alternates earlier/later by hop.",
                            )
                        with gr.Row():
                            acr_use_demucs = gr.Checkbox(
                                label="Use Demucs accompaniment stem",
                                value=False,
                            )
                            acr_demucs_model = gr.Dropdown(
                                ["htdemucs", "htdemucs_ft", "mdx_extra", "mdx_q"],
                                value="htdemucs",
                                label="Demucs model",
                            )
                            acr_demucs_device = gr.Dropdown(
                                ["cpu", "mps", "cuda"],
                                value="cpu",
                                label="Device",
                            )
                        acr_audio_state = gr.State({})
                        acr_prepare_button = gr.Button("Prepare preview audio")
                        acr_prepare_status = gr.Markdown("Prepare preview audio before sending an API request.")
                        with gr.Row():
                            acr_original_preview = gr.Audio(
                                label="Original selected audio",
                                type="filepath",
                            )
                            acr_recognition_preview = gr.Audio(
                                label="Recognition source audio after preprocessing",
                                type="filepath",
                            )
                        acr_request_preview_files = gr.Files(
                            label="Generated request-window clips before API call",
                            interactive=False,
                        )
                        acr_request_window_outputs = _request_window_preview_components()
                        acr_button = gr.Button("Run ACRCloud Fingerprinting", variant="primary")
                        acr_status = gr.Markdown()
                        acr_summary = gr.Dataframe(label="Summary metrics", wrap=True)
                        acr_results = gr.Dataframe(label="Window results", wrap=True)
                        acr_preview_files = gr.Files(
                            label="All generated processed clips",
                            interactive=False,
                        )
                        acr_preview_gallery = gr.HTML(
                            "<p>Run ACRCloud to generate processed clip previews.</p>"
                        )
                        acr_preview_choice = gr.Dropdown(
                            label="Focused processed clip preview",
                            choices=[],
                            interactive=True,
                        )
                        acr_preview = gr.Audio(label="Processed audio sent to ACRCloud", type="filepath")
                        acr_preview_status = gr.Markdown("Run ACRCloud to generate a processed-window preview.")
                        acr_interpretation = gr.Markdown("Run ACRCloud to generate interpretation.")
                        acr_details = gr.JSON(label="ACRCloud details")
                        acr_prepare_button.click(
                            prepare_fingerprinting_audio_preview_ui,
                            inputs=[
                                acr_source,
                                acr_upload,
                                current_take,
                                prepared_state,
                                acr_use_demucs,
                                acr_demucs_model,
                                acr_demucs_device,
                                acr_mode,
                                acr_window_s,
                                acr_hop_s,
                                acr_max_windows,
                                acr_start_offset_s,
                                acr_window_strategy,
                            ],
                            outputs=[
                                acr_prepare_status,
                                acr_original_preview,
                                acr_recognition_preview,
                                acr_request_preview_files,
                                acr_audio_state,
                                *acr_request_window_outputs,
                            ],
                        )
                        acr_button.click(
                            run_acrcloud_fingerprinting_lab_ui,
                            inputs=[
                                acr_audio_state,
                                acr_expected_title,
                                acr_expected_artist,
                                acr_mode,
                                acr_window_s,
                                acr_hop_s,
                                acr_max_windows,
                                acr_start_offset_s,
                                acr_window_strategy,
                            ],
                            outputs=[
                                acr_status,
                                acr_summary,
                                acr_results,
                                acr_original_preview,
                                acr_recognition_preview,
                                acr_preview_files,
                                acr_preview_gallery,
                                acr_preview_choice,
                                acr_preview,
                                acr_preview_status,
                                acr_interpretation,
                                acr_details,
                            ],
                        )
                        acr_preview_choice.change(
                            select_fingerprinting_preview,
                            inputs=[acr_preview_choice, acr_details],
                            outputs=[acr_preview, acr_preview_status],
                        )
    return demo


if __name__ == "__main__":
    port = int(os.environ["GRADIO_SERVER_PORT"]) if os.environ.get("GRADIO_SERVER_PORT") else None
    build_app().queue().launch(server_name="127.0.0.1", server_port=port)
