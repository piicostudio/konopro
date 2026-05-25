from __future__ import annotations

import hashlib
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import streamlit as st

RESEARCH_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = RESEARCH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from konopro_research.audio_io import load_audio  # noqa: E402
from konopro_research.baseline import (  # noqa: E402
    MelodyBaseline,
    baseline_from_rows,
    baseline_to_csv_text,
    baseline_to_rows,
    load_baseline_csv,
)
from konopro_research.contour_scoring import (  # noqa: E402
    compare_takes_to_reference_contour,
    score_take_against_reference_contour,
)
from konopro_research.demo_data import ensure_demo_data  # noqa: E402
from konopro_research.matching import (  # noqa: E402
    SectionMatchResult,
    build_demo_section_catalog,
    crop_contour,
    extract_matching_query,
    match_query_to_sections,
    sections_from_baseline,
    split_contour_into_sections,
)
from konopro_research.pitch import clean_pitch_contour, extract_pitch  # noqa: E402
from konopro_research.plots import (  # noqa: E402
    plot_contour_comparison,
    plot_contour_voiced_coverage,
    plot_reference_extraction,
    plot_section_match,
    plot_take_comparison,
    plot_voiced_coverage,
)
from konopro_research.quality import (  # noqa: E402
    AudioSummary,
    BaselineQuality,
    analyze_baseline_quality,
    duration_mismatch_warnings,
    summarize_audio,
)
from konopro_research.reference_audio import ReferenceExtraction, extract_reference_audio  # noqa: E402
from konopro_research.scoring import ComparisonResult, ScoreResult, compare_takes, score_take  # noqa: E402
from konopro_research.separation import (  # noqa: E402
    SeparationResult,
    prepare_vocal_analysis_audio,
)


st.set_page_config(page_title="Konopro Research Demo", layout="wide")

demo_paths = ensure_demo_data(RESEARCH_ROOT / "data" / "demo")
UPLOAD_CACHE_DIR = RESEARCH_ROOT / ".cache" / "uploads"
STEM_CACHE_DIR = RESEARCH_ROOT / ".cache" / "stems"
PROCESSING_STATE_KEY = "konopro_processing_state"
VERIFICATION_STATE_KEY = "konopro_verification_state"
MATCHING_STATE_KEY = "konopro_matching_state"

DEMO_SCENARIOS = {
    "Progress: cleaner current take": {
        "description": "Compares a rougher previous take with a cleaner current take.",
        "baseline": "demo_csv",
        "previous": "previous",
        "current": "current",
    },
    "Stable but wrong": {
        "description": "Shows that a stable take can still be wrong against the melody.",
        "baseline": "demo_csv",
        "previous": "previous",
        "current": "stable_wrong",
    },
    "Missing notes": {
        "description": "Shows how skipped notes reduce coverage even when sung notes are accurate.",
        "baseline": "demo_csv",
        "previous": "previous",
        "current": "missing_notes",
    },
    "Noisy room": {
        "description": "Shows the confidence and robustness caveats for noisy/reverberant audio.",
        "baseline": "demo_csv",
        "previous": "previous",
        "current": "noisy_room",
    },
    "Upload baseline CSV + takes": {
        "description": "Use when you already have a symbolic melody baseline. Previous take is optional.",
        "baseline": "upload_csv",
        "previous": "upload",
        "current": "upload",
    },
    "Upload reference audio + takes": {
        "description": "Experimental contour scoring from uploaded guide/reference audio. Previous take is optional.",
        "baseline": "upload_reference_audio",
        "previous": "upload",
        "current": "upload",
    },
}


@dataclass(frozen=True)
class InputState:
    scenario_name: str
    scenario: dict[str, str]
    baseline: MelodyBaseline | None
    reference_audio_path: Path | None
    previous_take_path: Path | None
    current_take_path: Path | None


@dataclass(frozen=True)
class ProcessingState:
    reference_analysis_path: Path | None
    previous_analysis_path: Path | None
    current_analysis_path: Path | None
    separation_results: tuple[SeparationResult, ...]
    separation_config: dict[str, object]


@dataclass(frozen=True)
class VerificationState:
    baseline: MelodyBaseline
    reference_extraction: ReferenceExtraction | None
    baseline_quality: BaselineQuality
    current_summary: AudioSummary
    previous_summary: AudioSummary | None
    duration_warnings: tuple[str, ...]
    current_score: ScoreResult
    comparison: ComparisonResult | None
    current_contour: object
    previous_contour: object | None


@dataclass(frozen=True)
class MatchingRunState:
    result: SectionMatchResult
    handoff_score: ScoreResult | None
    sections_count: int
    settings: dict[str, object]


def uploaded_file_to_cache(uploaded: Any, *, role: str) -> Path:
    data = uploaded.getvalue()
    digest = hashlib.sha256(data).hexdigest()[:16]
    suffix = Path(uploaded.name).suffix.lower() or ".bin"
    safe_stem = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in Path(uploaded.name).stem
    )
    safe_stem = safe_stem[:48] or role
    path = UPLOAD_CACHE_DIR / f"{role}_{digest}_{safe_stem}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)
    return path


def render_warnings(warnings: tuple[str, ...] | list[str]) -> None:
    for warning in warnings:
        st.warning(warning)


def metric_delta(value: float) -> str | None:
    if abs(value) < 0.05:
        return None
    return f"{value:+.1f}"


def render_metric(
    column,
    label: str,
    value: float,
    delta: float | None = None,
    help_text: str | None = None,
) -> None:
    column.metric(
        label,
        f"{value:.1f}",
        metric_delta(delta) if delta is not None else None,
        delta_color="normal" if delta is None or abs(delta) >= 0.05 else "off",
        help=help_text,
    )


def render_audio_summary(summary: AudioSummary) -> None:
    cols = st.columns(4)
    cols[0].metric("Duration", f"{summary.duration_s:.1f}s")
    cols[1].metric("Sample rate", f"{summary.sample_rate:,} Hz")
    cols[2].metric("RMS", f"{summary.rms:.4f}")
    cols[3].metric("Peak", f"{summary.peak:.3f}")
    st.caption(f"Clipping ratio: {summary.clipping_ratio:.5f}")
    render_warnings(summary.warnings)


def render_audio_preview(label: str, path: Path | None, *, required: bool = False) -> AudioSummary | None:
    st.write(f"**{label}**")
    if path is None:
        message = "Required audio is missing." if required else "No audio provided."
        st.info(message)
        return None

    try:
        summary = summarize_audio(path)
    except Exception as exc:
        st.error(f"Could not inspect audio: {exc}")
        return None

    st.audio(str(path))
    render_audio_summary(summary)
    return summary


def default_processing_state(
    inputs: InputState,
    separation_config: dict[str, object],
) -> ProcessingState:
    return ProcessingState(
        reference_analysis_path=inputs.reference_audio_path,
        previous_analysis_path=inputs.previous_take_path,
        current_analysis_path=inputs.current_take_path,
        separation_results=(),
        separation_config=separation_config,
    )


def empty_processing_state(separation_config: dict[str, object]) -> ProcessingState:
    return ProcessingState(
        reference_analysis_path=None,
        previous_analysis_path=None,
        current_analysis_path=None,
        separation_results=(),
        separation_config=separation_config,
    )


def input_signature(inputs: InputState) -> dict[str, object]:
    return {
        "scenario": inputs.scenario_name,
        "reference_audio_path": str(inputs.reference_audio_path) if inputs.reference_audio_path else None,
        "previous_take_path": str(inputs.previous_take_path) if inputs.previous_take_path else None,
        "current_take_path": str(inputs.current_take_path) if inputs.current_take_path else None,
        "baseline_title": inputs.baseline.title if inputs.baseline is not None else None,
        "baseline_duration_s": inputs.baseline.duration_s if inputs.baseline is not None else None,
    }


def stable_signature(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def processing_signature(inputs: InputState, separation_config: dict[str, object]) -> str:
    return stable_signature({"inputs": input_signature(inputs), "separation": separation_config})


def verification_signature(
    inputs: InputState,
    processing: ProcessingState,
    analysis_params: dict[str, object],
) -> str:
    return stable_signature(
        {
            "inputs": input_signature(inputs),
            "reference_analysis_path": (
                str(processing.reference_analysis_path) if processing.reference_analysis_path else None
            ),
            "previous_analysis_path": (
                str(processing.previous_analysis_path) if processing.previous_analysis_path else None
            ),
            "current_analysis_path": (
                str(processing.current_analysis_path) if processing.current_analysis_path else None
            ),
            "settings": analysis_params["settings"],
        }
    )


def get_stored_processing_state(signature: str) -> ProcessingState | None:
    stored = st.session_state.get(PROCESSING_STATE_KEY)
    if not stored or stored.get("signature") != signature:
        return None
    return stored["state"]


def set_stored_processing_state(signature: str, state: ProcessingState) -> None:
    st.session_state[PROCESSING_STATE_KEY] = {"signature": signature, "state": state}


def get_stored_verification_state(signature: str) -> VerificationState | None:
    stored = st.session_state.get(VERIFICATION_STATE_KEY)
    if not stored or stored.get("signature") != signature:
        return None
    return stored["state"]


def set_stored_verification_state(signature: str, state: VerificationState) -> None:
    st.session_state[VERIFICATION_STATE_KEY] = {"signature": signature, "state": state}


def get_stored_matching_state(signature: str) -> MatchingRunState | None:
    stored = st.session_state.get(MATCHING_STATE_KEY)
    if not stored or stored.get("signature") != signature:
        return None
    return stored["state"]


def set_stored_matching_state(signature: str, state: MatchingRunState) -> None:
    st.session_state[MATCHING_STATE_KEY] = {"signature": signature, "state": state}


def stale_state_notice(key: str, label: str) -> None:
    if key in st.session_state:
        st.warning(f"{label} is stale because inputs or settings changed. Rerun this stage.")


def render_baseline_preview(baseline: MelodyBaseline | None) -> BaselineQuality | None:
    st.write("**Reference baseline**")
    if baseline is None:
        st.info("Upload reference audio or a baseline CSV to create the baseline.")
        return None

    quality = analyze_baseline_quality(baseline)
    cols = st.columns(4)
    cols[0].metric("Quality", quality.level)
    cols[1].metric("Notes", quality.note_count)
    cols[2].metric("Coverage", f"{quality.voiced_coverage_ratio * 100:.1f}%")
    cols[3].metric("Duration", f"{quality.duration_s:.1f}s")
    render_warnings(quality.warnings)
    return quality


def apply_step_card_styles() -> None:
    st.markdown(
        """
<style>
div[data-testid="stExpander"] {
  border: 1px solid rgba(148, 163, 184, 0.32);
  border-radius: 8px;
  background: rgba(15, 23, 42, 0.22);
  margin-bottom: 1rem;
}
div[data-testid="stExpander"] details > summary {
  padding: 0.85rem 1rem;
}
div[data-testid="stExpander"] details > summary p {
  font-size: 1.05rem;
  font-weight: 700;
}
div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
  padding: 0 1rem 1rem;
}
</style>
        """.strip(),
        unsafe_allow_html=True,
    )


def render_step_1_import_files() -> InputState:
    scenario_name = st.radio(
        "Testing scenario",
        list(DEMO_SCENARIOS),
        horizontal=True,
        help="Choose a reliable demo case or upload your own reference/current/previous files.",
    )
    scenario = DEMO_SCENARIOS[scenario_name]
    st.caption(scenario["description"])

    baseline: MelodyBaseline | None = None
    reference_audio_path: Path | None = demo_paths.get(
        "reference",
        RESEARCH_ROOT / "data" / "demo" / "reference_melody.wav",
    )
    current_take_path: Path | None = None
    previous_take_path: Path | None = None

    if scenario["baseline"] == "demo_csv":
        baseline = load_baseline_csv(demo_paths["baseline"])
    elif scenario["baseline"] == "upload_csv":
        baseline_upload = st.file_uploader(
            "Reference baseline CSV",
            type=["csv"],
            help="CSV must include start_s, end_s, and midi columns.",
        )
        optional_reference_upload = st.file_uploader(
            "Optional reference audio for playback",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
        )
        if baseline_upload is not None:
            try:
                baseline = load_baseline_csv(
                    io.StringIO(baseline_upload.getvalue().decode("utf-8")),
                    title=baseline_upload.name,
                )
            except Exception as exc:
                st.error(f"Could not load baseline CSV: {exc}")
        if optional_reference_upload is not None:
            reference_audio_path = uploaded_file_to_cache(optional_reference_upload, role="reference")
        else:
            reference_audio_path = None
    else:
        reference_upload = st.file_uploader(
            "Reference audio",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
            help="Use a guide vocal, melody-only reference, or private BYO song clip.",
        )
        reference_audio_path = (
            uploaded_file_to_cache(reference_upload, role="reference")
            if reference_upload is not None
            else None
        )

    if scenario["current"] == "upload":
        current_upload = st.file_uploader(
            "Current vocal take",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
        )
        previous_upload = st.file_uploader(
            "Previous vocal take (optional)",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
        )
        current_take_path = (
            uploaded_file_to_cache(current_upload, role="current")
            if current_upload is not None
            else None
        )
        previous_take_path = (
            uploaded_file_to_cache(previous_upload, role="previous")
            if previous_upload is not None
            else None
        )
    else:
        current_take_path = demo_paths[scenario["current"]]
        previous_take_path = demo_paths[scenario["previous"]]

    st.subheader("File Preview")
    reference_col, previous_col, current_col = st.columns(3)
    with reference_col:
        render_baseline_preview(baseline)
        render_audio_preview(
            "Reference audio",
            reference_audio_path,
            required=scenario["baseline"] == "upload_reference_audio",
        )
    with previous_col:
        render_audio_preview("Previous take", previous_take_path, required=False)
    with current_col:
        render_audio_preview("Current take", current_take_path, required=True)

    return InputState(
        scenario_name=scenario_name,
        scenario=scenario,
        baseline=baseline,
        reference_audio_path=reference_audio_path,
        previous_take_path=previous_take_path,
        current_take_path=current_take_path,
    )


def render_step_2_prepare_audio(inputs: InputState) -> ProcessingState:
    st.caption("Choose which files pYIN should analyze: original uploads or separated vocal stems.")

    analysis_audio = st.radio(
        "Analysis audio",
        ["Original audio", "Demucs vocal stem"],
        horizontal=True,
        help="Demucs is slower but usually better for mixed songs with instruments.",
    )
    backend = "demucs" if analysis_audio.startswith("Demucs") else "none"

    apply_scope = st.radio(
        "Apply vocal isolation to",
        ["Reference audio only", "Reference and vocal takes"],
        horizontal=True,
        disabled=backend == "none",
        help="Use reference-only when your takes are clean vocals. Use both when takes include backing audio.",
    )
    cols = st.columns(4)
    model = cols[0].selectbox(
        "Demucs model",
        ["htdemucs", "htdemucs_ft", "mdx_extra", "mdx_q"],
        disabled=backend == "none",
    )
    device = cols[1].selectbox("Device", ["cpu", "mps", "cuda"], disabled=backend == "none")
    shifts = cols[2].slider("Shifts", 1, 4, 1, disabled=backend == "none")
    overlap = cols[3].slider("Overlap", 0.10, 0.50, 0.25, 0.05, disabled=backend == "none")

    separation_config = {
        "backend": backend,
        "apply_to_reference": backend == "demucs",
        "apply_to_takes": backend == "demucs" and apply_scope == "Reference and vocal takes",
        "model": model,
        "device": device,
        "shifts": int(shifts),
        "overlap": float(overlap),
    }

    if backend == "demucs":
        st.info(
            "First Demucs run may download model weights and take a while. "
            "Separated stems are cached for later reruns."
        )

    signature = processing_signature(inputs, separation_config)
    stored_state = get_stored_processing_state(signature)
    can_prepare = inputs.current_take_path is not None

    if backend == "none":
        processing_state = default_processing_state(inputs, separation_config)
        st.success("Original audio is ready for analysis. No preprocessing stage is needed.")
    else:
        if stored_state is None:
            stale_state_notice(PROCESSING_STATE_KEY, "Prepared audio")
        if st.button(
            "Prepare Analysis Audio",
            type="primary",
            disabled=not can_prepare,
            help="Runs optional vocal isolation and stores the analysis files for scoring.",
        ):
            with st.status("Preparing analysis audio...", expanded=True) as status:
                st.write("Checking reference audio")
                reference_analysis_path, reference_result = maybe_prepare_analysis_audio(
                    inputs.reference_audio_path,
                    label="reference",
                    kind="reference",
                    separation_config=separation_config,
                )
                st.write("Checking current take")
                current_analysis_path, current_result = maybe_prepare_analysis_audio(
                    inputs.current_take_path,
                    label="current take",
                    kind="take",
                    separation_config=separation_config,
                )
                st.write("Checking optional previous take")
                previous_analysis_path, previous_result = maybe_prepare_analysis_audio(
                    inputs.previous_take_path,
                    label="previous take",
                    kind="take",
                    separation_config=separation_config,
                )
                separation_results = tuple(
                    result
                    for result in (reference_result, current_result, previous_result)
                    if result is not None
                )
                processing_state = ProcessingState(
                    reference_analysis_path=reference_analysis_path,
                    previous_analysis_path=previous_analysis_path,
                    current_analysis_path=current_analysis_path,
                    separation_results=separation_results,
                    separation_config=separation_config,
                )
                set_stored_processing_state(signature, processing_state)
                status.update(label="Analysis audio is ready.", state="complete")
        else:
            processing_state = stored_state or empty_processing_state(separation_config)
            if stored_state is None:
                st.info("Press **Prepare Analysis Audio** before scoring with Demucs stems.")

    st.subheader("Processing Preview")
    render_processing_preview(
        "Reference",
        original_path=inputs.reference_audio_path,
        analysis_path=processing_state.reference_analysis_path,
        result=_separation_result_for(processing_state, inputs.reference_audio_path),
    )
    render_processing_preview(
        "Current take",
        original_path=inputs.current_take_path,
        analysis_path=processing_state.current_analysis_path,
        result=_separation_result_for(processing_state, inputs.current_take_path),
    )
    render_processing_preview(
        "Previous take",
        original_path=inputs.previous_take_path,
        analysis_path=processing_state.previous_analysis_path,
        result=_separation_result_for(processing_state, inputs.previous_take_path),
    )
    return processing_state


def maybe_prepare_analysis_audio(
    path: Path | None,
    *,
    label: str,
    kind: str,
    separation_config: dict[str, object],
) -> tuple[Path | None, SeparationResult | None]:
    if path is None:
        return None, None
    should_separate = separation_config["backend"] == "demucs" and (
        (kind == "reference" and separation_config["apply_to_reference"])
        or (kind == "take" and separation_config["apply_to_takes"])
    )
    if not should_separate:
        return path, None

    with st.spinner(f"Preparing {label} vocal stem with Demucs..."):
        result = prepare_vocal_analysis_audio(
            path,
            cache_dir=STEM_CACHE_DIR,
            backend=str(separation_config["backend"]),
            model=str(separation_config["model"]),
            device=str(separation_config["device"]),
            shifts=int(separation_config["shifts"]),
            overlap=float(separation_config["overlap"]),
    )
    return result.analysis_path, result


def _separation_result_for(
    processing_state: ProcessingState,
    source_path: Path | None,
) -> SeparationResult | None:
    if source_path is None:
        return None
    for result in processing_state.separation_results:
        if Path(result.source_path) == Path(source_path):
            return result
    return None


def render_processing_preview(
    label: str,
    *,
    original_path: Path | None,
    analysis_path: Path | None,
    result: SeparationResult | None,
) -> None:
    with st.expander(f"{label} processing", expanded=result is not None):
        status = processing_status(result)
        st.write(f"**Status:** {status}")
        cols = st.columns(2)
        with cols[0]:
            st.write("Original audio")
            if original_path is not None:
                st.audio(str(original_path))
            else:
                st.caption("No original audio for this slot.")
        with cols[1]:
            st.write("Audio analyzed by pYIN")
            if analysis_path is not None:
                st.audio(str(analysis_path))
            else:
                st.caption("No analysis audio for this slot.")
        if result is not None:
            render_warnings(result.warnings)
            with st.expander("Processing metadata", expanded=False):
                st.json(result.to_dict())
                if result.debug_output:
                    st.text_area(
                        "Raw Demucs output",
                        value=result.debug_output,
                        height=180,
                        disabled=True,
                    )


def processing_status(result: SeparationResult | None) -> str:
    if result is None:
        return "Original audio"
    if result.used_original and result.warnings:
        return "Fallback to original audio"
    if result.used_original:
        return "Original audio"
    if result.used_cache:
        return "Cached Demucs vocal stem"
    return "New Demucs vocal stem"


def render_step_3_hyperparameters(separation_config: dict[str, object]) -> dict[str, object]:
    st.caption("Changing these values affects the next explicit evaluation or matching run.")

    pitch_tab, clean_tab, reference_tab, match_tab, score_tab = st.tabs(
        ["Pitch detection", "Cleaning", "Reference extraction", "Matching", "Scoring"]
    )

    with pitch_tab:
        pitch_cols = st.columns(2)
        fmin_hz = pitch_cols[0].number_input(
            "Minimum pitch (Hz)",
            min_value=40.0,
            max_value=500.0,
            value=80.0,
            step=5.0,
            help="Lowest pitch pYIN will try to detect.",
        )
        fmax_hz = pitch_cols[1].number_input(
            "Maximum pitch (Hz)",
            min_value=120.0,
            max_value=2000.0,
            value=1000.0,
            step=10.0,
            help="Highest pitch pYIN will try to detect.",
        )
        frame_length = pitch_cols[0].select_slider(
            "Frame length",
            options=[512, 1024, 2048, 4096],
            value=2048,
            help="Audio window size per pitch estimate.",
        )
        hop_length = pitch_cols[1].select_slider(
            "Hop length",
            options=[128, 256, 512, 1024],
            value=256,
            help="Step size between pitch estimates.",
        )

    with clean_tab:
        clean_cols = st.columns(2)
        min_confidence = clean_cols[0].slider(
            "Minimum pitch confidence",
            min_value=0.0,
            max_value=0.95,
            value=0.25,
            step=0.05,
        )
        max_jump_cents = clean_cols[1].slider(
            "Maximum pitch jump (cents)",
            min_value=100.0,
            max_value=2400.0,
            value=700.0,
            step=50.0,
        )
        correct_octaves = st.checkbox("Correct local octave jumps", value=True)

    with reference_tab:
        reference_window_s = st.slider(
            "Reference baseline window (seconds)",
            min_value=0.05,
            max_value=1.00,
            value=0.20,
            step=0.05,
            help="Only affects the editable symbolic preview exported from reference audio.",
        )

    with match_tab:
        match_cols = st.columns(2)
        alignment_search_radius_s = match_cols[0].slider(
            "Symbolic alignment search radius (seconds)",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.1,
        )
        alignment_step_s = match_cols[1].slider(
            "Symbolic alignment step (seconds)",
            min_value=0.01,
            max_value=0.20,
            value=0.02,
            step=0.01,
        )
        dtw_time_weight = match_cols[0].slider(
            "DTW time weight",
            min_value=0.0,
            max_value=60.0,
            value=20.0,
            step=1.0,
        )
        dtw_band_radius = match_cols[1].slider(
            "DTW band radius",
            min_value=0.01,
            max_value=0.50,
            value=0.06,
            step=0.01,
        )
        max_dtw_frames = st.slider(
            "Maximum DTW frames",
            min_value=300,
            max_value=5000,
            value=2400,
            step=100,
        )

    with score_tab:
        score_cols = st.columns(2)
        note_coverage_min_ratio = score_cols[0].slider(
            "Note coverage threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.05,
        )
        transposition_warning_cents = score_cols[1].slider(
            "Transposition warning threshold (cents)",
            min_value=25.0,
            max_value=300.0,
            value=90.0,
            step=5.0,
        )
        pitch_error_penalty = score_cols[0].slider(
            "Pitch error penalty",
            min_value=0.10,
            max_value=2.00,
            value=0.70,
            step=0.05,
        )
        stability_penalty = score_cols[1].slider(
            "Stability penalty",
            min_value=0.10,
            max_value=3.00,
            value=1.10,
            step=0.05,
        )
        symbolic_timing_penalty = score_cols[0].slider(
            "Symbolic timing penalty",
            min_value=10.0,
            max_value=400.0,
            value=180.0,
            step=10.0,
        )
        contour_timing_penalty = score_cols[1].slider(
            "Contour timing penalty",
            min_value=10.0,
            max_value=240.0,
            value=90.0,
            step=5.0,
        )

    if fmax_hz <= fmin_hz:
        st.error("Maximum pitch must be higher than minimum pitch.")
        st.stop()

    pitch_kwargs = {
        "fmin_hz": fmin_hz,
        "fmax_hz": fmax_hz,
        "frame_length": int(frame_length),
        "hop_length": int(hop_length),
    }
    clean_kwargs = {
        "min_confidence": min_confidence,
        "max_jump_cents": max_jump_cents,
        "correct_octaves": correct_octaves,
    }
    settings = {
        "pitch": pitch_kwargs,
        "cleaning": clean_kwargs,
        "reference": {"window_s": reference_window_s},
        "vocal_isolation": separation_config,
        "symbolic_matching": {
            "search_radius_s": alignment_search_radius_s,
            "step_s": alignment_step_s,
            "note_coverage_min_ratio": note_coverage_min_ratio,
        },
        "contour_matching": {
            "dtw_time_weight": dtw_time_weight,
            "dtw_band_radius": dtw_band_radius,
            "max_dtw_frames": max_dtw_frames,
        },
        "scoring": {
            "pitch_error_penalty": pitch_error_penalty,
            "stability_penalty": stability_penalty,
            "symbolic_timing_penalty": symbolic_timing_penalty,
            "contour_timing_penalty": contour_timing_penalty,
            "transposition_warning_cents": transposition_warning_cents,
        },
    }

    with st.expander("Current settings", expanded=False):
        st.json(settings)
        st.download_button(
            "Download settings JSON",
            data=json.dumps(settings, indent=2).encode("utf-8"),
            file_name="konopro_analysis_settings.json",
            mime="application/json",
        )

    return {
        "pitch_kwargs": pitch_kwargs,
        "clean_kwargs": clean_kwargs,
        "reference_window_s": reference_window_s,
        "symbolic_score_kwargs": {
            "pitch_kwargs": pitch_kwargs,
            "clean_kwargs": clean_kwargs,
            "alignment_kwargs": {
                "search_radius_s": alignment_search_radius_s,
                "step_s": alignment_step_s,
            },
            "note_coverage_min_ratio": note_coverage_min_ratio,
            "pitch_error_penalty": pitch_error_penalty,
            "stability_penalty": stability_penalty,
            "timing_offset_penalty": symbolic_timing_penalty,
            "transposition_warning_cents": transposition_warning_cents,
        },
        "contour_score_kwargs": {
            "pitch_kwargs": pitch_kwargs,
            "clean_kwargs": clean_kwargs,
            "dtw_time_weight": dtw_time_weight,
            "dtw_band_radius": dtw_band_radius,
            "max_dtw_frames": int(max_dtw_frames),
            "pitch_error_penalty": pitch_error_penalty,
            "stability_penalty": stability_penalty,
            "timing_penalty": contour_timing_penalty,
            "transposition_warning_cents": transposition_warning_cents,
        },
        "settings": settings,
    }


def build_verification_state(
    inputs: InputState,
    processing: ProcessingState,
    analysis_params: dict[str, object],
    *,
    log: Callable[[str], None] | None = None,
) -> VerificationState | None:
    if inputs.current_take_path is None or processing.current_analysis_path is None:
        st.info("Upload or select a current take before verification can run.")
        return None
    if inputs.scenario["baseline"] == "upload_reference_audio" and processing.reference_analysis_path is None:
        st.info("Upload reference audio before verification can run.")
        return None
    if inputs.scenario["baseline"] == "upload_csv" and inputs.baseline is None:
        st.info("Upload a valid baseline CSV before verification can run.")
        return None

    try:
        reference_extraction = None
        reference_source_duration = None
        if inputs.scenario["baseline"] == "upload_reference_audio":
            if log:
                log("Extracting pitch contour from uploaded reference audio")
            reference_extraction = extract_reference_audio(
                processing.reference_analysis_path,
                title=processing.reference_analysis_path.name,
                window_s=analysis_params["reference_window_s"],
                pitch_kwargs=analysis_params["pitch_kwargs"],
                clean_kwargs=analysis_params["clean_kwargs"],
            )
            baseline = reference_extraction.baseline
            reference_source_duration = reference_extraction.audio_summary.duration_s
        else:
            if log:
                log("Using symbolic reference baseline")
            baseline = inputs.baseline

        if baseline is None:
            st.error("No reference baseline is available.")
            return None

        if log:
            log("Summarizing analysis audio and checking durations")
        baseline_quality = analyze_baseline_quality(
            baseline,
            source_duration_s=reference_source_duration,
        )
        current_summary = summarize_audio(processing.current_analysis_path)
        previous_summary = (
            summarize_audio(processing.previous_analysis_path)
            if processing.previous_analysis_path is not None
            else None
        )

        if previous_summary is not None:
            duration_warnings = duration_mismatch_warnings(
                baseline.duration_s,
                previous_summary.duration_s,
                current_summary.duration_s,
            )
        else:
            duration_warnings = single_take_duration_warnings(
                baseline.duration_s,
                current_summary.duration_s,
            )

        comparison = None
        if log:
            log("Scoring current take against the reference")
        if processing.previous_analysis_path is None:
            if reference_extraction is not None:
                current_score = score_take_against_reference_contour(
                    processing.current_analysis_path,
                    reference_extraction.contour,
                    name="current",
                    **analysis_params["contour_score_kwargs"],
                )
            else:
                current_score = score_take(
                    processing.current_analysis_path,
                    baseline,
                    name="current",
                    **analysis_params["symbolic_score_kwargs"],
                )
        elif reference_extraction is not None:
            comparison = compare_takes_to_reference_contour(
                processing.previous_analysis_path,
                processing.current_analysis_path,
                reference_extraction.contour,
                **analysis_params["contour_score_kwargs"],
            )
            current_score = comparison.current
        else:
            comparison = compare_takes(
                processing.previous_analysis_path,
                processing.current_analysis_path,
                baseline,
                **analysis_params["symbolic_score_kwargs"],
            )
            current_score = comparison.current

        if log:
            log("Extracting pitch contours for plots")
        current_audio, sample_rate = load_audio(processing.current_analysis_path)
        current_contour = clean_pitch_contour(
            extract_pitch(
                current_audio,
                sample_rate,
                name="current",
                **analysis_params["pitch_kwargs"],
            ),
            **analysis_params["clean_kwargs"],
        )
        previous_contour = None
        if processing.previous_analysis_path is not None:
            previous_audio, _ = load_audio(processing.previous_analysis_path, target_sr=sample_rate)
            previous_contour = clean_pitch_contour(
                extract_pitch(
                    previous_audio,
                    sample_rate,
                    name="previous",
                    **analysis_params["pitch_kwargs"],
                ),
                **analysis_params["clean_kwargs"],
            )

        return VerificationState(
            baseline=baseline,
            reference_extraction=reference_extraction,
            baseline_quality=baseline_quality,
            current_summary=current_summary,
            previous_summary=previous_summary,
            duration_warnings=duration_warnings,
            current_score=current_score,
            comparison=comparison,
            current_contour=current_contour,
            previous_contour=previous_contour,
        )
    except Exception as exc:
        st.error(f"Could not verify the takes: {exc}")
        return None


def single_take_duration_warnings(
    reference_duration_s: float,
    current_duration_s: float,
) -> tuple[str, ...]:
    if reference_duration_s <= 0 or current_duration_s <= 0:
        return ()
    diff = abs(current_duration_s - reference_duration_s)
    if diff > 2.0 and diff / max(reference_duration_s, 0.001) > 0.25:
        return (f"current duration differs from reference by {diff:.1f}s; trim files to the same section",)
    return ()


def render_evaluation_runner(
    inputs: InputState,
    processing: ProcessingState,
    analysis_params: dict[str, object],
) -> VerificationState | None:
    signature = verification_signature(inputs, processing, analysis_params)
    stored_state = get_stored_verification_state(signature)
    can_run = processing.current_analysis_path is not None
    if inputs.scenario["baseline"] == "upload_reference_audio" and processing.reference_analysis_path is None:
        can_run = False
    if inputs.scenario["baseline"] == "upload_csv" and inputs.baseline is None:
        can_run = False

    if stored_state is None:
        stale_state_notice(VERIFICATION_STATE_KEY, "Evaluation result")

    if st.button(
        "Run Evaluation",
        type="primary",
        disabled=not can_run,
        help="Extracts pitch, scores the current take, and computes progress if a previous take exists.",
    ):
        with st.status("Running singing evaluation...", expanded=True) as status:
            state = build_verification_state(
                inputs,
                processing,
                analysis_params,
                log=st.write,
            )
            if state is None:
                status.update(label="Evaluation could not finish.", state="error")
            else:
                set_stored_verification_state(signature, state)
                stored_state = state
                status.update(label="Evaluation complete.", state="complete")

    if not can_run:
        st.info("Prepare analysis audio and provide the required reference/current files before running.")
    elif stored_state is None:
        st.info("Press **Run Evaluation** to score the current setup.")
    return stored_state


def render_step_4_verify_results(state: VerificationState | None) -> None:
    if state is None:
        st.info("Complete the required inputs above to see scores and plots.")
        return

    st.subheader("Score Summary")
    comparison = state.comparison
    current_score = state.current_score
    cols = st.columns(5)
    render_metric(
        cols[0],
        "Overall",
        current_score.overall_score,
        comparison.overall_delta if comparison is not None else None,
        "Composite score from pitch accuracy, stability, coverage, and timing.",
    )
    render_metric(
        cols[1],
        "Pitch Accuracy",
        current_score.pitch_accuracy_score,
        comparison.pitch_accuracy_delta if comparison is not None else None,
    )
    render_metric(
        cols[2],
        "Stability",
        current_score.stability_score,
        comparison.stability_delta if comparison is not None else None,
    )
    render_metric(
        cols[3],
        "Coverage",
        current_score.coverage_score,
        comparison.coverage_delta if comparison is not None else None,
    )
    render_metric(
        cols[4],
        "Timing",
        current_score.timing_score,
        comparison.timing_delta if comparison is not None else None,
    )

    render_interpretation(state)
    render_trust_checklist(state)

    st.subheader("Reference Baseline")
    quality = state.baseline_quality
    quality_cols = st.columns(4)
    quality_cols[0].metric("Quality", quality.level)
    quality_cols[1].metric("Notes", quality.note_count)
    quality_cols[2].metric("Coverage", f"{quality.voiced_coverage_ratio * 100:.1f}%")
    quality_cols[3].metric("Duration", f"{quality.duration_s:.1f}s")
    render_warnings(quality.warnings)

    if state.reference_extraction is not None:
        st.subheader("Extracted Reference Preview")
        st.pyplot(
            plot_reference_extraction(
                state.reference_extraction.baseline,
                state.reference_extraction.contour,
            ),
            clear_figure=True,
        )

    with st.expander("Inspect or correct baseline", expanded=False):
        edited_rows = st.data_editor(
            pd.DataFrame(baseline_to_rows(state.baseline)),
            num_rows="dynamic",
            width="stretch",
            key="verification_baseline_editor",
        )
        try:
            edited_baseline = baseline_from_rows(
                edited_rows.to_dict("records"),
                title=f"{state.baseline.title} (edited)",
            )
        except Exception as exc:
            st.error(f"Baseline edit is invalid: {exc}")
        else:
            st.download_button(
                "Download baseline CSV",
                data=baseline_to_csv_text(edited_baseline).encode("utf-8"),
                file_name="konopro_baseline.csv",
                mime="text/csv",
            )

    st.subheader("Pitch Visualization")
    if state.reference_extraction is not None:
        st.caption("Uploaded reference contour vs detected pitch contours.")
        st.pyplot(
            plot_contour_comparison(
                state.reference_extraction.contour,
                state.previous_contour,
                state.current_contour,
            ),
            clear_figure=True,
        )
    else:
        st.caption("Reference melody vs detected pitch contours.")
        st.pyplot(
            plot_take_comparison(
                state.baseline,
                state.previous_contour,
                state.current_contour,
            ),
            clear_figure=True,
        )

    st.subheader("Voiced-Frame Coverage")
    st.caption("Shows where the reference expects notes and where each take has detected voiced frames.")
    if state.reference_extraction is not None:
        st.pyplot(
            plot_contour_voiced_coverage(
                state.reference_extraction.contour,
                state.previous_contour,
                state.current_contour,
            ),
            clear_figure=True,
        )
    else:
        st.pyplot(
            plot_voiced_coverage(state.baseline, state.previous_contour, state.current_contour),
            clear_figure=True,
        )

    with st.expander("Detailed metrics", expanded=False):
        if comparison is not None:
            st.json(comparison.to_dict())
        else:
            st.json(current_score.to_dict())


def render_interpretation(state: VerificationState) -> None:
    comparison = state.comparison
    current_score = state.current_score

    with st.expander("Interpretation", expanded=True):
        if comparison is not None:
            st.write(f"Verdict: **{comparison.verdict}**")
            for category, messages in comparison.feedback_by_category.items():
                label = {
                    "song_correctness": "Reference Match",
                    "technical_control": "Vocal Stability",
                    "recording_confidence": "Confidence",
                    "summary": "Summary",
                }.get(category, category.replace("_", " ").title())
                st.write(f"**{label}**")
                for message in messages:
                    st.write(f"- {message}")
            render_warnings(comparison.previous.warnings)
        else:
            st.caption("Add a previous take to compute progress deltas and an improved/declined verdict.")

        render_warnings(current_score.warnings)
        render_warnings(state.duration_warnings)

        if state.reference_extraction is not None:
            st.markdown(
                """
- **Overall** = `50% pitch accuracy + 20% stability + 15% coverage + 15% timing`.
- **Pitch accuracy** compares take contour to reference contour using DTW.
- **Stability** measures consistency of pitch deviations after contour matching.
- **Coverage** compares detected voiced-frame amount against reference voiced-frame amount.
                """.strip()
            )
        else:
            st.markdown(
                """
- **Overall** = `45% pitch accuracy + 20% stability + 20% coverage + 15% timing`.
- **Pitch accuracy** measures cents error against the reference melody.
- **Stability** measures how steady pitch is inside covered notes.
- **Coverage** measures how much of the expected melody was attempted.
                """.strip()
            )


def render_trust_checklist(state: VerificationState) -> None:
    enough_reference = state.baseline_quality.voiced_coverage_ratio >= 0.35
    enough_take = state.current_score.coverage_score >= 50.0
    no_duration_warnings = not state.duration_warnings
    reference_check = "Manual check" if state.reference_extraction is not None else "Symbolic baseline"

    rows = [
        ("Same song section", no_duration_warnings),
        ("Reference contour follows melody", reference_check),
        ("Enough reference voiced coverage", enough_reference),
        ("Enough current-take coverage", enough_take),
        ("No major duration mismatch", no_duration_warnings),
    ]

    with st.expander("Trust checklist", expanded=True):
        for label, value in rows:
            if value is True:
                st.success(f"{label}: pass")
            elif value is False:
                st.warning(f"{label}: check before trusting score")
            else:
                st.info(f"{label}: {value}")


def render_matching_prototype(
    inputs: InputState,
    processing: ProcessingState,
    analysis_params: dict[str, object],
    verification: VerificationState | None,
) -> None:
    st.caption(
        "Prototype B asks: what song or section does this recording look like before we score it?"
    )

    query_options = []
    if processing.current_analysis_path is not None:
        query_options.append("Use current take from Prototype A")
    query_options.append("Upload separate matching query")
    query_source = st.radio(
        "Query recording",
        query_options,
        horizontal=True,
        key="matching_query_source",
    )
    query_path = processing.current_analysis_path if query_source.startswith("Use current") else None
    if query_source.startswith("Upload"):
        query_upload = st.file_uploader(
            "Matching query audio",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
            key="matching_query_upload",
        )
        query_path = (
            uploaded_file_to_cache(query_upload, role="matching_query")
            if query_upload is not None
            else None
        )
        if query_path is not None:
            st.audio(str(query_path))

    catalog_options = ["Demo song catalog"]
    if verification is not None and verification.reference_extraction is not None:
        catalog_options.append("Uploaded reference split into sections")
    if inputs.baseline is not None:
        catalog_options.append("Active baseline as one section")
    catalog_source = st.radio(
        "Catalog to search",
        catalog_options,
        horizontal=True,
        key="matching_catalog_source",
    )

    st.write("**Matching controls**")
    cols = st.columns(4)
    section_window_s = cols[0].slider(
        "Reference section length",
        min_value=3.0,
        max_value=45.0,
        value=20.0,
        step=1.0,
        help="Only used when splitting uploaded reference audio into searchable sections.",
    )
    section_hop_s = cols[1].slider(
        "Reference section hop",
        min_value=1.0,
        max_value=20.0,
        value=10.0,
        step=1.0,
    )
    query_hop_s = cols[2].slider(
        "Query search hop",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help="For long recordings, scan the query in windows and keep the best match.",
    )
    top_k = cols[3].slider("Top matches", min_value=1, max_value=10, value=5, step=1)

    advanced_cols = st.columns(4)
    sample_count = advanced_cols[0].slider(
        "Shape samples",
        min_value=32,
        max_value=256,
        value=96,
        step=16,
        help="Number of normalized melody-shape samples compared per phrase.",
    )
    min_voiced_frames = advanced_cols[1].slider(
        "Minimum section voiced frames",
        min_value=4,
        max_value=120,
        value=24,
        step=4,
    )
    shape_error_penalty = advanced_cols[2].slider(
        "Shape error penalty",
        min_value=0.10,
        max_value=1.00,
        value=0.35,
        step=0.05,
    )
    transpose_invariant = advanced_cols[3].checkbox(
        "Ignore key/transposition",
        value=True,
        help="Subtracts each phrase median pitch before matching, useful for covers and different keys.",
    )

    matching_settings = {
        "catalog_source": catalog_source,
        "section_window_s": float(section_window_s),
        "section_hop_s": float(section_hop_s),
        "query_hop_s": float(query_hop_s),
        "top_k": int(top_k),
        "sample_count": int(sample_count),
        "min_voiced_frames": int(min_voiced_frames),
        "shape_error_penalty": float(shape_error_penalty),
        "transpose_invariant": bool(transpose_invariant),
    }

    if query_path is None:
        st.info("Choose or upload a query recording to run song/section matching.")
        return

    signature = stable_signature(
        {
            "inputs": input_signature(inputs),
            "query_path": str(query_path),
            "query_source": query_source,
            "matching": matching_settings,
            "analysis_settings": analysis_params["settings"],
            "processing_current": (
                str(processing.current_analysis_path) if processing.current_analysis_path else None
            ),
            "verification_available": verification is not None,
        }
    )
    stored_state = get_stored_matching_state(signature)
    if stored_state is None:
        stale_state_notice(MATCHING_STATE_KEY, "Matching result")

    if st.button(
        "Run Song/Section Matching",
        type="primary",
        help="Extracts the query contour, builds the section catalog, ranks matches, then scores the best phrase.",
    ):
        with st.status("Running song/section matching...", expanded=True) as status:
            try:
                st.write("Extracting query pitch contour")
                query_contour = _matching_query_contour(
                    query_path,
                    processing=processing,
                    verification=verification,
                    analysis_params=analysis_params,
                    query_source=query_source,
                )
                st.write("Building searchable section catalog")
                sections = _matching_catalog(
                    catalog_source,
                    inputs=inputs,
                    verification=verification,
                    section_window_s=section_window_s,
                    section_hop_s=section_hop_s,
                    min_voiced_frames=min_voiced_frames,
                )
                if not sections:
                    st.warning("No catalog sections are available for this matching mode.")
                    status.update(label="Matching could not run.", state="error")
                    return

                st.write(f"Comparing query against {len(sections)} section(s)")
                result = match_query_to_sections(
                    query_contour,
                    sections,
                    top_k=int(top_k),
                    sample_count=int(sample_count),
                    query_hop_s=float(query_hop_s),
                    transpose_invariant=transpose_invariant,
                    shape_error_penalty=float(shape_error_penalty),
                )

                handoff_score = None
                if result.best is not None:
                    st.write("Scoring the best matched query window")
                    query_window = crop_contour(
                        result.query,
                        result.best.query_start_s,
                        result.best.query_end_s,
                        name="matched query window",
                        shift_to_zero=True,
                    )
                    handoff_score = score_take_against_reference_contour(
                        query_window,
                        result.best.section.contour,
                        name="matched query",
                        **analysis_params["contour_score_kwargs"],
                    )

                stored_state = MatchingRunState(
                    result=result,
                    handoff_score=handoff_score,
                    sections_count=len(sections),
                    settings=matching_settings,
                )
                set_stored_matching_state(signature, stored_state)
                status.update(label="Song/section matching complete.", state="complete")
            except Exception as exc:
                st.error(f"Could not run song/section matching: {exc}")
                status.update(label="Matching failed.", state="error")
                return

    if stored_state is None:
        st.info("Press **Run Song/Section Matching** to identify the likely song section.")
        return

    render_matching_results(stored_state)


def render_matching_results(state: MatchingRunState) -> None:
    result = state.result

    st.subheader("Matching Results")
    st.caption(f"Compared against {state.sections_count} catalog section(s).")
    render_warnings(result.warnings)
    if not result.candidates:
        st.info("No match candidates passed the voiced-frame checks.")
        return

    st.dataframe(
        pd.DataFrame([candidate.to_dict() for candidate in result.candidates]),
        width="stretch",
        hide_index=True,
    )

    best = result.best
    if best is None:
        return

    st.subheader("Best Match")
    match_cols = st.columns(5)
    match_cols[0].metric("Match", f"{best.score:.1f}")
    match_cols[1].metric("Shape", f"{best.shape_score:.1f}")
    match_cols[2].metric("Coverage", f"{best.coverage_score:.1f}")
    match_cols[3].metric("Duration Fit", f"{best.duration_score:.1f}")
    match_cols[4].metric("Shape Error", f"{best.mean_shape_error_cents:.0f} cents")
    st.write(
        f"**{best.section.display_name}** "
        f"({best.section.start_s:.1f}s-{best.section.end_s:.1f}s reference, "
        f"{best.query_start_s:.1f}s-{best.query_end_s:.1f}s query)"
    )
    render_warnings(best.warnings)
    st.pyplot(
        plot_section_match(
            best.section.contour,
            result.query,
            query_start_s=best.query_start_s,
            query_end_s=best.query_end_s,
        ),
        clear_figure=True,
    )

    st.subheader("Prototype Handoff Score")
    st.caption("This scores the matched query window against the best matched section contour.")
    handoff_score = state.handoff_score
    if handoff_score is None:
        st.info("No handoff score is available because no best match was found.")
        return
    handoff_cols = st.columns(4)
    handoff_cols[0].metric("Overall", f"{handoff_score.overall_score:.1f}")
    handoff_cols[1].metric("Pitch Accuracy", f"{handoff_score.pitch_accuracy_score:.1f}")
    handoff_cols[2].metric("Stability", f"{handoff_score.stability_score:.1f}")
    handoff_cols[3].metric("Timing", f"{handoff_score.timing_score:.1f}")
    render_warnings(handoff_score.warnings)


def render_pipeline_timeline(
    inputs: InputState,
    processing: ProcessingState,
    verification: VerificationState | None,
) -> None:
    reference_ready = inputs.baseline is not None or inputs.reference_audio_path is not None
    import_ready = reference_ready and inputs.current_take_path is not None
    processing_ready = processing.current_analysis_path is not None
    processing_detail = (
        "original audio"
        if processing.separation_config["backend"] == "none"
        else ("prepared stems" if processing_ready else "needs Prepare Analysis Audio")
    )
    rows = [
        ("A1 Import", "ready" if import_ready else "waiting", "files selected"),
        ("A2 Prepare", "ready" if processing_ready else "waiting", processing_detail),
        ("A3 Tune", "ready", "settings selected"),
        ("A4 Evaluate", "ready" if verification is not None else "waiting", "stored result"),
    ]
    cols = st.columns(len(rows))
    for col, (label, state, detail) in zip(cols, rows, strict=True):
        message = f"**{label}**\n\n{detail}"
        if state == "ready":
            col.success(message)
        else:
            col.info(message)


def _matching_query_contour(
    query_path: Path,
    *,
    processing: ProcessingState,
    verification: VerificationState | None,
    analysis_params: dict[str, object],
    query_source: str,
):
    if (
        query_source.startswith("Use current")
        and verification is not None
        and processing.current_analysis_path == query_path
    ):
        return verification.current_contour
    return extract_matching_query(
        query_path,
        name="matching query",
        pitch_kwargs=analysis_params["pitch_kwargs"],
        clean_kwargs=analysis_params["clean_kwargs"],
    )


def _matching_catalog(
    catalog_source: str,
    *,
    inputs: InputState,
    verification: VerificationState | None,
    section_window_s: float,
    section_hop_s: float,
    min_voiced_frames: int,
):
    if catalog_source == "Demo song catalog":
        return build_demo_section_catalog()

    if catalog_source == "Active baseline as one section" and inputs.baseline is not None:
        return sections_from_baseline(inputs.baseline)

    if (
        catalog_source == "Uploaded reference split into sections"
        and verification is not None
        and verification.reference_extraction is not None
    ):
        title = (
            inputs.reference_audio_path.stem
            if inputs.reference_audio_path is not None
            else "Uploaded reference"
        )
        return split_contour_into_sections(
            verification.reference_extraction.contour,
            song_id="uploaded_reference",
            song_title=title,
            window_s=section_window_s,
            hop_s=section_hop_s,
            min_voiced_frames=int(min_voiced_frames),
        )

    return ()


def store_testing_state(
    inputs: InputState,
    processing: ProcessingState,
    analysis_params: dict[str, object],
) -> None:
    st.session_state["konopro_testing_state"] = {
        "scenario_name": inputs.scenario_name,
        "reference_audio_path": str(inputs.reference_audio_path) if inputs.reference_audio_path else None,
        "previous_take_path": str(inputs.previous_take_path) if inputs.previous_take_path else None,
        "current_take_path": str(inputs.current_take_path) if inputs.current_take_path else None,
        "reference_analysis_path": (
            str(processing.reference_analysis_path) if processing.reference_analysis_path else None
        ),
        "previous_analysis_path": (
            str(processing.previous_analysis_path) if processing.previous_analysis_path else None
        ),
        "current_analysis_path": (
            str(processing.current_analysis_path) if processing.current_analysis_path else None
        ),
        "settings": analysis_params["settings"],
    }


st.title("Konopro Singing Research Prototypes")
st.caption(
    "Prototype A evaluates singing against a known section. "
    "Prototype B tries to identify the song/section before evaluation."
)
apply_step_card_styles()

with st.expander("Recommended test flow", expanded=False):
    st.markdown(
        """
1. Use Prototype A when the reference section is already known.
2. Use Prototype B when you need to find the likely song/section first.
3. Press each stage button explicitly; changing settings does not auto-score.
4. Try original audio first; use Demucs when mixed tracks confuse pitch tracking.
5. Trust scores only after the plots and checklist look sane.
        """.strip()
    )

st.header("Prototype A: Singing Evaluation")
st.caption("Known reference section -> vocal take -> score and progress delta.")

with st.expander("A1. Import & Preview Files", expanded=True):
    input_state = render_step_1_import_files()

with st.expander("A2. Prepare Audio With Optional Demucs", expanded=False):
    processing_state = render_step_2_prepare_audio(input_state)

with st.expander("A3. Tune Analysis Hyperparameters", expanded=False):
    analysis_state = render_step_3_hyperparameters(processing_state.separation_config)

store_testing_state(input_state, processing_state, analysis_state)
verification_state = get_stored_verification_state(
    verification_signature(input_state, processing_state, analysis_state)
)

with st.expander("Pipeline Status", expanded=True):
    st.caption("Stages now run only when you press the explicit stage button.")
    render_pipeline_timeline(input_state, processing_state, verification_state)

with st.expander("A4. Verify Results", expanded=False):
    verification_state = render_evaluation_runner(input_state, processing_state, analysis_state)
    render_step_4_verify_results(verification_state)

st.header("Prototype B: Song & Section Matching")
st.caption("Unknown or long recording -> likely song/section -> handoff score against that section.")

with st.expander("B1. Match Song & Section", expanded=True):
    render_matching_prototype(input_state, processing_state, analysis_state, verification_state)
