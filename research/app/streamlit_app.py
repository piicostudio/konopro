from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = RESEARCH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from konopro_research.audio_io import load_audio  # noqa: E402
from konopro_research.baseline import (  # noqa: E402
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
from konopro_research.pitch import clean_pitch_contour, extract_pitch  # noqa: E402
from konopro_research.plots import (  # noqa: E402
    plot_contour_comparison,
    plot_contour_voiced_coverage,
    plot_reference_extraction,
    plot_take_comparison,
    plot_voiced_coverage,
)
from konopro_research.quality import (  # noqa: E402
    analyze_baseline_quality,
    duration_mismatch_warnings,
    summarize_audio,
)
from konopro_research.reference_audio import extract_reference_audio  # noqa: E402
from konopro_research.scoring import compare_takes, score_take  # noqa: E402


st.set_page_config(page_title="Konopro Research Demo", layout="wide")
st.title("Konopro Singing Progress Research Demo")

demo_paths = ensure_demo_data(RESEARCH_ROOT / "data" / "demo")

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

with st.sidebar:
    st.header("Demo Scenarios")
    scenario_name = st.radio(
        "Choose a feature to demo",
        list(DEMO_SCENARIOS),
        label_visibility="collapsed",
    )

scenario = DEMO_SCENARIOS[scenario_name]
st.caption(scenario["description"])

with st.expander("Recommended test flow", expanded=False):
    st.markdown(
        """
1. Start with **Progress: cleaner current take**.
2. Switch to **Stable but wrong** and confirm pitch accuracy drops.
3. For real tests, trim reference/current, and previous if used, to the same 20-40 second phrase.
4. Prefer a guide vocal or melody-only reference over a full mixed song.
5. Check the extracted reference plot before trusting any score.
        """.strip()
    )


def render_analysis_hyperparameters() -> dict[str, object]:
    with st.expander("Live analysis hyperparameters", expanded=False):
        st.caption("Changing these values reruns reference extraction, scoring, and the graphs.")
        pitch_tab, clean_tab, reference_tab, match_tab, score_tab = st.tabs(
            ["Pitch", "Cleaning", "Reference", "Matching", "Scoring"]
        )

        with pitch_tab:
            pitch_cols = st.columns(2)
            fmin_hz = pitch_cols[0].number_input(
                "Minimum pitch (Hz)",
                min_value=40.0,
                max_value=500.0,
                value=80.0,
                step=5.0,
                help="Lowest pitch pYIN will try to detect. Raise this to ignore low instruments/noise.",
            )
            fmax_hz = pitch_cols[1].number_input(
                "Maximum pitch (Hz)",
                min_value=120.0,
                max_value=2000.0,
                value=1000.0,
                step=10.0,
                help="Highest pitch pYIN will try to detect. Lower this to ignore high harmonics.",
            )
            frame_length = pitch_cols[0].select_slider(
                "Frame length",
                options=[512, 1024, 2048, 4096],
                value=2048,
                help="Audio window size per pitch estimate. Larger is smoother but less responsive.",
            )
            hop_length = pitch_cols[1].select_slider(
                "Hop length",
                options=[128, 256, 512, 1024],
                value=256,
                help="Step size between pitch estimates. Smaller gives denser graphs but costs more.",
            )

        with clean_tab:
            clean_cols = st.columns(2)
            min_confidence = clean_cols[0].slider(
                "Minimum pitch confidence",
                min_value=0.0,
                max_value=0.95,
                value=0.25,
                step=0.05,
                help="Frames below this pYIN voiced probability are removed from contours.",
            )
            max_jump_cents = clean_cols[1].slider(
                "Maximum pitch jump (cents)",
                min_value=100.0,
                max_value=2400.0,
                value=700.0,
                step=50.0,
                help="Large frame-to-frame pitch jumps above this are treated as tracking artifacts.",
            )
            correct_octaves = st.checkbox(
                "Correct local octave jumps",
                value=True,
                help="Try to fold obvious local octave errors back near the surrounding pitch.",
            )

        with reference_tab:
            reference_window_s = st.slider(
                "Reference baseline window (seconds)",
                min_value=0.05,
                max_value=1.00,
                value=0.20,
                step=0.05,
                help="Only affects the editable symbolic preview exported from uploaded reference audio.",
            )

        with match_tab:
            match_cols = st.columns(2)
            alignment_search_radius_s = match_cols[0].slider(
                "Symbolic alignment search radius (seconds)",
                min_value=0.0,
                max_value=3.0,
                value=0.5,
                step=0.1,
                help="How far symbolic scoring may shift a take earlier/later before comparing.",
            )
            alignment_step_s = match_cols[1].slider(
                "Symbolic alignment step (seconds)",
                min_value=0.01,
                max_value=0.20,
                value=0.02,
                step=0.01,
                help="Resolution of the symbolic alignment search.",
            )
            dtw_time_weight = match_cols[0].slider(
                "DTW time weight",
                min_value=0.0,
                max_value=60.0,
                value=20.0,
                step=1.0,
                help="Higher values force contour matching to stay closer to the same song position.",
            )
            dtw_band_radius = match_cols[1].slider(
                "DTW band radius",
                min_value=0.01,
                max_value=0.50,
                value=0.06,
                step=0.01,
                help="Allowed DTW warping width. Higher values permit more timing flexibility.",
            )
            max_dtw_frames = st.slider(
                "Maximum DTW frames",
                min_value=300,
                max_value=5000,
                value=2400,
                step=100,
                help="Contours longer than this are thinned before DTW to keep the app responsive.",
            )

        with score_tab:
            score_cols = st.columns(2)
            note_coverage_min_ratio = score_cols[0].slider(
                "Note coverage threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.35,
                step=0.05,
                help="Symbolic scoring: fraction of a note that must contain voiced frames to count as covered.",
            )
            transposition_warning_cents = score_cols[1].slider(
                "Transposition warning threshold (cents)",
                min_value=25.0,
                max_value=300.0,
                value=90.0,
                step=5.0,
                help="Warn when the median pitch offset is at least this large.",
            )
            pitch_error_penalty = score_cols[0].slider(
                "Pitch error penalty",
                min_value=0.10,
                max_value=2.00,
                value=0.70,
                step=0.05,
                help="Score points lost per cent of mean pitch error.",
            )
            stability_penalty = score_cols[1].slider(
                "Stability penalty",
                min_value=0.10,
                max_value=3.00,
                value=1.10,
                step=0.05,
                help="Score points lost per cent of pitch instability.",
            )
            symbolic_timing_penalty = score_cols[0].slider(
                "Symbolic timing penalty",
                min_value=10.0,
                max_value=400.0,
                value=180.0,
                step=10.0,
                help="Score points lost per second of global timing offset in symbolic scoring.",
            )
            contour_timing_penalty = score_cols[1].slider(
                "Contour timing penalty",
                min_value=10.0,
                max_value=240.0,
                value=90.0,
                step=5.0,
                help="Score points lost per second of median DTW timing mismatch in contour scoring.",
            )

        if fmax_hz <= fmin_hz:
            st.error("Maximum pitch must be higher than minimum pitch.")
            st.stop()

        settings = {
            "pitch": {
                "fmin_hz": fmin_hz,
                "fmax_hz": fmax_hz,
                "frame_length": frame_length,
                "hop_length": hop_length,
            },
            "cleaning": {
                "min_confidence": min_confidence,
                "max_jump_cents": max_jump_cents,
                "correct_octaves": correct_octaves,
            },
            "reference": {
                "window_s": reference_window_s,
            },
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
        with st.expander("Current hyperparameter values", expanded=False):
            st.json(settings)

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


analysis_params = render_analysis_hyperparameters()


def uploaded_audio_to_temp(uploaded, suffix: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def render_warnings(warnings: tuple[str, ...] | list[str]) -> None:
    for warning in warnings:
        st.warning(warning)


def metric_delta(value: float) -> str | None:
    if abs(value) < 0.05:
        return None
    return f"{value:+.1f}"


def render_metric(column, label: str, value: float, delta: float | None = None, help_text: str | None = None) -> None:
    column.metric(
        label,
        f"{value:.1f}",
        metric_delta(delta) if delta is not None else None,
        delta_color="normal" if delta is None or abs(delta) >= 0.05 else "off",
        help=help_text,
    )


def demo_take_path(key: str) -> Path:
    return demo_paths[key]


def single_take_duration_warnings(reference_duration_s: float, current_duration_s: float) -> tuple[str, ...]:
    if reference_duration_s <= 0 or current_duration_s <= 0:
        return ()
    diff = abs(current_duration_s - reference_duration_s)
    if diff > 2.0 and diff / max(reference_duration_s, 0.001) > 0.25:
        return (f"current duration differs from reference by {diff:.1f}s; trim files to the same section",)
    return ()


reference_extraction = None
reference_source_duration = None
demo_reference_path = RESEARCH_ROOT / "data" / "demo" / "reference_melody.wav"
reference_audio_path: Path | None = demo_paths.get("reference", demo_reference_path)

try:
    if scenario["baseline"] == "demo_csv":
        baseline = load_baseline_csv(demo_paths["baseline"])
    elif scenario["baseline"] == "upload_csv":
        st.subheader("Scenario Inputs")
        upload = st.file_uploader("Baseline CSV", type=["csv"])
        optional_reference_upload = st.file_uploader(
            "Optional reference audio for playback",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
        )
        if not upload:
            st.info("Upload a baseline CSV with start_s, end_s, and midi columns.")
            st.stop()
        baseline = load_baseline_csv(io.StringIO(upload.getvalue().decode("utf-8")), title=upload.name)
        reference_audio_path = (
            uploaded_audio_to_temp(optional_reference_upload, Path(optional_reference_upload.name).suffix)
            if optional_reference_upload
            else None
        )
    else:
        st.subheader("Scenario Inputs")
        upload = st.file_uploader("Reference audio", type=["wav", "mp3", "m4a", "flac", "ogg"])
        if not upload:
            st.info("Upload a clear guide vocal or melody-only reference audio file.")
            st.stop()
        reference_audio_path = uploaded_audio_to_temp(upload, Path(upload.name).suffix)
        reference_extraction = extract_reference_audio(
            reference_audio_path,
            title=upload.name,
            window_s=analysis_params["reference_window_s"],
            pitch_kwargs=analysis_params["pitch_kwargs"],
            clean_kwargs=analysis_params["clean_kwargs"],
        )
        baseline = reference_extraction.baseline
        reference_source_duration = reference_extraction.audio_summary.duration_s
except Exception as exc:
    st.error(f"Could not load reference input: {exc}")
    st.stop()

try:
    previous_take: Path | None
    previous_summary = None
    if scenario["previous"] == "upload":
        current_upload = st.file_uploader(
            "Current vocal take",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
        )
        previous_upload = st.file_uploader(
            "Previous vocal take (optional)",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
        )
        if not current_upload:
            st.info("Upload a current vocal take. Add a previous take only when you want progress deltas.")
            st.stop()
        current_take = uploaded_audio_to_temp(current_upload, Path(current_upload.name).suffix)
        previous_take = (
            uploaded_audio_to_temp(previous_upload, Path(previous_upload.name).suffix)
            if previous_upload
            else None
        )
    else:
        previous_take = demo_take_path(scenario["previous"])
        current_take = demo_take_path(scenario["current"])

    if previous_take is not None:
        previous_summary = summarize_audio(previous_take)
    current_summary = summarize_audio(current_take)
except Exception as exc:
    st.error(f"Could not load vocal take: {exc}")
    st.stop()

st.subheader("Reference Baseline")
quality = analyze_baseline_quality(baseline, source_duration_s=reference_source_duration)
metric_cols = st.columns(4)
metric_cols[0].metric("Quality", quality.level)
metric_cols[1].metric("Notes", quality.note_count)
metric_cols[2].metric("Coverage", f"{quality.voiced_coverage_ratio * 100:.1f}%")
metric_cols[3].metric("Duration", f"{quality.duration_s:.1f}s")
render_warnings(quality.warnings)

if reference_extraction is not None:
    st.subheader("Extracted Reference Preview")
    st.pyplot(
        plot_reference_extraction(reference_extraction.baseline, reference_extraction.contour),
        clear_figure=True,
    )

with st.expander("Inspect or correct baseline", expanded=scenario["baseline"] != "demo_csv"):
    edited_rows = st.data_editor(
        pd.DataFrame(baseline_to_rows(baseline)),
        num_rows="dynamic",
        width="stretch",
        key=f"baseline_editor_{scenario_name}",
    )
    try:
        baseline = baseline_from_rows(edited_rows.to_dict("records"), title=f"{baseline.title} (edited)")
    except Exception as exc:
        st.error(f"Baseline edit is invalid: {exc}")
        st.stop()

    st.download_button(
        "Download baseline CSV",
        data=baseline_to_csv_text(baseline).encode("utf-8"),
        file_name="konopro_baseline.csv",
        mime="text/csv",
    )

if previous_summary is not None:
    duration_warnings = duration_mismatch_warnings(
        baseline.duration_s,
        previous_summary.duration_s,
        current_summary.duration_s,
    )
else:
    duration_warnings = single_take_duration_warnings(baseline.duration_s, current_summary.duration_s)
render_warnings(duration_warnings)

with st.expander("Audio input checks", expanded=bool(duration_warnings)):
    if previous_summary is not None:
        st.write("Previous take")
        st.json(previous_summary.to_dict())
    st.write("Current take")
    st.json(current_summary.to_dict())

try:
    comparison = None
    if previous_take is None:
        if reference_extraction is not None:
            current_score = score_take_against_reference_contour(
                current_take,
                reference_extraction.contour,
                name="current",
                **analysis_params["contour_score_kwargs"],
            )
        else:
            current_score = score_take(
                current_take,
                baseline,
                name="current",
                **analysis_params["symbolic_score_kwargs"],
            )
    elif reference_extraction is not None:
        comparison = compare_takes_to_reference_contour(
            previous_take,
            current_take,
            reference_extraction.contour,
            **analysis_params["contour_score_kwargs"],
        )
        current_score = comparison.current
    else:
        comparison = compare_takes(
            previous_take,
            current_take,
            baseline,
            **analysis_params["symbolic_score_kwargs"],
        )
        current_score = comparison.current
except Exception as exc:
    st.error(f"Could not score the takes: {exc}")
    st.stop()

st.subheader("Progress Summary" if comparison is not None else "Current Take Summary")
cols = st.columns(5)
render_metric(
    cols[0],
    "Overall",
    current_score.overall_score,
    comparison.overall_delta if comparison is not None else None,
    "Weighted composite: 45% pitch accuracy, 20% stability, 20% coverage, 15% timing.",
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

with st.expander("How these scores are computed", expanded=False):
    if reference_extraction is not None:
        st.markdown(
            """
- **Overall** = `50% pitch accuracy + 20% stability + 15% coverage + 15% timing`.
- **Pitch accuracy** compares the uploaded take contour to the uploaded reference contour using DTW.
- **Stability** measures how consistent the take's pitch deviations are after contour matching.
- **Coverage** compares detected voiced-frame amount against the reference voiced-frame amount.
- **Timing** is based on the median timing difference along the DTW match path.
- **Confidence** is a recording-analysis confidence signal, not a singing score.
            """.strip()
        )
    else:
        st.markdown(
            """
- **Overall** = `45% pitch accuracy + 20% stability + 20% coverage + 15% timing`.
- **Pitch accuracy** measures cents error against the reference melody.
- **Stability** measures how steady the pitch is inside covered notes.
- **Coverage** measures how much of the expected melody was attempted.
- **Timing** currently measures global start offset, not full rhythm quality.
- **Confidence** is a recording-analysis confidence signal, not a singing score.
            """.strip()
        )

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

with st.expander("Detailed metrics", expanded=False):
    st.json(comparison.to_dict() if comparison is not None else current_score.to_dict())

current_audio, sr = load_audio(current_take)
current_contour = clean_pitch_contour(
    extract_pitch(current_audio, sr, name="current", **analysis_params["pitch_kwargs"]),
    **analysis_params["clean_kwargs"],
)
previous_contour = None
if previous_take is not None:
    previous_audio, _ = load_audio(previous_take, target_sr=sr)
    previous_contour = clean_pitch_contour(
        extract_pitch(previous_audio, sr, name="previous", **analysis_params["pitch_kwargs"]),
        **analysis_params["clean_kwargs"],
    )

st.subheader("Audio Playback")
playback_cols = st.columns(3 if previous_take is not None else 2)
with playback_cols[0]:
    st.write("Reference")
    if reference_audio_path:
        st.audio(str(reference_audio_path))
    else:
        st.caption("No reference audio provided.")
if previous_take is not None:
    with playback_cols[1]:
        st.write("Previous take")
        st.audio(str(previous_take))
    current_audio_column = playback_cols[2]
else:
    current_audio_column = playback_cols[1]
with current_audio_column:
    st.write("Current take")
    st.audio(str(current_take))

st.subheader("Pitch Visualization")
if reference_extraction is not None:
    st.caption("Uploaded reference contour vs detected pitch contours.")
    st.pyplot(
        plot_contour_comparison(reference_extraction.contour, previous_contour, current_contour),
        clear_figure=True,
    )
else:
    st.caption("Reference melody vs detected pitch contours.")
    st.pyplot(plot_take_comparison(baseline, previous_contour, current_contour), clear_figure=True)

st.subheader("Voiced-Frame Coverage")
st.caption("Shows where the reference expects notes and where each take has detected voiced frames.")
if reference_extraction is not None:
    st.pyplot(
        plot_contour_voiced_coverage(reference_extraction.contour, previous_contour, current_contour),
        clear_figure=True,
    )
else:
    st.pyplot(plot_voiced_coverage(baseline, previous_contour, current_contour), clear_figure=True)

if reference_extraction is not None:
    st.caption(
        "Reference-audio scoring uses contour-to-contour DTW. Trust the score only when the "
        "reference contour visibly follows the melody and all files cover the same section."
    )
