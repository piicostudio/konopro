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
from konopro_research.contour_scoring import compare_takes_to_reference_contour  # noqa: E402
from konopro_research.demo_data import ensure_demo_data  # noqa: E402
from konopro_research.pitch import extract_pitch  # noqa: E402
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
from konopro_research.scoring import compare_takes  # noqa: E402


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
        "description": "Use when you already have a symbolic melody baseline.",
        "baseline": "upload_csv",
        "previous": "upload",
        "current": "upload",
    },
    "Upload reference audio + takes": {
        "description": "Experimental: extracts a melody baseline from uploaded guide/reference audio.",
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
3. For real tests, trim reference/previous/current to the same 20-40 second phrase.
4. Prefer a guide vocal or melody-only reference over a full mixed song.
5. Check the extracted reference plot before trusting any score.
        """.strip()
    )


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
        reference_extraction = extract_reference_audio(reference_audio_path, title=upload.name)
        baseline = reference_extraction.baseline
        reference_source_duration = reference_extraction.audio_summary.duration_s
except Exception as exc:
    st.error(f"Could not load reference input: {exc}")
    st.stop()

try:
    if scenario["previous"] == "upload":
        previous_upload = st.file_uploader(
            "Previous vocal take",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
        )
        current_upload = st.file_uploader(
            "Current vocal take",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
        )
        if not previous_upload or not current_upload:
            st.info("Upload previous and current vocal takes for the same section.")
            st.stop()
        previous_take = uploaded_audio_to_temp(previous_upload, Path(previous_upload.name).suffix)
        current_take = uploaded_audio_to_temp(current_upload, Path(current_upload.name).suffix)
    else:
        previous_take = demo_take_path(scenario["previous"])
        current_take = demo_take_path(scenario["current"])

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

duration_warnings = duration_mismatch_warnings(
    baseline.duration_s,
    previous_summary.duration_s,
    current_summary.duration_s,
)
render_warnings(duration_warnings)

with st.expander("Audio input checks", expanded=bool(duration_warnings)):
    st.write("Previous take")
    st.json(previous_summary.to_dict())
    st.write("Current take")
    st.json(current_summary.to_dict())

try:
    if reference_extraction is not None:
        comparison = compare_takes_to_reference_contour(
            previous_take,
            current_take,
            reference_extraction.contour,
        )
    else:
        comparison = compare_takes(previous_take, current_take, baseline)
except Exception as exc:
    st.error(f"Could not score the takes: {exc}")
    st.stop()

st.subheader("Progress Summary")
cols = st.columns(5)
render_metric(
    cols[0],
    "Overall",
    comparison.current.overall_score,
    comparison.overall_delta,
    "Weighted composite: 45% pitch accuracy, 20% stability, 20% coverage, 15% timing.",
)
render_metric(cols[1], "Pitch Accuracy", comparison.current.pitch_accuracy_score, comparison.pitch_accuracy_delta)
render_metric(cols[2], "Stability", comparison.current.stability_score, comparison.stability_delta)
render_metric(cols[3], "Coverage", comparison.current.coverage_score, comparison.coverage_delta)
render_metric(cols[4], "Timing", comparison.current.timing_score, comparison.timing_delta)

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
render_warnings(comparison.current.warnings)

with st.expander("Detailed metrics", expanded=False):
    st.json(comparison.to_dict())

previous_audio, sr = load_audio(previous_take)
current_audio, _ = load_audio(current_take, target_sr=sr)
previous_contour = extract_pitch(previous_audio, sr, name="previous")
current_contour = extract_pitch(current_audio, sr, name="current")

st.subheader("Audio Playback")
playback_cols = st.columns(3)
with playback_cols[0]:
    st.write("Reference")
    if reference_audio_path:
        st.audio(str(reference_audio_path))
    else:
        st.caption("No reference audio provided.")
with playback_cols[1]:
    st.write("Previous take")
    st.audio(str(previous_take))
with playback_cols[2]:
    st.write("Current take")
    st.audio(str(current_take))

st.subheader("Pitch Visualization")
if reference_extraction is not None:
    st.caption("Uploaded reference contour vs detected pitch contours for the previous and current takes.")
    st.pyplot(
        plot_contour_comparison(reference_extraction.contour, previous_contour, current_contour),
        clear_figure=True,
    )
else:
    st.caption("Reference melody vs detected pitch contours for the previous and current takes.")
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
