from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

RESEARCH_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = RESEARCH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from konopro_research.baseline import demo_baseline  # noqa: E402
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
from konopro_research.pitch import clean_pitch_contour, extract_pitch  # noqa: E402
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


def load_demo_files() -> tuple[str, str, str, str]:
    return (
        str(DEMO_PATHS["reference"]),
        str(DEMO_PATHS["current"]),
        str(DEMO_PATHS["previous"]),
        "Loaded synthetic demo reference, current take, and previous take.",
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
) -> tuple[dict[str, Any], str, pd.DataFrame]:
    started = time.perf_counter()
    if current_take is None:
        return {}, "Upload or load a current take before preparing audio.", pd.DataFrame()

    config = {
        "backend": "demucs" if use_demucs else "none",
        "model": model,
        "device": device,
        "shifts": int(shifts),
        "overlap": float(overlap),
        "apply_to_takes": bool(apply_to_takes),
    }
    rows: list[dict[str, object]] = []

    def prepare_one(path: str | None, *, label: str, should_prepare: bool) -> str | None:
        if path is None:
            rows.append({"slot": label, "status": "missing", "analysis_path": ""})
            return None
        if not should_prepare:
            rows.append({"slot": label, "status": "original", "analysis_path": path})
            return path
        result = prepare_vocal_analysis_audio(
            path,
            cache_dir=STEM_CACHE_DIR,
            backend="demucs",
            model=model,
            device=device,
            shifts=int(shifts),
            overlap=float(overlap),
        )
        status = "fallback" if result.used_original and result.warnings else "stem"
        if result.used_cache:
            status = "cached stem"
        rows.append(
            {
                "slot": label,
                "status": status,
                "analysis_path": str(result.analysis_path),
                "warnings": "\n".join(result.warnings),
            }
        )
        return str(result.analysis_path)

    reference_analysis = prepare_one(
        reference_audio,
        label="reference",
        should_prepare=use_demucs and reference_audio is not None,
    )
    current_analysis = prepare_one(
        current_take,
        label="current",
        should_prepare=use_demucs and apply_to_takes,
    )
    previous_analysis = prepare_one(
        previous_take,
        label="previous",
        should_prepare=use_demucs and apply_to_takes and previous_take is not None,
    )

    elapsed = time.perf_counter() - started
    state = {
        "reference_analysis": reference_analysis,
        "current_analysis": current_analysis,
        "previous_analysis": previous_analysis,
        "config": config,
        "elapsed_s": elapsed,
    }
    status = f"Prepared analysis audio in {elapsed:.2f}s."
    if not use_demucs:
        status = f"Using original audio. Prepared in {elapsed:.2f}s."
    return state, status, pd.DataFrame(rows)


def run_evaluation(
    reference_mode: str,
    reference_audio: str | None,
    current_take: str | None,
    previous_take: str | None,
    prepared_state: dict[str, Any] | None,
) -> tuple[str, pd.DataFrame, str | None, str | None, dict[str, Any]]:
    started = time.perf_counter()
    prepared_state = prepared_state or {}
    reference_path = prepared_state.get("reference_analysis") or reference_audio
    current_path = prepared_state.get("current_analysis") or current_take
    previous_path = prepared_state.get("previous_analysis") or previous_take
    if current_path is None:
        return "Upload or load a current take before evaluation.", pd.DataFrame(), None, None, {}

    try:
        reference_extraction = None
        if reference_mode == "Uploaded reference audio":
            if reference_path is None:
                return "Upload or load reference audio for this mode.", pd.DataFrame(), None, None, {}
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
            details: dict[str, Any] = comparison.to_dict()
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
        return status, metrics, pitch_plot, coverage_plot, details
    except Exception as exc:
        return f"Evaluation failed after {time.perf_counter() - started:.2f}s: {exc}", pd.DataFrame(), None, None, {}


def run_matching(
    catalog_source: str,
    reference_audio: str | None,
    current_take: str | None,
    prepared_state: dict[str, Any] | None,
) -> tuple[str, pd.DataFrame, str | None, pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    prepared_state = prepared_state or {}
    query_path = prepared_state.get("current_analysis") or current_take
    reference_path = prepared_state.get("reference_analysis") or reference_audio
    if query_path is None:
        return "Upload or load a current take before matching.", pd.DataFrame(), None, pd.DataFrame(), {}

    try:
        query = extract_matching_query(
            query_path,
            name="matching query",
            pitch_kwargs=PITCH_KWARGS,
            clean_kwargs=CLEAN_KWARGS,
        )
        if catalog_source == "Uploaded reference sections":
            if reference_path is None:
                return "Upload or load reference audio for uploaded-reference matching.", pd.DataFrame(), None, pd.DataFrame(), {}
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

        result = match_query_to_sections(query, sections, top_k=5)
        candidates = pd.DataFrame([candidate.to_dict() for candidate in result.candidates])
        if result.best is None:
            elapsed = time.perf_counter() - started
            return f"No match found in {elapsed:.2f}s.", candidates, None, pd.DataFrame(), result.to_dict()

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
        return status, candidates, plot_path, handoff, result.to_dict()
    except Exception as exc:
        return f"Matching failed after {time.perf_counter() - started:.2f}s: {exc}", pd.DataFrame(), None, pd.DataFrame(), {}


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

Alternative audio-lab frontend using the same backend as the Streamlit prototype.
Run each stage explicitly and watch the elapsed processing time.
            """.strip()
        )
        prepared_state = gr.State({})

        with gr.Row():
            reference_audio = gr.Audio(label="Reference audio", type="filepath")
            current_take = gr.Audio(label="Current take", type="filepath")
            previous_take = gr.Audio(label="Previous take (optional)", type="filepath")

        load_demo = gr.Button("Load Synthetic Demo Files")
        demo_status = gr.Markdown()
        load_demo.click(
            load_demo_files,
            outputs=[reference_audio, current_take, previous_take, demo_status],
        )

        with gr.Tab("1. Prepare Audio"):
            use_demucs = gr.Checkbox(label="Use Demucs vocal stem", value=False)
            apply_to_takes = gr.Checkbox(label="Apply Demucs to takes too", value=False)
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
            prepare_status = gr.Markdown()
            prepare_table = gr.Dataframe(label="Preparation Results")
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
                outputs=[prepared_state, prepare_status, prepare_table],
            )

        with gr.Tab("2. Evaluate Singing"):
            reference_mode = gr.Radio(
                ["Demo symbolic baseline", "Uploaded reference audio"],
                value="Demo symbolic baseline",
                label="Reference mode",
            )
            evaluation_button = gr.Button("Run Evaluation", variant="primary")
            evaluation_status = gr.Markdown()
            evaluation_metrics = gr.Dataframe(label="Evaluation Metrics")
            with gr.Row():
                pitch_plot = gr.Image(label="Pitch plot", type="filepath")
                coverage_plot = gr.Image(label="Coverage plot", type="filepath")
            evaluation_json = gr.JSON(label="Evaluation Details")
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
                    evaluation_status,
                    evaluation_metrics,
                    pitch_plot,
                    coverage_plot,
                    evaluation_json,
                ],
            )

        with gr.Tab("3. Match Song / Section"):
            catalog_source = gr.Radio(
                ["Demo catalog", "Uploaded reference sections"],
                value="Demo catalog",
                label="Catalog source",
            )
            matching_button = gr.Button("Run Song/Section Matching", variant="primary")
            matching_status = gr.Markdown()
            match_table = gr.Dataframe(label="Candidate Matches")
            match_plot = gr.Image(label="Best match pitch-shape plot", type="filepath")
            handoff_metrics = gr.Dataframe(label="Handoff Score")
            matching_json = gr.JSON(label="Matching Details")
            matching_button.click(
                run_matching,
                inputs=[catalog_source, reference_audio, current_take, prepared_state],
                outputs=[
                    matching_status,
                    match_table,
                    match_plot,
                    handoff_metrics,
                    matching_json,
                ],
            )
    return demo


if __name__ == "__main__":
    build_app().queue().launch(server_name="127.0.0.1", server_port=7860)
