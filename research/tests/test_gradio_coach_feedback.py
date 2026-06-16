from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

import frontends.gradio.app as gradio_app
from konopro_research.audio_io import write_wav
from konopro_research.pitch import PitchContour


def test_coach_feedback_generates_explanation_and_problem_moments(tmp_path, monkeypatch) -> None:
    sample_rate = 22050
    times = np.arange(0.25, 3.25, 0.02)
    reference_hz = np.full_like(times, 220.0)
    current_hz = reference_hz.copy()
    current_hz[(times >= 1.0) & (times <= 1.35)] *= 2 ** (320.0 / 1200.0)
    current_hz[(times >= 2.05) & (times <= 2.35)] *= 2 ** (-260.0 / 1200.0)
    confidence = np.ones_like(times)

    reference_contour = PitchContour(times, reference_hz, confidence, name="reference")
    current_contour = PitchContour(times + 0.10, current_hz, confidence, name="current")

    audio_times = np.arange(int(sample_rate * 4.0)) / sample_rate
    reference_audio = 0.20 * np.sin(2.0 * np.pi * 220.0 * audio_times).astype(np.float32)
    current_audio = 0.20 * np.sin(2.0 * np.pi * 220.0 * audio_times).astype(np.float32)
    reference_path = tmp_path / "reference.wav"
    current_path = tmp_path / "current.wav"
    write_wav(reference_path, reference_audio, sample_rate)
    write_wav(current_path, current_audio, sample_rate)

    monkeypatch.setattr(gradio_app, "OUTPUT_DIR", tmp_path / "outputs")

    evaluation_details = {
        "overall_score": 62.0,
        "pitch_accuracy_score": 54.0,
        "timing_score": 82.0,
        "stability_score": 41.0,
        "coverage_score": 96.0,
        "recording_confidence_score": 88.0,
        "mean_pitch_error_cents": 72.0,
        "pitch_stability_cents": 180.0,
        "timing_offset_s": 0.10,
        "warnings": [],
    }
    timing_debug = {
        "global_offset_s": 0.10,
        "median_abs_local_error_s": 0.02,
        "raw_delta_s_mad_or_std": 0.02,
    }
    stability_debug = {"p95_abs_residual_cents": 320.0}
    contour_state = {
        "reference_mode": "Uploaded reference audio",
        "reference_path": str(reference_path),
        "current_path": str(current_path),
        "previous_path": None,
        "reference_contour": reference_contour,
        "current_contour": current_contour,
        "previous_contour": None,
        "raw_plots": {},
        "profile_rows": [],
    }

    explanation = gradio_app.run_explanation_generator(
        evaluation_details,
        timing_debug,
        stability_debug,
    )
    assert explanation[0]["available"] is True
    assert explanation[0]["primary_issue"] in {"pitch_accuracy", "pitch_consistency"}

    moments = gradio_app.run_problem_moment_detection(
        "Uploaded reference audio",
        str(reference_path),
        str(current_path),
        None,
        None,
        contour_state,
        evaluation_details,
        timing_debug,
        explanation[0],
    )

    assert "detected" in moments[0]
    assert not moments[1].empty
    assert moments[6]["available"] is True
    assert len(moments[6]["moments"]) <= 2
    assert "cents" in moments[6]["moments"][0]["practice_tip"]
    assert "target pitch" not in moments[2]
    assert moments[3] is not None
    assert Path(moments[3]).exists()
