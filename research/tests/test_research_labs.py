from __future__ import annotations

from konopro_research.demo_data import ensure_demo_data
from konopro_research.research_labs import (
    run_pitch_extractor_lab,
    run_reference_builder_lab,
    run_scoring_calibration_lab,
    run_song_identification_lab,
    run_stress_test_lab,
    run_timing_lab,
    write_synthetic_scenario_audio,
)


def test_pitch_extractor_lab_runs_pyin(tmp_path) -> None:
    paths = ensure_demo_data(tmp_path / "demo")

    status, table, excerpt_path, contours, notes = run_pitch_extractor_lab(
        str(paths["current"]),
        tmp_path,
        methods=["pYIN"],
        start_s=0.0,
        duration_s=2.0,
        min_confidence=0.25,
        max_jump_cents=700.0,
    )

    assert "complete" in status
    assert excerpt_path is not None
    assert "pYIN" in contours
    assert table.loc[0, "status"] == "ready"
    assert notes["pYIN"]


def test_reference_builder_lab_uses_demo_without_upload() -> None:
    status, table, baseline, contour, details = run_reference_builder_lab(None, window_s=0.2)

    assert "demo" in status.lower()
    assert baseline is not None
    assert contour is None
    assert not table.empty
    assert details["baseline"]["notes"] > 0


def test_reference_builder_lab_handles_uploaded_audio(tmp_path) -> None:
    paths = ensure_demo_data(tmp_path / "demo")

    status, table, baseline, contour, details = run_reference_builder_lab(
        str(paths["reference"]),
        window_s=0.2,
    )

    assert "Reference built" in status
    assert baseline is not None
    assert contour is not None
    assert not table.empty
    assert details["baseline"]["notes"] > 0
    assert "audio_summary" in details


def test_song_identification_lab_returns_candidates(tmp_path) -> None:
    paths = ensure_demo_data(tmp_path / "demo")

    status, table, details = run_song_identification_lab(
        None,
        str(paths["current"]),
        catalog_source="Demo catalog",
        top_k=3,
        window_s=20.0,
        hop_s=10.0,
    )

    assert "complete" in status
    assert not table.empty
    assert details["candidates"]


def test_timing_lab_returns_debug_chain(tmp_path) -> None:
    paths = ensure_demo_data(tmp_path / "demo")

    status, table, debug = run_timing_lab(
        str(paths["reference"]),
        str(paths["current"]),
        timing_penalty=90.0,
    )

    assert "complete" in status
    assert not table.empty
    assert "global_offset_s" in debug
    assert "new_timing_score_offset_corrected" in debug


def test_scoring_calibration_lab_handles_stable_wrong() -> None:
    status, table, details = run_scoring_calibration_lab(
        "stable but wrong",
        pitch_weight=0.45,
        stability_weight=0.25,
        coverage_weight=0.20,
        timing_weight=0.10,
    )

    assert "stable but wrong" in status
    assert not table.empty
    assert details["pitch_accuracy_score"] < 80


def test_stress_test_lab_runs_multiple_scenarios() -> None:
    status, table, details = run_stress_test_lab(
        scenarios=["accurate", "stable but wrong", "missing notes"],
    )

    assert "3 scenario" in status
    assert len(table) == 3
    assert len(details["scenarios"]) == 3


def test_synthetic_scenario_audio_reflects_late_start(tmp_path) -> None:
    accurate = write_synthetic_scenario_audio("accurate", tmp_path)
    late = write_synthetic_scenario_audio("late start", tmp_path)

    from konopro_research.audio_io import load_audio

    accurate_audio, sample_rate = load_audio(accurate)
    late_audio, _ = load_audio(late)

    assert len(late_audio) > len(accurate_audio) + int(2.0 * sample_rate)
    assert abs(float(late_audio[: int(2.0 * sample_rate)].max())) < 0.001
