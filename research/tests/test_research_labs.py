from __future__ import annotations

import numpy as np

from konopro_research.audio_io import write_wav
from konopro_research.demo_data import ensure_demo_data
from konopro_research.research_labs import (
    run_fingerprint_diagnostics_lab,
    run_long_session_segmentation_lab,
    run_matched_progress_lab,
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


def test_long_session_segmentation_lab_returns_intervals(tmp_path) -> None:
    sample_rate = 44100
    path = tmp_path / "long_session.wav"
    write_wav(path, np.zeros(sample_rate * 80, dtype=np.float32), sample_rate)

    def fake_recognizer(window_path):
        start_s = float(window_path.name.rsplit("_", 2)[1])
        if 20.0 <= start_s < 60.0:
            return {
                "status": "matched",
                "media": {
                    "title": "Lab Song",
                    "artist": "Demo Artist",
                    "isrc": "LAB-1",
                    "confidence": 0.92,
                },
            }
        return {"status": "no_match"}

    status, intervals, windows, details = run_long_session_segmentation_lab(
        str(path),
        tmp_path,
        provider="shazamkit",
        window_s=10.0,
        hop_s=10.0,
        max_windows=None,
        recognizer=fake_recognizer,
        rms_frame_s=0.5,
        rms_hop_s=0.25,
        tempo_window_s=5.0,
        tempo_hop_s=5.0,
    )

    assert "Intervals: 1" in status
    assert not intervals.empty
    assert not windows.empty
    assert details["best_interval_clip_path"]
    assert details["timeline_plot_path"]
    assert details["signal_diagnostic_params"]["rms_frame_s"] == 0.5


def test_long_session_segmentation_lab_can_skip_recognition_provider(tmp_path) -> None:
    sample_rate = 22050
    path = tmp_path / "diagnostic_only_session.wav"
    write_wav(path, np.zeros(sample_rate * 10, dtype=np.float32), sample_rate)

    status, intervals, windows, details = run_long_session_segmentation_lab(
        str(path),
        tmp_path,
        provider="none",
        window_s=10.0,
        hop_s=5.0,
        max_windows=None,
    )

    assert "Intervals: 0" in status
    assert intervals.empty
    assert windows.empty
    assert details["timeline_plot_path"]
    assert any("provider skipped" in warning for warning in details["warnings"])


def test_long_session_segmentation_lab_reports_no_match(tmp_path) -> None:
    sample_rate = 44100
    path = tmp_path / "unknown_session.wav"
    write_wav(path, np.zeros(sample_rate * 30, dtype=np.float32), sample_rate)

    status, intervals, windows, details = run_long_session_segmentation_lab(
        str(path),
        tmp_path,
        provider="shazamkit",
        window_s=10.0,
        hop_s=10.0,
        max_windows=None,
        recognizer=lambda _path: {"status": "no_match"},
    )

    assert "Intervals: 0" in status
    assert intervals.empty
    assert not windows.empty
    assert any("No recognized song intervals" in warning for warning in details["warnings"])


def test_long_session_segmentation_lab_reports_weak_candidates(tmp_path) -> None:
    sample_rate = 44100
    path = tmp_path / "weak_session.wav"
    write_wav(path, np.zeros(sample_rate * 80, dtype=np.float32), sample_rate)

    def fake_recognizer(window_path):
        start_s = float(window_path.name.rsplit("_", 2)[1])
        if start_s == 20.0:
            return {
                "status": "matched",
                "media": {
                    "title": "Weak Lab Song",
                    "artist": "Demo Artist",
                    "isrc": "WEAK-LAB-1",
                    "confidence": 0.34,
                },
            }
        return {"status": "no_match"}

    status, intervals, windows, details = run_long_session_segmentation_lab(
        str(path),
        tmp_path,
        provider="shazamkit",
        window_s=10.0,
        hop_s=10.0,
        max_windows=None,
        recognizer=fake_recognizer,
    )

    assert "Intervals: 0" in status
    assert "Weak candidates: 1" in status
    assert intervals.empty
    assert not windows.empty
    assert details["weak_candidates"][0]["reason"] == "singleton_match"


def test_fingerprint_diagnostics_lab_analyzes_headerless_csv(tmp_path) -> None:
    csv_text = "\n".join(
        [
            "acrcloud,0,5,no_match,false,,,,null,/tmp/a.wav,a.wav,",
            "acrcloud,30,35,no_match,false,,,,null,/tmp/b.wav,b.wav,",
            "acrcloud,480,485,matched,true,You(7525),KY Noraebang,isrc:qzncb2169572,0.34,/tmp/c.wav,c.wav,",
        ]
    )

    status, summary, weak, sweeps, recommendations, preview_files, first_preview, details = (
        run_fingerprint_diagnostics_lab(
            csv_text,
            tmp_path,
            provider="acrcloud",
            recording_duration_s=630,
            request_budget=120,
        )
    )

    assert "Fingerprint diagnostics complete" in status
    assert not bool(summary.loc[0, "can_segment"])
    assert "singleton_candidate" in summary.loc[0, "flags"]
    assert weak.loc[0, "reason"] == "singleton_match"
    assert not sweeps.empty
    assert not recommendations.empty
    assert preview_files == []
    assert first_preview is None
    assert details["weak_candidates"][0]["title"] == "You(7525)"


def test_fingerprint_diagnostics_lab_can_preview_recovery_windows(tmp_path) -> None:
    sample_rate = 44100
    audio_path = tmp_path / "source.wav"
    write_wav(audio_path, np.zeros(sample_rate * 120, dtype=np.float32), sample_rate)
    csv_text = "acrcloud,30,35,matched,true,Preview Song,Demo Artist,PREVIEW-1,0.34,/tmp/c.wav,c.wav,"

    _status, _summary, _weak, _sweeps, _recommendations, preview_files, first_preview, details = (
        run_fingerprint_diagnostics_lab(
            csv_text,
            tmp_path,
            provider="acrcloud",
            recording_duration_s=120,
            original_audio_path=str(audio_path),
            request_budget=20,
        )
    )

    assert preview_files
    assert first_preview == preview_files[0]
    assert details["preview_windows"]


def test_matched_progress_lab_returns_artifacts(tmp_path) -> None:
    paths = ensure_demo_data(tmp_path / "demo")

    status, summary, section, metrics, gates, pitch_plot, coverage_plot, ref_clip, prev_clip, cur_clip, details = (
        run_matched_progress_lab(
            str(paths["reference"]),
            str(paths["previous"]),
            str(paths["current"]),
            tmp_path,
            catalog_source="Uploaded reference sections",
            window_s=5.0,
            hop_s=2.0,
        )
    )

    assert "Matched progress" in status
    assert not summary.empty
    assert not section.empty
    assert not metrics.empty
    assert not gates.empty
    assert pitch_plot and coverage_plot
    assert ref_clip and prev_clip and cur_clip
    assert details["verdict"] in {"improved", "declined", "roughly unchanged", "insufficient confidence"}


def test_matched_progress_lab_reports_unsafe_result(tmp_path) -> None:
    paths = ensure_demo_data(tmp_path / "demo")

    status, _summary, _section, _metrics, gates, _pitch_plot, _coverage_plot, _ref_clip, _prev_clip, _cur_clip, details = (
        run_matched_progress_lab(
            str(paths["reference"]),
            str(paths["previous"]),
            str(paths["current"]),
            tmp_path,
            catalog_source="Uploaded reference sections",
            window_s=5.0,
            hop_s=2.0,
            min_match_score=101.0,
        )
    )

    assert "unsafe" in status.lower()
    assert details["confidence"]["can_score"] is False
    assert any("match score" in reason for reason in details["confidence"]["reasons"])
    assert not gates.empty


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
