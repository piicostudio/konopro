from __future__ import annotations

from pathlib import Path

import numpy as np

from konopro_research.audio_io import write_wav
from konopro_research.fingerprinting import (
    prepare_fingerprint_windows,
    recognize_with_acrcloud,
    recognize_with_audd,
    recognize_with_shazamkit,
    run_acrcloud_fingerprinting,
    run_audd_fingerprinting,
    run_shazam_fingerprinting,
)


def test_prepare_fingerprint_windows_caps_hop_spaced_windows(tmp_path) -> None:
    sample_rate = 44100
    path = tmp_path / "karaoke_attempt.wav"
    write_wav(path, np.zeros(sample_rate * 250, dtype=np.float32), sample_rate)

    windows = prepare_fingerprint_windows(
        path,
        tmp_path,
        mode="Sliding windows",
        window_s=10,
        hop_s=20,
        max_windows=3,
    )

    assert len(windows) == 3
    assert [window["start_s"] for window in windows] == [0.0, 20.0, 40.0]
    assert all(Path(window["audio_path"]).exists() for window in windows)


def test_prepare_fingerprint_windows_applies_start_offset(tmp_path) -> None:
    sample_rate = 44100
    path = tmp_path / "karaoke_attempt.wav"
    write_wav(path, np.zeros(sample_rate * 250, dtype=np.float32), sample_rate)

    windows = prepare_fingerprint_windows(
        path,
        tmp_path,
        mode="Sliding windows",
        window_s=10,
        hop_s=20,
        max_windows=3,
        start_offset_s=10,
    )

    assert [window["start_s"] for window in windows] == [10.0, 30.0, 50.0]


def test_prepare_fingerprint_windows_center_out_strategy(tmp_path) -> None:
    sample_rate = 44100
    path = tmp_path / "karaoke_attempt.wav"
    write_wav(path, np.zeros(sample_rate * 250, dtype=np.float32), sample_rate)

    windows = prepare_fingerprint_windows(
        path,
        tmp_path,
        mode="Sliding windows",
        window_s=10,
        hop_s=20,
        max_windows=5,
        window_strategy="Center-out",
    )

    assert [window["start_s"] for window in windows] == [120.0, 100.0, 140.0, 80.0, 160.0]


def test_fingerprinting_sliding_windows_find_expected_match(tmp_path) -> None:
    sample_rate = 44100
    audio = np.zeros(sample_rate * 30, dtype=np.float32)
    path = tmp_path / "karaoke_attempt.wav"
    write_wav(path, audio, sample_rate)

    def fake_recognizer(window_path):
        if "window_002" in window_path.name:
            return {
                "status": "matched",
                "media": {
                    "title": "Noraebang Song",
                    "artist": "Demo Artist",
                    "confidence": 0.91,
                    "shazam_id": "shazam-123",
                },
            }
        return {"status": "no_match"}

    result = run_shazam_fingerprinting(
        path,
        tmp_path,
        expected_title="Noraebang Song",
        expected_artist="Demo Artist",
        mode="Whole + sliding windows",
        window_s=10,
        hop_s=10,
        max_windows=5,
        recognizer=fake_recognizer,
    )

    assert result.summary["windows_tested"] == 4
    assert result.summary["whole_file_expected_match"] is False
    assert result.summary["sliding_window_expected_match"] is True
    assert result.summary["any_window_expected_match"] is True
    assert result.summary["first_expected_match_start_s"] == 10.0
    assert result.rows[1]["mode"] == "window"
    assert result.rows[1]["audio_file"].endswith("_window_001_0.00_10.00.wav")
    assert result.rows[1]["audio_path"].endswith("_window_001_0.00_10.00.wav")
    assert "expected song" in result.status


def test_fingerprinting_reports_missing_helper(tmp_path) -> None:
    path = tmp_path / "karaoke_attempt.wav"
    write_wav(path, np.zeros(44100 * 3, dtype=np.float32), 44100)

    result = run_shazam_fingerprinting(
        path,
        tmp_path,
        mode="Whole recording",
        helper_path=tmp_path / "missing.swift",
    )

    assert result.summary["windows_tested"] == 1
    assert result.summary["error_windows"] == 1
    assert result.rows[0]["status"] == "error"
    assert "helper not found" in result.rows[0]["error"]


def test_fingerprinting_clamps_windows_to_shazamkit_signature_limit(tmp_path) -> None:
    sample_rate = 44100
    path = tmp_path / "karaoke_attempt.wav"
    write_wav(path, np.zeros(sample_rate * 30, dtype=np.float32), sample_rate)
    seen_durations = []

    def fake_recognizer(window_path):
        import soundfile as sf

        info = sf.info(window_path)
        seen_durations.append(info.duration)
        return {"status": "no_match"}

    result = run_shazam_fingerprinting(
        path,
        tmp_path,
        mode="Whole + sliding windows",
        window_s=15,
        hop_s=10,
        max_windows=2,
        recognizer=fake_recognizer,
    )

    assert result.summary["windows_tested"] == 3
    assert result.rows[0]["mode"] == "whole"
    assert result.rows[0]["audio_file"].endswith("_whole_first_12s.wav")
    assert max(seen_durations) <= 12.0


def test_fingerprinting_rejects_audio_shorter_than_shazamkit_minimum(tmp_path) -> None:
    path = tmp_path / "short.wav"
    write_wav(path, np.zeros(44100, dtype=np.float32), 44100)

    result = run_shazam_fingerprinting(path, tmp_path, mode="Sliding windows")

    assert result.rows == ()
    assert result.summary["windows_tested"] == 0
    assert "too short" in result.status


def test_recognize_with_shazamkit_accepts_executable_helper(tmp_path) -> None:
    audio_path = tmp_path / "clip.wav"
    write_wav(audio_path, np.zeros(44100, dtype=np.float32), 44100)
    helper = tmp_path / "KonoproShazamHelper"
    helper.write_text("#!/bin/sh\nprintf '%s\\n' '{\"status\":\"no_match\"}'\n")
    helper.chmod(0o755)

    result = recognize_with_shazamkit(audio_path, helper_path=helper)

    assert result["status"] == "no_match"


def test_recognize_with_shazamkit_accepts_app_bundle_helper(tmp_path) -> None:
    audio_path = tmp_path / "clip.wav"
    write_wav(audio_path, np.zeros(44100, dtype=np.float32), 44100)
    app_path = tmp_path / "KonoproShazamHelper.app"
    helper = app_path / "Contents" / "MacOS" / "KonoproShazamHelper"
    helper.parent.mkdir(parents=True)
    helper.write_text("#!/bin/sh\nprintf '%s\\n' '{\"status\":\"matched\",\"media\":{\"title\":\"Demo\"}}'\n")
    helper.chmod(0o755)

    result = recognize_with_shazamkit(audio_path, helper_path=app_path)

    assert result["status"] == "matched"
    assert result["media"]["title"] == "Demo"


def test_audd_fingerprinting_maps_expected_match(tmp_path) -> None:
    sample_rate = 44100
    path = tmp_path / "karaoke_attempt.wav"
    write_wav(path, np.zeros(sample_rate * 12, dtype=np.float32), sample_rate)

    def fake_recognizer(window_path):
        return {
            "status": "matched",
            "media": {
                "title": "Noraebang Song",
                "artist": "Demo Artist",
                "spotify_id": "spotify-123",
            },
        }

    result = run_audd_fingerprinting(
        path,
        tmp_path,
        expected_title="Noraebang Song",
        expected_artist="Demo Artist",
        mode="Sliding windows",
        recognizer=fake_recognizer,
    )

    assert result.summary["any_window_expected_match"] is True
    assert result.rows[0]["status"] == "matched"
    assert result.rows[0]["spotify_id"] == "spotify-123"


def test_recognize_with_audd_requires_token(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("AUDD_API_TOKEN", raising=False)
    path = tmp_path / "clip.wav"
    write_wav(path, np.zeros(44100 * 3, dtype=np.float32), 44100)

    result = recognize_with_audd(path)

    assert result["status"] == "error"
    assert "AUDD_API_TOKEN" in result["error"]


def test_acrcloud_fingerprinting_maps_expected_match(tmp_path) -> None:
    sample_rate = 44100
    path = tmp_path / "karaoke_attempt.wav"
    write_wav(path, np.zeros(sample_rate * 12, dtype=np.float32), sample_rate)

    def fake_recognizer(window_path):
        return {
            "status": "matched",
            "media": {
                "title": "Noraebang Song",
                "artist": "Demo Artist",
                "acrid": "acr-123",
                "confidence": 91,
            },
        }

    result = run_acrcloud_fingerprinting(
        path,
        tmp_path,
        expected_title="Noraebang Song",
        expected_artist="Demo Artist",
        mode="Sliding windows",
        recognizer=fake_recognizer,
    )

    assert result.summary["any_window_expected_match"] is True
    assert result.rows[0]["status"] == "matched"
    assert result.rows[0]["acrid"] == "acr-123"
    assert result.rows[0]["confidence"] == 91.0


def test_recognize_with_acrcloud_requires_credentials(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ACRCLOUD_HOST", raising=False)
    monkeypatch.delenv("ACRCLOUD_ACCESS_KEY", raising=False)
    monkeypatch.delenv("ACRCLOUD_ACCESS_SECRET", raising=False)
    path = tmp_path / "clip.wav"
    write_wav(path, np.zeros(44100 * 3, dtype=np.float32), 44100)

    result = recognize_with_acrcloud(path)

    assert result["status"] == "error"
    assert "ACRCLOUD_HOST" in result["error"]
