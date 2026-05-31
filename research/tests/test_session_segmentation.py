from __future__ import annotations

import numpy as np

from konopro_research.audio_io import load_audio, write_wav
from konopro_research.session_segmentation import (
    SongIdentity,
    WeakSongCandidate,
    _break_likelihood_diagnostics,
    plot_session_segmentation,
    segment_long_recording,
    segment_recognized_windows,
    windows_from_fingerprint_rows,
    write_interval_clips,
)


def test_windows_from_rows_prefers_stable_ids_over_title_artist() -> None:
    rows = [
        fake_row(0, "Same Title", "Artist A", isrc="KR-A1B-26-00001"),
        fake_row(5, "Different Title", "Artist B", isrc="KR-A1B-26-00001"),
    ]

    windows = windows_from_fingerprint_rows(rows, provider="shazamkit")

    assert windows[0].identity is not None
    assert windows[0].identity.key == "isrc:kr-a1b-26-00001"
    assert windows[0].identity.key == windows[1].identity.key


def test_segment_recognized_windows_returns_one_song_interval() -> None:
    rows = [
        fake_row(0, status="no_match"),
        fake_row(5, status="no_match"),
        *song_rows(30, 90, title="Noraebang Song", artist="Demo Artist"),
        fake_row(95, status="no_match"),
    ]

    result = segment_recognized_windows(
        windows_from_fingerprint_rows(rows, provider="shazamkit"),
        recording_duration_s=120,
        hop_s=5,
    )

    assert len(result.intervals) == 1
    interval = result.intervals[0]
    assert interval.identity.title == "Noraebang Song"
    assert interval.start_s == 25.0
    assert interval.end_s == 95.0
    assert interval.confidence_score >= 80
    assert interval.to_dict()["recognized_windows"] == 12


def test_segment_recognized_windows_returns_two_song_intervals() -> None:
    rows = [
        *song_rows(30, 90, title="First Song", artist="Demo Artist", isrc="ISRC-FIRST"),
        *[fake_row(start, status="no_match") for start in range(95, 140, 5)],
        *song_rows(140, 210, title="Second Song", artist="Demo Artist", isrc="ISRC-SECOND"),
    ]

    result = segment_recognized_windows(
        windows_from_fingerprint_rows(rows, provider="shazamkit"),
        recording_duration_s=240,
        hop_s=5,
    )

    assert [interval.identity.title for interval in result.intervals] == ["First Song", "Second Song"]
    assert result.intervals[0].start_s == 25.0
    assert result.intervals[0].end_s == 95.0
    assert result.intervals[1].start_s == 135.0
    assert result.intervals[1].end_s == 215.0


def test_single_no_match_gap_does_not_split_song_interval() -> None:
    rows = [
        *song_rows(30, 55, title="Gap Song", artist="Demo Artist"),
        fake_row(55, status="no_match"),
        *song_rows(60, 90, title="Gap Song", artist="Demo Artist"),
    ]

    result = segment_recognized_windows(
        windows_from_fingerprint_rows(rows, provider="shazamkit"),
        recording_duration_s=120,
        hop_s=5,
        allowed_gap_windows=1,
    )

    assert len(result.intervals) == 1
    assert result.intervals[0].recognized_window_count == 11
    assert result.intervals[0].gap_window_count == 1
    assert result.intervals[0].confidence_score >= 75


def test_conflicting_windows_split_or_warn() -> None:
    rows = [
        *song_rows(30, 60, title="Original Song", artist="Demo Artist", isrc="ISRC-ORIGINAL"),
        *song_rows(60, 70, title="Wrong Song", artist="Other Artist", isrc="ISRC-WRONG"),
        *song_rows(70, 95, title="Original Song", artist="Demo Artist", isrc="ISRC-ORIGINAL"),
    ]

    result = segment_recognized_windows(
        windows_from_fingerprint_rows(rows, provider="shazamkit"),
        recording_duration_s=120,
        hop_s=5,
        conflict_split_windows=2,
    )

    assert len(result.intervals) >= 2
    assert any("conflicting" in warning for warning in result.warnings)
    assert any(interval.confidence_score < 90 for interval in result.intervals)


def test_all_unknown_recording_returns_no_intervals() -> None:
    rows = [fake_row(start, status="no_match") for start in range(0, 60, 5)]

    result = segment_recognized_windows(
        windows_from_fingerprint_rows(rows, provider="shazamkit"),
        recording_duration_s=60,
        hop_s=5,
    )

    assert result.intervals == ()
    assert result.weak_candidates == ()
    assert any("No recognized song intervals" in warning for warning in result.warnings)


def test_singleton_match_becomes_weak_candidate_not_interval() -> None:
    rows = [
        fake_row(0, status="no_match"),
        fake_row(30, title="Weak Clue", artist="Demo Artist", confidence=0.34, isrc="WEAK-1"),
        fake_row(60, status="no_match"),
    ]

    result = segment_recognized_windows(
        windows_from_fingerprint_rows(rows, provider="acrcloud"),
        recording_duration_s=120,
        hop_s=30,
    )

    assert result.intervals == ()
    assert len(result.weak_candidates) == 1
    candidate = result.weak_candidates[0]
    assert candidate.identity.title == "Weak Clue"
    assert candidate.reason == "singleton_match"
    assert candidate.recognized_window_count == 1
    assert candidate.provider_confidence == 0.34
    assert candidate.recovery_start_s <= 30 <= candidate.recovery_end_s
    details = result.to_dict()
    assert details["weak_candidates"][0]["reason"] == "singleton_match"


def test_low_confidence_cluster_is_visible_without_weakening_acceptance() -> None:
    rows = [
        fake_row(20, title="Low Confidence", artist="Demo Artist", confidence=0.2, isrc="LOW-1"),
        fake_row(25, title="Low Confidence", artist="Demo Artist", confidence=0.25, isrc="LOW-1"),
    ]

    result = segment_recognized_windows(
        windows_from_fingerprint_rows(rows, provider="acrcloud"),
        recording_duration_s=80,
        hop_s=5,
    )

    assert len(result.intervals) == 1
    assert result.intervals[0].confidence_level == "medium"
    assert result.intervals[0].warnings


def test_public_weak_candidate_import_is_available() -> None:
    assert WeakSongCandidate.__name__ == "WeakSongCandidate"


def test_public_identity_import_is_available() -> None:
    identity = SongIdentity(provider="test", key="title:demo", title="Demo", artist="")

    assert identity.display_name == "Demo"


def test_segment_long_recording_uses_fake_recognizer_without_external_service(tmp_path) -> None:
    audio_path = tmp_path / "long_session.wav"
    write_wav(audio_path, np.zeros(44100 * 80, dtype=np.float32), 44100)

    def fake_recognizer(window_path):
        name = window_path.name
        if "_window_" not in name:
            return {"status": "no_match"}
        start_s = float(name.rsplit("_", 2)[1])
        if 20.0 <= start_s < 60.0:
            return {
                "status": "matched",
                "media": {
                    "title": "Wrapper Song",
                    "artist": "Demo Artist",
                    "isrc": "WRAP-1",
                    "confidence": 0.91,
                },
            }
        return {"status": "no_match"}

    result = segment_long_recording(
        audio_path,
        tmp_path,
        provider="shazamkit",
        window_s=10,
        hop_s=10,
        recognizer=fake_recognizer,
    )

    assert len(result.intervals) == 1
    assert result.intervals[0].identity.title == "Wrapper Song"
    assert result.provider_result is not None


def test_write_interval_clips_creates_playable_files(tmp_path) -> None:
    audio_path = tmp_path / "long_session.wav"
    sample_rate = 22050
    audio = np.ones(sample_rate * 30, dtype=np.float32) * 0.05
    write_wav(audio_path, audio, sample_rate)
    result = segment_recognized_windows(
        windows_from_fingerprint_rows(song_rows(5, 20, title="Clip Song"), provider="test"),
        recording_duration_s=30,
        hop_s=5,
    )

    clips = write_interval_clips(audio_path, result.intervals, tmp_path / "clips")

    assert len(clips) == 1
    clip_path = clips[0]["clip_path"]
    clipped_audio, clipped_sr = load_audio(clip_path, target_sr=sample_rate)
    assert clipped_sr == sample_rate
    assert abs(len(clipped_audio) / sample_rate - 25.0) < 0.1


def test_plot_session_segmentation_can_save(tmp_path) -> None:
    result = segment_recognized_windows(
        windows_from_fingerprint_rows(song_rows(10, 40, title="Plot Song"), provider="test"),
        recording_duration_s=60,
        hop_s=5,
    )
    output_path = tmp_path / "timeline.png"

    fig = plot_session_segmentation(result, output_path=output_path)

    assert fig is not None
    assert output_path.exists()


def test_plot_session_segmentation_can_save_with_weak_candidate(tmp_path) -> None:
    result = segment_recognized_windows(
        windows_from_fingerprint_rows(
            [fake_row(10, title="Plot Weak", artist="Demo Artist", isrc="PLOT-WEAK")],
            provider="test",
        ),
        recording_duration_s=60,
        hop_s=5,
    )
    output_path = tmp_path / "weak_timeline.png"

    fig = plot_session_segmentation(result, output_path=output_path)

    assert fig is not None
    assert output_path.exists()
    assert len(result.weak_candidates) == 1


def test_plot_session_segmentation_can_include_rms_energy(tmp_path) -> None:
    sample_rate = 22050
    seconds = 4
    audio = np.zeros(sample_rate * seconds, dtype=np.float32)
    audio[sample_rate : sample_rate * 3] = 0.4
    audio_path = tmp_path / "energy_session.wav"
    write_wav(audio_path, audio, sample_rate)
    result = segment_recognized_windows(
        windows_from_fingerprint_rows(song_rows(1, 3, title="Energy Song"), provider="test"),
        recording_duration_s=float(seconds),
        hop_s=1,
    )
    output_path = tmp_path / "energy_timeline.png"

    fig = plot_session_segmentation(result, audio_path=audio_path, output_path=output_path)

    assert output_path.exists()
    assert len(fig.axes) >= 4
    assert fig.axes[1].get_ylabel() == "RMS"
    assert fig.axes[2].get_ylabel() == "BPM"
    assert fig.axes[3].get_ylabel() == "Break"


def test_break_likelihood_peaks_on_rms_and_tempo_disruption() -> None:
    times = np.arange(6, dtype=np.float32)
    rms = (times, np.array([0.6, 0.58, 0.55, 0.05, 0.56, 0.6], dtype=np.float32))
    tempo = (
        times,
        np.array([125, 126, 124, 205, 127, 126], dtype=np.float32),
        np.array([0.75, 0.76, 0.74, 0.15, 0.76, 0.75], dtype=np.float32),
    )

    result = _break_likelihood_diagnostics(rms, tempo)

    assert result is not None
    _, likelihood = result
    assert likelihood[3] > 0.75
    assert likelihood[0] < 0.25


def fake_row(
    start_s: float,
    title: str = "",
    artist: str = "",
    *,
    status: str = "matched",
    confidence: float | None = 0.9,
    isrc: str = "",
    shazam_id: str = "",
    acrid: str = "",
) -> dict[str, object]:
    recognized = status == "matched" and bool(title or artist or isrc or shazam_id or acrid)
    return {
        "mode": "window",
        "window_start_s": float(start_s),
        "window_end_s": float(start_s + 10.0),
        "matched_title": title,
        "matched_artist": artist,
        "status": status,
        "recognized": recognized,
        "confidence": confidence,
        "isrc": isrc,
        "shazam_id": shazam_id,
        "acrid": acrid,
        "spotify_id": "",
        "apple_music_id": "",
        "audio_path": f"/tmp/window_{start_s:.1f}.wav",
        "audio_file": f"window_{start_s:.1f}.wav",
        "error": "",
    }


def song_rows(
    start_s: int,
    end_s: int,
    *,
    title: str,
    artist: str = "Demo Artist",
    isrc: str = "",
) -> list[dict[str, object]]:
    return [
        fake_row(start, title=title, artist=artist, isrc=isrc)
        for start in range(start_s, end_s, 5)
    ]
