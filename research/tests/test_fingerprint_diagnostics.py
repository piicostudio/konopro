from __future__ import annotations

from konopro_research.fingerprint_diagnostics import (
    diagnose_fingerprint_rows,
    load_fingerprint_rows_csv,
    plan_recovery_sweeps,
)


def test_sparse_acrcloud_singleton_is_insufficient_evidence() -> None:
    rows = [
        fingerprint_row(start, status="no_match", confidence=None)
        for start in range(0, 630, 30)
    ]
    rows[16] = fingerprint_row(
        480,
        title="You(7525)",
        artist="KY Noraebang",
        isrc="qzncb2169572",
        confidence=0.34,
    )

    report = diagnose_fingerprint_rows(
        rows,
        provider="acrcloud",
        recording_duration_s=630,
        requested_window_s=5,
        requested_hop_s=30,
    )

    assert report.profile.tested_windows == 21
    assert report.profile.recognized_windows == 1
    assert report.can_segment is False
    assert flag_codes(report) >= {
        "sparse_scan",
        "short_windows",
        "high_no_match_rate",
        "low_confidence_match",
        "singleton_candidate",
    }
    assert len(report.weak_candidates) == 1
    assert report.weak_candidates[0].reason == "singleton_match"
    assert report.weak_candidates[0].confidence == 0.34


def test_dense_repeated_match_can_be_segmented() -> None:
    rows = [
        fingerprint_row(start, status="no_match", confidence=None)
        for start in range(0, 80, 5)
    ]
    rows[4] = fingerprint_row(20, title="Repeated", artist="Singer", isrc="ISRC-1", confidence=0.82)
    rows[5] = fingerprint_row(25, title="Repeated", artist="Singer", isrc="ISRC-1", confidence=0.84)
    rows[6] = fingerprint_row(30, title="Repeated", artist="Singer", isrc="ISRC-1", confidence=0.8)

    report = diagnose_fingerprint_rows(
        rows,
        provider="shazamkit",
        recording_duration_s=80,
        requested_window_s=10,
        requested_hop_s=5,
    )

    assert report.can_segment is True
    assert report.confidence_level == "recoverable"
    assert not {"sparse_scan", "short_windows", "singleton_candidate"} & flag_codes(report)
    assert report.weak_candidates[0].reason == "repeated_match"


def test_dense_all_no_match_recommends_source_investigation() -> None:
    rows = [
        fingerprint_row(start, status="no_match", confidence=None)
        for start in range(0, 80, 5)
    ]

    report = diagnose_fingerprint_rows(
        rows,
        provider="audd",
        recording_duration_s=80,
        requested_window_s=10,
        requested_hop_s=5,
    )

    assert report.can_segment is False
    assert "all_no_match" in flag_codes(report)
    assert not report.weak_candidates
    assert any(item["code"] == "source_provider_comparison" for item in report.recommendations)


def test_singleton_never_can_segment_even_with_high_confidence() -> None:
    report = diagnose_fingerprint_rows(
        [fingerprint_row(30, title="Confident Singleton", artist="Singer", confidence=0.99)],
        provider="acrcloud",
        recording_duration_s=100,
        requested_window_s=10,
        requested_hop_s=5,
    )

    assert report.can_segment is False
    assert "singleton_candidate" in flag_codes(report)


def test_repeated_sparse_match_still_needs_recovery() -> None:
    rows = [
        fingerprint_row(120, title="Sparse", artist="Singer", isrc="ISRC-SPARSE", confidence=0.9),
        fingerprint_row(180, title="Sparse", artist="Singer", isrc="ISRC-SPARSE", confidence=0.91),
    ]

    report = diagnose_fingerprint_rows(
        rows,
        provider="acrcloud",
        recording_duration_s=240,
        requested_window_s=5,
        requested_hop_s=60,
    )

    assert report.can_segment is False
    assert "sparse_scan" in flag_codes(report)


def test_headerless_csv_rows_are_normalized() -> None:
    csv_text = "\n".join(
        [
            "acrcloud,0,5,no_match,false,,,,null,/tmp/a.wav,a.wav,",
            "acrcloud,480,485,matched,true,You(7525),KY Noraebang,isrc:qzncb2169572,0.34,/tmp/b.wav,b.wav,",
        ]
    )

    rows = load_fingerprint_rows_csv(csv_text)

    assert len(rows) == 2
    assert rows[0]["window_start_s"] == 0.0
    assert rows[0]["recognized"] is False
    assert rows[1]["matched_title"] == "You(7525)"
    assert rows[1]["confidence"] == 0.34


def test_recovery_sweeps_are_budgeted_and_focused() -> None:
    rows = [
        fingerprint_row(start, status="no_match", confidence=None)
        for start in range(0, 630, 30)
    ]
    rows[16] = fingerprint_row(
        480,
        title="You(7525)",
        artist="KY Noraebang",
        isrc="qzncb2169572",
        confidence=0.34,
    )
    report = diagnose_fingerprint_rows(
        rows,
        provider="acrcloud",
        recording_duration_s=630,
        requested_window_s=5,
        requested_hop_s=30,
    )

    sweeps = plan_recovery_sweeps(report, recording_duration_s=630, request_budget=120)

    assert sweeps[0].name == "Dense full-session retry"
    assert sweeps[0].window_s >= 10
    assert sweeps[0].hop_s == 5
    assert sweeps[0].estimated_api_calls <= 120
    assert any(sweep.name == "Focused singleton recovery" for sweep in sweeps)
    focused = next(sweep for sweep in sweeps if sweep.name == "Focused singleton recovery")
    assert focused.start_s <= 480 <= focused.end_s
    assert focused.estimated_api_calls <= 120


def test_headered_csv_preserves_unknown_columns() -> None:
    csv_text = "\n".join(
        [
            "provider,window_start_s,window_end_s,status,recognized,matched_title,extra",
            "shazamkit,10,20,matched,true,Demo Song,kept",
        ]
    )

    rows = load_fingerprint_rows_csv(csv_text)

    assert rows[0]["provider"] == "shazamkit"
    assert rows[0]["window_start_s"] == 10.0
    assert rows[0]["extra"] == "kept"


def flag_codes(report) -> set[str]:
    return {flag.code for flag in report.flags}


def fingerprint_row(
    start_s: float,
    *,
    title: str = "",
    artist: str = "",
    isrc: str = "",
    status: str = "matched",
    confidence: float | None = 0.9,
) -> dict[str, object]:
    recognized = status == "matched" and bool(title or artist or isrc)
    return {
        "provider": "acrcloud",
        "window_start_s": float(start_s),
        "window_end_s": float(start_s + 5.0),
        "status": status,
        "recognized": recognized,
        "matched_title": title,
        "matched_artist": artist,
        "isrc": isrc,
        "confidence": confidence,
        "audio_path": f"/tmp/window_{start_s:.2f}.wav",
        "audio_file": f"window_{start_s:.2f}.wav",
        "error": "",
    }
