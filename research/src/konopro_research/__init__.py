"""Konopro singing-progress research prototype."""

from konopro_research.baseline import (
    MelodyBaseline,
    MelodyNote,
    baseline_from_rows,
    baseline_to_csv_text,
    baseline_to_rows,
    load_baseline_csv,
    write_baseline_csv,
)
from konopro_research.contour_scoring import (
    compare_takes_to_reference_contour,
    score_take_against_reference_contour,
)
from konopro_research.fingerprint_diagnostics import (
    FingerprintDiagnosticReport,
    FingerprintRecoverySweep,
    diagnose_fingerprint_rows,
    load_fingerprint_rows_csv,
    plan_recovery_sweeps,
)
from konopro_research.matching import build_demo_section_catalog, match_query_to_sections
from konopro_research.progress_scoring import (
    MatchedProgressScore,
    MatchedTakeWindow,
    ProgressConfidence,
    score_matched_section_progress,
)
from konopro_research.scoring import compare_takes, score_take
from konopro_research.session_segmentation import (
    RecognizedWindow,
    SessionSegmentationResult,
    SongIdentity,
    SongInterval,
    WeakSongCandidate,
    segment_long_recording,
    segment_recognized_windows,
    windows_from_fingerprint_rows,
)
from konopro_research.separation import prepare_vocal_analysis_audio

__all__ = [
    "MelodyBaseline",
    "MelodyNote",
    "FingerprintDiagnosticReport",
    "FingerprintRecoverySweep",
    "MatchedProgressScore",
    "MatchedTakeWindow",
    "ProgressConfidence",
    "RecognizedWindow",
    "SessionSegmentationResult",
    "SongIdentity",
    "SongInterval",
    "WeakSongCandidate",
    "baseline_from_rows",
    "baseline_to_csv_text",
    "baseline_to_rows",
    "build_demo_section_catalog",
    "compare_takes_to_reference_contour",
    "compare_takes",
    "diagnose_fingerprint_rows",
    "load_baseline_csv",
    "load_fingerprint_rows_csv",
    "match_query_to_sections",
    "segment_long_recording",
    "segment_recognized_windows",
    "score_take_against_reference_contour",
    "score_matched_section_progress",
    "score_take",
    "prepare_vocal_analysis_audio",
    "plan_recovery_sweeps",
    "windows_from_fingerprint_rows",
    "write_baseline_csv",
]
