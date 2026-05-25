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
from konopro_research.matching import build_demo_section_catalog, match_query_to_sections
from konopro_research.scoring import compare_takes, score_take
from konopro_research.separation import prepare_vocal_analysis_audio

__all__ = [
    "MelodyBaseline",
    "MelodyNote",
    "baseline_from_rows",
    "baseline_to_csv_text",
    "baseline_to_rows",
    "build_demo_section_catalog",
    "compare_takes_to_reference_contour",
    "compare_takes",
    "load_baseline_csv",
    "match_query_to_sections",
    "score_take_against_reference_contour",
    "score_take",
    "prepare_vocal_analysis_audio",
    "write_baseline_csv",
]
