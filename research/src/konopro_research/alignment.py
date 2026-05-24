from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from konopro_research.baseline import MelodyBaseline, cents_difference
from konopro_research.pitch import PitchContour


@dataclass(frozen=True)
class AlignmentResult:
    offset_s: float
    median_abs_error_cents: float
    matched_frame_ratio: float


def estimate_global_offset(
    contour: PitchContour,
    baseline: MelodyBaseline,
    *,
    search_radius_s: float = 0.50,
    step_s: float = 0.02,
) -> AlignmentResult:
    voiced = contour.voiced_mask
    if not np.any(voiced):
        return AlignmentResult(0.0, float("inf"), 0.0)

    best: AlignmentResult | None = None
    offsets = np.arange(-search_radius_s, search_radius_s + step_s / 2.0, step_s)
    for offset in offsets:
        reference_times = contour.times_s - offset
        expected = baseline.hz_at(reference_times)
        mask = voiced & np.isfinite(expected) & (expected > 0)
        if np.count_nonzero(mask) < 3:
            continue
        errors = np.abs(cents_difference(contour.frequencies_hz[mask], expected[mask]))
        matched_ratio = float(np.count_nonzero(mask) / max(1, np.count_nonzero(voiced)))
        # Keep alignment conservative. A wrong but stable pitch should not be
        # "fixed" by sliding the take onto a neighboring reference note.
        cost = float(
            np.nanmedian(errors)
            + (1.0 - matched_ratio) * 25.0
            + abs(offset) * 400.0
        )
        candidate = AlignmentResult(float(offset), cost, matched_ratio)
        if best is None or candidate.median_abs_error_cents < best.median_abs_error_cents:
            best = candidate

    return best or AlignmentResult(0.0, float("inf"), 0.0)
