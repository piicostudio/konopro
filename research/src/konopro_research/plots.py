from __future__ import annotations

import numpy as np

from konopro_research.baseline import MelodyBaseline, hz_to_midi
from konopro_research.pitch import PitchContour


def plot_take_comparison(
    baseline: MelodyBaseline,
    previous: PitchContour | None,
    current: PitchContour,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.8))
    _plot_baseline(ax, baseline)
    if previous is not None:
        _plot_contour(ax, previous, "Previous take", "#d97706")
    _plot_contour(ax, current, "Current take", "#2563eb")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch (MIDI note)")
    ax.set_title("Reference melody vs vocal takes")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_contour_comparison(
    reference: PitchContour,
    previous: PitchContour | None,
    current: PitchContour,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.8))
    _plot_contour(ax, reference, "Reference contour", "#111827", linewidth=2.0)
    if previous is not None:
        _plot_contour(ax, previous, "Previous take", "#d97706")
    _plot_contour(ax, current, "Current take", "#2563eb")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch (MIDI note)")
    ax.set_title("Reference contour vs vocal takes")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_reference_extraction(baseline: MelodyBaseline, reference_contour: PitchContour | None = None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3.8))
    _plot_baseline(ax, baseline)
    if reference_contour is not None:
        _plot_contour(ax, reference_contour, "Extracted pitch frames", "#6b7280")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch (MIDI note)")
    ax.set_title("Extracted reference baseline")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_voiced_coverage(
    baseline: MelodyBaseline,
    previous: PitchContour | None,
    current: PitchContour,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 2.8))
    if previous is None:
        _coverage_row(ax, baseline, y=1, label="Reference", color="#111827")
        _voiced_row(ax, current, y=0, label="Current voiced", color="#2563eb")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Current", "Reference"])
        ax.set_ylim(-0.6, 1.6)
    else:
        _coverage_row(ax, baseline, y=2, label="Reference", color="#111827")
        _voiced_row(ax, previous, y=1, label="Previous voiced", color="#d97706")
        _voiced_row(ax, current, y=0, label="Current voiced", color="#2563eb")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Current", "Previous", "Reference"])
        ax.set_ylim(-0.6, 2.6)
    ax.set_xlabel("Time (s)")
    ax.set_title("Voiced-frame coverage")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_contour_voiced_coverage(
    reference: PitchContour,
    previous: PitchContour | None,
    current: PitchContour,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 2.8))
    if previous is None:
        _voiced_row(ax, reference, y=1, label="Reference voiced", color="#111827")
        _voiced_row(ax, current, y=0, label="Current voiced", color="#2563eb")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Current", "Reference"])
        ax.set_ylim(-0.6, 1.6)
    else:
        _voiced_row(ax, reference, y=2, label="Reference voiced", color="#111827")
        _voiced_row(ax, previous, y=1, label="Previous voiced", color="#d97706")
        _voiced_row(ax, current, y=0, label="Current voiced", color="#2563eb")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Current", "Previous", "Reference"])
        ax.set_ylim(-0.6, 2.6)
    ax.set_xlabel("Time (s)")
    ax.set_title("Voiced-frame coverage")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def _plot_baseline(ax, baseline: MelodyBaseline) -> None:
    for note in baseline.notes:
        ax.hlines(
            note.midi,
            note.start_s,
            note.end_s,
            color="#111827",
            linewidth=3,
            label="Reference" if note == baseline.notes[0] else None,
        )


def _plot_contour(ax, contour: PitchContour, label: str, color: str, *, linewidth: float = 1.5) -> None:
    mask = contour.voiced_mask
    if not np.any(mask):
        return
    midi = hz_to_midi(contour.frequencies_hz[mask])
    ax.plot(contour.times_s[mask], midi, color=color, linewidth=linewidth, alpha=0.9, label=label)


def _coverage_row(ax, baseline: MelodyBaseline, *, y: int, label: str, color: str) -> None:
    for index, note in enumerate(baseline.notes):
        ax.hlines(
            y,
            note.start_s,
            note.end_s,
            color=color,
            linewidth=6,
            alpha=0.85,
            label=label if index == 0 else None,
        )


def _voiced_row(ax, contour: PitchContour, *, y: int, label: str, color: str) -> None:
    mask = contour.voiced_mask
    if not np.any(mask):
        return
    ax.scatter(
        contour.times_s[mask],
        np.full(np.count_nonzero(mask), y),
        color=color,
        s=5,
        alpha=0.65,
        label=label,
    )
