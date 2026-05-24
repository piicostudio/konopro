from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from konopro_research.audio_io import load_audio
from konopro_research.baseline import MelodyBaseline, baseline_from_pitch_contour
from konopro_research.pitch import clean_pitch_contour, extract_pitch
from konopro_research.pitch import PitchContour
from konopro_research.quality import AudioSummary, BaselineQuality, analyze_baseline_quality, summarize_audio


@dataclass(frozen=True)
class ReferenceExtraction:
    baseline: MelodyBaseline
    contour: PitchContour
    audio_summary: AudioSummary
    quality: BaselineQuality


def baseline_from_reference_audio(
    path_or_file: str | Path,
    *,
    title: str = "Experimental audio-derived baseline",
    window_s: float = 0.20,
    pitch_kwargs: dict[str, object] | None = None,
    clean_kwargs: dict[str, object] | None = None,
) -> MelodyBaseline:
    """Build a dense baseline from BYO reference audio.

    This is intentionally experimental. It is useful for private real-song tests,
    but symbolic baselines are the reliable demo path.
    """
    audio, sample_rate = load_audio(path_or_file)
    contour = clean_pitch_contour(
        extract_pitch(audio, sample_rate, name=title, **(pitch_kwargs or {})),
        **(clean_kwargs or {}),
    )
    return baseline_from_pitch_contour(
        contour.times_s,
        contour.frequencies_hz,
        title=title,
        window_s=window_s,
    )


def extract_reference_audio(
    path_or_file: str | Path,
    *,
    title: str = "Experimental audio-derived baseline",
    window_s: float = 0.20,
    pitch_kwargs: dict[str, object] | None = None,
    clean_kwargs: dict[str, object] | None = None,
) -> ReferenceExtraction:
    audio, sample_rate = load_audio(path_or_file)
    contour = clean_pitch_contour(
        extract_pitch(audio, sample_rate, name=title, **(pitch_kwargs or {})),
        **(clean_kwargs or {}),
    )
    baseline = baseline_from_pitch_contour(
        contour.times_s,
        contour.frequencies_hz,
        title=title,
        window_s=window_s,
    )
    if isinstance(path_or_file, (str, Path)):
        audio_summary = summarize_audio(path_or_file)
    else:
        duration = len(audio) / sample_rate if sample_rate else 0.0
        audio_summary = AudioSummary(
            duration_s=round(duration, 3),
            sample_rate=sample_rate,
            rms=0.0,
            peak=0.0,
            clipping_ratio=0.0,
            warnings=(),
        )
    quality = analyze_baseline_quality(baseline, source_duration_s=audio_summary.duration_s)
    return ReferenceExtraction(
        baseline=baseline,
        contour=contour,
        audio_summary=audio_summary,
        quality=quality,
    )
