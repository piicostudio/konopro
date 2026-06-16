from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from konopro_research.audio_io import load_audio
from konopro_research.baseline import MelodyBaseline, baseline_from_pitch_contour
from konopro_research.pitch import clean_pitch_contour, extract_pitch
from konopro_research.pitch import PitchContour
from konopro_research.quality import AudioSummary, BaselineQuality, analyze_baseline_quality, summarize_audio_from_array


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
    cache_dir: str | Path | None = None,
    source_hash: str | None = None,
) -> ReferenceExtraction:
    # Load audio once — reuse the array for pitch extraction and summary
    audio, sample_rate = load_audio(path_or_file)

    # Try to load a cached pitch contour
    contour = _load_cached_contour(
        cache_dir, source_hash, pitch_kwargs, clean_kwargs,
    )
    if contour is None:
        contour = clean_pitch_contour(
            extract_pitch(audio, sample_rate, name=title, **(pitch_kwargs or {})),
            **(clean_kwargs or {}),
        )
        _save_contour_cache(
            contour, cache_dir, source_hash, pitch_kwargs, clean_kwargs,
        )

    baseline = baseline_from_pitch_contour(
        contour.times_s,
        contour.frequencies_hz,
        title=title,
        window_s=window_s,
    )
    # Compute audio summary directly from the already-loaded array (avoids a re-read)
    audio_summary = summarize_audio_from_array(audio, sample_rate)
    quality = analyze_baseline_quality(baseline, source_duration_s=audio_summary.duration_s)
    return ReferenceExtraction(
        baseline=baseline,
        contour=contour,
        audio_summary=audio_summary,
        quality=quality,
    )


# ---------------------------------------------------------------------------
# Pitch contour caching helpers
# ---------------------------------------------------------------------------

def _contour_cache_key(
    source_hash: str | None,
    pitch_kwargs: dict[str, object] | None,
    clean_kwargs: dict[str, object] | None,
) -> str | None:
    """Build a deterministic cache key from the source hash and pitch parameters."""
    if not source_hash:
        return None
    key_data = {
        "source_hash": source_hash,
        "pitch_kwargs": json.dumps(pitch_kwargs or {}, sort_keys=True),
        "clean_kwargs": json.dumps(clean_kwargs or {}, sort_keys=True),
    }
    encoded = json.dumps(key_data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _load_cached_contour(
    cache_dir: str | Path | None,
    source_hash: str | None,
    pitch_kwargs: dict[str, object] | None,
    clean_kwargs: dict[str, object] | None,
) -> PitchContour | None:
    if cache_dir is None:
        return None
    key = _contour_cache_key(source_hash, pitch_kwargs, clean_kwargs)
    if key is None:
        return None
    cache_path = Path(cache_dir) / "pitch_contours" / f"{key}.npz"
    if not cache_path.exists():
        return None
    try:
        data = np.load(cache_path, allow_pickle=False)
        return PitchContour(
            times_s=data["times_s"],
            frequencies_hz=data["frequencies_hz"],
            confidence=data["confidence"],
            name=str(data.get("name", "cached")),
        )
    except Exception:
        return None


def _save_contour_cache(
    contour: PitchContour,
    cache_dir: str | Path | None,
    source_hash: str | None,
    pitch_kwargs: dict[str, object] | None,
    clean_kwargs: dict[str, object] | None,
) -> None:
    if cache_dir is None:
        return
    key = _contour_cache_key(source_hash, pitch_kwargs, clean_kwargs)
    if key is None:
        return
    cache_path = Path(cache_dir) / "pitch_contours" / f"{key}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.savez_compressed(
            cache_path,
            times_s=contour.times_s,
            frequencies_hz=contour.frequencies_hz,
            confidence=contour.confidence,
            name=np.array(contour.name),
        )
    except Exception:
        pass
