from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from konopro_research.audio_io import load_audio
from konopro_research.baseline import MelodyBaseline, MelodyNote, demo_baseline, hz_to_midi, midi_to_hz
from konopro_research.pitch import PitchContour, clean_pitch_contour, extract_pitch


@dataclass(frozen=True)
class SongSection:
    song_id: str
    song_title: str
    section_label: str
    start_s: float
    end_s: float
    contour: PitchContour
    source: str = "demo"

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    @property
    def display_name(self) -> str:
        return f"{self.song_title} - {self.section_label}"

    def to_dict(self) -> dict[str, object]:
        return {
            "song_id": self.song_id,
            "song_title": self.song_title,
            "section": self.section_label,
            "source": self.source,
            "start_s": round(self.start_s, 2),
            "end_s": round(self.end_s, 2),
            "duration_s": round(self.duration_s, 2),
        }


@dataclass(frozen=True)
class SectionMatch:
    section: SongSection
    score: float
    shape_score: float
    coverage_score: float
    duration_score: float
    mean_shape_error_cents: float
    query_start_s: float
    query_end_s: float
    duration_ratio: float
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "song": self.section.song_title,
            "section": self.section.section_label,
            "source": self.section.source,
            "score": round(self.score, 1),
            "shape": round(self.shape_score, 1),
            "coverage": round(self.coverage_score, 1),
            "duration_fit": round(self.duration_score, 1),
            "shape_error_cents": round(self.mean_shape_error_cents, 1),
            "reference_start_s": round(self.section.start_s, 2),
            "reference_end_s": round(self.section.end_s, 2),
            "query_start_s": round(self.query_start_s, 2),
            "query_end_s": round(self.query_end_s, 2),
            "duration_ratio": round(self.duration_ratio, 2),
        }


@dataclass(frozen=True)
class SectionMatchResult:
    query: PitchContour
    candidates: tuple[SectionMatch, ...]
    warnings: tuple[str, ...] = ()

    @property
    def best(self) -> SectionMatch | None:
        return self.candidates[0] if self.candidates else None

    def to_dict(self) -> dict[str, object]:
        return {
            "warnings": list(self.warnings),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def extract_matching_query(
    path_or_file: str | Path,
    *,
    name: str = "query",
    pitch_kwargs: dict[str, object] | None = None,
    clean_kwargs: dict[str, object] | None = None,
) -> PitchContour:
    audio, sample_rate = load_audio(path_or_file)
    return clean_pitch_contour(
        extract_pitch(audio, sample_rate, name=name, **(pitch_kwargs or {})),
        **(clean_kwargs or {}),
    )


def build_demo_section_catalog() -> tuple[SongSection, ...]:
    """Return a small public/synthetic catalog for TA-safe section retrieval."""
    sections = [
        _section_from_baseline(
            demo_baseline(),
            song_id="demo_city_lights",
            song_title="Konopro Demo Song",
            section_label="Chorus",
        ),
        _section_from_notes(
            [60, 62, 63, 65, 63, 62, 60, 58],
            song_id="demo_city_lights",
            song_title="Konopro Demo Song",
            section_label="Verse",
        ),
        _section_from_notes(
            [69, 67, 65, 64, 62, 60, 62, 64],
            song_id="demo_river_rain",
            song_title="River Rain Demo",
            section_label="Chorus",
        ),
        _section_from_notes(
            [55, 62, 59, 64, 60, 67, 64, 62],
            song_id="demo_neon_drive",
            song_title="Neon Drive Demo",
            section_label="Hook",
        ),
    ]
    return tuple(sections)


def sections_from_baseline(
    baseline: MelodyBaseline,
    *,
    song_id: str = "uploaded_baseline",
    song_title: str | None = None,
    section_label: str = "Full baseline",
) -> tuple[SongSection, ...]:
    return (
        _section_from_baseline(
            baseline,
            song_id=song_id,
            song_title=song_title or baseline.title,
            section_label=section_label,
            source="baseline",
        ),
    )


def split_contour_into_sections(
    contour: PitchContour,
    *,
    song_id: str = "uploaded_reference",
    song_title: str = "Uploaded reference",
    window_s: float = 20.0,
    hop_s: float = 10.0,
    min_voiced_frames: int = 24,
    max_sections: int = 24,
) -> tuple[SongSection, ...]:
    if contour.times_s.size == 0:
        return ()

    start = float(np.nanmin(contour.times_s))
    stop = float(np.nanmax(contour.times_s))
    if stop <= start:
        return ()

    window_s = max(0.5, float(window_s))
    hop_s = max(0.1, float(hop_s))
    sections: list[SongSection] = []
    cursor = start
    index = 1
    while cursor < stop and len(sections) < max_sections:
        end = min(cursor + window_s, stop)
        cropped = crop_contour(contour, cursor, end, name=f"{song_title} section {index}")
        if np.count_nonzero(cropped.voiced_mask) >= min_voiced_frames:
            sections.append(
                SongSection(
                    song_id=song_id,
                    song_title=song_title,
                    section_label=f"Section {index}",
                    start_s=round(cursor - start, 3),
                    end_s=round(end - start, 3),
                    contour=cropped,
                    source="uploaded_reference",
                )
            )
            index += 1
        cursor += hop_s

    return tuple(sections)


def match_query_to_sections(
    query: PitchContour,
    sections: tuple[SongSection, ...] | list[SongSection],
    *,
    top_k: int = 5,
    sample_count: int = 96,
    query_hop_s: float = 2.0,
    max_query_windows: int = 80,
    transpose_invariant: bool = True,
    shape_error_penalty: float = 0.35,
) -> SectionMatchResult:
    warnings: list[str] = []
    if np.count_nonzero(query.voiced_mask) < 8:
        warnings.append("query has too few voiced frames for reliable song/section matching")

    matches: list[SectionMatch] = []
    for section in sections:
        best = _best_section_match(
            query,
            section,
            sample_count=sample_count,
            query_hop_s=query_hop_s,
            max_query_windows=max_query_windows,
            transpose_invariant=transpose_invariant,
            shape_error_penalty=shape_error_penalty,
        )
        if best is not None:
            matches.append(best)

    matches.sort(key=lambda candidate: candidate.score, reverse=True)
    if not matches:
        warnings.append("no catalog sections contained enough voiced pitch to compare")
    return SectionMatchResult(query=query, candidates=tuple(matches[:top_k]), warnings=tuple(warnings))


def crop_contour(
    contour: PitchContour,
    start_s: float,
    end_s: float,
    *,
    name: str | None = None,
    shift_to_zero: bool = True,
) -> PitchContour:
    mask = (contour.times_s >= start_s) & (contour.times_s <= end_s)
    times = contour.times_s[mask]
    if shift_to_zero and times.size:
        times = times - start_s
    return PitchContour(
        times,
        contour.frequencies_hz[mask],
        contour.confidence[mask],
        name=name or contour.name,
    )


def _best_section_match(
    query: PitchContour,
    section: SongSection,
    *,
    sample_count: int,
    query_hop_s: float,
    max_query_windows: int,
    transpose_invariant: bool,
    shape_error_penalty: float,
) -> SectionMatch | None:
    best: SectionMatch | None = None
    for query_window, query_start, query_end in _query_windows(
        query,
        target_duration_s=section.duration_s,
        hop_s=query_hop_s,
        max_windows=max_query_windows,
    ):
        scored = _score_pair(
            query_window,
            query_start=query_start,
            query_end=query_end,
            section=section,
            sample_count=sample_count,
            transpose_invariant=transpose_invariant,
            shape_error_penalty=shape_error_penalty,
        )
        if scored is not None and (best is None or scored.score > best.score):
            best = scored
    return best


def _score_pair(
    query: PitchContour,
    *,
    query_start: float,
    query_end: float,
    section: SongSection,
    sample_count: int,
    transpose_invariant: bool,
    shape_error_penalty: float,
) -> SectionMatch | None:
    query_curve = _relative_midi_curve(query, sample_count=sample_count, transpose_invariant=transpose_invariant)
    section_curve = _relative_midi_curve(
        section.contour,
        sample_count=sample_count,
        transpose_invariant=transpose_invariant,
    )
    if query_curve is None or section_curve is None:
        return None

    mean_error_cents = _dtw_mean_abs_error_cents(section_curve, query_curve)
    shape_score = _clamp_score(100.0 - mean_error_cents * shape_error_penalty)

    query_voiced = _voiced_ratio(query)
    section_voiced = _voiced_ratio(section.contour)
    coverage_score = _clamp_score(100.0 * min(1.0, query_voiced / 0.35) * min(1.0, section_voiced / 0.35))

    query_duration = _contour_duration(query)
    section_duration = max(section.duration_s, 0.001)
    duration_ratio = query_duration / section_duration
    duration_score = _clamp_score(100.0 - abs(np.log(max(duration_ratio, 0.001))) * 45.0)

    score = 0.72 * shape_score + 0.18 * duration_score + 0.10 * coverage_score
    warnings: list[str] = []
    if duration_score < 60.0:
        warnings.append("query and section durations differ; phrase boundaries may be wrong")

    return SectionMatch(
        section=section,
        score=round(float(score), 2),
        shape_score=round(float(shape_score), 2),
        coverage_score=round(float(coverage_score), 2),
        duration_score=round(float(duration_score), 2),
        mean_shape_error_cents=round(float(mean_error_cents), 2),
        query_start_s=round(float(query_start), 3),
        query_end_s=round(float(query_end), 3),
        duration_ratio=round(float(duration_ratio), 3),
        warnings=tuple(warnings),
    )


def _query_windows(
    query: PitchContour,
    *,
    target_duration_s: float,
    hop_s: float,
    max_windows: int,
) -> list[tuple[PitchContour, float, float]]:
    duration = _contour_duration(query)
    if duration <= 0:
        return [(query, 0.0, 0.0)]
    if target_duration_s <= 0 or duration <= target_duration_s * 1.35:
        return [(query, 0.0, duration)]

    latest_start = max(0.0, duration - target_duration_s)
    starts = np.arange(0.0, latest_start + 0.001, max(0.1, hop_s))
    if starts.size > max_windows:
        starts = np.linspace(0.0, latest_start, num=max_windows)
    return [
        (
            crop_contour(query, float(start), float(start + target_duration_s), shift_to_zero=True),
            float(start),
            float(start + target_duration_s),
        )
        for start in starts
    ]


def _relative_midi_curve(
    contour: PitchContour,
    *,
    sample_count: int,
    transpose_invariant: bool,
) -> np.ndarray | None:
    mask = contour.voiced_mask
    if np.count_nonzero(mask) < 3:
        return None

    times = contour.times_s[mask]
    midi = hz_to_midi(contour.frequencies_hz[mask])
    order = np.argsort(times)
    times = times[order]
    midi = midi[order]
    unique_times, unique_indices = np.unique(times, return_index=True)
    midi = midi[unique_indices]
    if unique_times.size < 3:
        return None

    duration = max(float(unique_times[-1] - unique_times[0]), 0.001)
    normalized_times = (unique_times - unique_times[0]) / duration
    target = np.linspace(0.0, 1.0, max(8, sample_count))
    curve = np.interp(target, normalized_times, midi)
    if transpose_invariant:
        curve = curve - float(np.nanmedian(curve))
    return curve


def _dtw_mean_abs_error_cents(reference_curve: np.ndarray, query_curve: np.ndarray) -> float:
    try:
        import librosa

        _, path = librosa.sequence.dtw(
            X=reference_curve.reshape(1, -1),
            Y=query_curve.reshape(1, -1),
            metric="euclidean",
        )
        pairs = path[::-1]
        errors = np.abs(reference_curve[pairs[:, 0]] - query_curve[pairs[:, 1]])
    except Exception:
        length = min(len(reference_curve), len(query_curve))
        errors = np.abs(reference_curve[:length] - query_curve[:length])
    return float(np.nanmean(errors) * 100.0)


def _section_from_notes(
    midi_notes: list[float],
    *,
    song_id: str,
    song_title: str,
    section_label: str,
) -> SongSection:
    notes: list[MelodyNote] = []
    cursor = 0.0
    for midi in midi_notes:
        notes.append(MelodyNote(cursor, cursor + 0.55, midi))
        cursor += 0.60
    return _section_from_baseline(
        MelodyBaseline(tuple(notes), title=f"{song_title} {section_label}"),
        song_id=song_id,
        song_title=song_title,
        section_label=section_label,
    )


def _section_from_baseline(
    baseline: MelodyBaseline,
    *,
    song_id: str,
    song_title: str,
    section_label: str,
    source: str = "demo",
) -> SongSection:
    return SongSection(
        song_id=song_id,
        song_title=song_title,
        section_label=section_label,
        start_s=0.0,
        end_s=baseline.duration_s,
        contour=_contour_from_baseline(baseline, name=f"{song_title} {section_label}"),
        source=source,
    )


def _contour_from_baseline(baseline: MelodyBaseline, *, name: str) -> PitchContour:
    times: list[float] = []
    hz_values: list[float] = []
    confidence: list[float] = []
    for note in baseline.notes:
        note_times = np.arange(note.start_s + 0.03, note.end_s - 0.03, 0.025)
        if note_times.size == 0:
            continue
        times.extend(note_times.tolist())
        hz_values.extend(np.full(note_times.shape, midi_to_hz(note.midi)).tolist())
        confidence.extend(np.full(note_times.shape, 0.98).tolist())
    return PitchContour(np.asarray(times), np.asarray(hz_values), np.asarray(confidence), name=name)


def _voiced_ratio(contour: PitchContour) -> float:
    return float(np.mean(contour.voiced_mask)) if contour.times_s.size else 0.0


def _contour_duration(contour: PitchContour) -> float:
    if contour.times_s.size == 0:
        return 0.0
    return float(np.nanmax(contour.times_s) - np.nanmin(contour.times_s))


def _clamp_score(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))
