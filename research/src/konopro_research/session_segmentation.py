from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from konopro_research.audio_io import load_audio, write_wav


@dataclass(frozen=True)
class SongIdentity:
    provider: str
    key: str
    title: str
    artist: str
    isrc: str = ""
    provider_id: str = ""
    normalized_title: str = ""
    normalized_artist: str = ""
    quality: str = "high"

    @property
    def display_name(self) -> str:
        if self.title and self.artist:
            return f"{self.title} - {self.artist}"
        return self.title or self.artist or self.key

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecognizedWindow:
    provider: str
    start_s: float
    end_s: float
    status: str
    recognized: bool
    identity: SongIdentity | None = None
    confidence: float | None = None
    audio_path: str = ""
    audio_file: str = ""
    error: str = ""
    row: dict[str, Any] | None = None

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "status": self.status,
            "recognized": self.recognized,
            "title": self.identity.title if self.identity else "",
            "artist": self.identity.artist if self.identity else "",
            "identity_key": self.identity.key if self.identity else "",
            "confidence": self.confidence,
            "audio_path": self.audio_path,
            "audio_file": self.audio_file,
            "error": self.error,
        }


@dataclass(frozen=True)
class SongInterval:
    index: int
    identity: SongIdentity
    start_s: float
    end_s: float
    confidence_score: float
    confidence_level: str
    recognized_window_count: int
    total_window_count: int
    gap_window_count: int
    conflict_window_count: int
    provider_confidence: float
    warnings: tuple[str, ...] = ()

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "song": self.identity.title,
            "artist": self.identity.artist,
            "identity_key": self.identity.key,
            "start_s": round(self.start_s, 2),
            "end_s": round(self.end_s, 2),
            "duration_s": round(self.duration_s, 2),
            "confidence": round(self.confidence_score, 1),
            "confidence_level": self.confidence_level,
            "recognized_windows": self.recognized_window_count,
            "total_windows": self.total_window_count,
            "gap_windows": self.gap_window_count,
            "conflict_windows": self.conflict_window_count,
            "provider_confidence": round(self.provider_confidence, 3),
            "warnings": "; ".join(self.warnings),
        }


@dataclass(frozen=True)
class WeakSongCandidate:
    index: int
    identity: SongIdentity
    start_s: float
    end_s: float
    recognized_window_count: int
    total_window_count: int
    provider_confidence: float
    reason: str
    recovery_start_s: float
    recovery_end_s: float
    warnings: tuple[str, ...] = ()

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "song": self.identity.title,
            "artist": self.identity.artist,
            "identity_key": self.identity.key,
            "start_s": round(self.start_s, 2),
            "end_s": round(self.end_s, 2),
            "duration_s": round(self.duration_s, 2),
            "recognized_windows": self.recognized_window_count,
            "total_windows": self.total_window_count,
            "provider_confidence": round(self.provider_confidence, 3),
            "reason": self.reason,
            "recovery_start_s": round(self.recovery_start_s, 2),
            "recovery_end_s": round(self.recovery_end_s, 2),
            "warnings": "; ".join(self.warnings),
        }


@dataclass(frozen=True)
class SessionSegmentationResult:
    windows: tuple[RecognizedWindow, ...]
    intervals: tuple[SongInterval, ...]
    recording_duration_s: float
    hop_s: float
    weak_candidates: tuple[WeakSongCandidate, ...] = ()
    warnings: tuple[str, ...] = ()
    provider_result: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "recording_duration_s": round(self.recording_duration_s, 3),
            "hop_s": round(self.hop_s, 3),
            "warnings": list(self.warnings),
            "windows": [window.to_dict() for window in self.windows],
            "intervals": [interval.to_dict() for interval in self.intervals],
            "weak_candidates": [candidate.to_dict() for candidate in self.weak_candidates],
        }


def identity_from_row(row: dict[str, Any], provider: str) -> SongIdentity | None:
    title = _string(row.get("matched_title") or row.get("title"))
    artist = _string(row.get("matched_artist") or row.get("artist"))
    isrc = _string(row.get("isrc"))
    normalized_title = _normalized(title)
    normalized_artist = _normalized(artist)

    if isrc:
        return SongIdentity(
            provider=provider,
            key=f"isrc:{_normalized_id(isrc)}",
            title=title,
            artist=artist,
            isrc=isrc,
            normalized_title=normalized_title,
            normalized_artist=normalized_artist,
            quality="high",
        )

    for field in ("shazam_id", "acrid", "spotify_id", "apple_music_id"):
        provider_id = _string(row.get(field))
        if provider_id:
            return SongIdentity(
                provider=provider,
                key=f"{field}:{_normalized_id(provider_id)}",
                title=title,
                artist=artist,
                provider_id=provider_id,
                normalized_title=normalized_title,
                normalized_artist=normalized_artist,
                quality="high",
            )

    if normalized_title and normalized_artist:
        return SongIdentity(
            provider=provider,
            key=f"title_artist:{normalized_title}::{normalized_artist}",
            title=title,
            artist=artist,
            normalized_title=normalized_title,
            normalized_artist=normalized_artist,
            quality="medium",
        )

    if normalized_title:
        return SongIdentity(
            provider=provider,
            key=f"title:{normalized_title}",
            title=title,
            artist=artist,
            normalized_title=normalized_title,
            normalized_artist=normalized_artist,
            quality="low",
        )

    return None


def windows_from_fingerprint_rows(
    rows: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    provider: str,
) -> tuple[RecognizedWindow, ...]:
    windows: list[RecognizedWindow] = []
    for row in rows:
        status = _string(row.get("status") or "unknown")
        recognized = bool(row.get("recognized")) and status == "matched"
        identity = identity_from_row(row, provider) if recognized else None
        if identity is None:
            recognized = False
        windows.append(
            RecognizedWindow(
                provider=provider,
                start_s=_float(row.get("window_start_s")),
                end_s=_float(row.get("window_end_s")),
                status=status,
                recognized=recognized,
                identity=identity,
                confidence=_optional_confidence(row.get("confidence")),
                audio_path=_string(row.get("audio_path")),
                audio_file=_string(row.get("audio_file")),
                error=_string(row.get("error")),
                row=dict(row),
            )
        )
    return tuple(sorted(windows, key=lambda window: (window.start_s, window.end_s)))


def segment_recognized_windows(
    windows: tuple[RecognizedWindow, ...] | list[RecognizedWindow],
    *,
    recording_duration_s: float,
    hop_s: float,
    min_recognized_windows: int = 2,
    allowed_gap_windows: int = 1,
    conflict_split_windows: int = 2,
    provider_result: Any | None = None,
) -> SessionSegmentationResult:
    ordered = tuple(sorted(windows, key=lambda window: (window.start_s, window.end_s)))
    warnings: list[str] = []
    segments: list[tuple[SongIdentity, list[RecognizedWindow]]] = []
    active_identity: SongIdentity | None = None
    active_windows: list[RecognizedWindow] = []
    pending_conflicts: list[RecognizedWindow] = []
    gap_count = 0

    def finalize_active(reason: str = "") -> None:
        nonlocal active_identity, active_windows, pending_conflicts, gap_count
        if active_identity is not None and active_windows:
            segments.append((active_identity, list(active_windows)))
            if reason:
                warnings.append(reason)
        active_identity = None
        active_windows = []
        pending_conflicts = []
        gap_count = 0

    for window in ordered:
        if not window.recognized or window.identity is None:
            if active_identity is None:
                continue
            if gap_count < allowed_gap_windows:
                active_windows.append(window)
                gap_count += 1
                continue
            finalize_active()
            continue

        if active_identity is None:
            active_identity = window.identity
            active_windows = [window]
            pending_conflicts = []
            gap_count = 0
            continue

        if window.identity.key == active_identity.key:
            if pending_conflicts:
                active_windows.extend(pending_conflicts)
                pending_conflicts = []
                warnings.append(
                    f"Conflicting single-window recognition inside {active_identity.display_name}."
                )
            active_windows.append(window)
            gap_count = 0
            continue

        pending_conflicts.append(window)
        if len(pending_conflicts) >= conflict_split_windows:
            conflict_name = window.identity.display_name
            current_name = active_identity.display_name
            new_active_windows = list(pending_conflicts)
            new_active_identity = pending_conflicts[0].identity
            active_windows.extend(new_active_windows)
            finalize_active(
                f"Split interval after conflicting recognition: {current_name} -> {conflict_name}."
            )
            active_identity = new_active_identity
            active_windows = new_active_windows
            pending_conflicts = []
            gap_count = 0

    if pending_conflicts and active_windows:
        active_windows.extend(pending_conflicts)
        warnings.append(f"Conflicting trailing recognition inside {active_identity.display_name}.")
    finalize_active()

    intervals: list[SongInterval] = []
    weak_candidates: list[WeakSongCandidate] = []
    for identity, segment_windows in segments:
        interval = _interval_from_segment(
            len(intervals) + 1,
            identity,
            segment_windows,
            recording_duration_s=recording_duration_s,
            hop_s=hop_s,
            min_recognized_windows=min_recognized_windows,
        )
        if interval is None:
            weak_candidates.append(
                _weak_candidate_from_segment(
                    len(weak_candidates) + 1,
                    identity,
                    segment_windows,
                    recording_duration_s=recording_duration_s,
                    min_recognized_windows=min_recognized_windows,
                )
            )
            warnings.append(f"Rejected short/weak interval for {identity.display_name}.")
            continue
        intervals.append(interval)

    if not intervals:
        warnings.append("No recognized song intervals passed the confidence gates.")

    return SessionSegmentationResult(
        windows=ordered,
        intervals=tuple(intervals),
        recording_duration_s=float(recording_duration_s),
        hop_s=float(hop_s),
        weak_candidates=tuple(weak_candidates),
        warnings=tuple(dict.fromkeys(warnings)),
        provider_result=provider_result,
    )


def segment_long_recording(
    audio_path: str | Path | None,
    output_dir: str | Path,
    *,
    provider: str = "shazamkit",
    window_s: float = 10.0,
    hop_s: float = 5.0,
    max_windows: int | None = None,
    recognizer: Any | None = None,
    use_whole: bool = False,
    **provider_kwargs: Any,
) -> SessionSegmentationResult:
    if audio_path is None:
        return SessionSegmentationResult(
            windows=(),
            intervals=(),
            recording_duration_s=0.0,
            hop_s=float(hop_s),
            warnings=("No long recording selected.",),
        )

    audio, sample_rate = load_audio(audio_path, target_sr=44100)
    duration_s = len(audio) / sample_rate if sample_rate else 0.0
    if max_windows is None:
        max_windows = _window_count(duration_s, window_s=window_s, hop_s=hop_s)

    mode = "Whole + sliding windows" if use_whole else "Sliding windows"
    provider_key = provider.strip().casefold()
    if provider_key == "shazamkit":
        from konopro_research.fingerprinting import run_shazam_fingerprinting

        provider_result = run_shazam_fingerprinting(
            audio_path,
            output_dir,
            mode=mode,
            window_s=float(window_s),
            hop_s=float(hop_s),
            max_windows=int(max_windows),
            recognizer=recognizer,
            **provider_kwargs,
        )
    elif provider_key == "audd":
        from konopro_research.fingerprinting import run_audd_fingerprinting

        provider_result = run_audd_fingerprinting(
            audio_path,
            output_dir,
            mode=mode,
            window_s=float(window_s),
            hop_s=float(hop_s),
            max_windows=int(max_windows),
            recognizer=recognizer,
            **provider_kwargs,
        )
    elif provider_key == "acrcloud":
        from konopro_research.fingerprinting import run_acrcloud_fingerprinting

        provider_result = run_acrcloud_fingerprinting(
            audio_path,
            output_dir,
            mode=mode,
            window_s=float(window_s),
            hop_s=float(hop_s),
            max_windows=int(max_windows),
            recognizer=recognizer,
            **provider_kwargs,
        )
    else:
        raise ValueError(f"Unknown segmentation provider: {provider}")

    windows = windows_from_fingerprint_rows(provider_result.rows, provider=provider_key)
    return segment_recognized_windows(
        windows,
        recording_duration_s=duration_s,
        hop_s=float(hop_s),
        provider_result=provider_result,
    )


def write_interval_clips(
    audio_path: str | Path,
    intervals: tuple[SongInterval, ...] | list[SongInterval],
    output_dir: str | Path,
    *,
    pad_s: float = 0.0,
    max_clips: int | None = None,
) -> tuple[dict[str, Any], ...]:
    audio, sample_rate = load_audio(audio_path)
    duration_s = len(audio) / sample_rate if sample_rate else 0.0
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for interval in list(intervals)[:max_clips]:
        start_s = max(0.0, interval.start_s - pad_s)
        end_s = min(duration_s, interval.end_s + pad_s)
        if end_s <= start_s:
            continue
        start_index = int(round(start_s * sample_rate))
        end_index = int(round(end_s * sample_rate))
        slug = _slug(interval.identity.title or interval.identity.key)
        path = output_root / f"interval_{interval.index:02d}_{slug}_{start_s:.2f}_{end_s:.2f}.wav"
        write_wav(path, audio[start_index:end_index], sample_rate)
        rows.append(
            {
                "interval_index": interval.index,
                "clip_path": str(path),
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "title": interval.identity.title,
                "artist": interval.identity.artist,
                "confidence": interval.confidence_score,
                "warnings": "; ".join(interval.warnings),
            }
        )
    return tuple(rows)


def plot_session_segmentation(
    result: SessionSegmentationResult,
    *,
    audio_path: str | Path | None = None,
    rms_frame_s: float = 0.25,
    rms_hop_s: float = 0.1,
    tempo_window_s: float = 5.0,
    tempo_hop_s: float = 5.0,
    output_path: str | Path | None = None,
):
    import matplotlib.pyplot as plt

    rms = _rms_energy(audio_path, frame_s=rms_frame_s, hop_s=rms_hop_s) if audio_path else None
    tempo = (
        _rolling_tempo_diagnostics(audio_path, window_s=tempo_window_s, hop_s=tempo_hop_s)
        if audio_path
        else None
    )
    break_likelihood = _break_likelihood_diagnostics(rms, tempo) if rms and tempo else None
    if rms is None and tempo is None and break_likelihood is None:
        fig, ax = plt.subplots(figsize=(11, 3.8), constrained_layout=True)
        energy_ax = None
        tempo_ax = None
        break_ax = None
    else:
        plot_count = (
            1
            + int(rms is not None)
            + int(tempo is not None)
            + int(break_likelihood is not None)
        )
        height_ratios = [2.4]
        if rms is not None:
            height_ratios.append(1.0)
        if tempo is not None:
            height_ratios.append(1.0)
        if break_likelihood is not None:
            height_ratios.append(1.0)
        fig, axes = plt.subplots(
            plot_count,
            1,
            figsize=(11, 3.8 + 1.4 * (plot_count - 1)),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": height_ratios, "hspace": 0.08},
        )
        axes = np.atleast_1d(axes)
        ax = axes[0]
        next_axis = 1
        energy_ax = axes[next_axis] if rms is not None else None
        next_axis += int(rms is not None)
        tempo_ax = axes[next_axis] if tempo is not None else None
        next_axis += int(tempo is not None)
        break_ax = axes[next_axis] if break_likelihood is not None else None

    duration = max(result.recording_duration_s, 0.001)
    ax.hlines(0, 0, duration, color="#9ca3af", linewidth=2, label="Recording")

    for window in result.windows:
        if window.recognized:
            color = "#2563eb"
            marker = "s"
        elif window.status == "error":
            color = "#dc2626"
            marker = "x"
        else:
            color = "#9ca3af"
            marker = "|"
        ax.scatter(
            (window.start_s + window.end_s) / 2.0,
            0.35,
            color=color,
            marker=marker,
            s=28,
            alpha=0.85,
        )

    for interval in result.intervals:
        color = "#16a34a" if interval.confidence_score >= 75 else "#d97706"
        ax.axvspan(interval.start_s, interval.end_s, ymin=0.35, ymax=0.78, color=color, alpha=0.22)
        ax.text(
            (interval.start_s + interval.end_s) / 2.0,
            0.78,
            f"{interval.index}. {interval.identity.title or interval.identity.key}\n{interval.confidence_score:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#111827",
        )

    for candidate in result.weak_candidates:
        ax.axvspan(
            candidate.start_s,
            candidate.end_s,
            ymin=0.18,
            ymax=0.45,
            color="#f59e0b",
            alpha=0.2,
        )
        ax.text(
            (candidate.start_s + candidate.end_s) / 2.0,
            0.47,
            f"weak: {candidate.identity.title or candidate.identity.key}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#92400e",
        )

    ax.set_xlim(0, duration)
    ax.set_ylim(-0.2, 1.15)
    ax.set_yticks([])
    if energy_ax is None and tempo_ax is None and break_ax is None:
        ax.set_xlabel("Time (s)")
    else:
        ax.tick_params(labelbottom=False)
    ax.set_title("Long-session song interval segmentation")
    ax.grid(True, axis="x", alpha=0.25)

    if rms is not None and energy_ax is not None:
        times, values = rms
        energy_ax.plot(times, values, color="#0f766e", linewidth=1.1)
        energy_ax.fill_between(times, 0.0, values, color="#14b8a6", alpha=0.22)
        energy_ax.set_xlim(0, duration)
        energy_ax.set_ylim(0.0, 1.05)
        energy_ax.set_ylabel("RMS")
        if tempo_ax is None and break_ax is None:
            energy_ax.set_xlabel("Time (s)")
        else:
            energy_ax.tick_params(labelbottom=False)
        energy_ax.set_title("Normalized RMS energy", fontsize=10)
        energy_ax.grid(True, axis="x", alpha=0.25)
        energy_ax.grid(True, axis="y", alpha=0.18)

    if tempo is not None and tempo_ax is not None:
        times, bpm_values, confidence_values = tempo
        tempo_ax.plot(times, bpm_values, color="#7c3aed", linewidth=1.2, label="BPM")
        tempo_ax.scatter(
            times,
            bpm_values,
            c=confidence_values,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            s=20,
            alpha=0.85,
        )
        confidence_ax = tempo_ax.twinx()
        confidence_ax.fill_between(
            times,
            0.0,
            confidence_values,
            color="#f59e0b",
            alpha=0.18,
            label="Beat confidence",
        )
        confidence_ax.set_ylim(0.0, 1.05)
        confidence_ax.set_ylabel("Beat conf.")
        tempo_ax.set_xlim(0, duration)
        tempo_ax.set_ylim(40.0, 220.0)
        tempo_ax.set_ylabel("BPM")
        if break_ax is None:
            tempo_ax.set_xlabel("Time (s)")
        else:
            tempo_ax.tick_params(labelbottom=False)
        tempo_ax.set_title("Rolling tempo estimate and beat confidence", fontsize=10)
        tempo_ax.grid(True, axis="x", alpha=0.25)
        tempo_ax.grid(True, axis="y", alpha=0.18)

    if break_likelihood is not None and break_ax is not None:
        times, values = break_likelihood
        break_ax.plot(times, values, color="#dc2626", linewidth=1.25)
        break_ax.fill_between(times, 0.0, values, color="#ef4444", alpha=0.2)
        break_ax.axhline(0.5, color="#f59e0b", linewidth=0.8, linestyle="--", alpha=0.65)
        break_ax.axhline(0.75, color="#dc2626", linewidth=0.8, linestyle="--", alpha=0.65)
        break_ax.set_xlim(0, duration)
        break_ax.set_ylim(0.0, 1.05)
        break_ax.set_ylabel("Break")
        break_ax.set_xlabel("Time (s)")
        break_ax.set_title("Boundary / break likelihood", fontsize=10)
        break_ax.grid(True, axis="x", alpha=0.25)
        break_ax.grid(True, axis="y", alpha=0.18)

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160)
    return fig


def _rms_energy(
    audio_path: str | Path | None,
    *,
    frame_s: float = 0.25,
    hop_s: float = 0.1,
) -> tuple[np.ndarray, np.ndarray] | None:
    if audio_path is None:
        return None
    try:
        audio, sample_rate = load_audio(audio_path, target_sr=22050)
    except Exception:
        return None

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0 or sample_rate <= 0:
        return None

    frame_size = max(1, int(round(frame_s * sample_rate)))
    hop_size = max(1, int(round(hop_s * sample_rate)))
    if audio.size < frame_size:
        padded = np.pad(audio, (0, frame_size - audio.size))
        rms_values = np.array([float(np.sqrt(np.mean(np.square(padded))))], dtype=np.float32)
        times = np.array([0.0], dtype=np.float32)
    else:
        starts = np.arange(0, audio.size - frame_size + 1, hop_size)
        rms_values = np.empty(len(starts), dtype=np.float32)
        for index, start in enumerate(starts):
            frame = audio[start : start + frame_size]
            rms_values[index] = float(np.sqrt(np.mean(np.square(frame))))
        times = (starts + frame_size / 2.0) / float(sample_rate)

    peak = float(np.max(rms_values)) if rms_values.size else 0.0
    if peak > 0:
        rms_values = rms_values / peak
    return times, rms_values


def _rolling_tempo_diagnostics(
    audio_path: str | Path | None,
    *,
    window_s: float = 5.0,
    hop_s: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if audio_path is None:
        return None
    try:
        audio, sample_rate = load_audio(audio_path, target_sr=22050)
    except Exception:
        return None

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0 or sample_rate <= 0:
        return None

    window_size = max(1, int(round(window_s * sample_rate)))
    hop_size = max(1, int(round(hop_s * sample_rate)))
    if audio.size < max(sample_rate * 4, window_size):
        starts = np.array([0], dtype=int)
        window_size = audio.size
    else:
        starts = np.arange(0, audio.size - window_size + 1, hop_size)
    if starts.size == 0:
        return None

    times: list[float] = []
    tempos: list[float] = []
    confidences: list[float] = []
    beat_hop = 512
    for start in starts:
        segment = audio[start : start + window_size]
        if segment.size < sample_rate:
            continue
        tempo_bpm, confidence = _tempo_for_segment(segment, sample_rate, hop_length=beat_hop)
        times.append((start + segment.size / 2.0) / float(sample_rate))
        tempos.append(tempo_bpm)
        confidences.append(confidence)

    if not times:
        return None
    return (
        np.asarray(times, dtype=np.float32),
        np.asarray(tempos, dtype=np.float32),
        np.asarray(confidences, dtype=np.float32),
    )


def _tempo_for_segment(
    audio: np.ndarray,
    sample_rate: int,
    *,
    hop_length: int,
) -> tuple[float, float]:
    import librosa

    onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=hop_length)
    if onset_env.size < 3:
        return 0.0, 0.0

    onset_peak = float(np.percentile(onset_env, 95))
    onset_floor = float(np.percentile(onset_env, 50))
    onset_strength = max(0.0, onset_peak - onset_floor)
    if onset_strength <= 1e-6:
        return 0.0, 0.0

    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sample_rate,
        hop_length=hop_length,
        units="frames",
    )
    if isinstance(tempo, np.ndarray):
        tempo = float(np.ravel(tempo)[0]) if tempo.size else 0.0
    else:
        tempo = float(tempo)

    beat_frames = np.asarray(beat_frames, dtype=float)
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=hop_length)
    beat_count_score = float(np.clip(len(beat_times) / 8.0, 0.0, 1.0))
    if len(beat_times) >= 4:
        intervals = np.diff(beat_times)
        mean_interval = float(np.mean(intervals))
        jitter = float(np.std(intervals) / mean_interval) if mean_interval > 0 else 1.0
        regularity_score = float(np.clip(1.0 - jitter * 2.5, 0.0, 1.0))
    else:
        regularity_score = 0.0

    onset_score = float(np.clip(onset_strength / 2.0, 0.0, 1.0))
    confidence = float(np.clip(0.4 * onset_score + 0.3 * beat_count_score + 0.3 * regularity_score, 0.0, 1.0))
    if not 40.0 <= tempo <= 220.0:
        return 0.0, min(confidence, 0.2)
    return tempo, confidence


def _break_likelihood_diagnostics(
    rms: tuple[np.ndarray, np.ndarray],
    tempo: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray] | None:
    rms_times, rms_values = rms
    tempo_times, bpm_values, confidence_values = tempo
    if rms_times.size == 0 or tempo_times.size == 0:
        return None

    rms_at_tempo = np.interp(tempo_times, rms_times, rms_values)
    local_rms = _rolling_median(rms_at_tempo, radius=3)
    rms_low_score = np.clip((0.35 - rms_at_tempo) / 0.35, 0.0, 1.0)
    rms_dip_score = np.clip((local_rms - rms_at_tempo) / 0.3, 0.0, 1.0)
    rms_score = np.maximum(rms_low_score, rms_dip_score)

    bpm_values = np.asarray(bpm_values, dtype=np.float32)
    valid_bpm = bpm_values > 0
    if np.any(valid_bpm):
        indices = np.arange(len(bpm_values), dtype=float)
        bpm_filled = np.interp(indices, indices[valid_bpm], bpm_values[valid_bpm])
    else:
        bpm_filled = np.zeros_like(bpm_values)

    previous_bpm = np.r_[bpm_filled[0], bpm_filled[:-1]]
    next_bpm = np.r_[bpm_filled[1:], bpm_filled[-1]]
    bpm_jump = np.maximum(np.abs(bpm_filled - previous_bpm), np.abs(next_bpm - bpm_filled))
    bpm_jump_score = np.clip(bpm_jump / 35.0, 0.0, 1.0)
    local_bpm = _rolling_median(bpm_filled, radius=3)
    bpm_outlier_score = np.clip(np.abs(bpm_filled - local_bpm) / 35.0, 0.0, 1.0)
    invalid_bpm_score = np.where(valid_bpm, 0.0, 1.0)

    confidence_values = np.asarray(confidence_values, dtype=np.float32)
    previous_confidence = np.r_[confidence_values[0], confidence_values[:-1]]
    next_confidence = np.r_[confidence_values[1:], confidence_values[-1]]
    confidence_change = np.maximum(
        np.abs(confidence_values - previous_confidence),
        np.abs(next_confidence - confidence_values),
    )
    confidence_low_score = np.clip((0.45 - confidence_values) / 0.45, 0.0, 1.0)
    confidence_change_score = np.clip(confidence_change / 0.35, 0.0, 1.0)
    confidence_score = np.maximum(confidence_low_score, confidence_change_score)

    likelihood = np.clip(
        0.35 * rms_score
        + 0.3 * bpm_jump_score
        + 0.2 * bpm_outlier_score
        + 0.15 * confidence_score
        + 0.15 * invalid_bpm_score,
        0.0,
        1.0,
    )
    return tempo_times, likelihood.astype(np.float32)


def _rolling_median(values: np.ndarray, *, radius: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    medians = np.empty_like(values)
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        medians[index] = float(np.median(values[start:end]))
    return medians


def _interval_from_segment(
    index: int,
    identity: SongIdentity,
    segment_windows: list[RecognizedWindow],
    *,
    recording_duration_s: float,
    hop_s: float,
    min_recognized_windows: int,
) -> SongInterval | None:
    matching = [window for window in segment_windows if window.identity and window.identity.key == identity.key]
    if len(matching) < min_recognized_windows:
        return None

    recognized = [window for window in segment_windows if window.recognized]
    gap_count = sum(1 for window in segment_windows if not window.recognized)
    conflict_count = max(0, len(recognized) - len(matching))
    start_s = max(0.0, min(window.start_s for window in matching) - hop_s)
    end_s = min(float(recording_duration_s), max(window.end_s for window in matching))
    duration_s = max(0.0, end_s - start_s)
    if duration_s <= 0:
        return None

    provider_confidence = _mean_confidence(window.confidence for window in matching)
    identity_consistency = len(matching) / max(1, len(recognized))
    recognized_coverage = len(matching) / max(1, len(segment_windows))
    boundary_stability = max(0.0, 1.0 - (gap_count * 0.15 + conflict_count * 0.25))
    duration_plausibility = min(1.0, duration_s / 20.0)
    identity_quality = {"high": 1.0, "medium": 0.9, "low": 0.72}.get(identity.quality, 0.85)

    score = (
        35.0 * identity_consistency * identity_quality
        + 25.0 * recognized_coverage
        + 20.0 * provider_confidence
        + 10.0 * boundary_stability
        + 10.0 * duration_plausibility
    )
    if provider_confidence < 0.5:
        score = min(score, 69.0)
    score = float(np.clip(score, 0.0, 100.0))
    level = "high" if score >= 75.0 else "medium" if score >= 50.0 else "low"
    warnings: list[str] = []
    if identity.quality == "low":
        warnings.append("identity is based on title only")
    if conflict_count:
        warnings.append(f"{conflict_count} conflicting recognized window(s)")
    if gap_count:
        warnings.append(f"{gap_count} no-match gap window(s)")
    if provider_confidence < 0.5:
        warnings.append("provider confidence is low")
    if duration_s < 20.0:
        warnings.append("interval is shorter than 20s; boundary may be weak")
    if score < 50.0:
        warnings.append("segmentation confidence is low")

    return SongInterval(
        index=index,
        identity=identity,
        start_s=round(float(start_s), 3),
        end_s=round(float(end_s), 3),
        confidence_score=round(score, 2),
        confidence_level=level,
        recognized_window_count=len(matching),
        total_window_count=len(segment_windows),
        gap_window_count=gap_count,
        conflict_window_count=conflict_count,
        provider_confidence=round(provider_confidence, 3),
        warnings=tuple(warnings),
    )


def _weak_candidate_from_segment(
    index: int,
    identity: SongIdentity,
    segment_windows: list[RecognizedWindow],
    *,
    recording_duration_s: float,
    min_recognized_windows: int,
) -> WeakSongCandidate:
    matching = [window for window in segment_windows if window.identity and window.identity.key == identity.key]
    recognized = [window for window in segment_windows if window.recognized]
    if matching:
        start_s = min(window.start_s for window in matching)
        end_s = max(window.end_s for window in matching)
    else:
        start_s = min((window.start_s for window in segment_windows), default=0.0)
        end_s = max((window.end_s for window in segment_windows), default=start_s)
    center_s = (start_s + end_s) / 2.0
    recovery_start_s = max(0.0, center_s - 75.0)
    recovery_end_s = min(float(recording_duration_s), center_s + 75.0)
    if recovery_end_s <= recovery_start_s:
        recovery_end_s = max(recovery_start_s, center_s)

    provider_confidence = _mean_confidence(window.confidence for window in matching)
    warnings: list[str] = []
    if len(matching) == 1:
        reason = "singleton_match"
        warnings.append("only one recognized window")
    elif len(matching) < min_recognized_windows:
        reason = "insufficient_repeated_matches"
        warnings.append(f"requires at least {min_recognized_windows} recognized windows")
    elif provider_confidence < 0.5:
        reason = "low_provider_confidence"
        warnings.append("provider confidence is low")
    else:
        reason = "rejected_candidate"
    if len(recognized) > len(matching):
        warnings.append("candidate contains conflicting recognized windows")

    return WeakSongCandidate(
        index=index,
        identity=identity,
        start_s=round(float(start_s), 3),
        end_s=round(float(end_s), 3),
        recognized_window_count=len(matching),
        total_window_count=len(segment_windows),
        provider_confidence=round(provider_confidence, 3),
        reason=reason,
        recovery_start_s=round(float(recovery_start_s), 3),
        recovery_end_s=round(float(recovery_end_s), 3),
        warnings=tuple(warnings),
    )


def _window_count(duration_s: float, *, window_s: float, hop_s: float) -> int:
    if duration_s <= 0:
        return 1
    if duration_s <= window_s:
        return 1
    return max(1, int(np.floor((duration_s - window_s) / max(hop_s, 0.001))) + 1)


def _mean_confidence(values: Any) -> float:
    normalized = [_confidence for value in values if (_confidence := _confidence_0_1(value)) is not None]
    if not normalized:
        return 0.5
    return float(np.nanmean(normalized))


def _optional_confidence(value: Any) -> float | None:
    return _confidence_0_1(value)


def _confidence_0_1(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    if numeric > 1.0:
        numeric /= 100.0
    return float(np.clip(numeric, 0.0, 1.0))


def _float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(numeric):
        return 0.0
    return numeric


def _string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalized(value: str) -> str:
    return " ".join(re.sub(r"[^0-9a-zA-Z가-힣]+", " ", value.casefold()).split())


def _normalized_id(value: str) -> str:
    return re.sub(r"\s+", "", value.casefold())


def _slug(value: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z가-힣]+", "_", value.strip().casefold()).strip("_")
    return slug[:48] or "song"
