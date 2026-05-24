from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PitchContour:
    times_s: np.ndarray
    frequencies_hz: np.ndarray
    confidence: np.ndarray
    name: str = "take"

    def __post_init__(self) -> None:
        times = np.asarray(self.times_s, dtype=float)
        hz = np.asarray(self.frequencies_hz, dtype=float)
        confidence = np.asarray(self.confidence, dtype=float)
        if not (times.shape == hz.shape == confidence.shape):
            raise ValueError("PitchContour arrays must have matching shapes")
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "frequencies_hz", hz)
        object.__setattr__(self, "confidence", np.clip(confidence, 0.0, 1.0))

    @property
    def voiced_mask(self) -> np.ndarray:
        return np.isfinite(self.frequencies_hz) & (self.frequencies_hz > 0)

    def shifted(self, offset_s: float) -> "PitchContour":
        return PitchContour(
            self.times_s + offset_s,
            self.frequencies_hz,
            self.confidence,
            name=self.name,
        )


def extract_pitch(
    audio: np.ndarray,
    sample_rate: int,
    *,
    fmin_hz: float = 80.0,
    fmax_hz: float = 1000.0,
    frame_length: int = 2048,
    hop_length: int = 256,
    name: str = "take",
) -> PitchContour:
    """Extract a monophonic pitch contour from a vocal recording."""
    audio = _prepare_audio_for_pitch(audio)
    try:
        import librosa

        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio,
            fmin=fmin_hz,
            fmax=fmax_hz,
            sr=sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sample_rate, hop_length=hop_length)
        hz = np.where(voiced_flag, f0, np.nan)
        confidence = np.where(np.isfinite(voiced_prob), voiced_prob, 0.0)
        return PitchContour(times, hz, confidence, name=name)
    except Exception:
        return _extract_pitch_autocorrelation(
            audio,
            sample_rate,
            fmin_hz=fmin_hz,
            fmax_hz=fmax_hz,
            frame_length=frame_length,
            hop_length=hop_length,
            name=name,
        )


def clean_pitch_contour(
    contour: PitchContour,
    *,
    max_jump_cents: float = 700.0,
    min_confidence: float = 0.25,
    correct_octaves: bool = True,
) -> PitchContour:
    hz = contour.frequencies_hz.copy()
    hz[contour.confidence < min_confidence] = np.nan
    if correct_octaves:
        hz = _correct_local_octave_errors(hz)
    voiced_indices = np.flatnonzero(np.isfinite(hz) & (hz > 0))
    if voiced_indices.size < 3:
        return PitchContour(contour.times_s, hz, contour.confidence, name=contour.name)

    for prev_idx, idx in zip(voiced_indices[:-1], voiced_indices[1:]):
        jump = abs(1200.0 * np.log2(hz[idx] / hz[prev_idx]))
        if jump > max_jump_cents:
            hz[idx] = np.nan
    return PitchContour(contour.times_s, hz, contour.confidence, name=contour.name)


def _prepare_audio_for_pitch(audio: np.ndarray) -> np.ndarray:
    prepared = np.nan_to_num(np.asarray(audio, dtype=np.float32))
    if prepared.size == 0:
        return prepared
    prepared = prepared - float(np.mean(prepared))
    peak = float(np.max(np.abs(prepared)))
    if peak > 0:
        prepared = prepared / max(1.0, peak / 0.95)
    return prepared.astype(np.float32)


def _correct_local_octave_errors(
    hz: np.ndarray,
    *,
    fmin_hz: float = 80.0,
    fmax_hz: float = 1000.0,
) -> np.ndarray:
    corrected = hz.copy()
    voiced_indices = np.flatnonzero(np.isfinite(corrected) & (corrected > 0))
    if voiced_indices.size < 2:
        return corrected

    last_stable = corrected[voiced_indices[0]]
    for idx in voiced_indices[1:]:
        value = corrected[idx]
        raw_jump = abs(1200.0 * np.log2(value / last_stable))
        if raw_jump <= 900.0:
            last_stable = value
            continue

        candidates = [value]
        if value * 2 <= fmax_hz:
            candidates.append(value * 2)
        if value / 2 >= fmin_hz:
            candidates.append(value / 2)

        best = min(candidates, key=lambda candidate: abs(1200.0 * np.log2(candidate / last_stable)))
        best_jump = abs(1200.0 * np.log2(best / last_stable))
        if best_jump < 450.0 and best_jump + 250.0 < raw_jump:
            corrected[idx] = best
            last_stable = best
        else:
            last_stable = value
    return corrected


def _extract_pitch_autocorrelation(
    audio: np.ndarray,
    sample_rate: int,
    *,
    fmin_hz: float,
    fmax_hz: float,
    frame_length: int,
    hop_length: int,
    name: str,
) -> PitchContour:
    if len(audio) < frame_length:
        audio = np.pad(audio, (0, frame_length - len(audio)))

    window = np.hanning(frame_length)
    min_lag = max(1, int(sample_rate / fmax_hz))
    max_lag = min(frame_length - 1, int(sample_rate / fmin_hz))
    times: list[float] = []
    pitches: list[float] = []
    confidence: list[float] = []

    for start in range(0, len(audio) - frame_length + 1, hop_length):
        frame = audio[start : start + frame_length]
        rms = float(np.sqrt(np.mean(frame**2)))
        center = (start + frame_length / 2.0) / sample_rate
        times.append(center)
        if rms < 0.005:
            pitches.append(np.nan)
            confidence.append(0.0)
            continue

        frame = (frame - np.mean(frame)) * window
        corr = np.correlate(frame, frame, mode="full")[frame_length - 1 :]
        if corr[0] <= 0:
            pitches.append(np.nan)
            confidence.append(0.0)
            continue

        search = corr[min_lag:max_lag]
        if search.size == 0:
            pitches.append(np.nan)
            confidence.append(0.0)
            continue

        lag = int(np.argmax(search) + min_lag)
        peak_ratio = float(corr[lag] / corr[0])
        if peak_ratio < 0.25:
            pitches.append(np.nan)
            confidence.append(max(0.0, peak_ratio))
            continue
        pitches.append(float(sample_rate / lag))
        confidence.append(min(1.0, max(0.0, peak_ratio)))

    return PitchContour(
        np.asarray(times),
        np.asarray(pitches),
        np.asarray(confidence),
        name=name,
    )
