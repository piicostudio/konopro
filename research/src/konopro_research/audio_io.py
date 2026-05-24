from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import BinaryIO

import numpy as np


def load_audio(
    path_or_file: str | Path | BinaryIO,
    *,
    target_sr: int = 22050,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """Load audio as float32 in the range [-1, 1]."""
    try:
        import soundfile as sf

        if hasattr(path_or_file, "read"):
            data, sample_rate = sf.read(io.BytesIO(path_or_file.read()), always_2d=True)
        else:
            data, sample_rate = sf.read(str(path_or_file), always_2d=True)
        audio = data.astype(np.float32)
    except Exception as soundfile_error:
        try:
            import librosa

            if hasattr(path_or_file, "read"):
                raise soundfile_error
            audio, sample_rate = librosa.load(
                str(path_or_file),
                sr=target_sr,
                mono=mono,
            )
            return np.asarray(audio, dtype=np.float32), int(sample_rate)
        except Exception:
            pass

        if hasattr(path_or_file, "read"):
            raise RuntimeError("Install soundfile to load uploaded non-path audio objects")
        audio, sample_rate = _load_wav_fallback(Path(path_or_file))

    if mono and audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    elif audio.ndim == 2:
        audio = audio[:, 0]

    if target_sr and sample_rate != target_sr:
        audio = resample_linear(audio, sample_rate, target_sr)
        sample_rate = target_sr

    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(audio, dtype=float), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def resample_linear(audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    if original_sr == target_sr:
        return np.asarray(audio, dtype=np.float32)
    duration = len(audio) / float(original_sr)
    old_times = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    new_length = max(1, int(round(duration * target_sr)))
    new_times = np.linspace(0.0, duration, num=new_length, endpoint=False)
    return np.interp(new_times, old_times, audio).astype(np.float32)


def _load_wav_fallback(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if sampwidth != 2:
        raise RuntimeError("Fallback WAV loader only supports 16-bit PCM WAV files")
    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels)
    return pcm, sample_rate
