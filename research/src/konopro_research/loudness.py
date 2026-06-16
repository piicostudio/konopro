from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from konopro_research.audio_io import load_audio, write_wav


ACTIVE_RMS_CACHE_VERSION = 1


@dataclass(frozen=True)
class ActiveRmsNormalizationResult:
    source_path: Path
    analysis_path: Path
    target_rms: float
    active_percentile: float
    active_rms_before: float
    active_rms_after: float
    full_rms_after: float
    gain: float
    active_threshold: float
    active_ratio: float
    peak_after: float
    used_cache: bool
    cache_key: str
    status: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_path"] = str(self.source_path)
        data["analysis_path"] = str(self.analysis_path)
        return data


def normalize_active_rms_file(
    source_path: str | Path,
    *,
    cache_dir: str | Path,
    target_rms: float = 0.08,
    active_percentile: float = 60.0,
    source_hash: str | None = None,
) -> ActiveRmsNormalizationResult:
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(source)

    cache_path, cache_key = _active_rms_cache_path(
        source,
        cache_dir=cache_dir,
        target_rms=target_rms,
        active_percentile=active_percentile,
        source_hash=source_hash,
    )
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return _result_from_metadata(
                metadata,
                source_path=source,
                analysis_path=cache_path,
                used_cache=True,
                status="cache hit",
            )
        except Exception:
            pass

    audio, sample_rate = load_audio(source)
    normalized, metrics = normalize_active_rms(
        audio,
        sample_rate=sample_rate,
        target_rms=target_rms,
        active_percentile=active_percentile,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(cache_path, normalized, sample_rate)
    metadata = {
        **metrics,
        "source_path": str(source),
        "analysis_path": str(cache_path),
        "target_rms": float(target_rms),
        "active_percentile": float(active_percentile),
        "cache_key": cache_key,
        "status": "computed",
        "warnings": (),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return _result_from_metadata(
        metadata,
        source_path=source,
        analysis_path=cache_path,
        used_cache=False,
        status="computed",
    )


def normalize_active_rms(
    audio: np.ndarray,
    *,
    sample_rate: int,
    target_rms: float = 0.08,
    active_percentile: float = 60.0,
) -> tuple[np.ndarray, dict[str, float]]:
    source = np.nan_to_num(np.asarray(audio, dtype=np.float32))
    if source.size == 0:
        return source, {
            "active_rms_before": 0.0,
            "active_rms_after": 0.0,
            "full_rms_after": 0.0,
            "gain": 1.0,
            "active_threshold": 0.0,
            "active_ratio": 0.0,
            "peak_after": 0.0,
        }

    frame_length = min(max(256, int(round(0.05 * sample_rate))), max(1, source.size))
    hop_length = max(1, frame_length // 2)
    frame_rms = _frame_rms(source, frame_length=frame_length, hop_length=hop_length)
    if frame_rms.size == 0:
        before = _rms(source)
        active_threshold = 0.0
        active_ratio = 1.0
    else:
        percentile = float(np.clip(active_percentile, 0.0, 95.0))
        active_threshold = float(np.percentile(frame_rms, percentile))
        active_frames = frame_rms >= max(active_threshold, 1e-8)
        if np.count_nonzero(active_frames) < max(1, int(0.05 * frame_rms.size)):
            active_frames = frame_rms >= 1e-8
        before = float(np.sqrt(np.mean(np.square(frame_rms[active_frames])))) if np.any(active_frames) else _rms(source)
        active_ratio = float(np.mean(active_frames)) if frame_rms.size else 1.0

    gain = 1.0 if before <= 1e-8 else float(target_rms) / before
    normalized = source * gain
    peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    if peak > 0.98:
        normalized = normalized * (0.98 / peak)
        gain *= 0.98 / peak
        peak = 0.98

    after = _rms(normalized) if normalized.size else 0.0
    after_active = before * gain
    return normalized.astype(np.float32), {
        "active_rms_before": round(float(before), 6),
        "active_rms_after": round(float(after_active), 6),
        "full_rms_after": round(float(after), 6),
        "gain": round(float(gain), 4),
        "active_threshold": round(float(active_threshold), 6),
        "active_ratio": round(float(active_ratio), 4),
        "peak_after": round(float(peak), 6),
    }


def _result_from_metadata(
    metadata: dict[str, Any],
    *,
    source_path: Path,
    analysis_path: Path,
    used_cache: bool,
    status: str,
) -> ActiveRmsNormalizationResult:
    return ActiveRmsNormalizationResult(
        source_path=source_path,
        analysis_path=analysis_path,
        target_rms=float(metadata["target_rms"]),
        active_percentile=float(metadata["active_percentile"]),
        active_rms_before=float(metadata["active_rms_before"]),
        active_rms_after=float(metadata["active_rms_after"]),
        full_rms_after=float(metadata["full_rms_after"]),
        gain=float(metadata["gain"]),
        active_threshold=float(metadata["active_threshold"]),
        active_ratio=float(metadata["active_ratio"]),
        peak_after=float(metadata["peak_after"]),
        used_cache=used_cache,
        cache_key=str(metadata["cache_key"]),
        status=status,
        warnings=tuple(metadata.get("warnings", ())),
    )


def _active_rms_cache_path(
    source_path: Path,
    *,
    cache_dir: str | Path,
    target_rms: float,
    active_percentile: float,
    source_hash: str | None = None,
) -> tuple[Path, str]:
    file_hash = source_hash or _sha256_file(source_path)
    payload = {
        "version": ACTIVE_RMS_CACHE_VERSION,
        "source_hash": file_hash,
        "target_rms": round(float(target_rms), 5),
        "active_percentile": round(float(active_percentile), 3),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    cache_key = hashlib.sha256(encoded).hexdigest()[:24]
    return Path(cache_dir) / "active_rms" / f"{cache_key}.wav", cache_key


def _frame_rms(audio: np.ndarray, *, frame_length: int, hop_length: int) -> np.ndarray:
    if audio.size == 0:
        return np.asarray([], dtype=np.float32)
    n = audio.size
    if n < frame_length:
        return np.asarray([_rms(audio)], dtype=np.float32)
    n_frames = 1 + (n - frame_length) // hop_length
    if n_frames <= 0:
        return np.asarray([_rms(audio)], dtype=np.float32)
    # Vectorized frame RMS via stride tricks (avoids Python loop)
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, frame_length),
        strides=(audio.strides[0] * hop_length, audio.strides[0]),
    )
    return np.sqrt(np.mean(np.square(frames), axis=1)).astype(np.float32)


def _rms(audio: np.ndarray) -> float:
    values = np.asarray(audio, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fileobj:
        for chunk in iter(lambda: fileobj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
