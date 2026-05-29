from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests

from konopro_research.audio_io import load_audio, write_wav


FingerprintRecognizer = Callable[[Path], dict[str, Any]]
MIN_SHAZAM_SIGNATURE_S = 3.0
MAX_SHAZAM_SIGNATURE_S = 12.0
DEFAULT_SHAZAM_WINDOW_S = 10.0


@dataclass(frozen=True)
class FingerprintRunResult:
    status: str
    rows: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    interpretations: dict[str, str]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prepare_fingerprint_windows(
    audio_path: str | Path | None,
    output_dir: str | Path,
    *,
    mode: str = "Sliding windows",
    window_s: float = DEFAULT_SHAZAM_WINDOW_S,
    hop_s: float = 5.0,
    max_windows: int = 12,
    start_offset_s: float = 0.0,
    window_strategy: str = "From offset",
    namespace: str = "preview",
) -> tuple[dict[str, Any], ...]:
    if not audio_path:
        return ()

    source = Path(audio_path)
    output_root = Path(output_dir) / "fingerprinting" / namespace
    output_root.mkdir(parents=True, exist_ok=True)
    audio, sample_rate = load_audio(source, target_sr=44100)
    duration_s = len(audio) / sample_rate if sample_rate else 0.0
    windows = _build_windows(
        source,
        audio,
        sample_rate,
        output_root,
        mode=mode,
        duration_s=duration_s,
        window_s=window_s,
        hop_s=hop_s,
        max_windows=max_windows,
        start_offset_s=start_offset_s,
        window_strategy=window_strategy,
    )
    return tuple(
        {
            "mode": str(window["mode"]),
            "start_s": float(window["start_s"]),
            "end_s": float(window["end_s"]),
            "audio_path": str(window["path"]),
            "audio_file": Path(window["path"]).name,
        }
        for window in windows
    )


def run_shazam_fingerprinting(
    audio_path: str | Path | None,
    output_dir: str | Path,
    *,
    expected_title: str = "",
    expected_artist: str = "",
    mode: str = "Whole + sliding windows",
    window_s: float = DEFAULT_SHAZAM_WINDOW_S,
    hop_s: float = 5.0,
    max_windows: int = 12,
    start_offset_s: float = 0.0,
    window_strategy: str = "From offset",
    helper_path: str | Path | None = None,
    recognizer: FingerprintRecognizer | None = None,
    timeout_s: float = 45.0,
) -> FingerprintRunResult:
    if not audio_path:
        return FingerprintRunResult(
            status="No karaoke recording selected.",
            rows=(),
            summary=_empty_summary(expected_title, expected_artist),
            interpretations={},
            warnings=("upload or select a karaoke phone recording",),
        )

    source = Path(audio_path)
    output_root = Path(output_dir) / "fingerprinting"
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        audio, sample_rate = load_audio(source, target_sr=44100)
    except Exception as exc:
        return FingerprintRunResult(
            status=f"Could not load audio: {exc}",
            rows=(),
            summary=_empty_summary(expected_title, expected_artist),
            interpretations={},
            warnings=(str(exc),),
        )

    duration_s = len(audio) / sample_rate if sample_rate else 0.0
    windows = _build_windows(
        source,
        audio,
        sample_rate,
        output_root,
        mode=mode,
        duration_s=duration_s,
        window_s=window_s,
        hop_s=hop_s,
        max_windows=max_windows,
        start_offset_s=start_offset_s,
        window_strategy=window_strategy,
    )

    if not windows:
        return FingerprintRunResult(
            status="Audio is too short to fingerprint with ShazamKit.",
            rows=(),
            summary=_empty_summary(expected_title, expected_artist),
            interpretations={},
            warnings=(f"audio must be at least {MIN_SHAZAM_SIGNATURE_S:.0f}s",),
        )

    recognizer = recognizer or _make_shazamkit_recognizer(helper_path, timeout_s=timeout_s)
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for window in windows:
        started = time.perf_counter()
        try:
            payload = recognizer(window["path"])
        except Exception as exc:
            payload = {"status": "error", "error": str(exc)}
        elapsed = time.perf_counter() - started
        row = _row_from_payload(
            payload,
            test_case=source.stem,
            expected_title=expected_title,
            expected_artist=expected_artist,
            mode=str(window["mode"]),
            start_s=float(window["start_s"]),
            end_s=float(window["end_s"]),
            audio_path=Path(window["path"]),
            elapsed_s=elapsed,
        )
        if row["status"] == "error" and row["error"]:
            warnings.append(str(row["error"]))
        rows.append(row)

    summary = _summarize_rows(rows, expected_title, expected_artist)
    interpretations = _interpret_rows(rows, summary)
    status = _status_text(rows, summary)
    return FingerprintRunResult(
        status=status,
        rows=tuple(rows),
        summary=summary,
        interpretations=interpretations,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def run_audd_fingerprinting(
    audio_path: str | Path | None,
    output_dir: str | Path,
    *,
    expected_title: str = "",
    expected_artist: str = "",
    mode: str = "Whole + sliding windows",
    window_s: float = DEFAULT_SHAZAM_WINDOW_S,
    hop_s: float = 5.0,
    max_windows: int = 12,
    start_offset_s: float = 0.0,
    window_strategy: str = "From offset",
    api_token: str | None = None,
    recognizer: FingerprintRecognizer | None = None,
    timeout_s: float = 30.0,
) -> FingerprintRunResult:
    if not audio_path:
        return FingerprintRunResult(
            status="No karaoke recording selected.",
            rows=(),
            summary=_empty_summary(expected_title, expected_artist),
            interpretations={},
            warnings=("upload or select a karaoke phone recording",),
        )

    source = Path(audio_path)
    output_root = Path(output_dir) / "fingerprinting" / "audd"
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        audio, sample_rate = load_audio(source, target_sr=44100)
    except Exception as exc:
        return FingerprintRunResult(
            status=f"Could not load audio: {exc}",
            rows=(),
            summary=_empty_summary(expected_title, expected_artist),
            interpretations={},
            warnings=(str(exc),),
        )

    duration_s = len(audio) / sample_rate if sample_rate else 0.0
    windows = _build_windows(
        source,
        audio,
        sample_rate,
        output_root,
        mode=mode,
        duration_s=duration_s,
        window_s=window_s,
        hop_s=hop_s,
        max_windows=max_windows,
        start_offset_s=start_offset_s,
        window_strategy=window_strategy,
    )

    if not windows:
        return FingerprintRunResult(
            status="Audio is too short to fingerprint with AudD.",
            rows=(),
            summary=_empty_summary(expected_title, expected_artist),
            interpretations={},
            warnings=(f"audio must be at least {MIN_SHAZAM_SIGNATURE_S:.0f}s",),
        )

    cache_dir = output_root / "cache"
    recognizer = recognizer or (
        lambda window_path: recognize_with_audd(
            window_path,
            api_token=api_token,
            cache_dir=cache_dir,
            timeout_s=timeout_s,
        )
    )
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for window in windows:
        started = time.perf_counter()
        try:
            payload = recognizer(window["path"])
        except Exception as exc:
            payload = {"status": "error", "error": str(exc)}
        elapsed = time.perf_counter() - started
        row = _row_from_payload(
            payload,
            test_case=source.stem,
            expected_title=expected_title,
            expected_artist=expected_artist,
            mode=str(window["mode"]),
            start_s=float(window["start_s"]),
            end_s=float(window["end_s"]),
            audio_path=Path(window["path"]),
            elapsed_s=elapsed,
        )
        if row["status"] == "error" and row["error"]:
            warnings.append(str(row["error"]))
        rows.append(row)

    summary = _summarize_rows(rows, expected_title, expected_artist)
    interpretations = _interpret_rows(rows, summary)
    status = _status_text(rows, summary).replace("Fingerprinting", "AudD fingerprinting", 1)
    return FingerprintRunResult(
        status=status,
        rows=tuple(rows),
        summary=summary,
        interpretations=interpretations,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def run_acrcloud_fingerprinting(
    audio_path: str | Path | None,
    output_dir: str | Path,
    *,
    expected_title: str = "",
    expected_artist: str = "",
    mode: str = "Whole + sliding windows",
    window_s: float = DEFAULT_SHAZAM_WINDOW_S,
    hop_s: float = 5.0,
    max_windows: int = 12,
    start_offset_s: float = 0.0,
    window_strategy: str = "From offset",
    host: str | None = None,
    access_key: str | None = None,
    access_secret: str | None = None,
    recognizer: FingerprintRecognizer | None = None,
    timeout_s: float = 30.0,
) -> FingerprintRunResult:
    if not audio_path:
        return FingerprintRunResult(
            status="No karaoke recording selected.",
            rows=(),
            summary=_empty_summary(expected_title, expected_artist),
            interpretations={},
            warnings=("upload or select a karaoke phone recording",),
        )

    source = Path(audio_path)
    output_root = Path(output_dir) / "fingerprinting" / "acrcloud"
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        audio, sample_rate = load_audio(source, target_sr=44100)
    except Exception as exc:
        return FingerprintRunResult(
            status=f"Could not load audio: {exc}",
            rows=(),
            summary=_empty_summary(expected_title, expected_artist),
            interpretations={},
            warnings=(str(exc),),
        )

    duration_s = len(audio) / sample_rate if sample_rate else 0.0
    windows = _build_windows(
        source,
        audio,
        sample_rate,
        output_root,
        mode=mode,
        duration_s=duration_s,
        window_s=window_s,
        hop_s=hop_s,
        max_windows=max_windows,
        start_offset_s=start_offset_s,
        window_strategy=window_strategy,
    )

    if not windows:
        return FingerprintRunResult(
            status="Audio is too short to fingerprint with ACRCloud.",
            rows=(),
            summary=_empty_summary(expected_title, expected_artist),
            interpretations={},
            warnings=(f"audio must be at least {MIN_SHAZAM_SIGNATURE_S:.0f}s",),
        )

    cache_dir = output_root / "cache"
    recognizer = recognizer or (
        lambda window_path: recognize_with_acrcloud(
            window_path,
            host=host,
            access_key=access_key,
            access_secret=access_secret,
            cache_dir=cache_dir,
            timeout_s=timeout_s,
        )
    )
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for window in windows:
        started = time.perf_counter()
        try:
            payload = recognizer(window["path"])
        except Exception as exc:
            payload = {"status": "error", "error": str(exc)}
        elapsed = time.perf_counter() - started
        row = _row_from_payload(
            payload,
            test_case=source.stem,
            expected_title=expected_title,
            expected_artist=expected_artist,
            mode=str(window["mode"]),
            start_s=float(window["start_s"]),
            end_s=float(window["end_s"]),
            audio_path=Path(window["path"]),
            elapsed_s=elapsed,
        )
        if row["status"] == "error" and row["error"]:
            warnings.append(str(row["error"]))
        rows.append(row)

    summary = _summarize_rows(rows, expected_title, expected_artist)
    interpretations = _interpret_rows(rows, summary)
    status = _status_text(rows, summary).replace("Fingerprinting", "ACRCloud fingerprinting", 1)
    return FingerprintRunResult(
        status=status,
        rows=tuple(rows),
        summary=summary,
        interpretations=interpretations,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def recognize_with_shazamkit(
    audio_path: str | Path,
    *,
    helper_path: str | Path | None = None,
    timeout_s: float = 45.0,
) -> dict[str, Any]:
    helper = _resolve_helper_path(helper_path)
    if not helper.exists():
        return {
            "status": "error",
            "error": f"ShazamKit helper not found at {helper}",
        }

    command = _helper_command(helper, audio_path)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": f"ShazamKit helper timed out after {timeout_s:.0f}s"}
    except Exception as exc:
        return {"status": "error", "error": f"Could not run ShazamKit helper: {exc}"}

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    try:
        payload = json.loads(stdout) if stdout else {}
    except json.JSONDecodeError:
        payload = {
            "status": "error",
            "error": "ShazamKit helper returned non-JSON output",
            "stdout": stdout[-1200:],
        }

    if payload.get("status") == "error" and payload.get("error"):
        payload["raw_error"] = payload["error"]
        payload["error"] = _friendly_helper_error(str(payload["error"]))
    if completed.returncode != 0 and payload.get("status") != "error":
        payload["status"] = "error"
        payload["error"] = stderr or f"ShazamKit helper exited with code {completed.returncode}"
    if stderr and "stderr" not in payload:
        payload["stderr"] = stderr[-1200:]
    return payload


def recognize_with_acrcloud(
    audio_path: str | Path,
    *,
    host: str | None = None,
    access_key: str | None = None,
    access_secret: str | None = None,
    data_type: str = "audio",
    cache_dir: str | Path | None = None,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    resolved_host = (host or os.environ.get("ACRCLOUD_HOST") or "").strip()
    resolved_key = (access_key or os.environ.get("ACRCLOUD_ACCESS_KEY") or "").strip()
    resolved_secret = (access_secret or os.environ.get("ACRCLOUD_ACCESS_SECRET") or "").strip()
    if not resolved_host or not resolved_key or not resolved_secret:
        return {
            "status": "error",
            "error": "ACRCLOUD_HOST, ACRCLOUD_ACCESS_KEY, and ACRCLOUD_ACCESS_SECRET must be set.",
        }

    source = Path(audio_path)
    if not source.exists():
        return {"status": "error", "error": f"ACRCloud audio file not found at {source}"}

    endpoint = _acrcloud_endpoint(resolved_host)
    cache_path = _provider_cache_path(source, f"acrcloud:{endpoint}:{data_type}", cache_dir)
    if cache_path and cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text())
            payload["cached"] = True
            return payload
        except json.JSONDecodeError:
            pass

    timestamp = str(time.time())
    signature_version = "1"
    string_to_sign = "\n".join(
        ["POST", "/v1/identify", resolved_key, data_type, signature_version, timestamp]
    )
    signature = base64.b64encode(
        hmac.new(
            resolved_secret.encode("ascii"),
            string_to_sign.encode("ascii"),
            digestmod=hashlib.sha1,
        ).digest()
    ).decode("ascii")

    try:
        with source.open("rb") as handle:
            response = requests.post(
                endpoint,
                data={
                    "access_key": resolved_key,
                    "sample_bytes": str(source.stat().st_size),
                    "timestamp": timestamp,
                    "signature": signature,
                    "data_type": data_type,
                    "signature_version": signature_version,
                },
                files={"sample": (source.name, handle, "audio/wav")},
                timeout=float(timeout_s),
            )
        response.raise_for_status()
        payload = _normalize_acrcloud_response(response.json())
    except requests.Timeout:
        payload = {"status": "error", "error": f"ACRCloud request timed out after {timeout_s:.0f}s"}
    except requests.RequestException as exc:
        payload = {"status": "error", "error": f"ACRCloud request failed: {exc}"}
    except ValueError as exc:
        payload = {"status": "error", "error": f"ACRCloud returned invalid JSON: {exc}"}

    if cache_path and payload.get("status") != "error":
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def recognize_with_audd(
    audio_path: str | Path,
    *,
    api_token: str | None = None,
    endpoint: str = "https://api.audd.io/",
    return_metadata: str = "apple_music,spotify,deezer,musicbrainz",
    cache_dir: str | Path | None = None,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    token = (api_token or os.environ.get("AUDD_API_TOKEN") or "").strip()
    if not token:
        return {
            "status": "error",
            "error": "AUDD_API_TOKEN is not set.",
        }

    source = Path(audio_path)
    if not source.exists():
        return {"status": "error", "error": f"AudD audio file not found at {source}"}

    cache_path = _audd_cache_path(source, return_metadata, cache_dir)
    if cache_path and cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text())
            payload["cached"] = True
            return payload
        except json.JSONDecodeError:
            pass

    try:
        with source.open("rb") as handle:
            response = requests.post(
                endpoint,
                data={"api_token": token, "return": return_metadata},
                files={"file": (source.name, handle, "audio/wav")},
                timeout=float(timeout_s),
            )
        response.raise_for_status()
        payload = _normalize_audd_response(response.json())
    except requests.Timeout:
        payload = {"status": "error", "error": f"AudD request timed out after {timeout_s:.0f}s"}
    except requests.RequestException as exc:
        payload = {"status": "error", "error": f"AudD request failed: {exc}"}
    except ValueError as exc:
        payload = {"status": "error", "error": f"AudD returned invalid JSON: {exc}"}

    if cache_path and payload.get("status") != "error":
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _normalize_audd_response(response: dict[str, Any]) -> dict[str, Any]:
    if response.get("status") != "success":
        error = response.get("error")
        if isinstance(error, dict):
            message = error.get("error_message") or error.get("message") or str(error)
        else:
            message = str(error or response.get("status") or "unknown AudD error")
        return {"status": "error", "error": message, "raw_response": response}

    result = response.get("result")
    if not result:
        return {"status": "no_match", "raw_response": response}
    if not isinstance(result, dict):
        return {"status": "error", "error": "AudD result was not an object", "raw_response": response}

    media = {
        "title": result.get("title") or "",
        "artist": result.get("artist") or "",
        "album": result.get("album") or "",
        "label": result.get("label") or "",
        "release_date": result.get("release_date") or "",
        "isrc": result.get("isrc") or "",
        "web_url": result.get("song_link") or "",
        "timecode": result.get("timecode") or "",
    }
    if isinstance(result.get("apple_music"), dict):
        media["apple_music_id"] = str(result["apple_music"].get("id") or "")
        media["apple_music_url"] = result["apple_music"].get("url") or ""
    if isinstance(result.get("spotify"), dict):
        media["spotify_id"] = str(result["spotify"].get("id") or "")
        media["spotify_url"] = (
            result["spotify"].get("external_urls", {}).get("spotify")
            if isinstance(result["spotify"].get("external_urls"), dict)
            else ""
        )
    return {"status": "matched", "media": media, "raw_response": response}


def _normalize_acrcloud_response(response: dict[str, Any]) -> dict[str, Any]:
    status = response.get("status") if isinstance(response.get("status"), dict) else {}
    code = status.get("code")
    if code not in (0, "0", None):
        message = str(status.get("msg") or status.get("message") or f"ACRCloud status {code}")
        if "no result" in message.casefold() or code in (1001, "1001"):
            return {"status": "no_match", "raw_response": response}
        return {"status": "error", "error": message, "raw_response": response}

    match = _best_acrcloud_match(response)
    if match is None:
        return {"status": "no_match", "raw_response": response}

    media = _media_from_acrcloud_match(match)
    return {
        "status": "matched",
        "media": media,
        "confidence": _optional_float(match.get("score")),
        "raw_response": response,
    }


def _best_acrcloud_match(response: dict[str, Any]) -> dict[str, Any] | None:
    metadata = response.get("metadata") if isinstance(response.get("metadata"), dict) else {}
    candidates: list[dict[str, Any]] = []
    for key in ("music", "humming", "cover_songs", "custom_files"):
        values = metadata.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            result = item.get("result") if isinstance(item.get("result"), dict) else item
            candidates.append(result)
    if not candidates:
        return None
    return max(candidates, key=lambda item: _optional_float(item.get("score")) or 0.0)


def _media_from_acrcloud_match(match: dict[str, Any]) -> dict[str, Any]:
    external_ids = match.get("external_ids") if isinstance(match.get("external_ids"), dict) else {}
    external_metadata = (
        match.get("external_metadata")
        if isinstance(match.get("external_metadata"), dict)
        else {}
    )
    album = match.get("album") if isinstance(match.get("album"), dict) else {}
    artists = match.get("artists") if isinstance(match.get("artists"), list) else []
    artist = ", ".join(
        str(artist_item.get("name"))
        for artist_item in artists
        if isinstance(artist_item, dict) and artist_item.get("name")
    )
    spotify = external_metadata.get("spotify") if isinstance(external_metadata.get("spotify"), dict) else {}
    apple = (
        external_metadata.get("applemusic")
        if isinstance(external_metadata.get("applemusic"), dict)
        else {}
    )
    youtube = external_metadata.get("youtube") if isinstance(external_metadata.get("youtube"), dict) else {}
    spotify_track = spotify.get("track") if isinstance(spotify.get("track"), dict) else {}
    apple_track = apple.get("track") if isinstance(apple.get("track"), dict) else {}
    return {
        "title": match.get("title") or spotify_track.get("name") or apple_track.get("name") or "",
        "artist": artist,
        "album": album.get("name") or "",
        "label": match.get("label") or "",
        "release_date": match.get("release_date") or "",
        "isrc": external_ids.get("isrc") or "",
        "acrid": match.get("acrid") or "",
        "confidence": _optional_float(match.get("score")),
        "match_offset_s": (
            _optional_float(match.get("play_offset_ms")) / 1000.0
            if _optional_float(match.get("play_offset_ms")) is not None
            else None
        ),
        "spotify_id": str(spotify_track.get("id") or ""),
        "apple_music_id": str(apple_track.get("id") or ""),
        "web_url": f"https://www.youtube.com/watch?v={youtube.get('vid')}" if youtube.get("vid") else "",
    }


def _audd_cache_path(
    audio_path: Path,
    return_metadata: str,
    cache_dir: str | Path | None,
) -> Path | None:
    if cache_dir is None:
        return None

    digest = hashlib.sha256()
    digest.update(audio_path.read_bytes())
    digest.update(return_metadata.encode("utf-8"))
    return Path(cache_dir) / f"{digest.hexdigest()}.json"


def _acrcloud_endpoint(host: str) -> str:
    cleaned = host.strip().rstrip("/")
    if cleaned.startswith(("http://", "https://")):
        return f"{cleaned}/v1/identify"
    return f"https://{cleaned}/v1/identify"


def _provider_cache_path(
    audio_path: Path,
    provider_config: str,
    cache_dir: str | Path | None,
) -> Path | None:
    if cache_dir is None:
        return None
    digest = hashlib.sha256()
    digest.update(audio_path.read_bytes())
    digest.update(provider_config.encode("utf-8"))
    return Path(cache_dir) / f"{digest.hexdigest()}.json"


def _make_shazamkit_recognizer(
    helper_path: str | Path | None,
    *,
    timeout_s: float,
) -> FingerprintRecognizer:
    return lambda audio_path: recognize_with_shazamkit(
        audio_path,
        helper_path=helper_path,
        timeout_s=timeout_s,
    )


def _build_windows(
    source: Path,
    audio: np.ndarray,
    sample_rate: int,
    output_root: Path,
    *,
    mode: str,
    duration_s: float,
    window_s: float,
    hop_s: float,
    max_windows: int,
    start_offset_s: float,
    window_strategy: str,
) -> list[dict[str, Any]]:
    mode = mode.strip()
    include_whole = mode in {"Whole recording", "Whole + sliding windows"}
    include_sliding = mode in {"Sliding windows", "Whole + sliding windows"}
    windows: list[dict[str, Any]] = []
    if duration_s < MIN_SHAZAM_SIGNATURE_S:
        return windows

    if include_whole:
        whole_end_s = min(duration_s, MAX_SHAZAM_SIGNATURE_S)
        whole_path = source
        if duration_s > MAX_SHAZAM_SIGNATURE_S:
            whole_path = output_root / f"{source.stem}_whole_first_{MAX_SHAZAM_SIGNATURE_S:.0f}s.wav"
            end_index = min(len(audio), int(round(whole_end_s * sample_rate)))
            write_wav(whole_path, audio[:end_index], sample_rate)
        windows.append(
            {
                "mode": "whole",
                "start_s": 0.0,
                "end_s": round(whole_end_s, 3),
                "path": whole_path,
            }
        )

    if include_sliding:
        window_s = min(MAX_SHAZAM_SIGNATURE_S, max(MIN_SHAZAM_SIGNATURE_S, float(window_s)))
        hop_s = max(1.0, float(hop_s))
        starts = _sliding_starts(
            duration_s,
            window_s=window_s,
            hop_s=hop_s,
            max_windows=max_windows,
            start_offset_s=start_offset_s,
            window_strategy=window_strategy,
        )
        for index, start_s in enumerate(starts, start=1):
            end_s = min(duration_s, start_s + window_s)
            start_index = max(0, int(round(start_s * sample_rate)))
            end_index = min(len(audio), int(round(end_s * sample_rate)))
            if end_index <= start_index:
                continue
            window_path = output_root / f"{source.stem}_window_{index:03d}_{start_s:.2f}_{end_s:.2f}.wav"
            write_wav(window_path, audio[start_index:end_index], sample_rate)
            windows.append(
                {
                    "mode": "window",
                    "start_s": round(float(start_s), 3),
                    "end_s": round(float(end_s), 3),
                    "path": window_path,
                }
            )
    return windows


def _sliding_starts(
    duration_s: float,
    *,
    window_s: float,
    hop_s: float,
    max_windows: int,
    start_offset_s: float = 0.0,
    window_strategy: str = "From offset",
) -> list[float]:
    if duration_s <= 0:
        return []
    if duration_s <= window_s:
        return [0.0]

    latest_start = max(0.0, duration_s - window_s)
    max_count = max(1, int(max_windows))
    start_offset_s = min(latest_start, max(0.0, float(start_offset_s)))
    if window_strategy == "Center-out":
        center_start = min(latest_start, max(0.0, latest_start / 2.0 + start_offset_s))
        starts = [center_start]
        step = 1
        while len(starts) < max_count:
            added = False
            left = center_start - step * hop_s
            right = center_start + step * hop_s
            if left >= 0.0:
                starts.append(left)
                added = True
                if len(starts) >= max_count:
                    break
            if right <= latest_start:
                starts.append(right)
                added = True
            if not added:
                break
            step += 1
        return [round(float(value), 6) for value in starts[:max_count]]

    starts = np.arange(start_offset_s, latest_start + 0.001, hop_s)
    if starts.size == 0:
        starts = np.array([latest_start])
    if starts.size > max_count:
        starts = starts[:max_count]
    return [float(value) for value in starts]


def _row_from_payload(
    payload: dict[str, Any],
    *,
    test_case: str,
    expected_title: str,
    expected_artist: str,
    mode: str,
    start_s: float,
    end_s: float,
    audio_path: Path,
    elapsed_s: float,
) -> dict[str, Any]:
    status = str(payload.get("status") or "error")
    media = payload.get("media") if isinstance(payload.get("media"), dict) else {}
    title = _string(media.get("title") or payload.get("title"))
    artist = _string(media.get("artist") or payload.get("artist"))
    expected_match = _expected_match(title, artist, expected_title, expected_artist)
    recognized = status == "matched" and bool(title or artist)
    return {
        "test_case": test_case,
        "expected_title": expected_title,
        "expected_artist": expected_artist,
        "mode": mode,
        "window_start_s": round(start_s, 3),
        "window_end_s": round(end_s, 3),
        "audio_path": str(audio_path),
        "audio_file": audio_path.name,
        "matched_title": title,
        "matched_artist": artist,
        "matched_album": _string(media.get("album")),
        "matched_label": _string(media.get("label")),
        "expected_match": expected_match,
        "recognized": recognized,
        "status": status,
        "cached": bool(payload.get("cached", False)),
        "confidence": _optional_float(media.get("confidence") or payload.get("confidence")),
        "match_offset_s": _optional_float(media.get("match_offset_s") or payload.get("match_offset_s")),
        "frequency_skew": _optional_float(media.get("frequency_skew") or payload.get("frequency_skew")),
        "apple_music_id": _string(media.get("apple_music_id") or payload.get("apple_music_id")),
        "spotify_id": _string(media.get("spotify_id") or payload.get("spotify_id")),
        "acrid": _string(media.get("acrid") or payload.get("acrid")),
        "shazam_id": _string(media.get("shazam_id") or payload.get("shazam_id")),
        "isrc": _string(media.get("isrc") or payload.get("isrc")),
        "web_url": _string(media.get("web_url") or payload.get("web_url")),
        "spotify_url": _string(media.get("spotify_url") or payload.get("spotify_url")),
        "apple_music_url": _string(media.get("apple_music_url") or payload.get("apple_music_url")),
        "timecode": _string(media.get("timecode") or payload.get("timecode")),
        "elapsed_s": round(float(elapsed_s), 3),
        "error": _string(payload.get("error")),
    }


def _expected_match(
    matched_title: str,
    matched_artist: str,
    expected_title: str,
    expected_artist: str,
) -> str:
    expected_title = expected_title.strip()
    expected_artist = expected_artist.strip()
    if not expected_title and not expected_artist:
        return "not provided"

    title_ok = True
    artist_ok = True
    if expected_title:
        title_ok = _normalized(expected_title) in _normalized(matched_title)
    if expected_artist:
        artist_ok = _normalized(expected_artist) in _normalized(matched_artist)
    return "pass" if title_ok and artist_ok else "fail"


def _summarize_rows(
    rows: list[dict[str, Any]],
    expected_title: str,
    expected_artist: str,
) -> dict[str, Any]:
    total = len(rows)
    recognized = [row for row in rows if row["recognized"]]
    expected_passes = [row for row in rows if row["expected_match"] == "pass"]
    wrong_matches = [
        row
        for row in rows
        if row["recognized"] and row["expected_match"] == "fail"
    ]
    error_rows = [row for row in rows if row["status"] == "error"]
    no_match_rows = [row for row in rows if row["status"] == "no_match"]
    first_expected = expected_passes[0] if expected_passes else None
    first_recognized = recognized[0] if recognized else None
    whole_rows = [row for row in rows if row["mode"] == "whole"]
    window_rows = [row for row in rows if row["mode"] == "window"]

    return {
        "expected_title": expected_title,
        "expected_artist": expected_artist,
        "windows_tested": total,
        "recognized_windows": len(recognized),
        "expected_match_windows": len(expected_passes),
        "wrong_match_windows": len(wrong_matches),
        "no_match_windows": len(no_match_rows),
        "error_windows": len(error_rows),
        "top1_expected_match": rows[0]["expected_match"] == "pass" if rows else False,
        "any_window_expected_match": bool(expected_passes),
        "first_expected_match_start_s": first_expected["window_start_s"] if first_expected else None,
        "first_recognized_start_s": first_recognized["window_start_s"] if first_recognized else None,
        "no_match_rate": round(len(no_match_rows) / total, 3) if total else 0.0,
        "whole_file_recognized": any(row["recognized"] for row in whole_rows),
        "whole_file_expected_match": any(row["expected_match"] == "pass" for row in whole_rows),
        "sliding_window_recognized": any(row["recognized"] for row in window_rows),
        "sliding_window_expected_match": any(row["expected_match"] == "pass" for row in window_rows),
    }


def _interpret_rows(rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, str]:
    if not rows:
        return {}

    whole_tested = any(row["mode"] == "whole" for row in rows)
    windows_tested = any(row["mode"] == "window" for row in rows)
    any_expected = bool(summary["any_window_expected_match"])
    any_recognized = bool(summary["recognized_windows"])

    if whole_tested:
        full_file = "yes" if summary["whole_file_recognized"] else "no"
    else:
        full_file = "not tested"

    if windows_tested and whole_tested:
        if summary["sliding_window_expected_match"] and not summary["whole_file_expected_match"]:
            short_windows = "yes; a window matched when the whole file did not"
        elif summary["whole_file_expected_match"]:
            short_windows = "not necessary for this case; the whole file matched"
        else:
            short_windows = "inconclusive; neither mode found the expected song"
    elif windows_tested:
        short_windows = "tested without a whole-file baseline"
    else:
        short_windows = "not tested"

    if summary["first_expected_match_start_s"] is not None:
        audio_needed = f"expected song first matched at {summary['first_expected_match_start_s']:.1f}s"
    elif summary["first_recognized_start_s"] is not None:
        audio_needed = f"a song was first recognized at {summary['first_recognized_start_s']:.1f}s, but it did not match expectation"
    else:
        audio_needed = "no recognition in tested windows"

    if any_expected:
        backing_track = "likely enough for Shazam-style fingerprinting in this recording"
    elif any_recognized:
        backing_track = "recognizable audio exists, but it did not identify the expected song"
    else:
        backing_track = "not enough evidence from these windows"

    if summary["error_windows"] == summary["windows_tested"]:
        noise = "not measurable because every helper call failed"
    elif any_expected:
        noise = "did not prevent at least one expected match"
    elif any_recognized:
        noise = "may be causing wrong-match behavior; inspect matched windows"
    else:
        noise = "may be causing no-match behavior, or the backing track is not captured strongly enough"

    return {
        "does_full_file_recognition_work": full_file,
        "do_short_windows_work_better": short_windows,
        "how_much_audio_is_needed": audio_needed,
        "does_echo_or_noise_cause_failure": noise,
        "is_backing_track_bleed_enough": backing_track,
    }


def _status_text(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    if summary["error_windows"] == summary["windows_tested"]:
        return f"Fingerprinting failed for all {summary['windows_tested']} window(s)."
    if summary["any_window_expected_match"]:
        return (
            "Fingerprinting found the expected song in "
            f"{summary['expected_match_windows']} of {summary['windows_tested']} window(s)."
        )
    if summary["recognized_windows"]:
        return (
            "Fingerprinting recognized audio, but not the expected song "
            f"({summary['recognized_windows']} recognized window(s))."
        )
    return f"Fingerprinting finished with no matches across {summary['windows_tested']} window(s)."


def _empty_summary(expected_title: str, expected_artist: str) -> dict[str, Any]:
    return {
        "expected_title": expected_title,
        "expected_artist": expected_artist,
        "windows_tested": 0,
        "recognized_windows": 0,
        "expected_match_windows": 0,
        "wrong_match_windows": 0,
        "no_match_windows": 0,
        "error_windows": 0,
        "top1_expected_match": False,
        "any_window_expected_match": False,
        "first_expected_match_start_s": None,
        "first_recognized_start_s": None,
        "no_match_rate": 0.0,
        "whole_file_recognized": False,
        "whole_file_expected_match": False,
        "sliding_window_recognized": False,
        "sliding_window_expected_match": False,
    }


def _default_helper_path() -> Path:
    built_helper = _built_helper_path()
    if built_helper.exists():
        return built_helper
    return Path(__file__).resolve().parents[2] / "scripts" / "shazamkit_recognize.swift"


def _resolve_helper_path(helper_path: str | Path | None) -> Path:
    configured = helper_path or os.environ.get("KONOPRO_SHAZAMKIT_HELPER")
    helper = Path(configured).expanduser() if configured else _default_helper_path()
    if helper.suffix == ".app":
        return helper / "Contents" / "MacOS" / helper.stem
    return helper


def _helper_command(helper: Path, audio_path: str | Path) -> list[str]:
    if helper.suffix == ".swift":
        return ["swift", str(helper), str(audio_path)]
    return [str(helper), str(audio_path)]


def _built_helper_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "shazamkit_helper"
        / "build"
        / "KonoproShazamHelper.app"
        / "Contents"
        / "MacOS"
        / "KonoproShazamHelper"
    )


def _friendly_helper_error(error: str) -> str:
    if (
        "enabled the ShazamKit App Service" in error
        or "AMSStatusCode=401" in error
        or "Missing entitlements" in error
    ):
        return (
            "ShazamKit catalog access is not enabled for this helper. "
            "Apple requires a signed app/helper whose App ID has the ShazamKit App Service enabled."
        )
    return error


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _string(value: Any) -> str:
    return "" if value is None else str(value)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None
