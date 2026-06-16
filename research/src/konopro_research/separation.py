from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SeparationResult:
    source_path: Path
    analysis_path: Path
    backend: str
    stem: str
    model: str | None
    device: str
    used_original: bool
    used_cache: bool
    warnings: tuple[str, ...]
    debug_output: str = ""
    cache_key: str = ""
    cache_path: str = ""
    cache_status: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_path"] = str(self.source_path)
        data["analysis_path"] = str(self.analysis_path)
        return data


def is_demucs_available() -> bool:
    return importlib.util.find_spec("demucs") is not None


def prepare_vocal_analysis_audio(
    path: str | Path,
    *,
    cache_dir: str | Path,
    backend: str = "none",
    stem: str = "vocals",
    model: str = "htdemucs",
    device: str = "cpu",
    shifts: int = 1,
    overlap: float = 0.25,
    jobs: int = 0,
    timeout_s: int = 1800,
) -> SeparationResult:
    """Return the audio path that should be analyzed by downstream audio tasks.

    The default returns the original file. With the Demucs backend, the function
    creates a cached stem and returns that path. If Demucs is unavailable
    or fails, it falls back to the original file with a warning so the demo keeps
    running.
    """
    source_path = Path(path)
    backend = backend.lower().strip()
    stem = _normalize_stem(stem)
    if backend in {"", "none", "off", "original"}:
        return SeparationResult(
            source_path=source_path,
            analysis_path=source_path,
            backend="none",
            stem=stem,
            model=None,
            device=device,
            used_original=True,
            used_cache=False,
            warnings=(),
            debug_output="",
            cache_status="off",
        )
    if backend != "demucs":
        raise ValueError(f"Unsupported source-separation backend: {backend}")
    if stem not in {"vocals", "no_vocals"}:
        raise ValueError("Only the vocals and no_vocals stems are supported right now")
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not is_demucs_available():
        return _fallback_result(
            source_path,
            backend=backend,
            stem=stem,
            model=model,
            device=device,
            warning="Demucs is not installed; install with `uv sync --extra stems` and restart the app.",
        )

    cache_dir = Path(cache_dir)
    cache_key = _cache_key(source_path, model=model, device=device, stem=stem, shifts=shifts, overlap=overlap)
    target_path = cache_dir / "demucs" / model / cache_key / f"{stem}.wav"
    fallback_path = target_path.parent / "fallback.json"
    if target_path.exists() and target_path.stat().st_size > 0:
        return SeparationResult(
            source_path=source_path,
            analysis_path=target_path,
            backend=backend,
            stem=stem,
            model=model,
            device=device,
            used_original=False,
            used_cache=True,
            warnings=(),
            debug_output="",
            cache_key=cache_key,
            cache_path=str(target_path),
            cache_status="hit",
        )
    cached_fallback = _read_fallback_marker(fallback_path)
    if cached_fallback is not None and not _should_retry_cached_fallback(cached_fallback):
        return _fallback_result(
            source_path,
            backend=backend,
            stem=stem,
            model=model,
            device=device,
            warning=f"Using cached Demucs fallback; {cached_fallback['warning']}",
            debug_output=str(cached_fallback.get("debug_output", "")),
            used_cache=True,
            cache_key=cache_key,
            cache_path=str(fallback_path),
            cache_status="cached fallback",
        )

    run_dir = cache_dir / "_runs" / cache_key
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "demucs.separate",
        "--two-stems=vocals",
        "-n",
        model,
        "-d",
        device,
        "--out",
        str(run_dir),
        "--shifts",
        str(max(1, int(shifts))),
        "--overlap",
        str(float(overlap)),
    ]
    if jobs > 0:
        command.extend(["-j", str(jobs)])
    command.append(str(source_path))

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except Exception as exc:
        warning = f"Demucs could not run; using original audio for analysis. {exc}"
        return _store_fallback_result(
            source_path,
            fallback_path=fallback_path,
            backend=backend,
            stem=stem,
            model=model,
            device=device,
            warning=warning,
            debug_output="",
            cache_key=cache_key,
        )
    if completed.returncode != 0:
        detail = _demucs_failure_summary(completed.stderr, completed.stdout)
        debug_output = _compact_process_output(
            "\n".join(part for part in (completed.stderr, completed.stdout) if part),
            limit=2000,
        )
        return _store_fallback_result(
            source_path,
            fallback_path=fallback_path,
            backend=backend,
            stem=stem,
            model=model,
            device=device,
            warning=detail,
            debug_output=debug_output,
            cache_key=cache_key,
        )

    candidates = sorted(run_dir.glob(f"**/{stem}.wav"))
    if not candidates:
        return _store_fallback_result(
            source_path,
            fallback_path=fallback_path,
            backend=backend,
            stem=stem,
            model=model,
            device=device,
            warning="Demucs finished but no vocals stem was found; using original audio for analysis.",
            debug_output="",
            cache_key=cache_key,
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(candidates[0], target_path)
    return SeparationResult(
        source_path=source_path,
        analysis_path=target_path,
        backend=backend,
        stem=stem,
        model=model,
        device=device,
        used_original=False,
        used_cache=False,
        warnings=(),
        debug_output="",
        cache_key=cache_key,
        cache_path=str(target_path),
        cache_status="stored",
    )


def _normalize_stem(stem: str) -> str:
    stem = stem.lower().strip().replace("-", "_")
    if stem in {"instrumental", "instrumentals", "accompaniment", "backing", "bgm"}:
        return "no_vocals"
    return stem


def _fallback_result(
    source_path: Path,
    *,
    backend: str,
    stem: str,
    model: str,
    device: str,
    warning: str,
    debug_output: str = "",
    used_cache: bool = False,
    cache_key: str = "",
    cache_path: str = "",
    cache_status: str = "fallback",
) -> SeparationResult:
    return SeparationResult(
        source_path=source_path,
        analysis_path=source_path,
        backend=backend,
        stem=stem,
        model=model,
        device=device,
        used_original=True,
        used_cache=used_cache,
        warnings=(warning,),
        debug_output=debug_output,
        cache_key=cache_key,
        cache_path=cache_path,
        cache_status=cache_status,
    )


def _store_fallback_result(
    source_path: Path,
    *,
    fallback_path: Path,
    backend: str,
    stem: str,
    model: str,
    device: str,
    warning: str,
    debug_output: str,
    cache_key: str,
) -> SeparationResult:
    _write_fallback_marker(fallback_path, warning=warning, debug_output=debug_output)
    return _fallback_result(
        source_path,
        backend=backend,
        stem=stem,
        model=model,
        device=device,
        warning=warning,
        debug_output=debug_output,
        used_cache=False,
        cache_key=cache_key,
        cache_path=str(fallback_path),
        cache_status="stored fallback",
    )


def _cache_key(path: Path, **options: object) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    for key, value in sorted(options.items()):
        hasher.update(f"{key}={value}".encode("utf-8"))
    return hasher.hexdigest()[:20]


def _read_fallback_marker(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception:
        return None
    warning = str(data.get("warning", "")).strip()
    if not warning:
        return None
    return data


def _write_fallback_marker(path: Path, *, warning: str, debug_output: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": time.time(),
        "warning": warning,
        "debug_output": debug_output,
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _should_retry_cached_fallback(data: dict[str, Any]) -> bool:
    warning = str(data.get("warning", ""))
    if "TorchCodec" in warning and importlib.util.find_spec("torchcodec") is not None:
        return True
    return False


def _demucs_failure_summary(stderr: str | None, stdout: str | None) -> str:
    output = "\n".join(part for part in (stderr, stdout) if part)
    lines = [line.strip() for line in output.splitlines() if line.strip()]

    if _mentions_torchcodec_issue(output):
        return (
            "Demucs failed because TorchCodec is required by torchaudio for writing stems. "
            "Install it with `pip install torchcodec` (or your environment package manager), "
            "then rerun analysis."
        )

    meaningful = [
        _strip_carriage_progress(line)
        for line in lines
        if not _looks_like_progress_line(line)
    ]
    meaningful = [line for line in meaningful if line]

    if meaningful:
        detail = _compact_process_output("\n".join(meaningful[-6:]), limit=360)
        return f"Demucs failed; using original audio for analysis. {detail}"

    if _looks_like_progress_failure(output):
        return (
            "Demucs stopped during source separation; using original audio for analysis. "
            "Open processing metadata for raw Demucs output."
        )
    return (
        "Demucs failed; using original audio for analysis. "
        "Open processing metadata for raw Demucs output."
    )


def _mentions_torchcodec_issue(output: str) -> bool:
    lower = output.lower()
    return (
        "torchcodec is required" in lower
        or "importerror: torchcodec is required" in lower
        or "save_with_torchcodec" in lower
    )


def _looks_like_progress_failure(output: str) -> bool:
    return "%" in output and "seconds/s" in output


def _looks_like_progress_line(line: str) -> bool:
    compact = line.strip()
    return "%" in compact and ("seconds/s" in compact or "|" in compact)


def _strip_carriage_progress(line: str) -> str:
    return line.split("\r")[-1].strip()


def _compact_process_output(output: str, *, limit: int = 240) -> str:
    compact = " ".join(output.strip().split())
    if not compact:
        return "No process output was captured."
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."
