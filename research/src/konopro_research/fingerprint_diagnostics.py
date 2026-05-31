from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from statistics import median
from typing import Any


HEADERLESS_COLUMNS = (
    "provider",
    "window_start_s",
    "window_end_s",
    "status",
    "recognized",
    "matched_title",
    "matched_artist",
    "isrc",
    "confidence",
    "audio_path",
    "audio_file",
    "error",
)

NUMERIC_FIELDS = {
    "window_start_s",
    "window_end_s",
    "start_s",
    "end_s",
    "confidence",
    "elapsed_s",
    "match_offset_s",
}
BOOLEAN_FIELDS = {"recognized", "expected_match"}


@dataclass(frozen=True)
class FingerprintRunProfile:
    provider: str
    tested_windows: int
    recognized_windows: int
    no_match_windows: int
    error_windows: int
    no_match_rate: float
    error_rate: float
    median_window_s: float
    median_hop_s: float
    scan_coverage_pct: float
    first_window_start_s: float | None
    last_window_end_s: float | None
    recording_duration_s: float | None
    requested_window_s: float | None
    requested_hop_s: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FingerprintDiagnosticFlag:
    code: str
    severity: str
    message: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FingerprintWeakCandidate:
    provider: str
    identity_key: str
    title: str
    artist: str
    isrc: str
    start_s: float
    end_s: float
    center_s: float
    match_count: int
    confidence: float | None
    reason: str
    recovery_start_s: float
    recovery_end_s: float
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["warnings"] = list(self.warnings)
        return data


@dataclass(frozen=True)
class FingerprintRecoverySweep:
    name: str
    priority: int
    reason: str
    window_s: float
    hop_s: float
    start_s: float
    end_s: float
    max_windows: int
    estimated_api_calls: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FingerprintDiagnosticReport:
    provider: str
    profile: FingerprintRunProfile
    flags: tuple[FingerprintDiagnosticFlag, ...]
    weak_candidates: tuple[FingerprintWeakCandidate, ...]
    recommendations: tuple[dict[str, Any], ...]
    can_segment: bool
    confidence_level: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "profile": self.profile.to_dict(),
            "flags": [flag.to_dict() for flag in self.flags],
            "weak_candidates": [candidate.to_dict() for candidate in self.weak_candidates],
            "recommendations": [dict(recommendation) for recommendation in self.recommendations],
            "can_segment": self.can_segment,
            "confidence_level": self.confidence_level,
        }


def load_fingerprint_rows_csv(source: str | Path) -> list[dict[str, Any]]:
    text = _read_csv_source(source)
    if not text.strip():
        return []

    raw_rows = list(csv.reader(StringIO(text)))
    rows = [row for row in raw_rows if any(cell.strip() for cell in row)]
    if not rows:
        return []

    first = [cell.strip() for cell in rows[0]]
    has_header = bool(set(first) & {"window_start_s", "start_s", "status", "matched_title"})
    if has_header:
        reader = csv.DictReader(StringIO(text))
        return normalize_fingerprint_rows(row for row in reader if row)

    mapped: list[dict[str, Any]] = []
    for raw in rows:
        row: dict[str, Any] = {}
        for index, value in enumerate(raw):
            key = HEADERLESS_COLUMNS[index] if index < len(HEADERLESS_COLUMNS) else f"extra_{index}"
            row[key] = value
        mapped.append(row)
    return normalize_fingerprint_rows(mapped)


def normalize_fingerprint_rows(rows: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw_row in rows or []:
        row = dict(raw_row)
        if "start_s" in row and "window_start_s" not in row:
            row["window_start_s"] = row["start_s"]
        if "end_s" in row and "window_end_s" not in row:
            row["window_end_s"] = row["end_s"]
        if "title" in row and "matched_title" not in row:
            row["matched_title"] = row["title"]
        if "artist" in row and "matched_artist" not in row:
            row["matched_artist"] = row["artist"]

        for field in NUMERIC_FIELDS:
            if field in row:
                row[field] = _optional_float(row.get(field))
        for field in BOOLEAN_FIELDS:
            if field in row:
                row[field] = _bool(row.get(field))

        row["status"] = _string(row.get("status") or "unknown").lower()
        if "recognized" not in row:
            row["recognized"] = row["status"] == "matched" and _has_identity(row)
        row["matched_title"] = _string(row.get("matched_title"))
        row["matched_artist"] = _string(row.get("matched_artist"))
        row["isrc"] = _string(row.get("isrc"))
        row["error"] = _string(row.get("error"))
        normalized.append(row)
    return normalized


def diagnose_fingerprint_rows(
    rows: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    provider: str,
    recording_duration_s: float | None = None,
    requested_window_s: float | None = None,
    requested_hop_s: float | None = None,
    min_confidence: float = 0.6,
) -> FingerprintDiagnosticReport:
    normalized_rows = normalize_fingerprint_rows(rows)
    provider_key = provider.strip().casefold() or _provider_from_rows(normalized_rows)
    profile = _build_profile(
        normalized_rows,
        provider=provider_key,
        recording_duration_s=recording_duration_s,
        requested_window_s=requested_window_s,
        requested_hop_s=requested_hop_s,
    )
    weak_candidates = _build_candidates(
        normalized_rows,
        provider=provider_key,
        recording_duration_s=profile.recording_duration_s,
        min_confidence=float(min_confidence),
    )
    flags = _build_flags(profile, weak_candidates, min_confidence=float(min_confidence))
    flag_codes = {flag.code for flag in flags}
    can_segment = any(
        candidate.match_count >= 2
        and candidate.confidence is not None
        and candidate.confidence >= float(min_confidence)
        and "sparse_scan" not in flag_codes
        and "short_windows" not in flag_codes
        for candidate in weak_candidates
    )
    confidence_level = (
        "recoverable" if can_segment else "weak_clues" if weak_candidates else "failed"
    )
    recommendations = _build_recommendations(profile, flags, weak_candidates)
    return FingerprintDiagnosticReport(
        provider=provider_key,
        profile=profile,
        flags=flags,
        weak_candidates=weak_candidates,
        recommendations=recommendations,
        can_segment=can_segment,
        confidence_level=confidence_level,
    )


def plan_recovery_sweeps(
    report: FingerprintDiagnosticReport,
    *,
    recording_duration_s: float | None = None,
    request_budget: int = 120,
) -> tuple[FingerprintRecoverySweep, ...]:
    duration_s = _duration_for_recovery(report, recording_duration_s)
    if duration_s <= 0:
        return ()

    budget = max(1, int(request_budget))
    flags = {flag.code for flag in report.flags}
    sweeps: list[FingerprintRecoverySweep] = []
    if flags & {"sparse_scan", "short_windows", "high_no_match_rate"}:
        sweeps.append(
            _make_sweep(
                name="Dense full-session retry",
                priority=10,
                reason="The original scan was too sparse or short to rule out a song interval.",
                window_s=10.0,
                hop_s=5.0,
                start_s=0.0,
                end_s=duration_s,
                request_budget=budget,
            )
        )

    for candidate in report.weak_candidates:
        if candidate.reason in {"singleton_match", "low_confidence_cluster"}:
            start_s = max(0.0, candidate.center_s - 75.0)
            end_s = min(duration_s, candidate.center_s + 75.0)
            sweeps.append(
                _make_sweep(
                    name="Focused singleton recovery",
                    priority=20,
                    reason=f"Validate weak clue for {candidate.title or candidate.identity_key}.",
                    window_s=10.0,
                    hop_s=5.0,
                    start_s=start_s,
                    end_s=end_s,
                    request_budget=budget,
                )
            )

    deduped: dict[tuple[str, float, float], FingerprintRecoverySweep] = {}
    for sweep in sweeps:
        deduped[(sweep.name, sweep.start_s, sweep.end_s)] = sweep
    return tuple(sorted(deduped.values(), key=lambda sweep: (sweep.priority, sweep.start_s)))


def _read_csv_source(source: str | Path) -> str:
    if isinstance(source, Path):
        return source.read_text()
    text = str(source)
    if "\n" not in text and "\r" not in text:
        path = Path(text)
        if path.exists():
            return path.read_text()
    return text


def _build_profile(
    rows: list[dict[str, Any]],
    *,
    provider: str,
    recording_duration_s: float | None,
    requested_window_s: float | None,
    requested_hop_s: float | None,
) -> FingerprintRunProfile:
    starts = [_float(row.get("window_start_s")) for row in rows if row.get("window_start_s") is not None]
    ends = [_float(row.get("window_end_s")) for row in rows if row.get("window_end_s") is not None]
    durations = [
        max(0.0, _float(row.get("window_end_s")) - _float(row.get("window_start_s")))
        for row in rows
        if row.get("window_start_s") is not None and row.get("window_end_s") is not None
    ]
    sorted_starts = sorted(starts)
    hops = [
        sorted_starts[index + 1] - sorted_starts[index]
        for index in range(len(sorted_starts) - 1)
        if sorted_starts[index + 1] > sorted_starts[index]
    ]
    tested = len(rows)
    recognized = sum(1 for row in rows if _is_recognized(row))
    no_match = sum(1 for row in rows if _string(row.get("status")) == "no_match")
    errors = sum(1 for row in rows if _string(row.get("status")) == "error")
    median_window_s = float(requested_window_s) if requested_window_s else _median_or_zero(durations)
    median_hop_s = float(requested_hop_s) if requested_hop_s else _median_or_zero(hops)
    first_start = min(starts) if starts else None
    last_end = max(ends) if ends else None
    inferred_duration = last_end if last_end is not None else None
    duration = float(recording_duration_s) if recording_duration_s else inferred_duration
    scan_coverage_pct = 0.0
    if duration and duration > 0:
        scan_coverage_pct = min(100.0, 100.0 * tested * max(0.0, median_window_s) / duration)
    return FingerprintRunProfile(
        provider=provider,
        tested_windows=tested,
        recognized_windows=recognized,
        no_match_windows=no_match,
        error_windows=errors,
        no_match_rate=round(no_match / tested, 4) if tested else 0.0,
        error_rate=round(errors / tested, 4) if tested else 0.0,
        median_window_s=round(median_window_s, 3),
        median_hop_s=round(median_hop_s, 3),
        scan_coverage_pct=round(scan_coverage_pct, 2),
        first_window_start_s=round(first_start, 3) if first_start is not None else None,
        last_window_end_s=round(last_end, 3) if last_end is not None else None,
        recording_duration_s=round(duration, 3) if duration is not None else None,
        requested_window_s=float(requested_window_s) if requested_window_s is not None else None,
        requested_hop_s=float(requested_hop_s) if requested_hop_s is not None else None,
    )


def _build_candidates(
    rows: list[dict[str, Any]],
    *,
    provider: str,
    recording_duration_s: float | None,
    min_confidence: float,
) -> tuple[FingerprintWeakCandidate, ...]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not _is_recognized(row):
            continue
        identity_key = _identity_key(row)
        if identity_key:
            grouped[identity_key].append(row)

    candidates: list[FingerprintWeakCandidate] = []
    duration = float(recording_duration_s or 0.0)
    for identity_key, group in grouped.items():
        starts = [_float(row.get("window_start_s")) for row in group]
        ends = [_float(row.get("window_end_s")) for row in group]
        confidences = [
            value for value in (_optional_float(row.get("confidence")) for row in group) if value is not None
        ]
        confidence = round(sum(confidences) / len(confidences), 3) if confidences else None
        start_s = min(starts)
        end_s = max(ends)
        center_s = (start_s + end_s) / 2.0
        warnings: list[str] = []
        if len(group) < 2:
            reason = "singleton_match"
            warnings.append("only one recognized window")
        elif confidence is not None and confidence < min_confidence:
            reason = "low_confidence_cluster"
            warnings.append("provider confidence is below threshold")
        else:
            reason = "repeated_match"
        title = _string(group[0].get("matched_title"))
        artist = _string(group[0].get("matched_artist"))
        isrc = _clean_isrc(_string(group[0].get("isrc")))
        recovery_start_s = max(0.0, center_s - 75.0)
        recovery_end_s = min(duration, center_s + 75.0) if duration > 0 else center_s + 75.0
        candidates.append(
            FingerprintWeakCandidate(
                provider=provider,
                identity_key=identity_key,
                title=title,
                artist=artist,
                isrc=isrc,
                start_s=round(start_s, 3),
                end_s=round(end_s, 3),
                center_s=round(center_s, 3),
                match_count=len(group),
                confidence=confidence,
                reason=reason,
                recovery_start_s=round(recovery_start_s, 3),
                recovery_end_s=round(recovery_end_s, 3),
                warnings=tuple(warnings),
            )
        )
    return tuple(sorted(candidates, key=lambda candidate: (candidate.start_s, candidate.identity_key)))


def _build_flags(
    profile: FingerprintRunProfile,
    candidates: tuple[FingerprintWeakCandidate, ...],
    *,
    min_confidence: float,
) -> tuple[FingerprintDiagnosticFlag, ...]:
    flags: list[FingerprintDiagnosticFlag] = []
    if profile.tested_windows == 0:
        flags.append(_flag("no_rows", "error", "No fingerprint rows were provided.", {}))
        return tuple(flags)

    if profile.median_window_s and profile.median_window_s < 8.0:
        flags.append(
            _flag(
                "short_windows",
                "warning",
                "Fingerprint windows are shorter than the recommended 8-12 seconds.",
                {"median_window_s": profile.median_window_s},
            )
        )

    sparse_by_hop = profile.median_hop_s > max(10.0, profile.median_window_s * 1.5)
    sparse_by_coverage = profile.scan_coverage_pct < 35.0 and profile.tested_windows > 1
    if sparse_by_hop or sparse_by_coverage:
        flags.append(
            _flag(
                "sparse_scan",
                "warning",
                "The scan spacing is too sparse to rule out song intervals.",
                {
                    "median_hop_s": profile.median_hop_s,
                    "scan_coverage_pct": profile.scan_coverage_pct,
                },
            )
        )

    if profile.no_match_rate >= 0.8 and profile.tested_windows >= 3:
        flags.append(
            _flag(
                "high_no_match_rate",
                "warning",
                "Most tested windows returned no match.",
                {"no_match_rate": profile.no_match_rate},
            )
        )
    if profile.recognized_windows == 0:
        flags.append(
            _flag(
                "all_no_match",
                "warning",
                "No windows were recognized by the provider.",
                {"tested_windows": profile.tested_windows},
            )
        )
    if profile.error_rate >= 0.2:
        flags.append(
            _flag(
                "provider_errors",
                "error",
                "A significant share of provider requests errored.",
                {"error_rate": profile.error_rate},
            )
        )

    if any(candidate.match_count == 1 for candidate in candidates):
        flags.append(
            _flag(
                "singleton_candidate",
                "warning",
                "At least one song clue appears in only one window.",
                {"candidate_count": sum(1 for candidate in candidates if candidate.match_count == 1)},
            )
        )
    low_confidence = [
        candidate
        for candidate in candidates
        if candidate.confidence is not None and candidate.confidence < min_confidence
    ]
    if low_confidence:
        flags.append(
            _flag(
                "low_confidence_match",
                "warning",
                "At least one recognized clue is below the confidence threshold.",
                {
                    "min_confidence": min_confidence,
                    "candidate_count": len(low_confidence),
                },
            )
        )
    return tuple(flags)


def _build_recommendations(
    profile: FingerprintRunProfile,
    flags: tuple[FingerprintDiagnosticFlag, ...],
    candidates: tuple[FingerprintWeakCandidate, ...],
) -> tuple[dict[str, Any], ...]:
    flag_codes = {flag.code for flag in flags}
    recommendations: list[dict[str, Any]] = []
    if flag_codes & {"sparse_scan", "short_windows"}:
        recommendations.append(
            {
                "code": "dense_scan",
                "priority": 10,
                "action": "Run 10-12s windows with a 5s hop before judging the session.",
                "reason": "The current scan leaves large untested gaps.",
            }
        )
    if any(candidate.reason == "singleton_match" for candidate in candidates):
        recommendations.append(
            {
                "code": "focused_singleton_recovery",
                "priority": 20,
                "action": "Run a focused dense scan around the singleton timestamp.",
                "reason": "A single clue needs repeated neighboring matches before segmentation.",
            }
        )
    if "all_no_match" in flag_codes or (
        "high_no_match_rate" in flag_codes and profile.recognized_windows == 0
    ):
        recommendations.append(
            {
                "code": "source_provider_comparison",
                "priority": 30,
                "action": "Compare raw phone audio, accompaniment stem, and another provider.",
                "reason": "Dense no-match runs often point to source mix or provider catalog mismatch.",
            }
        )
    if "provider_errors" in flag_codes:
        recommendations.append(
            {
                "code": "provider_health_check",
                "priority": 5,
                "action": "Check credentials, provider quota, and network errors before rescanning.",
                "reason": "Provider errors make recognition evidence unreliable.",
            }
        )
    if not recommendations:
        recommendations.append(
            {
                "code": "segment_or_score",
                "priority": 50,
                "action": "Use repeated high-confidence intervals for segmentation and scoring handoff.",
                "reason": "The fingerprint rows contain enough repeated evidence for the prototype.",
            }
        )
    return tuple(sorted(recommendations, key=lambda item: int(item["priority"])))


def _make_sweep(
    *,
    name: str,
    priority: int,
    reason: str,
    window_s: float,
    hop_s: float,
    start_s: float,
    end_s: float,
    request_budget: int,
) -> FingerprintRecoverySweep:
    start_s = max(0.0, float(start_s))
    end_s = max(start_s, float(end_s))
    calls = _window_count(end_s - start_s, window_s=window_s, hop_s=hop_s)
    estimated = min(max(1, calls), max(1, int(request_budget)))
    if calls > request_budget:
        end_s = min(end_s, start_s + window_s + hop_s * (request_budget - 1))
    return FingerprintRecoverySweep(
        name=name,
        priority=int(priority),
        reason=reason,
        window_s=float(window_s),
        hop_s=float(hop_s),
        start_s=round(start_s, 3),
        end_s=round(end_s, 3),
        max_windows=estimated,
        estimated_api_calls=estimated,
    )


def _window_count(duration_s: float, *, window_s: float, hop_s: float) -> int:
    if duration_s <= 0:
        return 0
    if duration_s <= window_s:
        return 1
    return int((duration_s - window_s) // hop_s) + 1


def _duration_for_recovery(
    report: FingerprintDiagnosticReport,
    recording_duration_s: float | None,
) -> float:
    if recording_duration_s is not None and recording_duration_s > 0:
        return float(recording_duration_s)
    if report.profile.recording_duration_s is not None and report.profile.recording_duration_s > 0:
        return float(report.profile.recording_duration_s)
    if report.profile.last_window_end_s is not None:
        return float(report.profile.last_window_end_s)
    return 0.0


def _flag(code: str, severity: str, message: str, evidence: dict[str, Any]) -> FingerprintDiagnosticFlag:
    return FingerprintDiagnosticFlag(
        code=code,
        severity=severity,
        message=message,
        evidence=dict(evidence),
    )


def _is_recognized(row: dict[str, Any]) -> bool:
    return _bool(row.get("recognized")) and _string(row.get("status")) == "matched" and _has_identity(row)


def _has_identity(row: dict[str, Any]) -> bool:
    return bool(
        _string(row.get("matched_title"))
        or _string(row.get("matched_artist"))
        or _string(row.get("isrc"))
        or _string(row.get("acrid"))
        or _string(row.get("shazam_id"))
    )


def _identity_key(row: dict[str, Any]) -> str:
    isrc = _clean_isrc(_string(row.get("isrc")))
    if isrc:
        return f"isrc:{_normalized_id(isrc)}"
    for field in ("acrid", "shazam_id", "spotify_id", "apple_music_id"):
        value = _string(row.get(field))
        if value:
            return f"{field}:{_normalized_id(value)}"
    title = _normalized_text(_string(row.get("matched_title")))
    artist = _normalized_text(_string(row.get("matched_artist")))
    if title and artist:
        return f"title_artist:{title}::{artist}"
    if title:
        return f"title:{title}"
    return ""


def _provider_from_rows(rows: list[dict[str, Any]]) -> str:
    for row in rows:
        provider = _string(row.get("provider")).casefold()
        if provider:
            return provider
    return "unknown"


def _median_or_zero(values: list[float]) -> float:
    return float(median(values)) if values else 0.0


def _clean_isrc(value: str) -> str:
    lowered = value.strip()
    if lowered.casefold().startswith("isrc:"):
        return lowered.split(":", 1)[1].strip()
    return lowered


def _normalized_id(value: str) -> str:
    return "".join(char.casefold() for char in value if char.isalnum() or char in {"-", "_"})


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def _float(value: Any) -> float:
    parsed = _optional_float(value)
    return float(parsed) if parsed is not None else 0.0


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip()
    if not text or text.casefold() in {"none", "nan", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, int | float):
        return bool(value)
    return str(value).strip().casefold() in {"1", "true", "yes", "y", "matched"}


def _string(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.casefold() in {"none", "nan", "null"}:
        return ""
    return text
