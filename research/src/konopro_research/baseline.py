from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, TextIO

import numpy as np


@dataclass(frozen=True)
class MelodyNote:
    start_s: float
    end_s: float
    midi: float
    label: str = ""

    def __post_init__(self) -> None:
        if self.end_s <= self.start_s:
            raise ValueError("MelodyNote end_s must be greater than start_s")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def frequency_hz(self) -> float:
        return midi_to_hz(self.midi)


@dataclass(frozen=True)
class MelodyBaseline:
    notes: tuple[MelodyNote, ...]
    title: str = "Untitled baseline"

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.notes, key=lambda note: note.start_s))
        if not ordered:
            raise ValueError("MelodyBaseline requires at least one note")
        object.__setattr__(self, "notes", ordered)

    @property
    def duration_s(self) -> float:
        return max(note.end_s for note in self.notes)

    def hz_at(self, times_s: np.ndarray) -> np.ndarray:
        times = np.asarray(times_s, dtype=float)
        expected = np.full(times.shape, np.nan, dtype=float)
        for note in self.notes:
            mask = (times >= note.start_s) & (times < note.end_s)
            expected[mask] = note.frequency_hz
        return expected

    def midi_at(self, times_s: np.ndarray) -> np.ndarray:
        hz = self.hz_at(times_s)
        midi = np.full(hz.shape, np.nan, dtype=float)
        mask = np.isfinite(hz) & (hz > 0)
        midi[mask] = hz_to_midi(hz[mask])
        return midi


def midi_to_hz(midi: float | np.ndarray) -> float | np.ndarray:
    return 440.0 * (2.0 ** ((np.asarray(midi) - 69.0) / 12.0))


def hz_to_midi(hz: float | np.ndarray) -> float | np.ndarray:
    hz_array = np.asarray(hz, dtype=float)
    return 69.0 + 12.0 * np.log2(hz_array / 440.0)


def cents_difference(observed_hz: np.ndarray, expected_hz: np.ndarray) -> np.ndarray:
    observed = np.asarray(observed_hz, dtype=float)
    expected = np.asarray(expected_hz, dtype=float)
    return 1200.0 * np.log2(observed / expected)


def load_baseline_csv(path_or_file: str | Path | TextIO, title: str | None = None) -> MelodyBaseline:
    close_after = False
    if isinstance(path_or_file, (str, Path)):
        file_obj = open(path_or_file, newline="", encoding="utf-8")
        close_after = True
        resolved_title = title or Path(path_or_file).stem
    else:
        file_obj = path_or_file
        resolved_title = title or "Uploaded baseline"

    try:
        reader = csv.DictReader(file_obj)
        if not reader.fieldnames:
            raise ValueError("Baseline CSV must include a header row")
        notes = [_row_to_note(row) for row in reader]
    finally:
        if close_after:
            file_obj.close()

    return MelodyBaseline(tuple(notes), title=resolved_title)


def write_baseline_csv(baseline: MelodyBaseline, path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        file_obj.write(baseline_to_csv_text(baseline))


def baseline_to_csv_text(baseline: MelodyBaseline) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["start_s", "end_s", "midi", "label"])
    writer.writeheader()
    for row in baseline_to_rows(baseline):
        writer.writerow(row)
    return output.getvalue()


def baseline_to_rows(baseline: MelodyBaseline) -> list[dict[str, object]]:
    return [
        {
            "start_s": round(note.start_s, 6),
            "end_s": round(note.end_s, 6),
            "midi": round(note.midi, 6),
            "label": note.label,
        }
        for note in baseline.notes
    ]


def baseline_from_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    title: str = "Edited baseline",
) -> MelodyBaseline:
    notes: list[MelodyNote] = []
    for row in rows:
        if _empty_row(row):
            continue
        notes.append(
            MelodyNote(
                start_s=float(row["start_s"]),
                end_s=float(row["end_s"]),
                midi=float(row["midi"]),
                label=str(row.get("label", "") or ""),
            )
        )
    return MelodyBaseline(tuple(notes), title=title)


def baseline_from_pitch_contour(
    times_s: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    title: str = "Extracted reference",
    window_s: float = 0.20,
) -> MelodyBaseline:
    times = np.asarray(times_s, dtype=float)
    hz = np.asarray(frequencies_hz, dtype=float)
    if times.size == 0:
        raise ValueError("Cannot build a baseline from an empty pitch contour")

    voiced = np.isfinite(hz) & (hz > 0)
    notes: list[MelodyNote] = []
    start = float(np.nanmin(times))
    stop = float(np.nanmax(times))
    cursor = start
    while cursor < stop:
        next_cursor = min(cursor + window_s, stop)
        mask = voiced & (times >= cursor) & (times < next_cursor)
        if np.any(mask):
            midi = float(np.nanmedian(hz_to_midi(hz[mask])))
            notes.append(MelodyNote(cursor - start, next_cursor - start, midi))
        cursor = next_cursor

    if not notes:
        raise ValueError("Reference audio did not contain enough voiced pitch to form a baseline")
    return MelodyBaseline(tuple(notes), title=title)


def demo_baseline() -> MelodyBaseline:
    notes = [
        MelodyNote(0.00, 0.55, 60, "C4"),
        MelodyNote(0.60, 1.15, 62, "D4"),
        MelodyNote(1.20, 1.75, 64, "E4"),
        MelodyNote(1.80, 2.35, 65, "F4"),
        MelodyNote(2.40, 2.95, 67, "G4"),
        MelodyNote(3.00, 3.55, 69, "A4"),
        MelodyNote(3.60, 4.15, 67, "G4"),
        MelodyNote(4.20, 4.90, 65, "F4"),
    ]
    return MelodyBaseline(tuple(notes), title="Synthetic public-style demo melody")


def _row_to_note(row: dict[str, str]) -> MelodyNote:
    normalized = {key.strip().lower(): value for key, value in row.items()}
    start = _first_float(normalized, ["start_s", "start", "onset_s", "time_s"])
    end = _first_float(normalized, ["end_s", "end", "offset_s"])

    if "midi" in normalized and normalized["midi"]:
        midi = float(normalized["midi"])
    elif "midi_note" in normalized and normalized["midi_note"]:
        midi = float(normalized["midi_note"])
    elif "frequency_hz" in normalized and normalized["frequency_hz"]:
        midi = float(hz_to_midi(float(normalized["frequency_hz"])))
    elif "hz" in normalized and normalized["hz"]:
        midi = float(hz_to_midi(float(normalized["hz"])))
    else:
        raise ValueError("Baseline CSV must include midi, midi_note, frequency_hz, or hz")

    return MelodyNote(
        start_s=start,
        end_s=end,
        midi=midi,
        label=normalized.get("label", "") or normalized.get("note", ""),
    )


def _first_float(row: dict[str, str], keys: Iterable[str]) -> float:
    for key in keys:
        if key in row and row[key] != "":
            return float(row[key])
    raise ValueError(f"Missing one of required columns: {', '.join(keys)}")


def _empty_row(row: Mapping[str, object]) -> bool:
    return all(row.get(key) in (None, "") for key in ("start_s", "end_s", "midi"))
