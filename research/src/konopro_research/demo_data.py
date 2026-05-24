from __future__ import annotations

from pathlib import Path

import numpy as np

from konopro_research.audio_io import write_wav
from konopro_research.baseline import MelodyBaseline, demo_baseline, midi_to_hz, write_baseline_csv


SAMPLE_RATE = 22050


def ensure_demo_data(root: str | Path | None = None) -> dict[str, Path]:
    root_path = Path(root) if root else Path(__file__).resolve().parents[2] / "data" / "demo"
    root_path.mkdir(parents=True, exist_ok=True)

    baseline = demo_baseline()
    baseline_path = root_path / "demo_song_baseline.csv"
    write_baseline_csv(baseline, baseline_path)

    outputs = {
        "baseline": baseline_path,
        "reference": root_path / "reference_melody.wav",
        "previous": root_path / "previous_take.wav",
        "current": root_path / "current_take.wav",
        "stable_wrong": root_path / "stable_but_wrong_take.wav",
        "missing_notes": root_path / "missing_notes_take.wav",
        "noisy_room": root_path / "noisy_room_take.wav",
    }

    write_wav(outputs["reference"], synthesize_take(baseline, cents_error=0, vibrato_cents=0, offset_s=0), SAMPLE_RATE)
    write_wav(outputs["previous"], synthesize_take(baseline, cents_error=80, vibrato_cents=32, offset_s=0.08), SAMPLE_RATE)
    write_wav(outputs["current"], synthesize_take(baseline, cents_error=18, vibrato_cents=10, offset_s=0.03), SAMPLE_RATE)
    write_wav(outputs["stable_wrong"], synthesize_take(baseline, cents_error=200, vibrato_cents=3, offset_s=0.02), SAMPLE_RATE)
    write_wav(
        outputs["missing_notes"],
        synthesize_take(baseline, cents_error=18, vibrato_cents=10, offset_s=0.03, mute_note_indices={2, 5}),
        SAMPLE_RATE,
    )
    noisy = synthesize_take(baseline, cents_error=22, vibrato_cents=15, offset_s=0.05)
    noisy = add_noise_and_echo(noisy, noise_level=0.018, echo_delay_s=0.16, echo_gain=0.28)
    write_wav(outputs["noisy_room"], noisy, SAMPLE_RATE)

    return outputs


def synthesize_take(
    baseline: MelodyBaseline,
    *,
    cents_error: float = 0.0,
    vibrato_cents: float = 0.0,
    offset_s: float = 0.0,
    mute_note_indices: set[int] | None = None,
) -> np.ndarray:
    mute_note_indices = mute_note_indices or set()
    duration_s = baseline.duration_s + abs(offset_s) + 0.50
    audio = np.zeros(int(duration_s * SAMPLE_RATE), dtype=np.float32)

    for index, note in enumerate(baseline.notes):
        if index in mute_note_indices:
            continue
        start = max(0, int((note.start_s + offset_s) * SAMPLE_RATE))
        end = min(len(audio), int((note.end_s + offset_s) * SAMPLE_RATE))
        if end <= start:
            continue
        t = np.arange(end - start) / SAMPLE_RATE
        vibrato = vibrato_cents * np.sin(2.0 * np.pi * 5.2 * t)
        hz = midi_to_hz(note.midi + (cents_error + vibrato) / 100.0)
        phase = 2.0 * np.pi * np.cumsum(hz) / SAMPLE_RATE
        tone = 0.34 * np.sin(phase)
        tone += 0.05 * np.sin(2.0 * phase)
        tone *= _envelope(len(tone))
        audio[start:end] += tone.astype(np.float32)

    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / max(1.0, peak / 0.85)
    return audio.astype(np.float32)


def add_noise_and_echo(
    audio: np.ndarray,
    *,
    noise_level: float,
    echo_delay_s: float,
    echo_gain: float,
) -> np.ndarray:
    rng = np.random.default_rng(7)
    out = np.array(audio, dtype=np.float32)
    delay = int(echo_delay_s * SAMPLE_RATE)
    if delay > 0:
        out[delay:] += echo_gain * out[:-delay]
    out += rng.normal(0.0, noise_level, size=out.shape).astype(np.float32)
    peak = float(np.max(np.abs(out)))
    if peak > 0.95:
        out = out / peak * 0.95
    return out


def _envelope(length: int) -> np.ndarray:
    envelope = np.ones(length, dtype=np.float32)
    attack = min(length // 4, int(0.035 * SAMPLE_RATE))
    release = min(length // 4, int(0.050 * SAMPLE_RATE))
    if attack > 0:
        envelope[:attack] = np.linspace(0.0, 1.0, attack)
    if release > 0:
        envelope[-release:] = np.linspace(1.0, 0.0, release)
    return envelope
