from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

import frontends.gradio.app as gradio_app
from konopro_research.audio_io import write_wav
from konopro_research.pitch import PitchContour


def test_audible_alignment_check_loads_cached_bundle(tmp_path, monkeypatch) -> None:
    sample_rate = 22050
    duration_s = 3.0
    times = np.arange(0.25, 2.25, 0.02)
    frequencies = 220.0 + 20.0 * np.sin(2.0 * np.pi * times / 2.0)
    confidence = np.ones_like(times)

    reference_contour = PitchContour(times, frequencies, confidence, name="reference")
    current_contour = PitchContour(times + 0.12, frequencies, confidence, name="current")

    audio_times = np.arange(int(sample_rate * duration_s)) / sample_rate
    reference_audio = 0.25 * np.sin(2.0 * np.pi * 220.0 * audio_times).astype(np.float32)
    current_audio = np.zeros_like(reference_audio)
    offset_samples = int(round(0.12 * sample_rate))
    current_audio[offset_samples:] = reference_audio[:-offset_samples]

    reference_path = tmp_path / "reference.wav"
    current_path = tmp_path / "current.wav"
    write_wav(reference_path, reference_audio, sample_rate)
    write_wav(current_path, current_audio, sample_rate)

    monkeypatch.setattr(gradio_app, "ALIGNMENT_CHECK_CACHE_DIR", tmp_path / "alignment_checks")

    contour_state = {
        "reference_mode": "Uploaded reference audio",
        "reference_path": str(reference_path),
        "current_path": str(current_path),
        "previous_path": None,
        "reference_contour": reference_contour,
        "current_contour": current_contour,
        "previous_contour": None,
        "raw_plots": {},
        "profile_rows": [],
    }

    first = gradio_app.run_alignment_check_only(
        "Uploaded reference audio",
        str(reference_path),
        str(current_path),
        None,
        None,
        contour_state,
    )
    first_debug = first[9]
    assert first_debug["alignment_check_cache"]["status"] == "stored"

    second = gradio_app.run_alignment_check_only(
        "Uploaded reference audio",
        str(reference_path),
        str(current_path),
        None,
        None,
        None,
    )
    profile = second[2]
    second_debug = second[9]

    assert "loaded from cache" in second[1]
    assert profile.loc[0, "stage"] == "Alignment check cache"
    assert second_debug["alignment_check_cache"]["status"] == "hit"
    assert (
        second_debug["alignment_check_cache"]["cache_key"]
        == first_debug["alignment_check_cache"]["cache_key"]
    )
    for artifact_path in second[3:9]:
        assert artifact_path is not None
        assert Path(artifact_path).exists()
