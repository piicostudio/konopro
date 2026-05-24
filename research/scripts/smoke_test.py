from __future__ import annotations

import json
import sys
from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = RESEARCH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from konopro_research.audio_io import load_audio  # noqa: E402
from konopro_research.baseline import load_baseline_csv  # noqa: E402
from konopro_research.demo_data import ensure_demo_data  # noqa: E402
from konopro_research.pitch import extract_pitch  # noqa: E402
from konopro_research.plots import plot_take_comparison  # noqa: E402
from konopro_research.quality import analyze_baseline_quality  # noqa: E402
from konopro_research.scoring import compare_takes  # noqa: E402


def main() -> int:
    paths = ensure_demo_data(RESEARCH_ROOT / "data" / "demo")
    baseline = load_baseline_csv(paths["baseline"])

    comparison = compare_takes(paths["previous"], paths["current"], baseline)
    stable_wrong = compare_takes(paths["previous"], paths["stable_wrong"], baseline)

    if comparison.verdict != "improved":
        raise RuntimeError("Expected demo current take to improve over previous take")
    if stable_wrong.overall_delta >= 0:
        raise RuntimeError("Expected stable-but-wrong take to avoid being scored as improvement")

    previous_audio, sample_rate = load_audio(paths["previous"])
    current_audio, _ = load_audio(paths["current"], target_sr=sample_rate)
    previous_contour = extract_pitch(previous_audio, sample_rate, name="previous")
    current_contour = extract_pitch(current_audio, sample_rate, name="current")

    out_dir = RESEARCH_ROOT / "reports" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_take_comparison(baseline, previous_contour, current_contour)
    fig.savefig(out_dir / "demo_pitch_comparison.png", dpi=160)

    summary = {
        "status": "ok",
        "demo_verdict": comparison.verdict,
        "demo_overall_delta": comparison.overall_delta,
        "stable_wrong_delta": stable_wrong.overall_delta,
        "baseline_quality": analyze_baseline_quality(baseline).to_dict(),
        "plot": str(out_dir / "demo_pitch_comparison.png"),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
