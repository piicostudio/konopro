from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = RESEARCH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from konopro_research.baseline import load_baseline_csv  # noqa: E402
from konopro_research.baseline import write_baseline_csv  # noqa: E402
from konopro_research.audio_io import load_audio  # noqa: E402
from konopro_research.demo_data import ensure_demo_data  # noqa: E402
from konopro_research.pitch import extract_pitch  # noqa: E402
from konopro_research.plots import plot_take_comparison  # noqa: E402
from konopro_research.quality import analyze_baseline_quality  # noqa: E402
from konopro_research.reference_audio import extract_reference_audio  # noqa: E402
from konopro_research.scoring import compare_takes  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score previous/current takes against a baseline.")
    parser.add_argument("--baseline-csv", type=Path, help="Symbolic baseline CSV.")
    parser.add_argument("--reference-audio", type=Path, help="Experimental audio-derived baseline.")
    parser.add_argument("--previous", type=Path, help="Previous vocal take audio file.")
    parser.add_argument("--current", type=Path, help="Current vocal take audio file.")
    parser.add_argument("--plot-out", type=Path, help="Write a pitch comparison plot to this path.")
    parser.add_argument("--baseline-out", type=Path, help="Write the active baseline CSV to this path.")
    parser.add_argument("--quality", action="store_true", help="Include baseline quality metadata.")
    args = parser.parse_args()

    if not any([args.baseline_csv, args.reference_audio, args.previous, args.current]):
        paths = ensure_demo_data(RESEARCH_ROOT / "data" / "demo")
        baseline = load_baseline_csv(paths["baseline"])
        previous = paths["previous"]
        current = paths["current"]
    else:
        if bool(args.baseline_csv) == bool(args.reference_audio):
            raise SystemExit("Choose exactly one of --baseline-csv or --reference-audio.")
        if not args.previous or not args.current:
            raise SystemExit("Custom scoring requires --previous and --current.")
        if args.baseline_csv:
            baseline = load_baseline_csv(args.baseline_csv)
            reference_quality = analyze_baseline_quality(baseline)
        else:
            reference = extract_reference_audio(args.reference_audio)
            baseline = reference.baseline
            reference_quality = reference.quality
        previous = args.previous
        current = args.current

    if args.baseline_out:
        args.baseline_out.parent.mkdir(parents=True, exist_ok=True)
        write_baseline_csv(baseline, args.baseline_out)

    if args.plot_out:
        args.plot_out.parent.mkdir(parents=True, exist_ok=True)
        previous_audio, sample_rate = load_audio(previous)
        current_audio, _ = load_audio(current, target_sr=sample_rate)
        previous_contour = extract_pitch(previous_audio, sample_rate, name="previous")
        current_contour = extract_pitch(current_audio, sample_rate, name="current")
        fig = plot_take_comparison(baseline, previous_contour, current_contour)
        fig.savefig(args.plot_out, dpi=160)

    comparison = compare_takes(previous, current, baseline)
    output = comparison.to_dict()
    if args.quality:
        quality = reference_quality if "reference_quality" in locals() else analyze_baseline_quality(baseline)
        output["baseline_quality"] = quality.to_dict()
    print(json.dumps(output, indent=2))
