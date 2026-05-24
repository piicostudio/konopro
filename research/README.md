# Konopro Research Prototype

This Python prototype compares a previous and current vocal take against a
reference baseline. It is optimized for a TA-runnable demo first, with private
real-song experiments kept out of Git.

## Setup

Use Python 3.11 or 3.12 through `uv`.

```bash
uv sync --extra dev
uv run python scripts/generate_demo_data.py
uv run python scripts/smoke_test.py
uv run pytest
uv run streamlit run app/streamlit_app.py
```

## Demo Flow

1. Generate the synthetic demo data.
2. Open the Streamlit app.
3. Use the demo baseline, previous take, and current take.
4. Confirm that the current take improves against the reference.
5. Try the "stable but wrong" take to see that stability alone is not treated as
   song improvement.

You can also run the scorer from the command line:

```bash
uv run python scripts/run_demo_scoring.py
```

For private local files:

```bash
uv run python scripts/run_demo_scoring.py \
  --reference-audio data/private/reference.wav \
  --previous data/private/previous.wav \
  --current data/private/current.wav \
  --baseline-out data/private/extracted_baseline.csv \
  --plot-out data/private/comparison.png \
  --quality
```

Prefer `--baseline-csv` when you have a symbolic melody baseline. Use
`--reference-audio` for experimental real-song tests.

The Streamlit app also lets you inspect/edit the active baseline and download it
as CSV. This is especially useful after uploading reference audio, because the
extracted baseline should be treated as a hypothesis until the graph looks
reasonable.

## Private Real-Song Tests

Put private BYO reference audio and vocal takes in `data/private/`. The app can
experimentally extract a dense baseline from uploaded reference audio, but the
reliable TA path is the symbolic baseline in `data/demo/`.
