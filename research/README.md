# Konopro Research Prototype

This Python prototype now contains two linked research demos:

- **Prototype A: Singing Evaluation** compares a current vocal take, and
  optionally a previous take, against a known reference section.
- **Prototype B: Song & Section Matching** ranks likely song/section candidates
  before handing the best matched section into the scorer.

It is optimized for a TA-runnable demo first, with private real-song experiments
kept out of Git.

## Setup

Use Python 3.11 or 3.12 through `uv`.

```bash
uv sync --extra dev
uv run python scripts/generate_demo_data.py
uv run python scripts/smoke_test.py
uv run pytest
uv run streamlit run frontends/streamlit/app.py
uv run python frontends/gradio/app.py
```

Optional real-song vocal isolation uses Demucs:

```bash
uv sync --extra dev --extra stems
```

## Prototype A: Singing Evaluation

1. Generate the synthetic demo data.
2. Open the Streamlit app with `uv run streamlit run frontends/streamlit/app.py`.
3. Use **A1. Import & Preview Files** to select a demo scenario or upload private
   reference/current/previous audio.
4. Use **A2. Prepare Audio With Optional Demucs** when mixed songs need vocal
   isolation before pitch tracking.
5. Use **A3. Tune Analysis Hyperparameters** to adjust pitch extraction, contour
   cleaning, matching, and scoring.
6. Use **A4. Verify Results** to check the score summary, plots, detailed metrics,
   and trust checklist.

## Prototype B: Song & Section Matching

Use **B1. Match Song & Section** after selecting or uploading a current take. The
TA-safe path compares the current take against a small synthetic song-section
catalog. For private experiments, upload reference audio in Prototype A, then
search that reference split into overlapping phrase windows.

The matching prototype uses pitch-contour shape, not exact audio fingerprinting:

1. Extract and clean the query pitch contour.
2. Build a section catalog from synthetic demo sections, an active baseline, or
   uploaded reference-audio windows.
3. Normalize each phrase by time, optionally subtract median pitch to ignore
   key/transposition, and compare shape with DTW.
4. Rank candidate song sections and run a handoff score against the top match.

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

For uploaded mixed songs, open **Live analysis hyperparameters > Vocal
Isolation** and choose **Demucs vocal stem**. The first run may download model
weights and can be slow on CPU, but separated stems are cached under
`research/.cache/stems/`. Keep vocal isolation off for the TA-safe synthetic demo
unless you specifically want to test real-song uploads.

## Gradio Alternative

Run the smaller audio-lab frontend with:

```bash
uv run python frontends/gradio/app.py
```

The Gradio UI uses the same backend package as Streamlit, but focuses on manual
stage execution and visible processing time for audio preparation, evaluation,
and matching. Use it when Streamlit's rerun model feels too opaque for real-song
experiments.

## Private Real-Song Tests

Put private BYO reference audio and vocal takes in `data/private/`. The app can
experimentally extract a dense baseline from uploaded reference audio, but the
reliable TA path is the symbolic baseline in `data/demo/`.
