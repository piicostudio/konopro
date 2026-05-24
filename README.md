# Konopro

Konopro is a singing-progress tracker for iOS users. 

## Timeline (4 weeks)
```text
Week 0: Brainstorm required features, research how they can be implemented
Week 1: Validate feasibility of prototype 
Week 2: Algorithm-side: Improve scoring, alignment, and feedback, App-side: Create design for app
Week 3: Build the MVP
Week 4: Make the prototype easy to run, easy to inspect, and safe to demo. Prepare for final presentations.
```

## Repository Layout

```text
konopro/
  docs/       Product, copyright, and data notes
  research/   Prototype and feature demos
  ios/        Native iOS app
```

## Running the Research Prototype

```bash
cd research
uv sync --extra dev
uv run python scripts/generate_demo_data.py
uv run python scripts/smoke_test.py
uv run pytest
uv run streamlit run app/streamlit_app.py
```
