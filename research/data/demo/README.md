# Demo Data

`demo_song_baseline.csv` is a synthetic melody generated for this prototype. The
WAV files used by the app are generated with:

```bash
uv run python scripts/generate_demo_data.py
```

The generated takes include:

- `previous_take.wav`
- `reference_melody.wav`
- `current_take.wav`
- `stable_but_wrong_take.wav`
- `missing_notes_take.wav`
- `noisy_room_take.wav`
