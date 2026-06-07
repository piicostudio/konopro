# Konopro Backend

This backend is the concierge MVP foundation. It accepts uploaded
karaoke/session audio, stores metadata, creates durable processing jobs, runs
server-side fingerprinting/segmentation, and exposes experimental analysis
evidence for future app/admin workflows.

## Setup

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

## Run Locally

```bash
uv run uvicorn konopro_backend.app:create_app --factory --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Upload A Session

Phase 04 uses a beta header instead of full authentication:

```bash
curl -X POST http://127.0.0.1:8000/v1/sessions \
  -H "X-Konopro-Beta-User: peter-demo" \
  -F "file=@/path/to/session.wav;type=audio/wav" \
  -F "client_duration_s=120.0" \
  -F "source=karaoke_room"
```

List sessions:

```bash
curl http://127.0.0.1:8000/v1/sessions \
  -H "X-Konopro-Beta-User: peter-demo"
```

Poll the job returned by upload:

```bash
curl http://127.0.0.1:8000/v1/jobs/JOB_ID \
  -H "X-Konopro-Beta-User: peter-demo"
```

Fetch the experimental analysis after the worker processes the job:

```bash
curl http://127.0.0.1:8000/v1/sessions/SESSION_ID/analysis \
  -H "X-Konopro-Beta-User: peter-demo"
```

Request a verified report:

```bash
curl -X POST http://127.0.0.1:8000/v1/sessions/SESSION_ID/report-requests \
  -H "X-Konopro-Beta-User: peter-demo" \
  -H "Content-Type: application/json" \
  -d '{"request_type":"paid","user_notes":"Please check my chorus."}'
```

Track report request status:

```bash
curl http://127.0.0.1:8000/v1/report-requests \
  -H "X-Konopro-Beta-User: peter-demo"

curl http://127.0.0.1:8000/v1/report-requests/REPORT_REQUEST_ID \
  -H "X-Konopro-Beta-User: peter-demo"
```

Delete a session and its stored audio:

```bash
curl -X DELETE http://127.0.0.1:8000/v1/sessions/SESSION_ID \
  -H "X-Konopro-Beta-User: peter-demo"
```

## Worker

The worker processes queued `fingerprint_segmentation` jobs. It scans the stored
audio with the configured provider, persists raw fingerprint windows, accepted
song intervals, weak candidates, and diagnostic recommendations.

```bash
uv run python -m konopro_backend.worker --once
```

Local test flow:

```bash
uv run uvicorn konopro_backend.app:create_app --factory --reload
# upload a session in another terminal
uv run python -m konopro_backend.worker --once
# fetch /v1/sessions/SESSION_ID/analysis
```

## Environment

Settings use the `KONOPRO_` prefix:

```text
KONOPRO_DATABASE_URL=sqlite:///./.local/konopro_backend.db
KONOPRO_STORAGE_ROOT=./.local/storage
KONOPRO_PROCESSING_ROOT=./.local/processing
KONOPRO_MAX_UPLOAD_MB=500
KONOPRO_ENVIRONMENT=local
KONOPRO_ADMIN_API_KEY=change-me-locally
KONOPRO_FINGERPRINT_PROVIDER=acrcloud
KONOPRO_FINGERPRINT_WINDOW_S=10
KONOPRO_FINGERPRINT_HOP_S=5
KONOPRO_FINGERPRINT_MAX_WINDOWS=120
KONOPRO_FINGERPRINT_USE_WHOLE=false
KONOPRO_FINGERPRINT_TIMEOUT_S=30
```

Provider-specific settings:

```text
KONOPRO_ACRCLOUD_HOST=...
KONOPRO_ACRCLOUD_ACCESS_KEY=...
KONOPRO_ACRCLOUD_ACCESS_SECRET=...
KONOPRO_AUDD_API_TOKEN=...
KONOPRO_SHAZAMKIT_HELPER_PATH=/path/to/helper
```

Automated tests use fake recognizers and do not require provider credentials,
network calls, or copyrighted audio.

## Fingerprinting Semantics

Fingerprinting detects likely backing-track/song identity and approximate
intervals. It does not score singing quality. Weak provider matches are stored
as clues with recovery guidance, not as confirmed songs.

## Verified Report Queue

Verified reports are manually prepared in this concierge MVP. Users can request
a report and see status, but the report content is written/published by the
admin.

Admin endpoints use `X-Konopro-Admin-Key`, not the beta user header:

```bash
curl http://127.0.0.1:8000/v1/admin/report-requests \
  -H "X-Konopro-Admin-Key: change-me-locally"

curl http://127.0.0.1:8000/v1/admin/report-requests/REPORT_REQUEST_ID \
  -H "X-Konopro-Admin-Key: change-me-locally"
```

Update queue state:

```bash
curl -X PATCH http://127.0.0.1:8000/v1/admin/report-requests/REPORT_REQUEST_ID \
  -H "X-Konopro-Admin-Key: change-me-locally" \
  -H "Content-Type: application/json" \
  -d '{"status":"in_progress","admin_notes":"Preparing verified report."}'
```

Inspect evidence and playable audio:

```bash
curl http://127.0.0.1:8000/v1/admin/report-requests/REPORT_REQUEST_ID/evidence \
  -H "X-Konopro-Admin-Key: change-me-locally"

curl http://127.0.0.1:8000/v1/admin/sessions/SESSION_ID/audio \
  -H "X-Konopro-Admin-Key: change-me-locally" \
  -o session.wav
```

Publish a Markdown report:

```bash
curl -X POST http://127.0.0.1:8000/v1/admin/report-requests/REPORT_REQUEST_ID/artifacts \
  -H "X-Konopro-Admin-Key: change-me-locally" \
  -H "Content-Type: application/json" \
  -d '{"title":"Verified Report","body_text":"# Report\n\nManual findings...","visibility":"user_visible"}'

curl -X PATCH http://127.0.0.1:8000/v1/admin/report-requests/REPORT_REQUEST_ID \
  -H "X-Konopro-Admin-Key: change-me-locally" \
  -H "Content-Type: application/json" \
  -d '{"status":"delivered"}'
```

## Current Limitations

This backend still does not automate verified report scoring, perform
MIDI/melody scoring, enforce payments, or send notifications. Those are later
phases.
