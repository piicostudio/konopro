# Konopro iOS MVP

This folder contains the first native SwiftUI app for the concierge MVP. It is
built to work against the local backend from `../backend/`.

## Open And Build

```bash
xcodebuild -list -project ios/Konopro.xcodeproj
xcodebuild build \
  -project ios/Konopro.xcodeproj \
  -scheme Konopro \
  -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath .build/xcode \
  CODE_SIGNING_ALLOWED=NO
```

To run unit tests, use an installed simulator name:

```bash
xcrun simctl list devices available
xcodebuild test \
  -project ios/Konopro.xcodeproj \
  -scheme Konopro \
  -destination 'platform=iOS Simulator,name=iPhone 17' \
  -derivedDataPath .build/xcode \
  CODE_SIGNING_ALLOWED=NO
```

Open the project in Xcode for manual testing:

```bash
open ios/Konopro.xcodeproj
```

## Backend Setup

From the repo root:

```bash
cd backend
uv run uvicorn konopro_backend.app:create_app --factory --reload
```

In another terminal, queued fingerprinting jobs can be processed with:

```bash
cd backend
uv run python -m konopro_backend.worker --once
```

Use this backend URL in the iOS app:

- iOS Simulator: `http://127.0.0.1:8000`
- Physical device: `http://<your-mac-lan-ip>:8000`

For a device on the same Wi-Fi network, find the Mac IP with:

```bash
ipconfig getifaddr en0
```

The app asks for a beta identity. Any nonempty value is accepted by the local
MVP backend and is sent as `X-Konopro-Beta-User`.

## App Flow

1. Launch Konopro.
2. Enter a beta identity, for example `peter-demo`.
3. Enter the backend base URL.
4. Tap `Test Connection`, then `Save`.
5. On `Sessions`, tap `+`.
6. Record audio or import an audio file.
7. Upload the file.
8. Run the backend worker.
9. Refresh/open the session to see processing state and instant analysis.
10. Request a verified report from the session detail screen.
11. Use backend admin endpoints to publish a report artifact.
12. Open the `Reports` tab to track status and read delivered report text.

## Microphone Notes

The app includes a microphone permission description and records `.m4a` audio
with AVFoundation. Simulator microphone behavior can differ from device
behavior, so real recording should be smoke-tested on an iPhone before beta use.

## What Exists

- SwiftUI app shell with setup, sessions, reports, and settings tabs.
- Persisted beta identity and backend URL.
- `URLSession` API client for health, upload, sessions, jobs, analysis, and
  user report endpoints.
- Audio recording and audio-file import.
- Upload state handling.
- Session history and bounded job polling.
- Experimental fingerprint analysis display.
- Verified report request, status, and delivered text-artifact views.
- Unit tests for API decoding, request construction, multipart bodies, settings,
  setup, upload, session polling, and report state transitions.

## Current Limits

- No payment, credit, or plan enforcement.
- No push notifications.
- No offline/resumable upload.
- No background upload handling yet.
- Instant analysis is song/segment fingerprinting, not singing-quality scoring.
- Verified scoring still depends on manual report work and future melody/MIDI
  pipeline support.
