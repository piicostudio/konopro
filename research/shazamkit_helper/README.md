# Konopro ShazamKit Helper

This is a minimal signed macOS app bundle used by the Python research dashboard.
It behaves like a command-line tool:

```sh
KonoproShazamHelper.app/Contents/MacOS/KonoproShazamHelper path/to/audio.wav
```

The helper exists because `swift research/scripts/shazamkit_recognize.swift` runs
as Apple's `com.apple.swift-frontend` process. Shazam catalog matching needs a
real signed app identifier whose App ID has the ShazamKit App Service enabled.

## Setup

1. In Apple Developer, register or edit this App ID:

   ```text
   com.konopro.research.shazamhelper
   ```

2. Open the App Services tab and enable ShazamKit.

3. Build with your Apple Developer Team ID:

   ```sh
   KONOPRO_DEVELOPMENT_TEAM=TEAMID ./research/shazamkit_helper/build.sh
   ```

4. Run the Gradio research dashboard normally. The Python fingerprinting backend
   auto-detects the built helper at:

   ```text
   research/shazamkit_helper/build/KonoproShazamHelper.app/Contents/MacOS/KonoproShazamHelper
   ```

You can override the helper path with `KONOPRO_SHAZAMKIT_HELPER`. It may point
to the `.app` bundle or directly to the executable.

## Notes

- Do not add a `com.apple.developer.shazamkit` entitlement file. ShazamKit is an
  App Service on the App ID.
- If the runtime still returns a ShazamKit 401/App Service error, confirm that
  the bundle ID above is the one enabled for ShazamKit and rebuild the helper.
