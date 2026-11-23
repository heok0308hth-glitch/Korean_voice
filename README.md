# KoreanVoice

**On-device Korean neural TTS and voice cloning for Android.** After the model
download, everything — text normalization, the language model, the vocoder, the
speaker encoder — runs on the phone. No account, no cloud inference, no audio
leaving the device.

| | |
|---|---|
| Release | [`release/KoreanVoice-lite.apk`](release/KoreanVoice-lite.apk) |
| Version | 2.1.3-lite (`io.github.koreanvoice.app.lite`) |
| ABI | `arm64-v8a` |
| Runtime | ONNX Runtime, bundled (no Play Services, no ML Kit) |

## What it does

- **Korean TTS.** Text → speech with the Chatterbox-multilingual architecture
  exported to ONNX (`t3_lm` + `t3_cond` for text-to-token, `s3gen` for
  token-to-waveform), with a Korean G2P front end (CMU-style dictionary, hanja
  table, particle table).
- **Voice cloning.** Record or import a short reference clip; the speaker encoder
  (`campplus`, `ecapa`) and the voice-extraction pair (`ve_mel`, `ve_lstm`)
  condition the generator on that voice.
- **Style control and post-processing.** Emotion/style conditioning plus audio
  post-processing (gain, trimming, format export).
- **Two generation tiers.** `s3gen_turbo` for fast drafts and `s3gen_hq` for the
  final render, so a phone can trade latency for quality.

## Install and first run

1. Download the APK from [`release/`](release/) and allow installation from that
   source (the build is not Play-signed).
2. On first launch the app downloads the model pack (~1.2 GB across
   `t3_lm`, `s3gen_turbo`, `s3gen_hq`, `campplus`, `ecapa`, …) and verifies every
   file against the SHA-256 in `assets/model_manifest.json` before using it.
   Use Wi-Fi for this step.
3. After that the app works with the network off.

## Permissions and why

| Permission | Why |
|---|---|
| `INTERNET`, `ACCESS_NETWORK_STATE` | model download only |
| `RECORD_AUDIO` | reference clip for voice cloning |
| `FOREGROUND_SERVICE`, `FOREGROUND_SERVICE_MEDIA_PROCESSING`, `POST_NOTIFICATIONS`, `WAKE_LOCK` | keep a long synthesis alive while the screen is off |
| `FOREGROUND_SERVICE_DATA_SYNC` | resume an interrupted model download |

There is no analytics SDK, no ad SDK, and no third-party crash reporter in the
APK.

## How the models are shipped

The APK stays under 16 MB: the ONNX Runtime shared object is bundled and the
weights are not. Each model is split into `<name>.partN` segments listed in
`assets/model_manifest.json` with its size and SHA-256, so a download can be
resumed part by part and a corrupted part is rejected rather than loaded.

## Responsible use

Voice cloning is a dual-use capability. Use it with your own voice, or with the
explicit consent of the speaker, and do not use generated audio to impersonate
anyone. The app is provided for personal and research use.

## Credits and license

- Speech architecture: **Chatterbox** (Resemble AI, MIT) — the ONNX graph set is
  an export of that model family.
- Runtime: **ONNX Runtime** (MIT), AndroidX / Jetpack Compose (Apache-2.0).

This repository distributes the release APK and this note. It does not contain
the application source.
