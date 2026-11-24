# KoreanVoice model pack

This branch holds the on-device model files that the KoreanVoice app downloads on first launch.
Each model is split into parts of at most 40 MiB. `model_manifest.json` lists every model with its
size, SHA-256 and parts. The app joins the parts and checks the SHA-256 before it loads a model.

| File | Source | Licence |
|---|---|---|
| `t3_lm`, `t3_cond`, `t3_embed`, `tokenizer.json` | Chatterbox Multilingual T3 (`t3_mtl23ls_v2`) by Resemble AI, exported to ONNX; LM 4-bit (MatMulNBits) | MIT |
| `s3_tokenizer`, `s3_mel`, `campplus` | S3 speech tokenizer, mel front end and CAM++ speaker encoder from Chatterbox S3Gen (CosyVoice lineage) | MIT / Apache-2.0 |
| `ve_mel`, `ve_lstm` | Chatterbox VoiceEncoder | MIT |
| `ecapa` | SpeechBrain ECAPA-TDNN (`spkrec-ecapa-voxceleb`) | Apache-2.0 |
| `s3gen_turbo` | Chatterbox-Turbo conditional decoder (`ResembleAI/chatterbox-turbo-ONNX`, q4), fixed noise | MIT |
| `s3gen_hq` | Chatterbox Multilingual conditional decoder (`onnx-community/chatterbox-multilingual-ONNX`), 4-bit, fixed noise | MIT |

These files were regenerated in November 2026 from the original Hugging Face weights.
