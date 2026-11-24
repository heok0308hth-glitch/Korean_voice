# How this model pack and the v2.1.4 APK were rebuilt

The app's source repository was deleted, so the pack was regenerated from the released APK and the
original Hugging Face weights.

## Models (`tools/*.py`)
| Script | Output |
|---|---|
| `export_t3.py` | `t3_cond.onnx`, `t3_embed.bin/json`, `t3_lm_fp32.onnx` (custom Llama forward with KV cache, CFG batch of 2, aligned-head attentions) |
| `quant_lm.py` | `t3_lm.onnx`: 4-bit MatMulNBits, block 32, speech head fp32 |
| `export_enc.py` | `s3_tokenizer`, `s3_mel`, `campplus`, `ve_mel`, `ve_lstm` (STFT as framing conv + DFT matmul) |
| `export_ecapa.py` | `ecapa.onnx` (SpeechBrain ECAPA-TDNN with its Fbank) |
| `fp16_weights.py` | weight-only fp16 (compute stays fp32) for the tokenizer, ECAPA, CAM++, T3 cond |
| `patch_noise.py` | fixed noise in the S3Gen decoders (same tokens give the same audio) |
| `quant_dec.py` | 4-bit MatMuls in the HQ decoder |
| `build_pack.py` | 40 MiB parts and `model_manifest.json` |

Each export script checks its graph against the PyTorch reference before writing it.

## APK (`tools/apk/`)
- `patch_apk.py`: swaps `classes.dex` (the model base URLs were patched in smali), the manifest asset and the version, then writes an unsigned APK.
- `Sign.java`: v2+v3 signing and verification with apksig.
- `fix_inlined_ctors.py`: test-only fix for dex2jar, which translates R8's inlined constructors (`new-instance C` + `invoke-direct Super.<init>`) as `new Super`.

## End-to-end checks (`tools/eval/`)
`Synth.java` and `Flow.java` run the APK's own engine classes (dex2jar) on a desktop JVM: voice learning,
auto-learning, a marked-up script and MP3 output. `score.py` and `asr.py` measure Whisper CER and ECAPA similarity.
