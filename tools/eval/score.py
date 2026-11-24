"""CER (Whisper) + ECAPA speaker similarity for eval dirs. usage: score.py evaldir refs..."""
import sys, json, glob, os, re, warnings
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, librosa, onnxruntime as ort, torch
from transformers import pipeline
d = sys.argv[1]; refs = sys.argv[2:]
S = [l.strip() for l in open("eval/sentences.txt", encoding="utf-8") if l.strip()]
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device="cpu", torch_dtype=torch.float32)
ec = ort.InferenceSession("pack/ecapa.onnx")
def load16(p):
    w, sr = sf.read(p, dtype="float32")
    if w.ndim > 1: w = w.mean(1)
    return librosa.resample(w, orig_sr=sr, target_sr=16000).astype(np.float32)
def emb(w):
    e = ec.run(None, {"wav": w[None]})[0][0]; return e / np.linalg.norm(e)
def norm(s): return re.sub(r"[^가-힣0-9]", "", s)
def cer(r, h):
    r, h = norm(r), norm(h); dd = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        p, dd[0] = dd[0], i
        for j in range(1, len(h) + 1): p, dd[j] = dd[j], min(dd[j] + 1, dd[j - 1] + 1, p + (r[i - 1] != h[j - 1]))
    return dd[len(h)] / max(1, len(r))
R = {r: emb(load16(f"ref/{r}.wav")) for r in refs}
cers, same, cross = [], [], []
for r in refs:
    for k, s in enumerate(S):
        p = f"{d}/{r}/s{k:02d}.wav"
        w = load16(p); e = emb(w)
        hyp = asr({"raw": w, "sampling_rate": 16000}, generate_kwargs={"language": "korean"})["text"]
        c = cer(s, hyp); cers.append(c); same.append(float(e @ R[r]))
        cross += [float(e @ R[o]) for o in refs if o != r]
        print(f"{r} s{k} CER={c:.3f} sim={same[-1]:.3f} | {hyp}")
print(f"SUMMARY {d}: CER mean={np.mean(cers):.3f} median={np.median(cers):.3f} | ECAPA same-speaker={np.mean(same):.3f} vs other refs={np.mean(cross):.3f}")
