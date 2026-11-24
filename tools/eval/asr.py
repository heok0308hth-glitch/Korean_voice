"""Whisper CER for generated Korean audio. usage: asr.py ref_text|... -- wav ..."""
import sys, json, re, warnings
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, librosa, torch
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device="cpu", torch_dtype=torch.float32)

def norm(s): return re.sub(r"[^가-힣0-9a-zA-Z]", "", s)
def cer(r, h):
    r, h = norm(r), norm(h)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))

pairs = json.load(open(sys.argv[1]))      # [[wav, text], ...]
res = []
for wav, text in pairs:
    w, sr = sf.read(wav, dtype="float32")
    w = librosa.resample(w, orig_sr=sr, target_sr=16000)
    hyp = asr({"raw": w, "sampling_rate": 16000}, generate_kwargs={"language": "korean", "task": "transcribe"})["text"]
    c = cer(text, hyp); res.append(c)
    print(f"CER {c:.3f} | {wav} | {hyp}")
print("MEAN_CER", float(np.mean(res)))
