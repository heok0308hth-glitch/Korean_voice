"""SpeechBrain ECAPA-TDNN (spkrec-ecapa-voxceleb) -> ecapa.onnx: wav[1,N] @16 kHz -> [1,192].
Fbank (STFT as DFT matmul) + sentence mean norm + embedding model, matching EncoderClassifier.encode_batch."""
import math, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf, torchaudio
from speechbrain.lobes.features import Fbank
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from speechbrain.processing.features import InputNormalization

D = Path(sys.argv[1]); OUT = Path(sys.argv[2]); wav = sys.argv[3]
torch.set_grad_enabled(False)
fbank = Fbank(n_mels=80).eval()
norm = InputNormalization(norm_type="sentence", std_norm=False).eval()
emb = ECAPA_TDNN(80, channels=[1024, 1024, 1024, 1024, 3072], kernel_sizes=[5, 3, 3, 3, 1],
                 dilations=[1, 2, 3, 4, 1], attention_channels=128, lin_neurons=192)
emb.load_state_dict(torch.load(D / "embedding_model.ckpt", map_location="cpu"))
emb.eval()


class Ecapa(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.arange(400, dtype=torch.float64)[:, None]; f = torch.arange(201, dtype=torch.float64)[None]
        ang = 2 * math.pi * k * f / 400
        win = torch.hamming_window(400).double()[:, None]
        self.register_buffer("c", (torch.cos(ang) * win).float()); self.register_buffer("s", (-torch.sin(ang) * win).float())
        self.fb = fbank.compute_fbanks
        self.emb = emb

    def forward(self, wav):
        x = F.pad(wav[:, None, :], (200, 200))[:, 0]
        fr = F.conv1d(x[:, None, :], torch.eye(400)[:, None, :], stride=160)[0].transpose(0, 1)
        p = ((fr @ self.c) ** 2 + (fr @ self.s) ** 2)[None]                 # [1,T,201]
        feats = self.fb(p)
        feats = feats - feats.mean(dim=1, keepdim=True)
        return self.emb(feats)[:, 0]                                          # [1,192]


w, sr = sf.read(wav, dtype="float32")
if w.ndim > 1: w = w.mean(1)
x = torchaudio.functional.resample(torch.from_numpy(w)[None], sr, 16000)[:, : 16000 * 8]
ref_feats = norm(fbank(x), torch.ones(1))
ref = emb(ref_feats, torch.ones(1))[:, 0]
m = Ecapa().eval()
print("ecapa cos", F.cosine_similarity(m(x), ref).item(), "feat diff", (m.fb(((F.conv1d(F.pad(x[:, None], (200, 200)), torch.eye(400)[:, None], stride=160)[0].T @ m.c) ** 2 + (F.conv1d(F.pad(x[:, None], (200, 200)), torch.eye(400)[:, None], stride=160)[0].T @ m.s) ** 2)[None]) - fbank(x)).abs().max().item())
torch.onnx.export(m, (x,), str(OUT / "ecapa_fp32.onnx"), dynamo=False, opset_version=17, input_names=["wav"], output_names=["embedding"],
                  dynamic_axes={"wav": {1: "N"}})
print("exported")
