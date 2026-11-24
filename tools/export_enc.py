"""Export the voice-registration models used by the KoreanVoice app (all single utterance, batch 1).

  s3_tokenizer.onnx  audio_values[1,N] @24 kHz -> speech tokens [1,T] int64 (25 tokens/s)
  s3_mel.onnx        wav[1,N] @24 kHz -> log-mel [1,T,80] (S3Gen prompt features, 50 frames/s)
  campplus.onnx      wav[1,N] @16 kHz -> x-vector [1,192] (Kaldi fbank + mean norm inside)
  ve_mel.onnx        wav[1,N] @16 kHz -> mel [T,40] (VoiceEncoder input, power mel)
  ve_lstm.onnx       partials[n,160,40] -> [n,256] (L2-normalised partial embeddings)

STFTs are written as framing convolutions + DFT matmuls so the graphs need no STFT/complex ops.
"""
import math, sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from safetensors.torch import load_file

from chatterbox.models.s3gen import S3Gen
from chatterbox.models.s3gen.utils.mel import mel_spectrogram
from chatterbox.models.s3gen.xvector import extract_feature
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.models.voice_encoder.melspec import melspectrogram
from chatterbox.models.voice_encoder.config import VoiceEncConfig

CKPT = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)

s3gen = S3Gen()
s3gen.load_state_dict(load_file(str(CKPT / "s3gen.safetensors")), strict=False)
s3gen.eval()
ve = VoiceEncoder(); ve.load_state_dict(load_file(str(CKPT / "ve.safetensors"))); ve.eval()


def frames(x, n, hop):
    """[1,N] -> [F, n] strided frames (no padding) via an identity convolution."""
    eye = torch.eye(n)[:, None, :]
    return F.conv1d(x[:, None, :], eye, stride=hop)[0].transpose(0, 1)


def dft(n_fft, n_bins):
    k = torch.arange(n_fft, dtype=torch.float64)[:, None]
    f = torch.arange(n_bins, dtype=torch.float64)[None, :]
    ang = 2 * math.pi * k * f / n_fft
    return torch.cos(ang).float(), (-torch.sin(ang)).float()


# ---------------------------------------------------------------- resampler 24k -> 16k (torchaudio default kernel)
class Resample24to16(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchaudio.transforms.Resample(24000, 16000)
        self.register_buffer("kernel", r.kernel.clone())   # [2,1,W]
        self.width = r.width

    def forward(self, x):                                     # [1,N]
        n = x.shape[1]
        y = F.pad(x, (self.width, self.width + 3))
        y = F.conv1d(y[:, None, :], self.kernel, stride=3)    # [1,2,T]
        y = y.transpose(1, 2).reshape(1, -1)
        tgt = torch.div(2 * n + 2, 3, rounding_mode="floor")  # ceil(2n/3)
        return y[:, :tgt]


# ---------------------------------------------------------------- S3 tokenizer
tok = s3gen.tokenizer


class S3Tok(nn.Module):
    def __init__(self):
        super().__init__()
        self.rs = Resample24to16()
        c, s = dft(400, 201)
        win = torch.hann_window(400)
        self.register_buffer("c", c * win[:, None]); self.register_buffer("s", s * win[:, None])
        self.register_buffer("melfb", tok._mel_filters.clone())               # [128,201]
        enc = tok.encoder
        self.enc = enc
        fc = enc.freqs_cis[:, :]                                               # complex [2048, 64]
        self.register_buffer("cos", torch.view_as_real(fc)[..., 0].float())
        self.register_buffer("sin", torch.view_as_real(fc)[..., 1].float())
        self.q = tok.quantizer._codebook

    def logmel(self, x16):
        x = F.pad(x16[:, None, :], (200, 200), mode="reflect")[:, 0]
        fr = frames(x, 400, 160)                                               # [F,400]
        p = (fr @ self.c) ** 2 + (fr @ self.s) ** 2                            # [F,201]
        p = p[:-1]                                                             # drop last frame (stft[..., :-1])
        mel = self.melfb @ p.transpose(0, 1)                                   # [128,F-1]
        lg = torch.clamp(mel, min=1e-10).log10()
        lg = torch.maximum(lg, lg.max() - 8.0)
        return ((lg + 4.0) / 4.0)[None]                                        # [1,128,T]

    def forward(self, audio_values):
        mel = self.logmel(self.rs(audio_values))
        e = self.enc
        x = F.gelu(e.conv1(mel))
        x = F.gelu(e.conv2(x))
        x = x.permute(0, 2, 1)                                                 # [1,T,D]
        T = x.shape[1]
        cos = self.cos[:T][None, :, None, :]; sin = self.sin[:T][None, :, None, :]
        for b in e.blocks:
            a = b.attn
            y = b.attn_ln(x)
            q, k, v = a.query(y), a.key(y), a.value(y)
            H = a.n_head; D = q.shape[-1]; hd = D // H
            q = q.view(1, T, H, hd); k = k.view(1, T, H, hd); v = v.view(1, T, H, hd)
            q = q * cos + torch.cat((-q[..., hd // 2:], q[..., : hd // 2]), -1) * sin
            k = k * cos + torch.cat((-k[..., hd // 2:], k[..., : hd // 2]), -1) * sin
            vv = v.reshape(1, T, D)
            mem = a.fsmn_block(F.pad(vv.transpose(1, 2), (a.left_padding, a.right_padding))).transpose(1, 2) + vv
            scale = hd ** -0.25
            qh = q.permute(0, 2, 1, 3) * scale
            kh = k.permute(0, 2, 3, 1) * scale
            w = torch.softmax(qh @ kh, dim=-1)
            o = (w @ v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3).reshape(1, T, D)
            x = x + a.out(o) + mem
            x = x + b.mlp(b.mlp_ln(x))
        h = self.q.project_down(x).tanh() * 0.9990000128746033
        h = h.round() + 1
        powers = torch.tensor([3.0 ** i for i in range(8)])
        return (h * powers).sum(-1).to(torch.int64)                            # [1,T]


# ---------------------------------------------------------------- S3Gen 24 kHz mel
class S3Mel(nn.Module):
    def __init__(self):
        super().__init__()
        c, s = dft(1920, 961)
        win = torch.hann_window(1920)
        self.register_buffer("c", c * win[:, None]); self.register_buffer("s", s * win[:, None])
        self.register_buffer("fb", torch.from_numpy(librosa.filters.mel(sr=24000, n_fft=1920, n_mels=80, fmin=0, fmax=8000)).float())

    def forward(self, wav):
        x = F.pad(wav[:, None, :], (720, 720), mode="reflect")[:, 0]
        fr = frames(x, 1920, 480)
        mag = torch.sqrt((fr @ self.c) ** 2 + (fr @ self.s) ** 2 + 1e-9)      # [F,961]
        mel = mag @ self.fb.transpose(0, 1)                                    # [F,80]
        return torch.log(torch.clamp(mel, min=1e-5))[None]                     # [1,F,80]


# ---------------------------------------------------------------- CAM++ with Kaldi fbank
class Camp(nn.Module):
    def __init__(self):
        super().__init__()
        c, s = dft(512, 257)
        self.register_buffer("c", c); self.register_buffer("s", s)
        win = torch.hann_window(400, periodic=False).pow(0.85)
        self.register_buffer("win", win)
        banks, _ = torchaudio.compliance.kaldi.get_mel_banks(80, 512, 16000.0, 20.0, 0.0, 100.0, -500.0, 1.0)
        self.register_buffer("banks", F.pad(banks, (0, 1)).float())           # [80,257]
        self.model = s3gen.speaker_encoder

    def forward(self, wav):
        fr = frames(wav, 400, 160)                                             # [F,400] snip_edges
        fr = fr - fr.mean(dim=1, keepdim=True)
        fr = fr - 0.97 * torch.cat((fr[:, :1], fr[:, :-1]), dim=1)
        fr = F.pad(fr * self.win, (0, 112))                                    # -> 512
        p = (fr @ self.c) ** 2 + (fr @ self.s) ** 2
        fb = torch.log(torch.clamp(p @ self.banks.transpose(0, 1), min=float(torch.finfo(torch.float32).eps)))
        fb = fb - fb.mean(dim=0, keepdim=True)
        return self.model(fb[None])                                            # [1,192]


# ---------------------------------------------------------------- VoiceEncoder
class VeMel(nn.Module):
    def __init__(self):
        super().__init__()
        c, s = dft(400, 201)
        win = torch.hann_window(400)
        self.register_buffer("c", c * win[:, None]); self.register_buffer("s", s * win[:, None])
        hp = VoiceEncConfig()
        self.register_buffer("fb", torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=400, n_mels=40, fmin=0, fmax=8000)).float())

    def forward(self, wav):
        x = F.pad(wav[:, None, :], (200, 200), mode="reflect")[:, 0]
        fr = frames(x, 400, 160)
        p = (fr @ self.c) ** 2 + (fr @ self.s) ** 2                            # |X|^2
        return p @ self.fb.transpose(0, 1)                                     # [F,40]


class VeLstm(nn.Module):
    def __init__(self):
        super().__init__()
        self.ve = ve

    def forward(self, partials):
        return self.ve(partials)


# ---------------------------------------------------------------- checks against the PyTorch / librosa reference
rng = np.random.default_rng(0)
import soundfile as sf
ref_path = next((a for a in sys.argv[3:] if not a.startswith("--")), None)
if ref_path:
    w, sr = sf.read(ref_path, dtype="float32")
    if w.ndim > 1: w = w.mean(1)
    w24 = librosa.resample(w, orig_sr=sr, target_sr=24000)[: 10 * 24000]
else:
    t = np.arange(24000 * 6) / 24000
    w24 = (0.3 * np.sin(2 * np.pi * 180 * t * (1 + 0.1 * np.sin(2 * np.pi * 3 * t))) + 0.02 * rng.standard_normal(t.size)).astype(np.float32)
x24 = torch.from_numpy(w24)[None]
x16 = torchaudio.transforms.Resample(24000, 16000)(x24)

m_tok, m_mel, m_camp, m_vem, m_vel = S3Tok().eval(), S3Mel().eval(), Camp().eval(), VeMel().eval(), VeLstm().eval()

print("resample max|diff|", (m_tok.rs(x24) - x16).abs().max().item(), m_tok.rs(x24).shape, x16.shape)
ref_tok, _ = tok(x16)
mine = m_tok(x24)
n = min(ref_tok.shape[1], mine.shape[1])
print("tokens", ref_tok.shape, mine.shape, "agree", (ref_tok[0, :n] == mine[0, :n]).float().mean().item())
ref_mel = mel_spectrogram(x24).transpose(1, 2)
print("s3mel max|diff|", (m_mel(x24) - ref_mel).abs().max().item(), ref_mel.shape)
feat, _, _ = extract_feature([x16[0]])
ref_xv = s3gen.speaker_encoder(feat)
print("campplus cos", F.cosine_similarity(m_camp(x16), ref_xv).item(), "fbank max|diff|")
ref_vm = torch.from_numpy(melspectrogram(x16[0].numpy(), VoiceEncConfig()).T.astype(np.float32))
my_vm = m_vem(x16)
print("ve_mel rel diff", ((my_vm - ref_vm).abs().max() / ref_vm.abs().max()).item(), my_vm.shape, ref_vm.shape)
parts = torch.randn(3, 160, 40).abs()
print("ve_lstm max|diff|", (m_vel(parts) - ve(parts)).abs().max().item())

if "--check" in sys.argv: sys.exit()

kw = dict(dynamo=False, opset_version=17)
torch.onnx.export(m_tok, (x24,), str(OUT / "s3_tokenizer_fp32.onnx"), input_names=["audio_values"], output_names=["speech_tokens"],
                  dynamic_axes={"audio_values": {1: "N"}, "speech_tokens": {1: "T"}}, **kw)
torch.onnx.export(m_mel, (x24,), str(OUT / "s3_mel.onnx"), input_names=["wav"], output_names=["mel"],
                  dynamic_axes={"wav": {1: "N"}, "mel": {1: "T"}}, **kw)
torch.onnx.export(m_camp, (x16,), str(OUT / "campplus.onnx"), input_names=["wav"], output_names=["xvector"],
                  dynamic_axes={"wav": {1: "N"}}, **kw)
torch.onnx.export(m_vem, (x16,), str(OUT / "ve_mel.onnx"), input_names=["wav"], output_names=["mel"],
                  dynamic_axes={"wav": {1: "N"}, "mel": {0: "T"}}, **kw)
torch.onnx.export(m_vel, (parts,), str(OUT / "ve_lstm.onnx"), input_names=["partials"], output_names=["embeds"],
                  dynamic_axes={"partials": {0: "n"}, "embeds": {0: "n"}}, **kw)
print("exported")
