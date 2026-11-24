"""Export Chatterbox Multilingual T3 for the KoreanVoice app.

Produces (in OUT):
  t3_cond.onnx   speaker_emb[1,256], prompt_tokens[1,N] int64, exaggeration[1] -> cond_emb[1,34,1024]
  t3_lm_fp32.onnx (+data)  inputs_embeds[B,T,1024], past_{k,v}_i[B,16,P,64]
                 -> logits[B,8194] (last position), attentions[3,T,P+T] (batch 0, aligned heads),
                    present_{k,v}_i[B,16,P+T,64]
  t3_embed.bin / t3_embed.json   fp16 tables text_emb, text_pos_emb, speech_emb, speech_pos_emb
"""
import json, math, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file

from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.t3.inference.alignment_stream_analyzer import LLAMA_ALIGNED_HEADS

CKPT = Path(sys.argv[1])
OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)

t3 = T3(T3Config.multilingual())
sd = load_file(str(CKPT / "t3_mtl23ls_v2.safetensors"))
if "model" in sd: sd = sd["model"][0]
t3.load_state_dict(sd)
t3.eval()
H, NH, HD, L = 1024, 16, 64, 30


def llama3_inv_freq():
    base, dim = 500000.0, HD
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    factor, low, high, old = 8.0, 1.0, 4.0, 8192.0
    low_wl, high_wl = old / low, old / high
    wl = 2 * math.pi / inv
    out = torch.where(wl > low_wl, inv / factor, inv)
    smooth = (old / wl - low) / (high - low)
    smoothed = (1 - smooth) * out / factor + smooth * out
    medium = ~(wl < high_wl) & ~(wl > low_wl)
    return torch.where(medium, smoothed, out).float()


INV_FREQ = llama3_inv_freq()
HEADS = {layer: head for layer, head in LLAMA_ALIGNED_HEADS}


def rot(x):
    a, b = x[..., : HD // 2], x[..., HD // 2:]
    return torch.cat((-b, a), dim=-1)


class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = t3.tfmr.layers
        self.norm = t3.tfmr.norm
        self.head = t3.speech_head
        self.register_buffer("inv_freq", INV_FREQ)

    def forward(self, x, *past):
        B, T = x.shape[0], x.shape[1]
        P = past[0].shape[2]
        pos = torch.arange(T, dtype=torch.float32) + P
        f = pos[:, None] * self.inv_freq[None, :]
        emb = torch.cat((f, f), dim=-1)
        cos, sin = emb.cos()[None, None], emb.sin()[None, None]
        kpos = torch.arange(T, dtype=torch.int64) + P
        allk = torch.arange(P + T, dtype=torch.int64)
        mask = (allk[None, :] > kpos[:, None]).float() * -1e9          # [T, P+T]
        h = x
        presents, atts = [], {}
        for i, ly in enumerate(self.layers):
            r = h
            y = ly.input_layernorm(h)
            a = ly.self_attn
            q = a.q_proj(y).view(B, T, NH, HD).transpose(1, 2)
            k = a.k_proj(y).view(B, T, NH, HD).transpose(1, 2)
            v = a.v_proj(y).view(B, T, NH, HD).transpose(1, 2)
            q = q * cos + rot(q) * sin
            k = k * cos + rot(k) * sin
            k = torch.cat((past[2 * i], k), dim=2)
            v = torch.cat((past[2 * i + 1], v), dim=2)
            presents += [k, v]
            s = torch.matmul(q, k.transpose(2, 3)) * (1.0 / math.sqrt(HD)) + mask
            p = torch.softmax(s, dim=-1)
            if i in HEADS: atts[i] = p[0, HEADS[i]]
            o = torch.matmul(p, v).transpose(1, 2).reshape(B, T, H)
            h = r + a.o_proj(o)
            r = h
            h = r + ly.mlp(ly.post_attention_layernorm(h))
        logits = self.head(self.norm(h[:, -1]))
        att = torch.stack([atts[l] for l, _ in LLAMA_ALIGNED_HEADS])
        return (logits, att, *presents)


class Cond(nn.Module):
    def __init__(self):
        super().__init__()
        self.speech_emb = t3.speech_emb
        self.speech_pos = t3.speech_pos_emb.emb
        self.enc = t3.cond_enc

    def forward(self, speaker_emb, prompt_tokens, exaggeration):
        n = prompt_tokens.shape[1]
        e = self.speech_emb(prompt_tokens) + self.speech_pos(torch.arange(n, dtype=torch.int64))[None]
        spk = self.enc.spkr_enc(speaker_emb.view(-1, 256))[:, None]
        pr = self.enc.perceiver(e)
        emo = self.enc.emotion_adv_fc(exaggeration.view(-1, 1, 1))
        return torch.cat((spk, pr, emo), dim=1)


def check_lm(lm):
    """Compare the export wrapper against the HF Llama backbone (prefill + one cached step)."""
    torch.manual_seed(0)
    x = torch.randn(2, 12, H) * 0.5
    past = [torch.zeros(2, NH, 0, HD) for _ in range(2 * L)]
    out = lm(x, *past)
    ref = t3.tfmr(inputs_embeds=x, use_cache=True, output_attentions=False)
    hl = t3.speech_head(ref.last_hidden_state[:, -1])
    print("prefill logits max|diff|", (out[0] - hl).abs().max().item(), "scale", hl.abs().max().item())
    x2 = torch.randn(2, 1, H) * 0.5
    out2 = lm(x2, *out[2:])
    ref2 = t3.tfmr(inputs_embeds=torch.cat((x, x2), 1), use_cache=False)
    hl2 = t3.speech_head(ref2.last_hidden_state[:, -1])
    print("step logits max|diff|", (out2[0] - hl2).abs().max().item())
    print("att shape", tuple(out[1].shape), tuple(out2[1].shape), "rows sum", out2[1].sum(-1).flatten()[:3].tolist())


lm = LM().eval()
check_lm(lm)

# ---- embedding tables (fp16) ----
tables = [("text_emb", t3.text_emb.weight), ("text_pos_emb", t3.text_pos_emb.emb.weight),
          ("speech_emb", t3.speech_emb.weight), ("speech_pos_emb", t3.speech_pos_emb.emb.weight)]
hdr, off = {}, 0
with open(OUT / "t3_embed.bin", "wb") as f:
    for name, w in tables:
        a = w.detach().numpy().astype(np.float16)
        f.write(a.tobytes())
        hdr[name] = {"offset": off, "shape": list(a.shape), "dtype": "float16"}
        off += a.nbytes
(OUT / "t3_embed.json").write_text(json.dumps(hdr))
print("embed", hdr)

# ---- cond ----
cond = Cond().eval()
spk = torch.randn(1, 256); spk = spk / spk.norm()
pt = torch.randint(0, 6561, (1, 150)); ex = torch.tensor([0.5])
ref_t3cond = __import__("chatterbox.models.t3.modules.cond_enc", fromlist=["T3Cond"]).T3Cond(
    speaker_emb=spk, cond_prompt_speech_tokens=pt, emotion_adv=ex.view(1, 1, 1))
ref = t3.prepare_conditioning(ref_t3cond)
print("cond max|diff|", (cond(spk, pt, ex) - ref).abs().max().item(), tuple(ref.shape))
torch.onnx.export(cond, (spk, pt, ex), str(OUT / "t3_cond.onnx"), dynamo=False, opset_version=17,
                  input_names=["speaker_emb", "prompt_tokens", "exaggeration"], output_names=["cond_emb"],
                  dynamic_axes={"prompt_tokens": {1: "N"}})

# ---- lm ----
names_in = ["inputs_embeds"] + [f"past_{kv}_{i}" for i in range(L) for kv in ("k", "v")]
names_out = ["logits", "attentions"] + [f"present_{kv}_{i}" for i in range(L) for kv in ("k", "v")]
dyn = {"inputs_embeds": {0: "B", 1: "T"}, "logits": {0: "B"}, "attentions": {1: "T", 2: "S"}}
for i in range(L):
    for kv in ("k", "v"):
        dyn[f"past_{kv}_{i}"] = {0: "B", 2: "P"}
        dyn[f"present_{kv}_{i}"] = {0: "B", 2: "S"}
x = torch.randn(2, 5, H)
past = tuple(torch.zeros(2, NH, 3, HD) for _ in range(2 * L))
torch.onnx.export(lm, (x, *past), str(OUT / "t3_lm_fp32.onnx"), dynamo=False, opset_version=17,
                  input_names=names_in, output_names=names_out, dynamic_axes=dyn)
print("exported")
