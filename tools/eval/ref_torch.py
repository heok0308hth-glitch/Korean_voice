"""Reference: original Chatterbox Multilingual (PyTorch) on the same refs/sentences."""
import sys, os, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import torch, soundfile as sf
from safetensors.torch import load_file
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.s3gen import S3Gen
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.models.tokenizers import MTLTokenizer
C = Path("hf/chatterbox")
ve = VoiceEncoder(); ve.load_state_dict(load_file(str(C / "ve.safetensors"))); ve.eval()
t3 = T3(T3Config.multilingual()); t3.load_state_dict(load_file(str(C / "t3_mtl23ls_v2.safetensors"))); t3.eval()
s3 = S3Gen(); s3.load_state_dict(load_file(str(C / "s3gen.safetensors")), strict=False); s3.eval()
tts = ChatterboxMultilingualTTS(t3, s3, ve, MTLTokenizer(str(C / "grapheme_mtl_merged_expanded_v1.json")), "cpu")
S = [l.strip() for l in open("eval/sentences.txt", encoding="utf-8") if l.strip()]
torch.manual_seed(0)
for r in sys.argv[1:]:
    os.makedirs(f"eval/torch/{r}", exist_ok=True)
    tts.prepare_conditionals(f"ref/{r}.wav", exaggeration=0.5)
    for k, s in enumerate(S):
        if os.path.exists(f"eval/torch/{r}/s{k:02d}.wav"): continue
        w = tts.generate(s, language_id="ko", exaggeration=0.5, cfg_weight=0.5, temperature=0.8)
        sf.write(f"eval/torch/{r}/s{k:02d}.wav", w[0].numpy(), 24000)
        print("done", r, k, w.shape[-1] / 24000, flush=True)
