"""fp16 S3 tokenizer: the transformer encoder runs in fp16; the mel front end and FSQ head stay fp32."""
import sys
import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
import librosa
from onnxconverter_common import float16

src, dst, wav = sys.argv[1], sys.argv[2], sys.argv[3:]
m = onnx.load(src)
block = [n.name for n in m.graph.node if not (n.name.startswith("/enc/blocks") or n.name.startswith("/enc/conv"))]
m16 = float16.convert_float_to_float16(m, keep_io_types=True, node_block_list=block)
onnx.save(m16, dst)

so = ort.SessionOptions()
a = ort.InferenceSession(src, so)
so16 = ort.SessionOptions(); so16.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
b = ort.InferenceSession(dst, so16)
for p in wav:
    w, sr = sf.read(p, dtype="float32")
    if w.ndim > 1: w = w.mean(1)
    w = librosa.resample(w, orig_sr=sr, target_sr=24000)[: 10 * 24000][None].astype(np.float32)
    ta = a.run(None, {"audio_values": w})[0]; tb = b.run(None, {"audio_values": w})[0]
    print(p, ta.shape, "fp16 agreement", float((ta == tb).mean()))
