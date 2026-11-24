"""Split the model files into <=48 MiB parts and write the app's model_manifest.json (same schema as v2)."""
import hashlib, json, os, sys
src, dst = sys.argv[1], sys.argv[2]
ORDER = ["t3_lm.onnx", "t3_cond.onnx", "t3_embed.bin", "t3_embed.json", "tokenizer.json", "s3_tokenizer.onnx",
         "campplus.onnx", "s3_mel.onnx", "ve_mel.onnx", "ve_lstm.onnx", "ecapa.onnx", "s3gen_turbo.onnx", "s3gen_hq.onnx"]
PART = 40 * 1024 * 1024   # jsDelivr serves files up to 50 MB
os.makedirs(dst, exist_ok=True)
files = {}
for name in ORDER:
    data = open(os.path.join(src, name), "rb").read()
    parts = []
    for i in range(0, max(1, len(data)), PART):
        pn = f"{name}.part{i // PART}"
        open(os.path.join(dst, pn), "wb").write(data[i:i + PART])
        parts.append(pn)
    files[name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "parts": parts}
    print(name, len(data), len(parts))
m = {"version": 2, "engine": "chatterbox-multilingual", "files": files}
json.dump(m, open(os.path.join(dst, "model_manifest.json"), "w"), indent=1)
print("total MB", sum(f["size"] for f in files.values()) / 1e6)
