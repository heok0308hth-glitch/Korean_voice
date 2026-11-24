"""Replace RandomNormalLike / RandomUniformLike in an S3Gen decoder with fixed noise so the same
tokens always give the same audio (the app compares variants under equal seeds).

Each random op becomes: Tile(fixed flat noise) -> Slice(first N) -> Reshape(Shape(like)).
usage: patch_noise.py src.onnx dst.onnx"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)                       # loads external data too
g = m.graph
rng = np.random.default_rng(1234)
pool = {
    "normal_big": rng.standard_normal(80 * 15000).astype(np.float32),
    "normal_audio": rng.standard_normal(1 << 20).astype(np.float32),
    "uniform": rng.random(4096).astype(np.float32),
}
for k, a in pool.items():
    g.initializer.append(numpy_helper.from_array(a, f"fixed_{k}"))
g.initializer.append(numpy_helper.from_array(np.array([0], np.int64), "fixed_zero1"))
g.initializer.append(numpy_helper.from_array(np.array([0], np.int64), "fixed_axes0"))

out_nodes = []
count = 0
for n in g.node:
    if n.op_type not in ("RandomNormalLike", "RandomUniformLike"):
        out_nodes.append(n); continue
    attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
    if n.op_type == "RandomNormalLike":
        key = "normal_audio" if "sin_gen" in n.name else "normal_big"
        mean, scale = attrs.get("mean", 0.0), attrs.get("scale", 1.0)
    else:
        key = "uniform"
        lo, hi = attrs.get("low", 0.0), attrs.get("high", 1.0)
    p = f"{n.name}_fixed"
    M = pool[key].size
    sh, numel, numel1, reps, tiled, sl = (p + s for s in ("/shape", "/numel", "/numel1", "/reps", "/tiled", "/slice"))
    nodes = [
        helper.make_node("Shape", [n.input[0]], [sh], name=sh),
        helper.make_node("ReduceProd", [sh], [numel], keepdims=0, name=numel),
        helper.make_node("Reshape", [numel, p + "/one_shape"], [numel1], name=numel1),
        helper.make_node("Add", [numel1, p + "/Mm1"], [p + "/num"], name=p + "/num"),
        helper.make_node("Div", [p + "/num", p + "/M"], [reps], name=reps),
        helper.make_node("Tile", [f"fixed_{key}", reps], [tiled], name=tiled),
        helper.make_node("Slice", [tiled, "fixed_zero1", numel1, "fixed_axes0"], [sl], name=sl),
        helper.make_node("Reshape", [sl, sh], [p + "/noise"], name=p + "/reshape"),
    ]
    g.initializer.append(numpy_helper.from_array(np.array([1], np.int64), p + "/one_shape"))
    g.initializer.append(numpy_helper.from_array(np.array([M - 1], np.int64), p + "/Mm1"))
    g.initializer.append(numpy_helper.from_array(np.array([M], np.int64), p + "/M"))
    if n.op_type == "RandomNormalLike":
        g.initializer.append(numpy_helper.from_array(np.array(scale, np.float32), p + "/scale"))
        g.initializer.append(numpy_helper.from_array(np.array(mean, np.float32), p + "/mean"))
        nodes += [helper.make_node("Mul", [p + "/noise", p + "/scale"], [p + "/scaled"], name=p + "/mul"),
                  helper.make_node("Add", [p + "/scaled", p + "/mean"], [n.output[0]], name=p + "/add")]
    else:
        g.initializer.append(numpy_helper.from_array(np.array(hi - lo, np.float32), p + "/range"))
        g.initializer.append(numpy_helper.from_array(np.array(lo, np.float32), p + "/low"))
        nodes += [helper.make_node("Mul", [p + "/noise", p + "/range"], [p + "/scaled"], name=p + "/mul"),
                  helper.make_node("Add", [p + "/scaled", p + "/low"], [n.output[0]], name=p + "/add")]
    out_nodes += nodes
    count += 1
del g.node[:]
g.node.extend(out_nodes)
onnx.save(m, dst, save_as_external_data=False) if m.ByteSize() < 2_000_000_000 else onnx.save(m, dst, save_as_external_data=True)
print("patched", count, "random ops")
