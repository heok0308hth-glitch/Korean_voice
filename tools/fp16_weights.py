"""Weight-only fp16: large fp32 initializers are stored as fp16 and Cast back to fp32 in the graph.
Compute stays fp32 (bit-identical outputs up to weight rounding); the file and the unoptimised
resident size halve. Loaded with graph optimisation off, ORT keeps the fp16 copies.
usage: fp16_weights.py src.onnx dst.onnx [min_elements]"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

src, dst = sys.argv[1], sys.argv[2]
min_el = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
m = onnx.load(src)
g = m.graph
new_init, casts = [], []
for t in g.initializer:
    if t.data_type == TensorProto.FLOAT and np.prod(t.dims) >= min_el:
        a = numpy_helper.to_array(t).astype(np.float16)
        h = numpy_helper.from_array(a, t.name + "_fp16")
        new_init.append(h)
        casts.append(helper.make_node("Cast", [h.name], [t.name], to=TensorProto.FLOAT, name=t.name + "_cast"))
    else:
        new_init.append(t)
del g.initializer[:]
g.initializer.extend(new_init)
nodes = casts + list(g.node)
del g.node[:]
g.node.extend(nodes)
onnx.save(m, dst)
print("converted", len(casts), "initializers")
