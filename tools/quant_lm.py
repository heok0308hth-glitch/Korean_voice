"""4-bit (MatMulNBits, block 32, symmetric) quantization of t3_lm; the speech head stays fp32."""
import sys
import onnx
from onnx import numpy_helper
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
inits = {i.name: i for i in m.graph.initializer}
exclude = []
for n in m.graph.node:
    if n.op_type == "MatMul" and n.input[1] in inits:
        dims = list(inits[n.input[1]].dims)
        if dims[-1] == 8194:
            exclude.append(n.name)
print("excluded (speech head):", exclude)
cfg = DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, accuracy_level=4)
q = MatMulNBitsQuantizer(m, block_size=32, is_symmetric=True, accuracy_level=4, nodes_to_exclude=exclude, algo_config=cfg)
q.process()
onnx.save(q.model.model, dst)
print("saved", dst)
