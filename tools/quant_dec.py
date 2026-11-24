"""4-bit MatMulNBits (block 32, symmetric) for the HQ decoder's MatMuls; convs stay fp32."""
import sys, onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig
m = onnx.load(sys.argv[1])
cfg = DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, accuracy_level=4)
q = MatMulNBitsQuantizer(m, block_size=32, is_symmetric=True, accuracy_level=4, algo_config=cfg)
q.process()
onnx.save(q.model.model, sys.argv[2])
