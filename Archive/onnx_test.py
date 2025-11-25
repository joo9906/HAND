from onnxruntime import InferenceSession
import numpy as np
import onnx

model = onnx.load(r"G:\SSAFY\About_Code\S13P31A106\ai\FastAPI\onnx\model.onnx")
for i in model.graph.input:
    print("INPUT:", i.name, i.type.tensor_type.shape)
for o in model.graph.output:
    print("OUTPUT:", o.name, o.type.tensor_type.shape)
