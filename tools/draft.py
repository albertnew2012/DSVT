import onnx
from onnx import shape_inference

onnx_path = "./deploy_files/dsvt.onnx"
inferred_model_path = "./deploy_files/inferred_model.onnx"

# Load the original model
model = onnx.load(onnx_path)

# Run shape inference
inferred_model = shape_inference.infer_shapes(model)

# Save the inferred model
onnx.save(inferred_model, inferred_model_path)

import onnxruntime as ort

providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
ort_session = ort.InferenceSession(inferred_model_path, providers=providers)
