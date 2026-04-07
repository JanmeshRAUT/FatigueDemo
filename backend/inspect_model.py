import onnxruntime as ort
import os

model_path = os.path.join(os.path.dirname(__file__), "model", "model.onnx")
session = ort.InferenceSession(model_path)

print("ONNX Model Inputs:")
for input_meta in session.get_inputs():
    print(f"Name: {input_meta.name}")
    print(f"Shape: {input_meta.shape}")
    print(f"Type: {input_meta.type}")
    print("---")

print("ONNX Model Outputs:")
for output_meta in session.get_outputs():
    print(f"Name: {output_meta.name}")
    print(f"Shape: {output_meta.shape}")
    print(f"Type: {output_meta.type}")
    print("---")