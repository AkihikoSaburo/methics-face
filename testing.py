import onnxruntime as ort

session = ort.InferenceSession("models/face_recognition/arcface_r50_fp16.onnx")

for img_input in session.get_inputs():
    print(f"Nama Input: {img_input.name} | Tipe Data: {img_input.type}")

