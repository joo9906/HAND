from transformers import (
    AutoTokenizer,
)
import torch
import os
import onnxruntime as ort

print("[ONNX MODEL LOADER] 모델 로드 중...")

onnx_model_path = os.path.join(os.path.dirname(__file__), "onnx", "model.onnx")
tokenizer_path = os.path.join(os.path.dirname(__file__), "Classifier_Model")
Device = "cuda" if torch.cuda.is_available() else "cpu"

session = ort.InferenceSession(
    onnx_model_path,
    providers=["CPUExecutionProvider"]
)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

if session:
    print(f"모델 로드 완료")
else:
    print(f"모델 로드 실패")