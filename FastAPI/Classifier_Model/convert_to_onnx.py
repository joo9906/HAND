import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

def convert_to_onnx():
    # Get the directory where this script is located
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths
    model_path = base_path
    output_path = os.path.join(r"G:\SSAFY\About_Code\S13P31A106\ai\FastAPI\onnx", "model.onnx")

    print(f"Loading model from {model_path}...")
    
    # Load model and tokenizer
    try:
        # Load model on CPU for conversion
        model = AutoModelForSequenceClassification.from_pretrained(model_path, dtype=torch.float16, device_map="cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create dummy input for tracing
    # Using a simple sentence to generate valid input_ids and attention_mask
    dummy_text = "안녕하세요, 테스트 문장입니다."
    inputs = tokenizer(dummy_text, return_tensors="pt")
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    print("Starting ONNX conversion...")
    print(f"Input shape: {input_ids.shape}")

    try:
        torch.onnx.export(
            model,                                      # model being run
            (input_ids, attention_mask),                # model input (or a tuple for multiple inputs)
            output_path,                                # where to save the model (can be a file or file-like object)
            export_params=True,                         # store the trained parameter weights inside the model file
            opset_version=17,                           # the ONNX version to export the model to
            do_constant_folding=True,                   # whether to execute constant folding for optimization
            input_names=['input_ids', 'attention_mask'], # the model's input names
            output_names=['logits'],                    # the model's output names
            dynamic_axes={                              # variable length axes
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size'}
            }
        )
        print(f"Successfully converted model to: {output_path}")
        
    except Exception as e:
        print(f"Failed to convert model: {e}")

if __name__ == "__main__":
    convert_to_onnx()
