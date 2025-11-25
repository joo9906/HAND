from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_cpp import Llama
import time

model_path = r"C:\Users\SSAFY\Desktop\WANG\S13P31A106\ai\FastAPI\Advice_Model\llama-3-Korean-Bllossom-8B-Q4_K_M.gguf"
tokenizer = AutoTokenizer.from_pretrained("MLP-KTLim/llama-3-Korean-Bllossom-8B")
model = AutoModelForCausalLM.from_pretrained("MLP-KTLim/llama-3-Korean-Bllossom-8B")

# 2. 테스트용 입력 문장
user_text = "친구와 점심을 먹으니 기분이 나아졌다. 그런데 막상 일을 시작하니 상사가 자꾸 지적한다. 짜증나고 불쾌한 하루였다. "

# 3. Llama 3.2 / Bllossom용 채팅 템플릿
prompt = f"""
    당신은 따뜻하고 현실적인 한국인 조언가입니다.
    
    [출력 조건]
    - 무조건 존댓말 사용.
    - 반드시 "~하는게 어떠세요?" 라고 끝낼 것.
    - 감정에 공감하면서 간단한 행동 조언을 주세요.
    - 반드시 최대 5문장 이내로 말해 주세요.

    [사용자 감정 요약]
    {user_text}

    [조언]
    """

messages = [
    {"role": "user", "content": prompt},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

print("✅ 추론 시작...")
start = time.time()
outputs = model.generate(**inputs, max_new_tokens=100)
reply = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
end = time.time()


print("\n====== 모델 응답 ======\n")
print(reply)
print(f"실제 걸린 시간은 : {end-start}")
print(f"현재 돌아가고 있는 하드웨어는 {model.device}")
print("\n=======================")