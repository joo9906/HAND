<<<<<<< HEAD
=======
from model_loader import model, tokenizer
>>>>>>> 1c0419e150ea1f5bf9bf98ccbd8708b2bc14ae22
from transformers import pipeline
import torch
import re
import emoji
from soynlp.normalizer import repeat_normalize
from model_loader import session, tokenizer
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

emojis = ''.join(emoji.EMOJI_DATA.keys())
pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣{emojis}]+')
url_pattern = re.compile(
    r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

# 문장을 더 깔끔하게 가공하는 함수
def clean(x):
    x = pattern.sub(' ', x)
    x = emoji.replace_emoji(x, replace='') #emoji 삭제
    x = url_pattern.sub('', x)
    x = x.strip()
    x = repeat_normalize(x, num_repeats=2)
    return x

# 감정 라벨 매핑
id2label = {
    0: "기쁨",       # happy
    1: "당황",       # embarrass
    2: "분노",       # anger
    3: "불안",       # unrest
    4: "상처",       # damaged
    5: "슬픔"        # sadness
}
label2id = {v: k for k, v in id2label.items()}


# onnx로 바꾸면서 softmax가 풀렸으므로 다시 numpy를 사용해 만들어줌
def softmax(x):
    x = np.array(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def predict(text: str):
    text = clean(text)

    inputs = tokenizer(text, return_tensors="np")

    ort_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }

    logits = session.run(["logits"], ort_inputs)[0]  # (1, 6)
    probs = softmax(logits)[0]  # shape: (6,)

    result = {id2label[i]: float(probs[i]) for i in range(6)}
    return result

# 감정별 가중치 (심리 영향 기반)
emotion_weights = {
    "기쁨": +3.0,
    "당황": -0.7,
    "분노": -0.7,
    "불안": -1.2,
    "상처": -1.4,
    "슬픔": -1.7
}

# 들어온 텍스트를 onnx 변환 된 감정 분류 모델로 판정 내림.
def emotionClassifying(texts: list[str]) -> dict:
    try:
        all_scores = {label: 0.0 for label in id2label.values()}

        for text in texts:
            preds = predict(clean(text))  # ← dict 형태

            # preds = {"분노":0.12, "슬픔":0.22 ...} 형태
            for label, score in preds.items():
                all_scores[label] += score

        # 전체 평균
        for k in all_scores:
            all_scores[k] /= len(texts)

        # 가중치 반영
        weighted_sum = sum(
            all_scores[e] * emotion_weights[e] for e in all_scores
        )

        base_score = 70
        scale = 30
        final_score = base_score + (weighted_sum * scale)

        # 0~100 constrain
        final_score = max(0, min(100, final_score))
        final_score = round(final_score, 3)

        return {
            "sentiment": {k: round(v, 5) for k, v in all_scores.items()},
            "score": final_score,
            "type": "emotion_score",
        }

    except Exception as e:
        print(f'"error": 감정 분석 중 오류 : {str(e)}')
        return { "sentiment": {}, "score": 60, "type": "error" }