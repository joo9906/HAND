# import numpy as np
# import time
# import re
# import emoji
# import os
# from soynlp.normalizer import repeat_normalize
# from model_loader import session, tokenizer

# emojis = ''.join(emoji.EMOJI_DATA.keys())
# pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣{emojis}]+')
# url_pattern = re.compile(
#     r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
# )

# def clean(x):
#     x = pattern.sub(' ', x)
#     x = emoji.replace_emoji(x, replace='')
#     x = url_pattern.sub('', x)
#     x = x.strip()
#     x = repeat_normalize(x, num_repeats=2)
#     return x

# id2label = {
#     0: "기쁨",
#     1: "당황",
#     2: "분노",
#     3: "불안",
#     4: "상처",
#     5: "슬픔"
# }
# label2id = {v: k for k, v in id2label.items()}

# # onnx로 바꾸면서 softmax가 풀렸으므로 다시 numpy를 사용해 만들어줌
# def softmax(x):
#     x = np.array(x)
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum(axis=-1, keepdims=True)

# def predict(text: str):
#     text = clean(text)

#     inputs = tokenizer(text, return_tensors="np")

#     ort_inputs = {
#         "input_ids": inputs["input_ids"],
#         "attention_mask": inputs["attention_mask"]
#     }

#     logits = session.run(["logits"], ort_inputs)[0]  # (1, 6)
#     probs = softmax(logits)[0]  # shape: (6,)

#     result = {id2label[i]: float(probs[i]) for i in range(6)}
#     return result

# start = time.time()
# sentence = "오늘 상사한테 깨져서 너무 우울해"
# result = predict(sentence)
# end = time.time()

# print("추론 시간:", round(end - start, 4), "초")
# print("감정 확률:", result)
