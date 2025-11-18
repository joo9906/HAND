import numpy as np
import os
import json
from dotenv import load_dotenv
import requests
import weaviate
import weaviate.classes as wvc
load_dotenv()

API_KEY = os.getenv("GMS_KEY")
EMB_MODEL = os.getenv("EMBEDDING_MODEL")
EMB_URL = os.getenv("EMBEDDING_GMS_URL")

# Weaviate랑 연결 - 환경 변수 사용
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

print(f"🔗 Weaviate 연결: {WEAVIATE_HOST}:{WEAVIATE_HTTP_PORT}")

client = weaviate.connect_to_custom(
    http_host=WEAVIATE_HOST,
    http_port=WEAVIATE_HTTP_PORT,
    grpc_host=WEAVIATE_HOST,
    grpc_port=WEAVIATE_GRPC_PORT,
    http_secure=False,
    grpc_secure=False,
)

# Class 생성 및 필드 설정
existing = client.collections.list_all()

if "SingleCounsel" not in existing:
    single_collection = client.collections.create(
        name="SingleCounsel",
        description="단일 상담 데이터",
        vector_config=wvc.config.Configure.Vectors.self_provided(),
        properties=[
            wvc.config.Property(name="input", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="output", data_type=wvc.config.DataType.TEXT),
        ],
    )
    print("✅ SingleCounsel 컬렉션 생성 완료")
else:
    print("✅ SingleCounsel 이미 존재. 생성 생략.")

if "MultiCounsel" not in existing:
    client.collections.create(
        name="MultiCounsel",
        description="멀티턴 상담 데이터",
        vector_config=wvc.config.Configure.Vectors.self_provided(),
        properties=[
            wvc.config.Property(name="patient", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="counselor", data_type=wvc.config.DataType.TEXT),
        ],
    )
    print("✅ MultiCounsel 컬렉션 생성 완료")
else:
    print("✅ MultiCounsel 이미 존재. 생성 생략.")

# jsonl 파일 가공을 위한 함수
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

class Embedding():
    def __init__(self):
        self.api_key = API_KEY
        self.emb_model = EMB_MODEL
        self.emb_url = EMB_URL

    # 임베딩
    def embed(self, text: str) -> list:
        headers = {
            "Content-type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {"model": self.emb_model, "input": text}

        res = requests.post(self.emb_url, headers=headers, json=payload)
        res.raise_for_status()
        
        result = res.json()["data"][0]["embedding"]

        return [float(v) for v in result]  # 1536 or 3072 dim


    # 벡터 검증 및 변환 함수
    def validate_and_convert_vector(self, vector):
        """벡터를 list[float]로 변환"""
        if vector is None:
            return None

        # numpy 배열 → list 변환
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()

        # list로 변환 확인
        if not isinstance(vector, list):
            raise TypeError(f"Vector must be list, got {type(vector)}")

        # 차원 확인
        if len(vector) == 0:
            raise ValueError("Vector cannot be empty")

        # float 검증
        vector = [float(v) for v in vector]

        return vector

    def validated_embed(self, text):
        vect = self.embed(text)
        validated_vector = self.validate_and_convert_vector(vect)

        return validated_vector
emb = Embedding()

# 수정된 코드
single_data = load_jsonl("./total_kor_counsel_bot.jsonl")
single_collection = client.collections.get("SingleCounsel")

for idx, d in enumerate(single_data):
    try:
        # 1. 임베딩 생성 (에러 처리 포함)
        vector_embedding = emb.validated_embed(d["input"])

        # 2. 벡터 검증
        if not vector_embedding or len(vector_embedding) == 0:
            print(f"❌ Row {idx}: 벡터가 비어있습니다")
            continue

        # 3. 데이터 객체 생성
        data_object = {
            "input": d["input"].strip(),
            "output": d["output"].strip()
        }

        # 4. Insert 및 반환값 확인
        uuid = single_collection.data.insert(
            properties=data_object,
            vector=vector_embedding
        )
        print(f"✅ Single {idx}: 저장 완료 (UUID: {uuid})")

    except Exception as e:
        print(f"❌ Row {idx} 에러: {str(e)}")
        continue

print("✅ SingleCounsel 업로드 완료")

multi_data = load_jsonl("./total_kor_multiturn_counsel_bot.jsonl")
multi_collection = client.collections.get("MultiCounsel")

for idx, d in enumerate(multi_data):
    try:
        patient = ""
        counselor = ""

        for t in d:
            if t['speaker'] == "내담자":
                patient += t['utterance']
            elif t['speaker'] == "상담사":
                counselor += t['utterance']

        # 1. 임베딩 생성 (에러 처리 포함)
        vector_embedding = emb.validated_embed(patient)

        # 2. 벡터 검증
        if not vector_embedding or len(vector_embedding) == 0:
            print(f"❌ Row {idx}: 벡터가 비어있습니다")
            continue

        # 3. 데이터 객체 생성
        data_object = {
            "patient": patient.strip(),
            "counselor" : counselor.strip()
        }

        # 계속 안돼서 DataObject를 사용하여 삽입 시도.
        obj = multi_collection.data.insert(
            properties=data_object,
            vector=vector_embedding
            )

        # 4. Insert 및 반환값 확인
        print(f"✅ Multi {idx}: 저장 완료 (UUID: {uuid})")

    except Exception as e:
        print(f"❌ Multi {idx} 에러: {str(e)}")
        continue

print("✅ MultiCounsel 업로드 완료")

# 제대로 들어가있는지 확인
for name in ["SingleCounsel", "MultiCounsel"]:
    col = client.collections.get(name)
    res = col.query.fetch_objects(limit = 1, include_vector=True)

    for i, obj in enumerate(res.objects, start=1):
        print(f"\n=== Object {i} ===")
        print("ID:", obj.uuid)
        print("properties:", obj.properties)
        print("vector exists:", obj.vector is not None)
        print("vector length:", obj.vector.get("default")[:5])