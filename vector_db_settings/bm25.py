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

if client:
    print('🔗 Weaviate 연결 성공')

# client = weaviate.connect_to_local()
single = client.collections.use("SingleCounsel")

from weaviate.classes.config import Reconfigure

# Update the BM25 parameters for the inverted index
single.config.update(
    inverted_index_config=Reconfigure.inverted_index(
        bm25_b=0.8,
        bm25_k1=1.2
    )
)

check = single.config.get()
print(check)
client.close()