import weaviate
from weaviate.classes.config import Property, DataType, Configure

# ✅ v4 클라이언트 연결
try:
    weaviate_client = weaviate.connect_to_local(
        host="localhost",
        port=8080,
        grpc_port=50051,
    )
    print("✅ Weaviate 연결 성공")
    
except Exception as e:
    print(f"⚠️ Weaviate 연결 실패: {e}")
    weaviate_client = None


# ✅ 클래스(컬렉션) 존재 여부 확인 후 생성
def init_schema():
    if weaviate_client is None:
        return

# 서버 시작 시 스키마 초기화
init_schema()
