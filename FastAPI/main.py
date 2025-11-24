from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import route
from model_loader import model, tokenizer
from contextlib import asynccontextmanager
import torch
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print("🚀 서버 시작 중… 모델 Warm-up 중입니다.")
        
        # 토크나이저 입력 준비
        inputs = tokenizer("오늘 해가 나와서 기분 좋아.", return_tensors="pt")
        
        # 디바이스 설정 및 이동
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # gradient 비활성화 후 모델 실행
        model.eval()
        with torch.no_grad():
            _ = model(**inputs)
        
        print("✅ 모델 로드 및 Warm-up 완료")

    except Exception as e:
        print(f"❌ Warm-up 실패: {e}")

    # FastAPI 앱이 실행되는 동안 유지
    yield

    # 서버 종료 시 리소스 정리
    try:
        torch.cuda.empty_cache()
        print("🛑 서버 종료 중… GPU 캐시 정리 완료.")
    except Exception as e:
        print(f"⚠️ 종료 중 문제 발생: {e}")


app = FastAPI(lifespan=lifespan, title="AI Server")

# CORS 설정
allowed_origins = os.getenv("CORS_ORIGINS", "https://gatewaytohand.store").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://gatewaytohand.store/api/v1/",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 라우터 등록
app.include_router(route.router, prefix="/ai")