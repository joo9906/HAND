# Python 3.10 slim 이미지 사용
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치 (torch, opencv 등을 위한 필수 라이브러리)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 1단계: 가장 무거운 패키지 먼저 설치 (캐시 최적화)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    transformers==4.57.1

# 2단계: ML/데이터 관련 패키지
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    soynlp==0.0.493 \
    mlflow==3.5.1

# 3단계: 나머지 애플리케이션 패키지
RUN pip install --no-cache-dir \
    fastapi==0.120.0 \
    uvicorn==0.38.0 \
    python-dotenv==1.1.0 \
    weaviate-client==4.17.0 \
    openai==1.70.0 \
    httpx==0.28.1 \
    requests==2.32.4 \
    emoji==2.15.0 \
    langsmith==0.4.41 \
    langchain-core==1.0.3 \
    langchain==1.0.4

# FastAPI 애플리케이션 코드 복사
COPY FastAPI/ ./FastAPI/

# 작업 디렉토리를 FastAPI로 이동
WORKDIR /app/FastAPI

# 포트 노출
EXPOSE 8000

# FastAPI 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
