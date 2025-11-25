# 프로젝트 개요
### 감정 완화 프로젝트 — AI 심리·생체 데이터 분석 시스템

HAND Project – Emotion & Stress Early Detection AI Server

이 프로젝트는 사용자의 감정 일기, 생체 데이터(HRV·심박·체온 등), 이상 징후 분석, 그리고 유사 상담 사례 기반 RAG 조언 생성을 통합한 정서 안정 AI 분석 서버이다.
개인 사용자와 관리자(팀장·조직) 모두를 위한 맞춤형 심리 보고서 + 안전한 AI 조언을 생성하고, 생성된 조언의 품질을 자동 평가하여 고품질 조언만 VectorDB(Weaviate)에 축적한다.

### 실제 사용 영상

<div style="display: flex; justify-content: center; gap: 40px;">
  <img src="./example/다이어리%20작성.gif" width="30%"; margin-right="50px">
  <img src="./example/관리자%20기능.gif" width="30%"; margin-right = "50px">
</div> 

# 핵심 기능 정리

## 1. 📘 감정 분석 (Emotion Classification)

- 사용자가 작성한 하루 다이어리 텍스트를 기반으로 감정 점수와 6가지 감정 확률을 분석합니다.

- 모델: KcELECTRA 기반 커스텀 감정 분류 모델

- 감정 라벨: 기쁨 / 당황 / 분노 / 불안 / 상처 / 슬픔

- 감정별 확률을 계산 후 가중치를 적용하여 0~100 감정 점수 산출

- 전처리: URL 제거, 이모지 제거, 텍스트 정규화 등

- 최적화 및 경량화
    - 기존 FP32 모델을 FP 16으로 변환(모델 크기 511MB -> 213MB로 감소)
    - CPU 추론 최적화를 위해 ONNX 변환 (20토큰 기준 추론 속도 0.03초 -> 0.015초)

## 2. ✏️ AI 기반 텍스트 요약

다이어리 텍스트를 두 형태로 요약해 제공합니다.

✔️ 짧은 요약 (20글자 내, ~한 날로 끝남)

- 사용자에게 직접 보여주는 요약

- 사건 중심으로 감정 톤을 극도로 압축

✔️ 긴 요약 (길이 제한 없음)

- 내부 분석용 요약

- 사건/감정/맥락을 모두 담은 상세 요약

- 보고서 생성과 조언 생성을 위한 기반 데이터로 사용

## 3. 🩺 주간 심리·생체 보고서 생성 (Weekly Psychological/Bio Report)

다음 데이터를 종합해 전문가 스타일의 주간 분석 리포트를 생성합니다.

- 입력: 다이어리 요약들(long/short), 감정 점수 변화, Galaxy Watch 기반 생체 데이터(baseline, anomalies), 사용자 정보(age, gender, job 등)

- 출력: 감정 패턴 분석, 생체 스트레스 지표 해석, 감정–생체 반응 상관성 분석, 개인 특성 기반 코멘트, 주의 필요한 부분 / 회복 신호

- 항목별 글자 수 조건을 맞춘 정제된 보고서

## 4. 💬 조언 생성 (Advice Generation)

서버는 사용자 및 관리자(팀장)에게 각각 맞춤 조언을 제공합니다.

✔️ 일간 조언 (daily_advice)

- 오늘의 다이어리 내용을 기반으로 2줄 내외 조언 생성

✔️ 개인용 주간 조언 (private_advice)

- 주간 보고서 + 유사 상담 기록을 기반으로 조언

✔️ 관리자용 주간 조언 (manager_advice)

- 팀원의 상태 보고서 기반으로 관리자만 할 수 있는 톤의 조언을 추천

## 5. 🔍 RAG 기반 유사 상담 사례 검색

Weaviate 벡터 DB를 활용하여 과거 상담 기록에서 유사한 상황을 탐색합니다.

- Hybrid Search (BM25 + Vector)

- 입력: 감정 요약 및 사용자 정보(나이, 직업, 질병력 등), 일일 다이어리 요약 합본

- 단일 상담/멀티턴 상담 모두 검색 >> 조언 생성 시 참고 자료로 활용

## 6. 🧠 벡터 임베딩 시스템

GMS Embedding API 사용 (gpt-3-embedding)

- 텍스트 → 벡터 변환

- 조언/요약/리포트 등 모든 텍스트를 vector로 저장해 RAG 활용

## 7. 🗄️ Weaviate 벡터 DB 연동

다음 목적의 벡터를 저장/조회합니다.

- 조언 데이터 저장 (SingleCounsel, MultiCounsel)

- 저장 조건: 생성된 조언이 8 종류의 평가 기준 점수 평균 0.7 이상일 때만 저장

## 8. 📊 조언 품질 평가 시스템 (RAGAS-like + ARES)

조언의 품질을 자동으로 평가하여 0~1 스코어로 측정합니다.

- 포함된 평가 지표:

    Answer Relevancy (답변 관련성)

    Faithfulness (왜곡 없음)

    Context Relevancy (문맥 적합도)

    Empathy (공감도)

    Safety (안전성)

    Actionability (실행 가능성)

- ARES 7개 항목 (helpfulness, coherence, groundedness, safety 등)

- 모든 평가는 GMS 모델로 수행하고 MLflow에 자동 기록됩니다.

## 9. 📑 MLflow 자동 기록 (MLOps)

- 모든 조언 평가 metric을 MLflow에 기록

- 실험 관리 / 모델 품질 추적

- mlruns 디렉토리 자동 관리


# AI 폴더 구성
```
├── FastAPI/
│   ├── app/
│   │   ├── api/
│   │   │   ├── route.py           # 엔드포인트 라우팅
│   │   │   └── weav.py            # weaviate 연결 모듈
│   │   ├── core/
│   │   │   └── vector_embedding.py # GPT embedding 3 + GMS 요청 모듈
│   │   ├── models/
│   │   │   └── schemas.py         # Pydantic 스키마
│   │   └── services/
│   │       ├── advice.py          # 관리자 조언 생성
│   │       ├── emotion_classify.py# 감정 분석
│   │       ├── report.py          # 주간 보고서 생성
│   │       └── summary.py         # 요약 기능 모듈
│   └── __init__.py
│
├── Classifier_Model/                 # 감정 분류 모델(KcELECTRA) 관련 파일들
│   ├── config.json
│   ├── model.safetensors             # 감정 분석을 위한 핵심 파라미터 파일
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json                # 토큰화를 위한 핵심 파일
│   ├── training_args.bin
│   ├── main.py
│   ├── model_loader.py
│   └── README.md
│
├── Pretraining/                      # 사전학습 및 추가 파인튜닝 관련 노트북
│   └── KcELECTRA_shortsentence_finetuning.ipynb
│
├── Quantization/                     # 모델 경량화/양자화 실험
│   └── KcELECTRA_Quantization.ipynb
│
├── vector_db_settings/               # Weaviate 벡터DB 설정 및 데이터 적재
│   ├── db_setting.py
│   ├── docker-compose.yml
│   ├── insert.ipynb
│   ├── total_kor_counsel_bot.jsonl
│   └── total_kor_multiturn_counsel_bot.jsonl
│
├── mlflow.db                          # MLflow 로컬 DB
├── README.md
└── requirements.txt
```
