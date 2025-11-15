# HAND
SSAFY 자율 프로젝트 : 감정 완화 서비스의 AI 서버입니다.

## 감정 분석 모델
모델은 허깅페이스에 업로드 되어 있습니다.
사용 방법도 함께 기재하였습니다.

혹은 아래의 주소에서 safetensor 파일만 다운로드 후 FastAPI/Classifier_Model 폴더에  업로드 하면 사용 가능합니다.

모델 URL : https://huggingface.co/noridorimari/emotion_classifier

## 폴더 구성
```
── ai/                                # FastAPI 백엔드 전체
│   ├── FastAPI/
│   │   ├── app/
│   │   │   ├── api/
│   │   │   │   ├── route.py           # 엔드포인트 라우팅
│   │   │   │   └── weav.py            # weaviate 연결 모듈
│   │   │   ├── core/
│   │   │   │   └── vector_embedding.py # GPT embedding 3 + GMS 요청 모듈
│   │   │   ├── models/
│   │   │   │   └── schemas.py         # Pydantic 스키마
│   │   │   └── services/
│   │   │       ├── advice.py          # 관리자 조언 생성
│   │   │       ├── emotion_classify.py# 감정 분석
│   │   │       ├── report.py          # 주간 보고서 생성
│   │   │       └── summary.py         # 요약 기능 모듈
│   │   └── __init__.py
│   │
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
├── .env                               # 환경 변수(gms key, url 등)
├── README.md
└── requirements.txt
```