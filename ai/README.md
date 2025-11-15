# AI
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

# 주소 봐줘야 할 파일들

## FastAPI 폴더 쪽
### FastAPI/main.py
- FastAPI 주소 맞추기

### ai\FastAPI\app\services\RAG.py
- 맨 윗부분의 client = weaviate.connect_to_custom 주석을 풀고 Weaviate로 주소 맞춰주기

### Weaviate 관련
FastAPI\app\api\weav.py
vector_db_settings\docker-compose.yml

### Weaviate에 넣을 데이터
- total_kor_counsel_bot.jsonl
- total_kor_multiturn_counsel_bot.jsonl

- vector_db_settings\db_setting.py : 얘 주소값을 매칭해줘야 데이터 올릴 때 서버에 있는 weaviate로 드르감.

### Weaviate 도커로 띄운 다음 db_setting 실행. Weaviate에 데이터 삽입 후 print찍히는거 확인해서 vector 리스트가 나와야함.
- 이거 아마 시간 엄청 걸릴거임. 대충.... 2~3 시간?