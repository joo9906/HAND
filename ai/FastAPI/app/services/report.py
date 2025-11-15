import os
import httpx
from dotenv import load_dotenv
load_dotenv()

GMS_API_KEY = os.getenv("GMS_KEY")
GMS_URL = os.getenv("SUMMARY_GMS_URL")
MODEL = os.getenv("REPORT_MODEL")

async def create_report(diary: dict, biodata: dict, total_summary: str) -> str:
    system_prompt = f"""
        당신은 사용자의 정서 변화와 생체 데이터 패턴을 종합 분석하는 심리·생체 데이터 분석가입니다.
        입력으로는 사용자의 감정 일기 요약(diaries), 생체 데이터(biometrics), 이상 징후(anomalies), 개인 정보(userInfo)가 주어집니다.

        - diaries에는 날짜별 감정 요약(상세 요약 : longSummary, 단순 요약 : shortSummary)과 감정 점수(depressionScore)가 포함되어 있습니다.  
        각 일자의 정서적 톤(긍정/부정/중립), 감정 기복, 일관성, 회복력 등을 분석하세요. 감정 점수는 베이스 점수가 70점이고 숫자가 클수록 긍정적인 감정입니다.

        - biometrics에는 갤럭시 워치로 측정된 사용자의 신체 지표가 포함됩니다.  
        → baseline은 한 달치 평균값(mean, std)으로 개인의 안정 상태를 보여줍니다.  
        → anomalies는 스트레스 지표(stressIndex, heartRate, HRV 등)가 평소와 다르게 상승한 시점을 나타냅니다.  
        → 이를 통해 생리적 스트레스 반응의 빈도, 강도, 회복 속도를 판단하세요.

        - userInfo에는 나이, 성별, 직업, 신체 정보 등이 포함되어 있습니다. 분석 시 개인 특성과 직업 환경을 고려해, 생체 반응이 자연스러운 수준인지 혹은 위험 신호인지 구분하세요.

        당신의 역할은 단순한 요약이 아니라,  
        1) 심리적 패턴과 생체적 반응 간의 상관성을 해석하고,  
        2) 전반적인 주간 감정 안정성(또는 불안정성)을 평가하며,  
        3) 필요시 주의가 필요한 징후나 회복 추세를 명확하게 짚어주는 것입니다.

        출력은 반드시 한국어로 작성하고,  
        전문가다운 분석적이고 따뜻한 어조로 정리하세요.
        """
    
    prompt = f"""
        다음은 한 사용자의 주간 감정 일기, 생체 데이터, 이상 징후, 개인 정보입니다.
        이 정보를 기반으로 사용자의 전반적인 감정적 안정성과 스트레스 상태를 종합 분석하세요.

        [사용자 다이어리 요약 데이터]
        {total_summary}
        
        [사용자 신체 데이터]
        {biodata}

        분석 시 다음 항목을 반드시 포함해 주세요:
        1. {len(diary)}일간의 전반적인 정서 상태 요약 (예: 안정적, 불안정, 회복 중 등)
        2. 감정 일기(diaries)를 통한 주요 감정 패턴 및 우울 점수 변화 요약
        3. 생체 데이터(biometrics, anomalies)를 통한 신체적 스트레스 반응 분석
        4. 감정 변화와 생체 반응 간의 상관성 또는 불일치 지점
        5. 사용자(userInfo)의 개인 특성을 고려한 짧은 해석 (직업, 연령, 생활 환경 등)
        6. 전반적 평가 및 주의가 필요한 부분, 또는 회복 징후에 대한 결론
        
        1, 2, 3, 4번 문항은 최대 50자, 5, 6번 문항은 최소 50자 ~ 최대 100자로 작성해주세요.
        출력은 반드시 한국어로 작성하고, 분석적이고 따뜻한 어조로 정리하세요.
        """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GMS_API_KEY}",
    }
    
    messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {   
                "role": "user", 
                "content": prompt,
            },
        ]

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.7,
    }
    try:
        async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
            response = await client.post(GMS_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

        reason = result["choices"][0]["message"]["content"].strip()
        return reason

    except httpx.HTTPError as e:
        print(f"❌ HTTP 요청 실패: {e}")
        return "서버 요청 중 오류가 발생했습니다."
    except Exception as e:
        print(f"❌ 예기치 못한 오류: {e}")
        return "예기치 못한 오류가 발생했습니다."
