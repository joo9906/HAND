import os
import weaviate
import httpx
from fastapi import HTTPException
from dotenv import load_dotenv
from app.core.vector_embedding import embed 
from app.services.report import create_report

load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
ADVICE_URL = os.getenv("COUNSELING_GMS_URL")
ADVICE_MODEL = os.getenv("COUNSELING_MODEL")
GMS_KEY = os.getenv("GMS_KEY")

# Weaviate 연결
# 여기를 db_setting.py 에서 맞춰둔 주소랑 일치하게 해줘야 함. 바꾸기 전에는 로컬이랑 연결된 상태
client = weaviate.connect_to_custom(
    http_host="localhost",
    http_port=8080,
    grpc_host="localhost",
    grpc_port=50051,
    http_secure=False,
    grpc_secure=False,
    )

# 유사 상담내용 검색
async def retrieve_similar_cases(query: str, top_k: int = 2):
    try:
        # 쿼리 임베딩 생성
        query_vector = embed(query)
        
        # 뭔가 오류가 터지는데 뭔지 몰라서 찍어보는 것.
        if query_vector is None or not isinstance(query_vector, list):
            raise ValueError("Embedding 함수가 벡터를 반환하지 않았습니다.")

        # 단일 상담 검색
        single_col = client.collections.get("SingleCounsel")
        single_res = single_col.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_properties=["output"],
        )

        # 멀티턴 상담 검색
        multi_coll = client.collections.get("MultiCounsel")
        multi_res = multi_coll.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_properties=["counselor"],
        )

        # 결과만 텍스트로 추출
        single_texts = [o.properties.get("output", "") for o in single_res.objects]
        multi_texts = [o.properties.get("counselor", "") for o in multi_res.objects]

        return single_texts or [], multi_texts or []

    except Exception as e:
        print(f"❌ 상담 검색 중 오류: {e}")
        return [], []
    
    finally:
        client.close()

# 관리자 조언 생성 함수
async def manager_advice(report: str, summary: str):
    single, multi = await retrieve_similar_cases(summary)
    
    single_text = "\n".join([f"- {s}" for s in single])
    multi_text = "\n".join([f"- {m}" for m in multi])
    
    prompt = f"""
        당신은 팀장으로서 팀원의 상태를 보고 조언을 제시하는 역할입니다.
        - 팀장만 할 수 있는 조언을 위주로 작성할 것. 개인에게도 추천할 수 있는 방법보다는 관리자 입장에서의 조언을 만들어야 함.
        - 존댓말로 조언 작성
        - 불필요한 감정 표현은 피하고, 현실적이고 따뜻하게 조언할 것
        - 팀장은 상담 전문가가 아니므로 보다 안전하고 조심스러운 접근 방법을 제시할 것.
        - 유사한 상담 사례를 참고할 것.
        - 답변은 최소 300자, 최대 500자를 넘기지 말것.
        
        [팀원의 일주일치 상태 보고서]
        {report}

        [팀원의 상태와 유사한 사람과의 상담 사례]
        단일 상담사의 답변 : 
        {single_text}
        
        멀티턴 상담사의 답변 : 
        {multi_text}
        
        답변 생성 시 두가지 상담의 예시를 모두 참고하세요. 만약 유사 상담이 없을 경우 알아서 조언을 생성해주세요.
        아래의 형식을 참고하여 비슷한 형태로 생성하되, 아래의 형식의 내용은 참고하지 마세요.
        제안은 최대 3개까지만 제공해주세요.
        상태 요약을 짧고 간략하게 핵심만 뽑아주세요.

        당신의 답변 : 
        
        상태 요약 : 요즘 화재 출동이 많아지면서 스트레스가 누적되고, 수면 부족까지 겹쳐 많이 힘드실 것 같습니다. 누구라도 이런 상황이 지속되면 집중력이 떨어질 수밖에 없습니다.
        현재 본인의 상태를 스스로 인지하고 계신 것은 정말 중요한 부분이라고 생각합니다. 업무 특성상 긴장 상태가 길게 이어지면 몸과 마음 모두 지치기 쉽기  때문에, 작은 변화라도 시도해보는 것이 필요합니다.

        이런 제안을 해주는건 어떨까요?

        제안:
        1. 짧은 휴식이라도 챙기기
        바쁜 와중에도 잠깐이라도 눈을 감고 숨을 고르거나, 스트레칭을 해보시길 권합니다. 짧은 시간이더라도 반복적으로 휴식을 취하면 몸이 조금은 회복하는 데 도움이 될 수 있습니다.

        2. 수면 환경 점검하기
        퇴근 후에는 가급적 전자기기 사용을 줄이고, 밝은 조명을 피하는 등 잠자기 좋은 환경을 만들어보세요. 잠이 부족하면 업무 집중력에 더 큰 영향을 줄 수 있으니, 수면 시간을 조금이라도 확보하는 것이 중요합니다.

        3. 주변에 도움 요청하기
        혼자서 모든 부담을 안으려고 하지 않으셨으면 합니다. 팀 내에서 업무 분담이 조정이 가능한 부분이 있다면 꼭 말씀해주셔도 좋고, 서로 힘든 부분을 공유하는 것만으로도 심리적으로 도움이 될 수 있습니다.
        """
        
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GMS_KEY}",
    }
    
    messages = [
        {
            "role": "system",
            "content": "당신은 정서적으로 불안정한 팀원에게 상담을 해줘야 하는 팀장에게 가이드라인을 제시하는 상담 코치입니다. 한국어로 대답해 주세요. 관리자만이 할 수 있는 조언 위주로 답변을 만들어주세요.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    payload = {
        "model": ADVICE_MODEL,
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.6,
    }

    try:
        async with httpx.AsyncClient(verify=False, timeout=30.0) as cli:
            response = await cli.post(ADVICE_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
        advice = result["choices"][0]["message"]["content"].strip()
        
        return advice

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GMS 요청 중 오류 발생: {e}")
    


# 개인용 조언 생성 함수
async def private_advice(report: str, summary: str):
    single, multi = await retrieve_similar_cases(summary)
    single_text = "\n".join([f"- {s}" for s in single]) if single else "유사 단일 상담 없음"
    multi_text = "\n".join([f"- {m}" for m in multi]) if multi else "유사 멀티 상담 없음"
    
    prompt = f"""
        당신은 정서적으로 불안정할 수 있는 사람에게 작은 조언을 주는 역할입니다.
        - 존댓말로 조언 작성
        - 불필요한 감정 표현은 피하고, 현실적이고 따뜻하게 조언할 것
        - 당신은 상담 전문가가 아니므로 보다 안전하고 조심스러운 접근 방법을 제시할 것.
        - 유사한 상담 사례를 참고할 것.
        - 답변은 최소 100자, 최대 300자를 넘기지 말것.

        [사용자의 일주일치 다이어리 보고서]
        {report}

        [사용자의 상태와 유사한 사람과의 상담 사례]
        
        단일 상담사의 답변 : 
        {single_text}
        
        멀티턴 상담사의 답변 : 
        {multi_text}
        
        답변 생성 시 두가지 종류의 상담 사례를 모두 참고하세요.
        아래의 형식을 참고하여 비슷한 형태로 생성하되, 아래의 형식의 내용은 참고하지 마세요.
        제안은 최대 3개까지만 짧게 제공해주세요.

        당신의 답변 : 
        
        제안:
        1. 짧은 휴식이라도 챙기기
        바쁜 와중에도 잠깐이라도 눈을 감고 숨을 고르거나, 스트레칭을 해보시길 권합니다. 짧은 시간이더라도 반복적으로 휴식을 취하면 몸이 조금은 회복하는 데 도움이 될 수 있습니다.

        2. 수면 환경 점검하기
        퇴근 후에는 가급적 전자기기 사용을 줄이고, 밝은 조명을 피하는 등 잠자기 좋은 환경을 만들어보세요. 잠이 부족하면 업무 집중력에 더 큰 영향을 줄 수 있으니, 수면 시간을 조금이라도 확보하는 것이 중요합니다.

        3. 주변에 도움 요청하기
        혼자서 모든 부담을 안으려고 하지 않으셨으면 합니다. 팀 내에서 업무 분담이 조정이 가능한 부분이 있다면 꼭 말씀해주셔도 좋고, 서로 힘든 부분을 공유하는 것만으로도 심리적으로 도움이 될 수 있습니다.
        """
        
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GMS_KEY}",
    }
    
    messages = [
        {
            "role": "system",
            "content": "당신은 정서적으로 불안정한 팀원에게 상담을 해주는 코치입니다. 한국어로 대답해 주세요.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    payload = {
        "model": ADVICE_MODEL,
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.6,
    }

    try:
        async with httpx.AsyncClient(verify=False, timeout=20.0) as cli:
            response = await cli.post(ADVICE_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
        advice = result["choices"][0]["message"]["content"].strip()
        client.close()
        return advice

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GMS 요청 중 오류 발생: {e}")
    
# 개인용 조언 생성 함수
async def daily_advice(text: str):  
    prompt = f"""
        당신은 정서적으로 불안정할 수 있는 사람에게 매우 짧은 조언을 주는 역할입니다. 아래의 조건을 참고하세요.
        
        [조건]
        - 존댓말로 조언 작성
        - 불필요한 감정 표현은 피하고, 현실적이고 따뜻하게 조언할 것
        - 당신은 상담 전문가가 아니므로 보다 안전하고 조심스러운 접근 방법을 제시할 것.
        - 유사한 상담 사례를 참고할 것.
        - 답변은 아래의 예시를 참고하되, 각 조언 당 50글자를 넘지 않을 것.

        
        [예시]
        
        예시 다이어리 내용 : 오늘 회사를 다녀오는 길에 어떤 사람이 술에 취해서 시비를 걸었어. 너무 불쾌한데 어쩔 수 없다는게 화나. 계속 머릿속에 맴돌아서 고통스러워.
        
        [출력]
        
        오늘 술에 취한 사람 때문에 기분이 좋지 않으시군요. 이렇게 해보는건 어떠신가요?
        
        조언 1 : 가볍게 산책하며 머리를 비우기.
        조언 2 : 따듯하고 맛있는 음식 먹으며 소소한 행복 찾기.
        
        [실제 사용자의 다이어리]
        {text}
        
        """
        
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GMS_KEY}",
    }
    
    messages = [
        {
            "role": "system",
            "content": "당신은 정서적으로 불안정한 사용자에게 조언을 해주는 친구입니다. 한국어로 대답해 주세요.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    payload = {
        "model": ADVICE_MODEL,
        "messages": messages,
        "max_tokens": 200,
        "temperature": 0.6,
    }

    try:
        async with httpx.AsyncClient(verify=False, timeout=20.0) as cli:
            response = await cli.post(ADVICE_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
        advice = result["choices"][0]["message"]["content"].strip()
        client.close()
        return advice

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GMS 요청 중 오류 발생: {e}")
