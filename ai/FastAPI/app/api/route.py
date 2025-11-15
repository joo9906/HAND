
from fastapi import APIRouter, HTTPException
from app.models.schemas import DiaryOutput, DiaryInput, ManageAdviceInput, ManageAdviceOutput, PersonalAdviceOutput, PersonalAdviceInput, ReportInput, ReportOutput
from app.services.emotion_classify import emotionClassifying
from app.services.report import create_report
from app.services.summary import longSummarize, shortSummarize
from app.services.advice import daily_advice
from app.services.advice import manager_advice as generate_manager_advice
from app.services.advice import private_advice as generate_private_advice
from app.core.vector_embedding import embed
import os
import weaviate
from RAGAS_eval.ragas import AdviceQualityEvaluator
import asyncio

router = APIRouter()

# Weaviate 세팅
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
def get_client():
    return weaviate.connect_to_custom(
        http_host="localhost",
        http_port=8080,
        grpc_host="localhost",
        grpc_port=50051,
        http_secure=False,
        grpc_secure=False,
    )

@router.get("/ai-server/health", response_model=str)
async def health():
    """서버의 상태를 확인합니다."""
    return "OK"

# 사용자의 다이어리 문장들을 받아와 오늘의 감정 점수 + 일간 요약(짧은 요약, 긴 요약)을 반환
@router.post("/diary/summary", response_model=DiaryOutput)
async def diary_classification(input_data: DiaryInput):
    try:
        user_id = input_data.user_id
        text_list = input_data.texts
        texts = " ".join(input_data.texts)
        
        # 텍스트 예외처리
        if not texts:
            raise ValueError("입력된 일기 텍스트가 없습니다.")

        # 감정 분석은 CPU/GPU 연산이므로 별도 스레드로 실행
        classify_task = emotionClassifying(text_list)
        short_task = shortSummarize(texts)
        long_task = longSummarize(texts)
        short_advice = daily_advice(texts)

        short_summary, long_summary, short_advice = await asyncio.gather(
            short_task, long_task, short_advice
        )

        if "error" in classify_task:
            raise ValueError(classify_task["error"])

        result = {
            "user_id": user_id,
            "result": {
                "score": classify_task["score"],
                "sentiment": classify_task["sentiment"],
                "short_summary": short_summary,
                "long_summary": long_summary,
                "short_advice": short_advice,
            },
        }
        return result
    
    except Exception as e:
        print(f"❌ diary_summary 오류: {e}")
        raise HTTPException(status_code=500, detail=f"오류 코드는 {e}")

# 팀장급에게 1주일치의 보고서와 조언을 제공
@router.post("/manager/advice", response_model = ManageAdviceOutput)
async def group_advice(input_data: ManageAdviceInput):
    try:
        user_id = input_data.user_id
        diaries = input_data.diaries
        biodata = input_data.biometrics
        total_summary = input_data.total_summary

        # 1) 보고서 생성
        report = await create_report(
            diary=diaries,
            biodata=biodata,
            total_summary=total_summary
        )

        evaluator = AdviceQualityEvaluator()
        
        # 2) 조언 생성 + 평가 반복
        MAX_RETRY = 3
        best_advice = None
        best_score = 0
        
        for attempt in range(MAX_RETRY):
            advice = await generate_manager_advice(report=report, summary=total_summary)

            # 평가
            eval_result = await evaluator.evaluate(
                summary=total_summary,
                report=report,
                advice=advice
            )

            final_score = evaluator.calc_final_score(eval_result)
            print(f"👉 Attempt {attempt+1} Score: {final_score}")

            # 최고 점수 기록
            if final_score > best_score:
                best_score = final_score
                best_advice = advice

            # 기준 통과하면 즉시 종료
            if final_score >= 0.7:
                break

        # 3) 최종 조언 결정 및 임베딩
        advice = best_advice
        
        # 최종 조언을 Weaviate에 집어넣어서 나중에 쓸 수 있도록.
        data_object = {
        "input": total_summary,
        "output": advice,
        }
        
        if best_score >= 0.7:
            embedding_advice = embed(total_summary)
            client = get_client()
            try:
                col = client.collections.get("SingleCounsel")
                uuid = col.data.insert(properties=data_object, vector=embedding_advice)
            finally:
                client.close()
                
            print(f"벡터 DB에 새로운 상담 데이터 저장. UUID : {uuid}, 백터는 : {embedding_advice[:5]}")
            print(f"평가 점수는 : {best_score}, 평가된 조언은 : {best_advice}, 상세 점수는 : {eval_result}")
        else:
            print(f"평가 점수가 낮아 Weaviate에 저장은 하지 않음. 점수 : {best_score}")

        return ManageAdviceOutput(
            user_id=user_id,
            report=report,
            advice=advice
        )

    except Exception as e:
        print(f"❌ manager_advice 오류: {e}")
        raise HTTPException(status_code=500, detail=f"관리자 조언 생성 중 오류: {e}")


# 개인에게 보고서와 조언 제공(1주일 치)
@router.post("/individual-users/report", response_model = PersonalAdviceOutput)
async def personal_advice(data: PersonalAdviceInput):
    try:
        user_id = data.user_id
        diary = data.diaries
        biodata = data.biometrics
        total_summary = data.total_summary

        report = await create_report(
            diary=diary,
            biodata=biodata,
            total_summary=total_summary
        )

        evaluator = AdviceQualityEvaluator()
        
        MAX_RETRY = 3
        best_advice = None
        best_score = 0
        best_eval = None

        for attempt in range(MAX_RETRY):
            advice = await generate_private_advice(report=report, summary=total_summary)

            eval_result = await evaluator.evaluate(
                summary=total_summary,
                report=report,
                advice=advice
            )

            final_score = evaluator.calc_final_score(eval_result)
            print(f"👉 [IND] Attempt {attempt+1} Score: {final_score}")

            if final_score > best_score:
                best_score = final_score
                best_advice = advice
                best_eval = eval_result

            if final_score >= 0.7:
                break

        advice = best_advice

        # 최종 조언을 Weaviate에 집어넣어서 나중에 쓸 수 있도록.
        data_object = {
        "input": total_summary,
        "output": advice,
        }
        
        if best_score >= 0.7:
            embedding_advice = embed(total_summary)
            client = get_client()
            try:
                col = client.collections.get("SingleCounsel")
                uuid = col.data.insert(properties=data_object, vector=embedding_advice)
            finally:
                client.close()
                        
            print(f"벡터 DB에 새로운 상담 데이터 저장. UUID : {uuid}, 백터는 : {embedding_advice[:5]}")
        else:
            print(f"평가 점수가 낮아 Weaviate에 저장은 하지 않음. 점수 : {best_score}")
        
        return PersonalAdviceOutput(
            user_id=user_id,
            report=report,
            advice=advice
        )

    except Exception as e:
        print(f"❌ personal_advice 오류: {e}")
        raise HTTPException(status_code=500, detail=f"개인 조언 생성 중 오류: {e}")
