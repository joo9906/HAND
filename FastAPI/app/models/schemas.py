from pydantic import BaseModel
from typing import List, Dict, Any

# 감정 분석 후 요약본 제공
class DiaryInput(BaseModel):
    user_id : int
    texts : List[str]

class DiaryOutput(BaseModel):
    user_id : int
    result: Dict[str, Any]


# 주간(월간) 보고서
class DiaryItem(BaseModel):
    date: str
    longSummary: str
    shortSummary: str
    depressionScore: float

class BaselineItem(BaseModel):
    version: int
    measurementCount: int
    dataStartDate: str
    dataEndDate: str
    hrvSdnn: Dict[str, float]
    hrvRmssd: Dict[str, float]
    heartRate: Dict[str, float]
    objectTemp: Dict[str, float]

class AnomalyItem(BaseModel):
    detectedAt: str
    measurementId: int
    stressIndex: float
    stressLevel: int
    heartRate: float
    hrvSdnn: float
    hrvRmssd: float

class UserInfo(BaseModel):
    age: int
    gender: str
    job: str
    height: float
    weight: float
    disease: str
    
class BM25User(BaseModel):
    age: int
    gender: str
    job: str
    disease: str
    family: str

class BM25User(BaseModel):
    age: int
    gender: str
    job: str
    disease: str
    family: str

class Biometrics(BaseModel):
    baseline: BaselineItem
    anomalies: List[AnomalyItem]
    userInfo: UserInfo

class ReportInput(BaseModel):
    user_id: int
    diaries: List[DiaryItem]
    biometrics: Biometrics

class ReportOutput(BaseModel):
    user_id: int
    result : str

# 개인 조언 전용
class PersonalAdviceInput(BaseModel):
    user_id: int
    diaries: List[DiaryItem]
    biometrics: Biometrics
    user_info: BM25User
    total_summary : str

class PersonalAdviceOutput(BaseModel):
    user_id : int
    report : str
    advice : str
    
# 관리자 상담 전용
class ManageAdviceInput(BaseModel):
    user_id: int
    diaries: List[DiaryItem]
    biometrics: Biometrics
    user_info: BM25User
    total_summary : str

class ManageAdviceOutput(BaseModel):
    user_id : int
    report : str
    advice : str