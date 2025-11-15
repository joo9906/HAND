from dotenv import load_dotenv
import os
import httpx
load_dotenv()

API_KEY = os.getenv("GMS_KEY")
EMB_MODEL = os.getenv("EMBEDDING_MODEL")
EMB_URL = os.getenv("EMBEDDING_GMS_URL")

def embed(text: str) -> list[float]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    payload = {
        "model": EMB_MODEL,
        "input": text,
    }

    try:
        response = httpx.post(
            EMB_URL,
            headers=headers,
            json=payload,
            timeout=10.0,
            verify=False,
        )
        response.raise_for_status()
        data = response.json()
        if "data" not in data:
            embedding = []
        else:
            embedding = data["data"][0]["embedding"]

        if not isinstance(embedding, list):
            raise ValueError("embedding이 list 형태가 아닙니다.")
        
        return [float(v) for v in embedding]

    except Exception as e:
        print("Embedding 요청 오류 발생:", e)
        raise e
