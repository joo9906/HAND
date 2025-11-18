import os
import json
import zipfile
import glob
from tqdm import tqdm

# ===============================
# 1. 데이터 경로 설정
# ===============================
BASE_DIR = "/workspace/empathy_dataset"  # GPU 서버 업로드 위치
EXTRACT_DIR = os.path.join(BASE_DIR, "unzipped")
OUTPUT_FILE = os.path.join(BASE_DIR, "empathy_pairs.jsonl")

os.makedirs(EXTRACT_DIR, exist_ok=True)

# ===============================
# 2. ZIP 파일 자동 해제
# ===============================
zip_files = glob.glob(os.path.join(BASE_DIR, "*.zip"))
print(f"📦 발견된 zip 파일 개수: {len(zip_files)}")

for zip_path in tqdm(zip_files, desc="압축 해제 중"):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        extract_folder = os.path.join(EXTRACT_DIR, os.path.splitext(os.path.basename(zip_path))[0])
        os.makedirs(extract_folder, exist_ok=True)
        zf.extractall(extract_folder)

print("✅ 모든 zip 파일 해제 완료")

# ===============================
# 3. JSON 파일 파싱 함수 정의
# ===============================
def extract_pairs_from_json(json_path):
    pairs = []
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        situation = data["info"].get("situation", "").strip()
        utterances = data.get("utterances", [])

        for i in range(len(utterances) - 1):
            u1, u2 = utterances[i], utterances[i + 1]
            if u1.get("role") == "speaker" and u2.get("role") == "listener":
                if not u2.get("listener_empathy"):
                    continue
                input_text = f"[상황] {situation}\n[화자] {u1['text'].strip()}\n[답변]"
                output_text = u2["text"].strip()
                pairs.append({"input": input_text, "output": output_text})
        return pairs

    except Exception as e:
        print(f"❌ {json_path} 처리 중 오류: {e}")
        return []

# ===============================
# 4. 모든 JSON 파일 순회 및 파싱
# ===============================
json_files = glob.glob(os.path.join(EXTRACT_DIR, "**/*.json"), recursive=True)
print(f"📂 JSON 파일 총 {len(json_files)}개 탐색됨")

pairs = []
for json_path in tqdm(json_files, desc="JSON 파싱 중"):
    pairs.extend(extract_pairs_from_json(json_path))

print(f"✅ 총 {len(pairs)}개의 대화 쌍 추출 완료")

# ===============================
# 5. 결과 저장
# ===============================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"💾 저장 완료: {OUTPUT_FILE}")
