import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
from qdrant_client import QdrantClient

app = FastAPI()

# เปิดรับ CORS จากทุกโดเมน
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ข้อมูลเชื่อมต่อ Qdrant (ใส่ URL และ API Key ของคุณ)
QDRANT_URL = "https://your-cluster-url.qdrant.tech"
QDRANT_API_KEY = "your-api-key"
COLLECTION_NAME = "event_faces"

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Face Search API is running"}

@app.post("/search")
async def search_face(file: UploadFile = File(...)):
    try:
        # อ่านไฟล์รูปภาพ
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # ปรับขนาดย่อภาพลงหากใหญ่เกินไปเพื่อประหยัด RAM บน Render Free
        image.thumbnail((800, 800))
        img_np = np.array(image)

        # สกัดเวกเตอร์ใบหน้า โดยตั้ง enforce_detection=False ป้องกันแครชหากตรวจไม่พบหน้าตรงเป๊ะ
        embeddings = DeepFace.represent(
            img_path=img_np,
            model_name="Facenet512",
            detector_backend="opencv",
            enforce_detection=False
        )

        if not embeddings or len(embeddings) == 0:
            return {"status": "success", "results": []}

        query_vector = embeddings[0]["embedding"]

        # ค้นหาเวกเตอร์ที่ใกล้เคียงที่สุดจาก Qdrant
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=12,
            score_threshold=0.40  # ปรับเกณฑ์ความเหมือนตามความเหมาะสม
        )

        results = [hit.payload for hit in search_result]
        return {"status": "success", "results": results}

    except Exception as e:
        print(f"Error processing search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
