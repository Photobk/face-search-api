import os
# สั่งปิดการจองแรมส่วนเกินของ TensorFlow และลด Log ที่ไม่จำเป็น
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
from qdrant_client import QdrantClient

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ข้อมูลเชื่อมต่อ Qdrant
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
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # ย่อขนาดรูปภาพให้เล็กลงเหลือ 500px เพื่อลดการใช้แรมขณะประมวลผล
        image.thumbnail((500, 500))
        img_np = np.array(image)

        # สลับมาใช้ detector opencv และ enforce_detection=False
        embeddings = DeepFace.represent(
            img_path=img_np,
            model_name="OpenFace",
            detector_backend="opencv",
            enforce_detection=False
        )

        if not embeddings or len(embeddings) == 0:
            return {"status": "success", "results": []}

        query_vector = embeddings[0]["embedding"]

        # ค้นหาใน Qdrant
        # ใช้คำสั่ง query_points สำหรับ qdrant-client ยุคใหม่
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=12,
            score_threshold=0.40,
        )

        results = [hit.payload for hit in response.points]
        )

        results = [hit.payload for hit in search_result]
        return {"status": "success", "results": results}

    except Exception as e:
        print(f"Error processing search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
