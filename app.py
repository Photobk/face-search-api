import io
import os
import uuid
import requests
import numpy as np
from PIL import Image
from pydantic import BaseModel
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# ป้องกัน TensorFlow จองหน่วยความจำเกินพิกัดบน Render Free Tier
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

app = FastAPI()

# ตั้งค่าเปิดรับ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ข้อมูลเชื่อมต่อ Qdrant
QDRANT_URL = "https://03d32c2a-f1d5-4076-bd7b-1ce3b4cc8e70.us-west-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6ZDkwMzU5OTItYzg5ZC00MzM5LWI3OWItNDkxNjQzZDM2MTcwIn0.3pkA4knu0EVaHoCn9HAlunY0WrGKde195WykbsFjcc8"
COLLECTION_NAME = "event_faces"

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


# โครงสร้างข้อมูลที่รับจาก Google Apps Script
class IndexPhotoRequest(BaseModel):
    photo_url: str
    event_title: str = "กิจกรรมสัมพันธ์"
    event_date: str = ""


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Face Search API is running"}


# 1. ระบบรับรูปจาก Google Apps Script เข้าสู่ Qdrant (ตรงกับที่ GAS เรียกหา)
@app.post("/index-photo")
async def index_photo(data: IndexPhotoRequest):
    try:
        # ดาวน์โหลดรูปภาพจาก Google Drive ผ่าน photo_url
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(data.photo_url, headers=headers, timeout=20)
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Cannot download image from Drive")

        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        image.thumbnail((500, 500))  # ย่อภาพประหยัด RAM
        img_np = np.array(image)

        # สกัดเวกเตอร์ใบหน้าขนาด 128 มิติด้วย OpenFace
        embeddings = DeepFace.represent(
            img_path=img_np,
            model_name="OpenFace",
            detector_backend="opencv",
            enforce_detection=False,
        )

        if not embeddings or len(embeddings) == 0:
            return {"status": "no_face_found", "message": "No face found in image"}

        points = []
        for face in embeddings:
            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=face["embedding"],
                    payload={
                        "photo_url": data.photo_url,
                        "event_title": data.event_title,
                        "event_date": data.event_date,
                    },
                )
            )

        # บันทึกข้อมูลเข้า Qdrant
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

        return {"status": "success", "indexed_faces": len(points)}

    except Exception as e:
        print(f"Error indexing photo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# 2. ระบบค้นหาใบหน้าจากหน้าเว็บ
@app.post("/search")
async def search_face(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # ย่อขนาดรูปภาพเพื่อประหยัดหน่วยความจำ
        image.thumbnail((500, 500))
        img_np = np.array(image)

        # สกัดเวกเตอร์ขนาด 128 มิติด้วย OpenFace
        embeddings = DeepFace.represent(
            img_path=img_np,
            model_name="OpenFace",
            detector_backend="opencv",
            enforce_detection=False,
        )

        if not embeddings or len(embeddings) == 0:
            return {"status": "success", "results": []}

        query_vector = embeddings[0]["embedding"]

        # ค้นหาเวกเตอร์ที่ตรงกันจาก Qdrant (ปรับ score_threshold ให้เหมาะกับ OpenFace)
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=12,
            score_threshold=0.68,
        )

        results = [hit.payload for hit in response.points]
        return {"status": "success", "results": results}

    except Exception as e:
        print(f"Error processing search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
