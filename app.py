import io
import os
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
from qdrant_client import QdrantClient

# ปิด Log ส่วนเกินและประหยัด RAM
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = FastAPI()

# เปิดรับ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ข้อมูลเชื่อมต่อ Qdrant (อย่าลืมตรวจเช็ก URL และ API KEY ของคุณ)
QDRANT_URL = "https://03d32c2a-f1d5-4076-bd7b-1ce3b4cc8e70.us-west-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6ZDkwMzU5OTItYzg5ZC00MzM5LWI3OWItNDkxNjQzZDM2MTcwIn0.3pkA4knu0EVaHoCn9HAlunY0WrGKde195WykbsFjcc8"
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

    # ย่อภาพเหลือ 500px เพื่อประหยัด RAM บน Render Free
    image.thumbnail((500, 500))
    img_np = np.array(image)

    # สกัดเวกเตอร์ใบหน้าด้วย OpenFace (โมเดลขนาดเล็ก 15MB ไม่กิน RAM)
    embeddings = DeepFace.represent(
        img_path=img_np,
        model_name="Facenet512",
        detector_backend="opencv",
        enforce_detection=False,
    )

    if not embeddings or len(embeddings) == 0:
      return {"status": "success", "results": []}

    query_vector = embeddings[0]["embedding"]

    # ค้นหาใน Qdrant ด้วย query_points (ไวยากรณ์เวอร์ชันใหม่)
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=12,
        score_threshold=0.40,
    )

    results = [hit.payload for hit in response.points]
    return {"status": "success", "results": results}

  except Exception as e:
    print(f"Error processing search: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))
