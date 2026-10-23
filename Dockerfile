FROM python:3.10-slim

WORKDIR /app

# ติดตั้งไลบรารีพื้นฐานสำหรับ OpenCV (เปลี่ยนมาใช้ libgl1)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# กำหนด Port สำหรับ Render
ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
