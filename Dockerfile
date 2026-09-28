FROM python:3.10-slim

# ติดตั้ง System Dependencies สำหรับ dlib และ OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# คัดลอกและติดตั้ง Python Packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกโค้ดโปรเจกต์ทั้งหมด
COPY . .

# เปิดพอร์ตและรัน FastAPI (หากชื่อไฟล์ไม่ใช่ school.py ให้เปลี่ยนชื่อตรง school:app)
EXPOSE 8000
CMD ["uvicorn", "school:app", "--host", "0.0.0.0", "--port", "8000"]
