FROM python:3.10-slim

# ติดตั้ง System Dependencies ที่จำเป็นสำหรับการ Build
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    libx11-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# จำกัดการใช้ CPU/RAM ขณะคอมไพล์ dlib ไม่ให้เกินโควตา Render
ENV MAKEFLAGS="-j1"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "school:app", "--host", "0.0.0.0", "--port", "10000"]
