# face-search-server

เซิร์ฟเวอร์ค้นหารูปตัวเองจากคลังรูปภาพ (เช่น รูปที่ดึงมาจากเพจโรงเรียน) โดยใช้การจดจำใบหน้าจริง

## ติดตั้ง

ต้องมี Python 3.9+ และ CMake (สำหรับ compile dlib ซึ่ง `face_recognition` ใช้อยู่ข้างใต้)

```bash
# macOS
brew install cmake

# Ubuntu/Debian
sudo apt-get install -y cmake build-essential

pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

## วิธีใช้

**1. ลงทะเบียนใบหน้าตัวเอง** (ยิ่งส่งหลายรูป มุม/แสงต่างกัน ยิ่งแม่น)

```bash
curl -X POST http://localhost:8000/enroll \
  -F "files=@me1.jpg" -F "files=@me2.jpg"
```

**2. สแกนคลังรูป** — ส่งรูปที่ดึงมาจากเพจได้ทีละหลายไฟล์ ระบบจะคืนเฉพาะรูปที่ตรง รูปที่ไม่ตรงจะไม่ถูกส่งกลับหรือเก็บไว้เลย

```bash
curl -X POST http://localhost:8000/scan \
  -F "files=@post1.jpg" -F "files=@post2.jpg" -F "files=@post3.jpg"
```

ตอบกลับเป็น JSON บอกว่าแมตช์กี่รูป พร้อม `download_url` ของแต่ละรูป

**3. ดูรูปที่แมตช์ไว้ทั้งหมด / ดาวน์โหลด**

```bash
curl http://localhost:8000/matches
curl -O http://localhost:8000/download/<filename>
```

เชื่อมกับหน้าเว็บ (เช่นต้นแบบ HTML ที่ทำไว้ก่อนหน้า) ได้โดยเรียก endpoint เหล่านี้ผ่าน `fetch()` แทนการคำนวณในเบราว์เซอร์

## สถาปัตยกรรม

```
รูปตัวเอง ──▶ /enroll ──▶ ถอด embedding (128 มิติ) ──▶ เก็บใน profile.json
                                                              │
คลังรูปจากเพจ ──▶ /scan ──▶ ตรวจจับใบหน้าทุกรูป ──▶ เทียบ embedding กับ │
                              (face_locations)         profile ◀────────┘
                                    │
                         ตรงตามเกณฑ์ (tolerance) ──▶ copy ไป /matches ──▶ คืน download_url
                         ไม่ตรง ──▶ ทิ้ง ไม่เก็บ ไม่ส่งกลับ
```

- **โมเดล**: `face_recognition` (dlib ResNet embedding 128 มิติ) — ติดตั้งง่าย แม่นพอสำหรับต้นแบบ/รูปหลักร้อย-พันรูป
- **เกณฑ์ความคล้าย**: ปรับผ่านพารามิเตอร์ `tolerance` ตอนเรียก `/scan` (ค่าเริ่มต้น 0.55 ยิ่งต่ำยิ่งเข้มงวด)
- **เก็บข้อมูล**: ต้นแบบนี้เก็บ embedding เป็นไฟล์ `profile.json` และรูปที่แมตช์เป็นไฟล์ในโฟลเดอร์ `matches/` ธรรมดา — ถ้าจะขยายเป็นระบบจริงที่มีผู้ใช้หลายคน ควรย้ายไป PostgreSQL + pgvector หรือ vector DB อย่าง FAISS/Milvus/Chroma เพื่อค้นหาเร็วขึ้นเมื่อรูปมีจำนวนมาก

## เมื่อรูปมีปริมาณมาก (หลักหมื่นขึ้นไป)

`face_recognition`/dlib จะช้าลงเมื่อข้อมูลเยอะ ทางเลือกสำหรับ production:
- **InsightFace** (ONNX Runtime) — เร็วกว่า dlib หลายเท่า รองรับ GPU
- **Vector database** (FAISS, Milvus, Qdrant) แทนการวน loop เทียบทีละรูปใน `profile.json`
- แยก **queue/worker** (เช่น Celery + Redis) สำหรับสแกนรูปจำนวนมากแบบ background แทนการรอสด ๆ ใน request เดียว

## ข้อควรระวังเรื่องความเป็นส่วนตัว

- รูปจากเพจโรงเรียนมักมีใบหน้าของผู้เยาว์ ควรดึงข้อมูลรูปผ่านช่องทาง/สิทธิ์ที่โรงเรียนหรือเพจอนุญาตจริง ๆ ไม่ใช่การสแครปโดยไม่ได้รับอนุญาต
- เก็บ embedding และรูปเท่าที่จำเป็น และมีกลไกให้ลบข้อมูลได้ตามคำขอ
- ถ้าจะเปิดให้คนอื่นใช้ระบบนี้ (ไม่ใช่แค่ตัวเอง) ควรมีการแจ้งและขอความยินยอมเรื่องการประมวลผลใบหน้า
