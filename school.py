"""
school.py  -  ระบบค้นหารูปของฉันจากคลังรูปโรงเรียน

  แอดมิน:   เปิด /admin ใส่รหัสผ่าน สร้าง "งาน" (เช่น กีฬาสี 2569) แล้วอัปโหลดรูปเข้าแต่ละงาน
            (ระบบตรวจจับใบหน้าและเก็บลักษณะใบหน้าไว้ล่วงหน้าตอนอัปโหลด)
  นักเรียน: เปิดหน้าแรก เลือกงาน (หรือทุกงาน) อัปโหลดรูปตัวเอง 1-3 รูป ระบบคืนรูปของนักเรียนที่อยู่ในคลัง
            แยกหัวข้อตามงาน พร้อมปุ่มดาวน์โหลด (รูปที่ใช้ค้นหา "ไม่ถูกบันทึก" ประมวลผลในหน่วยความจำแล้วทิ้ง)

รัน:  uvicorn school:app --port 8000
รหัสผ่านแอดมินจะถูกสร้างครั้งแรกและเก็บไว้ที่ school_data/admin_password.txt (แก้ไฟล์นี้เพื่อเปลี่ยนรหัส
ใช้ตัวอักษรอังกฤษ/ตัวเลขเท่านั้น)
"""

import datetime
import hashlib
import html
import io
import json
import re
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import List

import face_recognition
import numpy as np
from fastapi import Body, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, ImageOps

try:  # ถ้าติดตั้ง pillow-heif ไว้จะอ่าน HEIC ได้ด้วย ถ้าไม่ได้ติดตั้งก็ไม่เป็นไร
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# ====== ตั้งค่า (แก้ได้) ======
SCHOOL_NAME = "ชื่อโรงเรียนของคุณ"
CONTACT_TEXT = "ถ้าไม่ต้องการให้รูปของคุณอยู่ในระบบนี้ ติดต่อผู้ดูแลเพจเพื่อขอนำรูปออก"
INDEX_MAX_SIDE = 2000    # ย่อรูปคลังก่อนตรวจจับใบหน้า (ทำครั้งเดียวตอนอัปโหลด เลยใช้ค่าสูงหน่อย)
QUERY_MAX_SIDE = 1200    # ย่อรูปที่นักเรียนอัปโหลดค้นหา
LOOSE = 0.65             # เกณฑ์กว้างสุดที่ส่งกลับ หน้าเว็บกรองต่อด้วยแถบเลื่อน
DEFAULT_TOL = 0.48       # ค่าเริ่มต้นของแถบความเข้มงวดฝั่งนักเรียน
MAX_QUERY_PHOTOS = 3
RATE_LIMIT = 15          # ค้นหาได้กี่ครั้ง ต่อ IP
RATE_WINDOW = 600        # ในช่วงกี่วินาที (600 = 10 นาที)
# ==============================

BASE = Path(__file__).parent / "school_data"
PHOTOS_DIR = BASE / "photos"
THUMBS_DIR = BASE / "thumbs"
INDEX_FILE = BASE / "index.jsonl"
EVENTS_FILE = BASE / "events.json"
PW_FILE = BASE / "admin_password.txt"
for d in (PHOTOS_DIR, THUMBS_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="school-photo-finder")

# ---------- รหัสผ่านแอดมิน ----------
def _load_admin_password() -> str:
    if PW_FILE.exists():
        return PW_FILE.read_text(encoding="utf-8").strip()
    pw = secrets.token_urlsafe(6)
    PW_FILE.write_text(pw, encoding="utf-8")
    return pw

ADMIN_PW = _load_admin_password()
print("=" * 54)
print(" รหัสผ่านผู้ดูแล (ใช้ที่หน้า /admin):", ADMIN_PW)
print(" เก็บอยู่ที่ไฟล์:", PW_FILE)
print("=" * 54)


def require_admin(pw):
    if not pw or not secrets.compare_digest(pw.encode(), ADMIN_PW.encode()):
        raise HTTPException(401, "รหัสผ่านไม่ถูกต้อง")


# ---------- ฐานข้อมูลลักษณะใบหน้า (เก็บในไฟล์ jsonl + โหลดขึ้นหน่วยความจำ) ----------
LOCK = threading.Lock()
ENTRIES = {}      # photo_id -> {"id","name","sha1","event","encodings":[[128 ค่า],...]}
SHAS = set()
EVENTS = {}       # event_id -> {"id","name","created"}   (รูปที่ไม่มีงาน = event "" แสดงเป็น "ไม่ระบุงาน")
NO_EVENT = "ไม่ระบุงาน"
_cache = {"dirty": True, "G": None, "OWNER": None, "IDS": []}


def _load_events():
    if EVENTS_FILE.exists():
        for e in json.loads(EVENTS_FILE.read_text(encoding="utf-8")):
            if not e.get("date"):  # อัลบั้มเก่าที่ยังไม่มีวันที่ ใช้วันที่สร้างแทน
                e["date"] = datetime.date.fromtimestamp(e.get("created", time.time())).isoformat()
            EVENTS[e["id"]] = e


def _save_events():
    EVENTS_FILE.write_text(json.dumps(list(EVENTS.values()), ensure_ascii=False, indent=1), encoding="utf-8")


def _load_index():
    if not INDEX_FILE.exists():
        return
    for line in INDEX_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            e = json.loads(line)
            if (PHOTOS_DIR / f"{e['id']}.jpg").exists():
                if e.get("event") not in EVENTS:
                    e["event"] = ""          # รูปเก่าที่ไม่มีงาน
                ENTRIES[e["id"]] = e
                SHAS.add(e["sha1"])


_load_events()
_load_index()


def _rewrite_index():
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        for e in ENTRIES.values():
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def _snapshot():
    """สร้างเมทริกซ์ลักษณะใบหน้าทั้งหมดใหม่ถ้ามีการเพิ่ม/ลบรูป"""
    with LOCK:
        if _cache["dirty"]:
            ids = list(ENTRIES.keys())
            encs, owner = [], []
            for i, pid in enumerate(ids):
                for e in ENTRIES[pid]["encodings"]:
                    encs.append(e)
                    owner.append(i)
            _cache["G"] = np.array(encs, dtype=np.float64) if encs else np.zeros((0, 128))
            _cache["OWNER"] = np.array(owner, dtype=np.int64)
            _cache["IDS"] = ids
            _cache["dirty"] = False
        return _cache["G"], _cache["OWNER"], _cache["IDS"]


# ---------- ตัวช่วย ----------
def open_image(raw: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(raw))
        return ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        raise HTTPException(400, "อ่านไฟล์รูปไม่ได้ ถ้าเป็นไฟล์ HEIC กรุณาแปลงเป็น JPG ก่อน")


def valid_id(pid: str):
    if not re.fullmatch(r"[0-9a-f]{32}", pid):
        raise HTTPException(404, "ไม่พบไฟล์")


HITS = defaultdict(deque)


def check_rate(ip: str):
    now = time.time()
    q = HITS[ip]
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        raise HTTPException(429, "ค้นหาถี่เกินไป กรุณารอสักครู่แล้วลองใหม่")
    q.append(now)


def _parse_date(value) -> str:
    """รับวันที่รูปแบบ YYYY-MM-DD (ว่าง = วันนี้) คืนสตริง ISO"""
    if not value:
        return datetime.date.today().isoformat()
    try:
        return datetime.date.fromisoformat(str(value)).isoformat()
    except ValueError:
        raise HTTPException(400, "วันที่ไม่ถูกต้อง")


def _events_list(include_empty: bool):
    """รายการงาน (ใหม่สุดก่อน) พร้อมจำนวนรูป; รูปที่ไม่มีงานใช้ id = "none" ใน API"""
    counts = defaultdict(int)
    for e in ENTRIES.values():
        counts[e["event"]] += 1
    out = [{"id": ev["id"], "name": ev["name"], "date": ev["date"], "count": counts[ev["id"]]}
           for ev in sorted(EVENTS.values(), key=lambda x: (x["date"], x["created"]), reverse=True)
           if include_empty or counts[ev["id"]] > 0]
    if counts[""]:
        out.append({"id": "none", "name": NO_EVENT, "date": "", "count": counts[""]})
    return out


def _api_event(eid: str) -> str:
    return "none" if eid == "" else eid


def _event_name(eid: str) -> str:
    return EVENTS[eid]["name"] if eid in EVENTS else NO_EVENT


def _internal_event(api_id: str) -> str:
    return "" if api_id == "none" else api_id


# ---------- ฝั่งแอดมิน ----------
@app.post("/admin/login")
def admin_login(x_admin_password: str = Header(None)):
    require_admin(x_admin_password)
    return {"ok": True}


@app.post("/admin/upload")
def admin_upload(file: UploadFile = File(...), event: str = Form(...), x_admin_password: str = Header(None)):
    require_admin(x_admin_password)
    if event not in EVENTS:
        raise HTTPException(400, "ไม่พบงานนี้ กรุณาเลือกงานที่จะอัปโหลดเข้า")
    raw = file.file.read()
    sha = hashlib.sha1(raw).hexdigest()
    if sha in SHAS:
        return {"status": "duplicate"}

    pil = open_image(raw)
    det = pil.copy()
    det.thumbnail((INDEX_MAX_SIDE, INDEX_MAX_SIDE))
    arr = np.array(det)
    locs = face_recognition.face_locations(arr)
    if not locs:
        locs = face_recognition.face_locations(arr, number_of_times_to_upsample=2)
    if not locs:
        return {"status": "no_face"}
    encs = face_recognition.face_encodings(arr, locs, model="large")

    pid = uuid.uuid4().hex
    pil.save(PHOTOS_DIR / f"{pid}.jpg", "JPEG", quality=92)
    th = pil.copy()
    th.thumbnail((480, 480))
    th.save(THUMBS_DIR / f"{pid}.jpg", "JPEG", quality=80)

    entry = {"id": pid, "name": file.filename or pid, "sha1": sha, "event": event,
             "encodings": [e.tolist() for e in encs]}
    with LOCK:
        ENTRIES[pid] = entry
        SHAS.add(sha)
        with open(INDEX_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        _cache["dirty"] = True
    return {"status": "added", "faces": len(encs)}


@app.get("/admin/events")
def admin_events(x_admin_password: str = Header(None)):
    require_admin(x_admin_password)
    return {"events": _events_list(include_empty=True)}


@app.post("/admin/event")
def create_event(payload: dict = Body(...), x_admin_password: str = Header(None)):
    require_admin(x_admin_password)
    name = str(payload.get("name", "")).strip()[:80]
    if not name:
        raise HTTPException(400, "ใส่ชื่ออัลบั้มก่อน")
    date = _parse_date(payload.get("date"))
    eid = uuid.uuid4().hex[:12]
    with LOCK:
        EVENTS[eid] = {"id": eid, "name": name, "date": date, "created": time.time()}
        _save_events()
    return EVENTS[eid]


@app.put("/admin/event/{eid}")
def update_event(eid: str, payload: dict = Body(...), x_admin_password: str = Header(None)):
    require_admin(x_admin_password)
    if eid not in EVENTS:
        raise HTTPException(404, "ไม่พบอัลบั้มนี้")
    name = str(payload.get("name", "")).strip()[:80]
    date = _parse_date(payload["date"]) if payload.get("date") else ""
    if not name and not date:
        raise HTTPException(400, "ข้อมูลไม่ถูกต้อง")
    with LOCK:
        if name:
            EVENTS[eid]["name"] = name
        if date:
            EVENTS[eid]["date"] = date
        _save_events()
    return EVENTS[eid]


@app.delete("/admin/event/{eid}")
def delete_event(eid: str, x_admin_password: str = Header(None)):
    """ลบงาน พร้อมรูปทั้งหมดในงานนั้น (ทั้งไฟล์รูปและข้อมูลใบหน้า)"""
    require_admin(x_admin_password)
    if eid not in EVENTS:
        raise HTTPException(404, "ไม่พบงานนี้")
    with LOCK:
        pids = [p for p, e in ENTRIES.items() if e["event"] == eid]
        for p in pids:
            SHAS.discard(ENTRIES.pop(p)["sha1"])
        EVENTS.pop(eid)
        _save_events()
        _rewrite_index()
        _cache["dirty"] = True
    for p in pids:
        for f in (PHOTOS_DIR / f"{p}.jpg", THUMBS_DIR / f"{p}.jpg"):
            if f.exists():
                f.unlink()
    return {"deleted_photos": len(pids)}


@app.get("/admin/photos")
def admin_photos(event: str = "", x_admin_password: str = Header(None)):
    """event: "" = ทุกงาน, "none" = ไม่ระบุงาน, หรือ id ของงาน"""
    require_admin(x_admin_password)
    items = ENTRIES.values()
    if event:
        target = _internal_event(event)
        items = [e for e in items if e["event"] == target]
    return [{"id": e["id"], "name": e["name"], "faces": len(e["encodings"])} for e in items]


@app.post("/admin/photo/{pid}/move")
def admin_move(pid: str, payload: dict = Body(...), x_admin_password: str = Header(None)):
    require_admin(x_admin_password)
    valid_id(pid)
    target = str(payload.get("event", ""))
    if pid not in ENTRIES or target not in EVENTS:
        raise HTTPException(400, "ข้อมูลไม่ถูกต้อง")
    with LOCK:
        ENTRIES[pid]["event"] = target
        _rewrite_index()
    return {"ok": True}


@app.delete("/admin/photo/{pid}")
def admin_delete(pid: str, x_admin_password: str = Header(None)):
    require_admin(x_admin_password)
    valid_id(pid)
    with LOCK:
        e = ENTRIES.pop(pid, None)
        if e:
            SHAS.discard(e["sha1"])
            _rewrite_index()
            _cache["dirty"] = True
    for p in (PHOTOS_DIR / f"{pid}.jpg", THUMBS_DIR / f"{pid}.jpg"):
        if p.exists():
            p.unlink()
    return {"deleted": bool(e)}


# ---------- ฝั่งนักเรียน ----------
@app.get("/api/events")
def public_events():
    return {"total": len(ENTRIES), "events": _events_list(include_empty=False)}


@app.post("/api/search")
def search(request: Request, files: List[UploadFile] = File(...), event: str = "", consent: str = Form("")):
    if consent != "1":
        raise HTTPException(400, "กรุณาติ๊กยินยอมก่อนค้นหา")
    check_rate(request.client.host if request.client else "unknown")
    if len(files) > MAX_QUERY_PHOTOS:
        raise HTTPException(400, f"เลือกรูปได้ไม่เกิน {MAX_QUERY_PHOTOS} รูป")

    queries = []
    for f in files:
        pil = open_image(f.file.read())
        pil.thumbnail((QUERY_MAX_SIDE, QUERY_MAX_SIDE))
        faces = face_recognition.face_encodings(np.array(pil), num_jitters=3, model="large")
        if not faces:
            raise HTTPException(400, "ไม่พบใบหน้าในรูป ลองใช้รูปที่เห็นหน้าตรงและชัดกว่านี้")
        if len(faces) > 1:
            raise HTTPException(400, "พบหลายใบหน้าในรูป กรุณาใช้รูปที่มีเราคนเดียว")
        queries.append(faces[0])
    # รูปที่นักเรียนอัปโหลดอยู่ในหน่วยความจำเท่านั้น ไม่ถูกเขียนลงดิสก์

    G, OWNER, IDS = _snapshot()
    if not IDS:
        return {"total_photos": 0, "results": []}

    best = np.full(len(IDS), np.inf)
    for q in queries:
        np.minimum.at(best, OWNER, np.linalg.norm(G - q, axis=1))
    idx = np.where(best <= LOOSE)[0]
    if event:  # จำกัดเฉพาะงานที่เลือก
        target = _internal_event(event)
        idx = np.array([i for i in idx if ENTRIES[IDS[i]]["event"] == target], dtype=np.int64)
    idx = idx[np.argsort(best[idx])]
    results = [{"id": IDS[i], "distance": round(float(best[i]), 3),
                "event": _api_event(ENTRIES[IDS[i]]["event"]), "event_name": _event_name(ENTRIES[IDS[i]]["event"]),
                "thumb": f"/thumb/{IDS[i]}", "download": f"/photo/{IDS[i]}"} for i in idx]
    return {"total_photos": len(IDS), "results": results}


@app.get("/thumb/{pid}")
def thumb(pid: str):
    valid_id(pid)
    p = THUMBS_DIR / f"{pid}.jpg"
    if not p.exists():
        raise HTTPException(404, "ไม่พบไฟล์")
    return FileResponse(p)


@app.get("/photo/{pid}")
def photo(pid: str):
    valid_id(pid)
    p = PHOTOS_DIR / f"{pid}.jpg"
    if not p.exists():
        raise HTTPException(404, "ไม่พบไฟล์")
    stem = Path(ENTRIES[pid]["name"]).stem if pid in ENTRIES else pid
    return FileResponse(p, filename=f"{stem}.jpg")


# ---------- หน้าเว็บ ----------
STYLE = r"""
body{margin:0;background:#11141b;color:#eee7d8;font-family:'Segoe UI',Tahoma,sans-serif}
.w{max-width:800px;margin:0 auto;padding:28px 18px 60px}
h1{font-size:26px;margin:0 0 6px}
.box{background:#171b24;border:1px solid #2a2f3b;border-radius:10px;padding:16px;margin:14px 0}
h2{font-size:16px;color:#c99a4a;margin:0 0 6px}
p{font-size:13.5px;color:#9aa1af;margin:0 0 10px;line-height:1.7}
button{padding:11px 18px;border:0;border-radius:8px;background:#c99a4a;color:#1a1408;font-weight:600;cursor:pointer;font-size:14px}
button:disabled{opacity:.4;cursor:not-allowed}
button.sm{padding:5px 10px;font-size:12px}
button.del,button.del2{background:#7a2f2f;color:#fff}
button.del{padding:5px 8px;font-size:12px;margin-top:6px;width:100%}
button.link{background:none;color:#c99a4a;padding:0;margin-bottom:14px;font-size:14px}
button.big{padding:16px 46px;font-size:19px;border-radius:12px}
input[type=file]{color:#9aa1af;font-size:13px}
input[type=password],input[type=text],input[type=date],select{padding:9px;background:#11141b;border:1px solid #2a2f3b;border-radius:6px;color:#eee7d8;font-size:13px;max-width:100%;color-scheme:dark}
input[type=range]{width:220px;vertical-align:middle;accent-color:#c99a4a}
.bar{display:none;margin-top:16px}
.track{height:22px;background:#2a2f3b;border-radius:11px;overflow:hidden;position:relative}
.fill{height:100%;width:0;background:#c99a4a;transition:width .2s}
.pct{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:12.5px;font-weight:700;color:#fff;text-shadow:0 0 3px #000}
.st{font-size:13.5px;color:#9aa1af;margin-top:8px;line-height:1.7}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px;margin-top:12px}
.card{background:#11141b;border:1px solid #2a2f3b;border-radius:8px;overflow:hidden}
.card img{width:100%;height:140px;object-fit:cover;display:block}
.card div{padding:8px;font-size:12.5px;color:#9aa1af;word-break:break-all}
.card a{display:block;margin-top:6px;padding:6px;text-align:center;background:#c99a4a;color:#1a1408;border-radius:6px;text-decoration:none;font-weight:600}
.card select{width:100%;margin-top:6px;padding:5px;font-size:12px}
.evrow{display:flex;align-items:center;flex-wrap:wrap;gap:8px;padding:10px 0;border-bottom:1px solid #2a2f3b;font-size:14px}
.evname{flex:1;min-width:160px}
.note{font-size:12.5px;color:#9aa1af;line-height:1.7;margin-top:22px;padding-top:14px;border-top:1px solid #2a2f3b}
.hero{text-align:center;padding:50px 10px 20px}
.hero h1{font-size:30px;line-height:1.4;margin-bottom:10px}
.steps{max-width:420px;margin:26px auto;text-align:left}
.step{display:flex;gap:12px;align-items:center;margin:12px 0;font-size:15px}
.step i{flex-shrink:0;width:30px;height:30px;border-radius:50%;background:#c99a4a;color:#1a1408;display:flex;align-items:center;justify-content:center;font-style:normal;font-weight:700}
.cal{background:#171b24;border:1px solid #2a2f3b;border-radius:10px;padding:12px;margin:14px 0}
.calhead{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;font-weight:600}
.calhead button{padding:4px 14px}
.calgrid{display:grid;grid-template-columns:repeat(7,1fr);gap:4px;text-align:center}
.dn{font-size:11.5px;color:#9aa1af;padding:4px 0}
.day{padding:8px 0;border-radius:6px;font-size:13px;color:#4f5666;position:relative}
.day.has{color:#eee7d8;background:#232938;cursor:pointer;font-weight:600}
.day.has::after{content:"";position:absolute;bottom:3px;left:50%;width:5px;height:5px;margin-left:-2.5px;border-radius:50%;background:#c99a4a}
.day.sel{background:#c99a4a;color:#1a1408}
.day.sel::after{background:#1a1408}
.day.today{outline:1px solid #c99a4a}
.chip{display:inline-block;background:#232938;border:1px solid #c99a4a;border-radius:16px;padding:4px 12px;font-size:12.5px;margin:0 8px 10px 0}
.chip b{cursor:pointer;margin-left:6px;color:#c99a4a}
.album{display:flex;align-items:center;gap:14px;padding:12px;background:#171b24;border:1px solid #2a2f3b;border-radius:10px;margin-bottom:10px;cursor:pointer}
.album:hover{border-color:#c99a4a}
.badge{width:54px;text-align:center;background:#11141b;border:1px solid #2a2f3b;border-radius:8px;padding:6px 0;flex-shrink:0}
.badge b{display:block;font-size:20px;color:#c99a4a;line-height:1.15}
.badge span{font-size:11px;color:#9aa1af}
.ainfo{flex:1;min-width:0}
.aname{font-size:15px;font-weight:600}
.asub{font-size:12.5px;color:#9aa1af;margin-top:2px}
.arrow{color:#c99a4a;font-size:22px}
"""

STUDENT_PAGE = r"""<!DOCTYPE html>
<html lang="th"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ค้นหารูปของฉัน - __SCHOOL__</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Prompt:wght@500;600&family=Sarabun:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{--navy:#0f2557;--blue:#2456c9;--sky:#eaf1ff;--line:#d3def4;--ink:#0d1b3e;--mute:#586a8f;--bg:#f5f8ff}
*{box-sizing:border-box}[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--ink);font:400 16px/1.65 'Sarabun',Tahoma,sans-serif}
h1,h2,.cta,.steps,.gt{font-family:'Prompt','Sarabun',sans-serif}
button{font:inherit;cursor:pointer}
:focus-visible{outline:3px solid #8fb0ff;outline-offset:2px}
.top{background:var(--navy);color:#fff;padding:28px 18px 40px}
.in,.w{max-width:720px;margin:0 auto}
.school{font-size:14px;opacity:.75;margin-bottom:4px}
.top h1{font-size:clamp(26px,6vw,36px);line-height:1.3;margin:0 0 10px;font-weight:600}
.lead{margin:0 0 16px;color:#cfdcff;max-width:34em}
.priv{display:flex;gap:10px;align-items:center;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.25);border-radius:12px;padding:10px 14px;font-size:14.5px}
.priv svg{flex:none}
.w{padding:0 16px 60px}
.steps{list-style:none;display:flex;gap:6px;margin:-20px 0 22px;padding:0;position:relative}
.steps li{flex:1;background:#fff;border:1px solid var(--line);border-radius:12px;padding:8px 10px;font-size:13px;color:var(--mute);display:flex;align-items:center;gap:8px}
.steps li b{width:22px;height:22px;border-radius:50%;background:var(--sky);color:var(--blue);display:grid;place-items:center;font-size:12px;flex:none}
.steps li.on{border-color:var(--blue);color:var(--ink);font-weight:600}
.steps li.on b,.steps li.done b{background:var(--blue);color:#fff}
.steps li.done{color:var(--ink)}
@media(max-width:480px){.steps li span{display:none}.steps li.on span{display:inline}}
h2{font-size:21px;margin:0 0 4px;font-weight:600}
.sub{color:var(--mute);margin:0 0 14px}
.link{background:none;border:0;color:var(--blue);padding:6px 0;font-weight:600;margin:0 0 6px}
.allbtn{width:100%;display:flex;align-items:center;justify-content:space-between;text-align:left;background:var(--blue);color:#fff;border:0;border-radius:16px;padding:18px 20px;margin:8px 0 4px;box-shadow:0 8px 20px -10px rgba(36,86,201,.7)}
.allbtn b{display:block;font:600 18px 'Prompt',sans-serif}
.allbtn small{color:#d6e2ff;font-size:14px}
.allbtn i{font-style:normal;font-size:28px}
.sep{text-align:center;color:var(--mute);font-size:14px;margin:18px 0 10px}
input[type=search]{width:100%;padding:13px 16px;border:1px solid var(--line);border-radius:12px;font:inherit;background:#fff;color:var(--ink);margin-bottom:6px}
.cal{background:#fff;border:1px solid var(--line);border-radius:14px;padding:12px;margin:8px 0 12px}
.calhead{display:flex;justify-content:space-between;align-items:center;font-weight:600;margin-bottom:6px}
.calhead button{background:var(--sky);color:var(--blue);border:0;border-radius:8px;width:36px;height:32px;font-size:18px}
.calgrid{display:grid;grid-template-columns:repeat(7,1fr);gap:4px;text-align:center}
.dn{font-size:12px;color:var(--mute)}
.day{padding:8px 0;border-radius:8px;font-size:14px;color:#a8b4cf;position:relative}
.day.has{color:var(--ink);background:var(--sky);font-weight:600;cursor:pointer}
.day.has::after{content:"";position:absolute;bottom:3px;left:50%;width:5px;height:5px;margin-left:-2.5px;border-radius:50%;background:var(--blue)}
.day.sel{background:var(--blue);color:#fff}.day.sel::after{background:#fff}
.day.today{outline:1px solid var(--blue)}
.chip{display:inline-flex;align-items:center;gap:6px;background:var(--sky);border:1px solid var(--blue);color:var(--navy);border-radius:999px;padding:3px 6px 3px 14px;font-size:14px;margin:0 8px 10px 0}
.chip button{background:none;border:0;color:var(--blue);font-size:16px;padding:0 6px}
.album{display:flex;align-items:center;gap:14px;width:100%;text-align:left;padding:12px;background:#fff;border:1px solid var(--line);border-radius:14px;margin-bottom:10px;color:inherit}
.album:hover{border-color:var(--blue)}
.badge{width:56px;flex:none;text-align:center;background:var(--sky);border-radius:10px;padding:6px 0;color:var(--blue)}
.badge b{display:block;font:600 20px/1.15 'Prompt',sans-serif}.badge span{font-size:11.5px;color:var(--mute)}
.aname{font-weight:600}.asub{font-size:13.5px;color:var(--mute)}
.arrow{margin-left:auto;color:var(--blue);font-size:24px}
.panel{background:#fff;border:1px solid var(--line);border-radius:18px;padding:16px;margin:8px 0 18px}
.drop{display:flex;flex-direction:column;align-items:center;text-align:center;gap:4px;padding:30px 16px;background:var(--sky);border-radius:14px;cursor:pointer;position:relative;color:var(--navy)}
.drop i{position:absolute;width:26px;height:26px;border:3px solid var(--blue)}
.drop i.a{top:10px;left:10px;border-right:0;border-bottom:0;border-radius:8px 0 0 0}
.drop i.b{top:10px;right:10px;border-left:0;border-bottom:0;border-radius:0 8px 0 0}
.drop i.c{bottom:10px;left:10px;border-right:0;border-top:0;border-radius:0 0 0 8px}
.drop i.d{bottom:10px;right:10px;border-left:0;border-top:0;border-radius:0 0 8px 0}
.drop b{font:600 17px 'Prompt',sans-serif}.drop span{font-size:14px;color:var(--mute)}
.drop.over{background:#dbe8ff}
.prev{display:flex;gap:12px;flex-wrap:wrap;margin-top:14px}
.pv{position:relative}.pv img{width:72px;height:72px;border-radius:50%;object-fit:cover;border:3px solid var(--blue);display:block}
.pv button{position:absolute;top:-4px;right:-4px;width:24px;height:24px;border-radius:50%;border:2px solid #fff;background:var(--navy);color:#fff;font-size:12px;line-height:1;padding:0}
.consent{display:flex;gap:12px;align-items:flex-start;margin:16px 0;padding:12px 14px;border:1px solid var(--line);border-radius:12px;font-size:14.5px;cursor:pointer}
.consent input{width:22px;height:22px;flex:none;margin-top:2px;accent-color:var(--blue)}
.cta{width:100%;padding:15px;border:0;border-radius:14px;background:var(--blue);color:#fff;font-size:17px;font-weight:600;display:block;text-align:center;text-decoration:none}
.cta:disabled{background:#b9c7e6;cursor:not-allowed}
.cta.busy::after{content:"";display:inline-block;width:14px;height:14px;margin-left:10px;border:2px solid #fff;border-right-color:transparent;border-radius:50%;vertical-align:-2px;animation:sp .8s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
.msg{margin-top:10px;font-size:14.5px;color:var(--mute)}.msg.err{color:#b3261e}
.tool{position:sticky;top:0;z-index:5;background:rgba(245,248,255,.95);backdrop-filter:blur(6px);padding:10px 0;border-bottom:1px solid var(--line);margin-bottom:12px}
.tl{display:flex;align-items:center;gap:10px;font-size:14px}
.tl input{flex:1;accent-color:var(--blue)}
.hint{display:flex;justify-content:space-between;font-size:12.5px;color:var(--mute)}
.rh{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin:8px 0}
.ghost{background:#fff;color:var(--blue);border:1px solid var(--blue);border-radius:10px;padding:8px 14px;font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px;margin-bottom:14px}
.card{background:#fff;border:1px solid var(--line);border-radius:12px;overflow:hidden}
.card button{display:block;width:100%;padding:0;border:0;background:none}
.card img{width:100%;aspect-ratio:1;object-fit:cover;display:block}
.card a{display:block;text-align:center;padding:9px;color:var(--blue);font-weight:600;text-decoration:none;font-size:14.5px}
.empty{text-align:center;padding:26px 10px;color:var(--mute)}
.gt{font-size:16px;font-weight:600;margin:18px 0 8px;color:var(--navy)}
.note{font-size:13px;color:var(--mute);margin-top:22px;padding-top:14px;border-top:1px solid var(--line)}
.lb{position:fixed;inset:0;background:rgba(9,20,50,.92);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;padding:16px;z-index:20}
.lb img{max-width:100%;max-height:74vh;border-radius:10px}
.lb .cta{width:auto;padding:12px 28px}
.lbx{position:absolute;top:14px;right:14px;background:#fff;border:0;border-radius:50%;width:40px;height:40px;font-size:18px}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style></head><body>

<header class="top"><div class="in">
  <div class="school">__SCHOOL__</div>
  <h1>ค้นหารูปของคุณ<br>ในอัลบั้มโรงเรียน</h1>
  <p class="lead">อัปโหลดรูปหน้าตัวเอง ระบบจะหารูปที่มีคุณให้ และดาวน์โหลดได้ทันที</p>
  <div class="priv"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/><path d="M9 12l2 2 4-4"/></svg>
  <span>รูปที่คุณอัปโหลดจะ<b>ไม่ถูกเก็บไว้</b> ระบบใช้ค้นหาเสร็จแล้วทิ้งทันที</span></div>
</div></header>

<div class="w">
<ol class="steps" id="steps">
  <li><b>1</b><span>เลือกอัลบั้ม</span></li>
  <li><b>2</b><span>อัปโหลดรูป</span></li>
  <li><b>3</b><span>รับรูปของคุณ</span></li>
</ol>

<section id="s1">
  <h2>เลือกอัลบั้ม</h2>
  <p class="sub">ไม่แน่ใจว่าอยู่งานไหน ให้ค้นหาจากทุกอัลบั้มได้เลย</p>
  <button class="allbtn" id="allalb"><span><b>ค้นหาจากทุกอัลบั้ม</b><small id="allsub"></small></span><i>&rsaquo;</i></button>
  <div class="sep">หรือเลือกอัลบั้มเดียว</div>
  <input type="search" id="q" placeholder="พิมพ์ชื่ออัลบั้ม เช่น กีฬาสี">
  <button class="link" id="caltog">เลือกจากปฏิทิน</button>
  <div class="cal" id="calbox" hidden>
    <div class="calhead"><button id="pm" aria-label="เดือนก่อน">&lsaquo;</button><span id="calt"></span><button id="nm" aria-label="เดือนถัดไป">&rsaquo;</button></div>
    <div class="calgrid" id="calg"></div>
    <p class="sub" style="margin:8px 0 0;font-size:13px">วันที่มีจุดสีน้ำเงินคือวันที่มีอัลบั้ม</p>
  </div>
  <div id="chips"></div>
  <div id="albums"></div>
</section>

<section id="s2" hidden>
  <button class="link" id="back">&lsaquo; เปลี่ยนอัลบั้ม</button>
  <h2 id="atitle"></h2><p class="sub" id="asubt"></p>

  <div class="panel">
    <label class="drop" id="drop" for="me"><i class="a"></i><i class="b"></i><i class="c"></i><i class="d"></i>
      <b>แตะเพื่อเลือกรูปหน้าตัวเอง</b>
      <span>รูปหน้าตรง ชัด มีคุณคนเดียว 1-3 รูป (JPG / PNG / WEBP)</span>
    </label>
    <input type="file" id="me" accept="image/jpeg,image/png,image/webp" multiple hidden>
    <div class="prev" id="prev"></div>

    <label class="consent"><input type="checkbox" id="ok"><span>ฉันยินยอมให้ระบบใช้รูปนี้ค้นหาใบหน้าของฉันในอัลบั้ม และยืนยันว่าเป็นรูปของตัวเอง (หากอายุต่ำกว่า 18 ปี ได้รับอนุญาตจากผู้ปกครองแล้ว) ฉันรับทราบว่า<b>รูปที่อัปโหลดจะไม่ถูกเก็บไว้</b> ระบบประมวลผลชั่วคราวแล้วทิ้งทันที</span></label>
    <button class="cta" id="go" disabled>สแกนหารูปของฉัน</button>
    <div class="msg" id="st" role="status"></div>
  </div>

  <div id="resbox" hidden>
    <div class="tool">
      <div class="tl"><span>ความเข้มงวด</span><input type="range" id="tv" min="0.35" max="0.65" step="0.01" value="__DEFTOL__" aria-label="ความเข้มงวด"></div>
      <div class="hint"><span>เข้มงวด: ตรงกว่า</span><span>ผ่อนปรน: เจอมากขึ้น</span></div>
    </div>
    <div class="rh"><b id="count"></b><button class="ghost" id="dlall" hidden></button></div>
    <div id="out"></div>
  </div>
  <p class="note">__CONTACT__</p>
</section>
</div>

<div class="lb" id="lb" hidden><button class="lbx" id="lbx" aria-label="ปิด">&#10005;</button><img id="lbi" alt=""><a class="cta" id="lba" href="#">ดาวน์โหลดรูปนี้</a></div>

<script>
const $ = id => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if(cls) e.className = cls; if(text !== undefined) e.textContent = text; return e; };
const TH_M = ["มกราคม","กุมภาพันธ์","มีนาคม","เมษายน","พฤษภาคม","มิถุนายน","กรกฎาคม","สิงหาคม","กันยายน","ตุลาคม","พฤศจิกายน","ธันวาคม"];
const TH_S = ["ม.ค.","ก.พ.","มี.ค.","เม.ย.","พ.ค.","มิ.ย.","ก.ค.","ส.ค.","ก.ย.","ต.ค.","พ.ย.","ธ.ค."];
const TH_D = ["อา","จ","อ","พ","พฤ","ศ","ส"];
const pad = n => String(n).padStart(2, "0");
const fmtDate = iso => { const p = iso.split("-"); return (+p[2]) + " " + TH_S[+p[1] - 1] + " " + (+p[0] + 543); };

let events = [], total = 0, results = [], sel = null, filterDate = "", files = [], lastShown = [];
let calY = new Date().getFullYear(), calM = new Date().getMonth();

function setStep(n){ Array.from($("steps").children).forEach((li, i) => { li.classList.toggle("on", i + 1 === n); li.classList.toggle("done", i + 1 < n); }); }
function show(n){ $("s1").hidden = n !== 1; $("s2").hidden = n !== 2; setStep(n); window.scrollTo(0, 0); }

async function loadEvents(){
  try{
    const j = await (await fetch("/api/events")).json();
    events = j.events; total = j.total;
    const dated = events.filter(e => e.date).map(e => e.date).sort();
    if(dated.length){ const last = dated[dated.length - 1].split("-"); calY = +last[0]; calM = +last[1] - 1; }
    $("allsub").textContent = "รวม " + total + " รูป จาก " + events.length + " อัลบั้ม";
    renderCal(); renderAlbums();
  }catch(e){ $("allsub").textContent = "โหลดรายการอัลบั้มไม่ได้ ลองรีเฟรชหน้านี้"; }
}

function renderCal(){
  $("calt").textContent = TH_M[calM] + " " + (calY + 543);
  const g = $("calg"); g.innerHTML = "";
  for(const d of TH_D) g.appendChild(el("div", "dn", d));
  const first = new Date(calY, calM, 1).getDay(), days = new Date(calY, calM + 1, 0).getDate();
  const have = new Set(events.map(e => e.date).filter(Boolean));
  const now = new Date(), todayIso = now.getFullYear() + "-" + pad(now.getMonth() + 1) + "-" + pad(now.getDate());
  for(let i = 0; i < first; i++) g.appendChild(el("div"));
  for(let d = 1; d <= days; d++){
    const iso = calY + "-" + pad(calM + 1) + "-" + pad(d);
    const c = el("div", "day", String(d));
    if(have.has(iso)){
      c.classList.add("has");
      if(iso === filterDate) c.classList.add("sel");
      c.onclick = () => { filterDate = (filterDate === iso) ? "" : iso; renderCal(); renderAlbums(); };
    }
    if(iso === todayIso) c.classList.add("today");
    g.appendChild(c);
  }
}
$("pm").onclick = () => { calM--; if(calM < 0){ calM = 11; calY--; } renderCal(); };
$("nm").onclick = () => { calM++; if(calM > 11){ calM = 0; calY++; } renderCal(); };
$("caltog").onclick = () => { const h = $("calbox").hidden; $("calbox").hidden = !h; $("caltog").textContent = h ? "ซ่อนปฏิทิน" : "เลือกจากปฏิทิน"; };

function renderAlbums(){
  const q = $("q").value.trim().toLowerCase();
  const list = events.filter(e => (!q || e.name.toLowerCase().includes(q)) && (!filterDate || e.date === filterDate));
  $("chips").innerHTML = "";
  if(filterDate){
    const chip = el("span", "chip", "วันที่ " + fmtDate(filterDate));
    const x = el("button", "", "\u2715"); x.setAttribute("aria-label", "ล้างวันที่");
    x.onclick = () => { filterDate = ""; renderCal(); renderAlbums(); };
    chip.appendChild(x); $("chips").appendChild(chip);
  }
  const box = $("albums"); box.innerHTML = "";
  if(!list.length){ box.appendChild(el("p", "empty", events.length ? "ไม่พบอัลบั้มที่ตรงกัน ลองล้างคำค้นหรือวันที่ หรือกด “ค้นหาจากทุกอัลบั้ม”" : "ยังไม่มีอัลบั้มในระบบ")); return; }
  for(const e of list){
    const a = el("button", "album");
    const badge = el("div", "badge");
    if(e.date){ const p = e.date.split("-"); badge.append(el("b", "", String(+p[2])), el("span", "", TH_S[+p[1] - 1] + " " + String((+p[0] + 543) % 100))); }
    else badge.append(el("b", "", "-"), el("span", "", "ไม่ระบุ"));
    const info = el("div");
    info.append(el("div", "aname", e.name), el("div", "asub", (e.date ? fmtDate(e.date) + " \u00b7 " : "") + e.count + " รูป"));
    a.append(badge, info, el("span", "arrow", "\u203a"));
    a.onclick = () => pick(e);
    box.appendChild(a);
  }
}
$("q").oninput = renderAlbums;

function pick(e){
  sel = e || {id: "", name: "ทุกอัลบั้ม", date: "", count: total};
  $("atitle").textContent = sel.name;
  $("asubt").textContent = sel.id === "" ? "ค้นหาจากทุกอัลบั้มพร้อมกัน" : ((sel.date ? fmtDate(sel.date) + " \u00b7 " : "") + sel.count + " รูปในอัลบั้ม");
  files = []; renderPrev(); $("ok").checked = false; updateGo();
  $("st").textContent = ""; $("st").className = "msg"; $("resbox").hidden = true; results = [];
  show(2);
}
$("allalb").onclick = () => pick(null);
$("back").onclick = () => show(1);

function updateGo(){ $("go").disabled = !(files.length && $("ok").checked); }
function renderPrev(){
  $("prev").innerHTML = "";
  files.forEach((f, i) => {
    const w = el("div", "pv"), im = el("img");
    im.src = URL.createObjectURL(f); im.alt = "รูปที่เลือก " + (i + 1);
    const x = el("button", "", "\u2715"); x.type = "button"; x.setAttribute("aria-label", "เอารูปนี้ออก");
    x.onclick = () => { files.splice(i, 1); renderPrev(); };
    w.append(im, x); $("prev").appendChild(w);
  });
  updateGo();
}
function addFiles(list){
  const room = 3 - files.length;
  const arr = Array.from(list).filter(f => /^image\//.test(f.type));
  if(arr.length > room){ $("st").className = "msg"; $("st").textContent = "เลือกได้ไม่เกิน 3 รูป"; }
  files = files.concat(arr.slice(0, room)); renderPrev();
}
$("me").onchange = () => { addFiles($("me").files); $("me").value = ""; };
$("drop").ondragover = e => { e.preventDefault(); $("drop").classList.add("over"); };
$("drop").ondragleave = () => $("drop").classList.remove("over");
$("drop").ondrop = e => { e.preventDefault(); $("drop").classList.remove("over"); addFiles(e.dataTransfer.files); };
$("ok").onchange = updateGo;

function render(){
  const tol = parseFloat($("tv").value);
  const shown = results.filter(r => r.distance <= tol);
  lastShown = shown;
  $("count").textContent = "พบ " + shown.length + " รูป";
  $("dlall").hidden = !shown.length;
  $("dlall").textContent = "ดาวน์โหลดทั้งหมด (" + shown.length + ")";
  const out = $("out"); out.innerHTML = "";
  if(!shown.length){
    out.appendChild(el("p", "empty", results.length ? "ยังไม่พบรูปที่ตรงกัน ลองเลื่อนแถบความเข้มงวดไปทางขวา" : "ไม่พบรูปของคุณในอัลบั้มนี้ ลองเลือกอัลบั้มอื่น หรืออัปโหลดรูปที่เห็นหน้าชัดกว่านี้"));
    return;
  }
  const order = events.map(e => e.id), groups = new Map();
  for(const r of shown){
    if(!groups.has(r.event)) groups.set(r.event, {name: r.event_name, items: []});
    groups.get(r.event).items.push(r);
  }
  const keys = Array.from(groups.keys()).sort((a, b) => { const ia = order.indexOf(a), ib = order.indexOf(b); return (ia < 0 ? 999 : ia) - (ib < 0 ? 999 : ib); });
  for(const k of keys){
    const g = groups.get(k);
    if(keys.length > 1) out.appendChild(el("div", "gt", g.name + " (" + g.items.length + " รูป)"));
    const grid = el("div", "grid");
    for(const r of g.items){
      const card = el("div", "card"), b = el("button"), img = el("img");
      b.type = "button"; b.setAttribute("aria-label", "ดูรูปใหญ่");
      img.loading = "lazy"; img.src = r.thumb; img.alt = "รูปที่พบ";
      b.appendChild(img); b.onclick = () => openLb(r);
      const a = el("a", "", "ดาวน์โหลด"); a.href = r.download;
      card.append(b, a); grid.appendChild(card);
    }
    out.appendChild(grid);
  }
}
$("tv").oninput = render;

$("dlall").onclick = async () => {
  for(const r of lastShown){
    const a = document.createElement("a"); a.href = r.download; a.download = "";
    document.body.appendChild(a); a.click(); a.remove();
    await new Promise(s => setTimeout(s, 400));
  }
};

function openLb(r){ $("lbi").src = r.download; $("lba").href = r.download; $("lb").hidden = false; }
function closeLb(){ $("lb").hidden = true; $("lbi").src = ""; }
$("lbx").onclick = closeLb;
$("lb").onclick = e => { if(e.target === $("lb")) closeLb(); };
document.addEventListener("keydown", e => { if(e.key === "Escape") closeLb(); });

$("go").onclick = async () => {
  if(!files.length || !$("ok").checked) return;
  $("go").disabled = true; $("go").classList.add("busy"); $("go").textContent = "กำลังสแกน";
  $("st").className = "msg"; $("st").textContent = "อาจใช้เวลาสักครู่ กรุณาอย่าปิดหน้านี้";
  const fd = new FormData();
  for(const f of files) fd.append("files", f);
  fd.append("consent", "1");
  try{
    const r = await fetch("/api/search?event=" + encodeURIComponent(sel ? sel.id : ""), {method: "POST", body: fd});
    let j = {}; try{ j = await r.json(); }catch(e){}
    if(!r.ok){ $("st").className = "msg err"; $("st").textContent = j.detail || ("เกิดข้อผิดพลาด (" + r.status + ") ลองใหม่อีกครั้ง"); }
    else{
      results = j.results;
      $("st").textContent = j.total_photos ? "สแกนเสร็จแล้ว รูปของคุณไม่ได้ถูกเก็บไว้" : "ยังไม่มีรูปในคลัง";
      $("resbox").hidden = false; setStep(3); render();
      $("resbox").scrollIntoView({behavior: "smooth", block: "start"});
    }
  }catch(e){ $("st").className = "msg err"; $("st").textContent = "เชื่อมต่อเซิร์ฟเวอร์ไม่ได้ ตรวจสอบอินเทอร์เน็ตแล้วลองใหม่"; }
  $("go").classList.remove("busy"); $("go").textContent = "สแกนหารูปของฉัน"; updateGo();
};

show(1); loadEvents();
</script></body></html>"""

ADMIN_PAGE = r"""<!DOCTYPE html>
<html lang="th"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ผู้ดูแล - __SCHOOL__</title><style>__STYLE__</style></head><body><div class="w">
<h1>หน้าผู้ดูแลคลังรูป</h1>

<div class="box" id="loginbox"><h2>เข้าสู่ระบบ</h2>
<p>ใส่รหัสผ่านผู้ดูแล (ดูได้ที่หน้าต่าง PowerShell ตอนเปิดเซิร์ฟเวอร์ หรือในไฟล์ school_data/admin_password.txt)</p>
<input type="password" id="pw" placeholder="รหัสผ่าน"> <button id="login">เข้าสู่ระบบ</button>
<div class="st" id="lmsg"></div></div>

<div id="panel" style="display:none">

<div class="box"><h2>เพิ่มอัลบั้มใหม่</h2>
<input type="text" id="evname" placeholder="ชื่ออัลบั้ม เช่น กีฬาสี 2569" size="28">
<input type="date" id="evdate">
<button id="mkev">+ เพิ่มอัลบั้ม</button>
<div id="evlist" style="margin-top:14px"></div></div>

<div class="box" id="upbox"><h2>อัปโหลดรูปเข้าอัลบั้ม</h2>
<p>เลือกอัลบั้ม แล้วเลือกรูปทั้งหมดของงานนั้น (ทีละหลายร้อยรูปได้) ระบบจะตรวจจับใบหน้าและจดจำไว้ รูปที่ซ้ำจะถูกข้าม<br>รองรับ JPG / PNG / WEBP (ถ้าเป็น HEIC ให้แปลงเป็น JPG ก่อน)</p>
อัลบั้ม: <select id="upev"></select><br><br>
<input type="file" id="files" accept="image/jpeg,image/png,image/webp" multiple><br><br>
<button id="up">อัปโหลด</button>
<div class="bar" id="bar"><div class="track"><div class="fill" id="fill"></div><div class="pct" id="pct">0%</div></div>
<div class="st" id="ust"></div></div></div>

<div class="box"><h2>รูปในคลัง <span id="n"></span></h2>
<p>ถ้านักเรียนขอนำรูปออก กดลบรูปนั้นได้เลย (ลบทั้งรูปและข้อมูลใบหน้า) รูปที่อยู่ผิดอัลบั้มย้ายได้จากเมนูใต้ภาพ</p>
ดูอัลบั้ม: <select id="galev"></select>
<div class="grid" id="gal"></div></div>
</div>
</div>
<script>
const $ = id => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if(cls) e.className = cls; if(text !== undefined) e.textContent = text; return e; };
const TH_S = ["ม.ค.","ก.พ.","มี.ค.","เม.ย.","พ.ค.","มิ.ย.","ก.ค.","ส.ค.","ก.ย.","ต.ค.","พ.ย.","ธ.ค."];
const fmtDate = iso => { if(!iso) return ""; const p = iso.split("-"); return (+p[2]) + " " + TH_S[+p[1] - 1] + " " + (+p[0] + 543); };
const now = new Date();
$("evdate").value = now.getFullYear() + "-" + String(now.getMonth() + 1).padStart(2, "0") + "-" + String(now.getDate()).padStart(2, "0");
let PW = "", EV = [];
const H = () => ({"X-Admin-Password": PW});
const HJ = () => ({"X-Admin-Password": PW, "Content-Type": "application/json"});

$("login").onclick = async () => {
  PW = $("pw").value.trim();
  try{
    const r = await fetch("/admin/login", {method: "POST", headers: H()});
    if(!r.ok){ $("lmsg").textContent = "รหัสผ่านไม่ถูกต้อง"; return; }
    $("loginbox").style.display = "none"; $("panel").style.display = "block"; init();
  }catch(e){ $("lmsg").textContent = "เชื่อมต่อเซิร์ฟเวอร์ไม่ได้"; }
};

async function init(){ await loadEvents(); await loadGallery(); }

function fillSelect(sel, items, first){
  const old = sel.value;
  sel.innerHTML = "";
  if(first){ const o = el("option", "", first); o.value = ""; sel.appendChild(o); }
  for(const e of items){
    const o = el("option", "", e.name + (e.date ? " \u00b7 " + fmtDate(e.date) : "") + " (" + e.count + " รูป)");
    o.value = e.id; sel.appendChild(o);
  }
  if(old && Array.from(sel.options).some(o => o.value === old)) sel.value = old;
}

async function loadEvents(){
  const r = await fetch("/admin/events", {headers: H()});
  EV = (await r.json()).events;
  const box = $("evlist"); box.innerHTML = "";
  if(!EV.length) box.appendChild(el("p", "", "ยังไม่มีอัลบั้ม ตั้งชื่อและกด + เพิ่มอัลบั้มด้านบนได้เลย"));
  for(const e of EV){
    const row = el("div", "evrow");
    row.appendChild(el("span", "evname", e.name + " (" + e.count + " รูป)"));
    if(e.id !== "none"){
      const dt = el("input"); dt.type = "date"; dt.value = e.date;
      dt.onchange = async () => {
        if(!dt.value) return;
        await fetch("/admin/event/" + e.id, {method: "PUT", headers: HJ(), body: JSON.stringify({date: dt.value})});
        init();
      };
      const add = el("button", "sm", "เพิ่มรูป");
      add.onclick = () => { $("upev").value = e.id; $("upbox").scrollIntoView({behavior: "smooth"}); };
      const rn = el("button", "sm", "เปลี่ยนชื่อ");
      rn.onclick = async () => {
        const n = prompt("ชื่ออัลบั้มใหม่", e.name);
        if(!n || !n.trim()) return;
        await fetch("/admin/event/" + e.id, {method: "PUT", headers: HJ(), body: JSON.stringify({name: n})});
        init();
      };
      const dl = el("button", "sm del2", "ลบ");
      dl.onclick = async () => {
        if(!confirm("ลบอัลบั้ม " + e.name + " และรูปทั้งหมดในอัลบั้มนี้ (" + e.count + " รูป) ใช่ไหม? ย้อนกลับไม่ได้")) return;
        await fetch("/admin/event/" + e.id, {method: "DELETE", headers: H()});
        init();
      };
      row.append(dt, add, rn, dl);
    }
    box.appendChild(row);
  }
  fillSelect($("upev"), EV.filter(e => e.id !== "none"), "");
  fillSelect($("galev"), EV, "ทุกอัลบั้ม");
}

$("mkev").onclick = async () => {
  const n = $("evname").value.trim();
  if(!n){ alert("ใส่ชื่ออัลบั้มก่อน"); return; }
  const r = await fetch("/admin/event", {method: "POST", headers: HJ(), body: JSON.stringify({name: n, date: $("evdate").value})});
  if(r.ok){
    const created = await r.json();
    $("evname").value = "";
    await init();
    $("upev").value = created.id;
  } else { alert("เพิ่มอัลบั้มไม่สำเร็จ"); }
};
$("galev").onchange = loadGallery;

async function loadGallery(){
  const r = await fetch("/admin/photos?event=" + encodeURIComponent($("galev").value), {headers: H()});
  const list = await r.json();
  $("n").textContent = "(" + list.length + " รูป)";
  $("gal").innerHTML = "";
  const targets = EV.filter(e => e.id !== "none");
  for(const p of list){
    const d = el("div", "card");
    const img = el("img"); img.loading = "lazy"; img.src = "/thumb/" + p.id; d.appendChild(img);
    const info = el("div", "", p.faces + " ใบหน้า");
    const mv = el("select");
    const o0 = el("option", "", "ย้ายไปอัลบั้ม..."); o0.value = ""; mv.appendChild(o0);
    for(const t of targets){ const o = el("option", "", t.name); o.value = t.id; mv.appendChild(o); }
    mv.onchange = async () => {
      if(!mv.value) return;
      await fetch("/admin/photo/" + p.id + "/move", {method: "POST", headers: HJ(), body: JSON.stringify({event: mv.value})});
      init();
    };
    const b = el("button", "del", "ลบรูปนี้");
    b.onclick = async () => {
      if(!confirm("ลบรูปนี้ออกจากคลัง?")) return;
      await fetch("/admin/photo/" + p.id, {method: "DELETE", headers: H()});
      init();
    };
    info.append(mv, b); d.appendChild(info); $("gal").appendChild(d);
  }
}

function fmt(sec){ sec = Math.max(0, Math.round(sec)); const m = Math.floor(sec / 60), s = sec % 60; return m ? m + " นาที " + s + " วินาที" : s + " วินาที"; }

$("up").onclick = async () => {
  const files = Array.from($("files").files);
  const ev = $("upev").value;
  if(!ev){ alert("เพิ่มอัลบั้มและเลือกอัลบั้มปลายทางก่อน"); return; }
  if(!files.length){ alert("เลือกรูปก่อน"); return; }
  $("up").disabled = true; $("bar").style.display = "block";
  $("fill").style.width = "0%"; $("pct").textContent = "0%";
  let done = 0, added = 0, dup = 0, noface = 0, failed = 0, stop = "";
  const t0 = Date.now();
  for(const f of files){
    try{
      const fd = new FormData(); fd.append("file", f); fd.append("event", ev);
      const r = await fetch("/admin/upload", {method: "POST", headers: H(), body: fd});
      let j = {}; try{ j = await r.json(); }catch(e){}
      if(r.status === 401){ stop = "รหัสผ่านไม่ถูกต้อง"; break; }
      if(!r.ok) failed++;
      else if(j.status === "added") added++;
      else if(j.status === "duplicate") dup++;
      else if(j.status === "no_face") noface++;
    }catch(e){ stop = "เชื่อมต่อเซิร์ฟเวอร์ไม่ได้"; break; }
    done++;
    const p = Math.round(done / files.length * 100);
    $("fill").style.width = p + "%"; $("pct").textContent = p + "%";
    const left = done < files.length ? " | เหลืออีกประมาณ " + fmt((Date.now() - t0) / 1000 / done * (files.length - done)) : "";
    $("ust").textContent = "ประมวลผลแล้ว " + done + " / " + files.length + " | เพิ่มเข้าอัลบั้ม " + added +
      " | ซ้ำ " + dup + " | ไม่พบใบหน้า " + noface + (failed ? " | อ่านไม่ได้ " + failed : "") + left;
  }
  if(stop) $("ust").textContent = "หยุด: " + stop;
  else $("ust").textContent += " | เสร็จแล้ว ใช้เวลา " + fmt((Date.now() - t0) / 1000);
  $("up").disabled = false; init();
};
</script></body></html>"""


def _page(tpl: str) -> str:
    return (tpl.replace("__STYLE__", STYLE)
               .replace("__SCHOOL__", html.escape(SCHOOL_NAME))
               .replace("__CONTACT__", html.escape(CONTACT_TEXT))
               .replace("__DEFTOL__", str(DEFAULT_TOL)))


@app.get("/", response_class=HTMLResponse)
def home():
    return _page(STUDENT_PAGE)


@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    return _page(ADMIN_PAGE)
