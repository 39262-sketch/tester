"""
face-search-server
-------------------
ค้นหาตัวเองในรูปจากคลังรูป (เช่นรูปที่โหลดมาจากเพจโรงเรียน) ด้วยการจดจำใบหน้าจริง

  GET  /         หน้าเว็บใช้งาน (เลือกรูป กดปุ่ม ดูผล ดาวน์โหลด)
  POST /enroll   ลงทะเบียนใบหน้าตัวเอง (รูปเดียวหรือหลายรูป)
  POST /scan     สแกนรูป คืนรูปที่อาจใช่พร้อมค่าระยะ (distance) ของแต่ละรูป
  POST /clear    ล้างรูปที่แมตช์ไว้จากการสแกนรอบก่อน
  GET  /matches  รายการรูปที่แมตช์ไว้ตอนนี้
  GET  /download/{filename}  ดาวน์โหลดรูป

แนวคิดการกรอง: สแกนครั้งเดียวด้วยเกณฑ์กว้าง แล้วให้หน้าเว็บกรองตามแถบเลื่อนแบบทันที
ไม่ต้องสแกนซ้ำเวลาปรับความเข้มงวด
"""

import io
import json
import uuid
from pathlib import Path
from typing import List

import face_recognition
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, ImageOps

try:  # ถ้าติดตั้ง pillow-heif ไว้ จะอ่าน HEIC ได้ด้วย ถ้าไม่ได้ติดตั้งก็ไม่เป็นไร
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

BASE_DIR = Path(__file__).parent
MATCH_DIR = BASE_DIR / "matches"
PROFILE_PATH = BASE_DIR / "profile.json"
MATCH_DIR.mkdir(exist_ok=True)

MAX_SIDE = 1600          # ย่อรูปใหญ่ก่อนสแกนเพื่อความเร็ว (ถ้าหน้าในรูปหมู่เล็กเกินไป ลองเพิ่มเป็น 2400)
LOOSE_TOLERANCE = 0.65   # เกณฑ์กว้างสุดตอนสแกน หน้าเว็บจะกรองต่อเองด้วยแถบเลื่อน

app = FastAPI(title="face-search-server")


def load_profile() -> List[list]:
    if PROFILE_PATH.exists():
        return json.loads(PROFILE_PATH.read_text())
    return []


def read_image_to_array(raw: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        raise HTTPException(400, "อ่านไฟล์รูปไม่ได้ ถ้าเป็นไฟล์ HEIC กรุณาแปลงเป็น JPG ก่อน")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return np.array(img)


@app.post("/enroll")
async def enroll(files: List[UploadFile] = File(...)):
    """ลงทะเบียนใบหน้าตัวเอง ยิ่งหลายรูป (มุม/แสงต่างกัน) ยิ่งแม่น"""
    encodings = []
    for f in files:
        arr = read_image_to_array(await f.read())
        # num_jitters=5: ถอดลักษณะใบหน้าหลายรอบแล้วเฉลี่ย แม่นขึ้น (ช้าลงนิดหน่อยเพราะมีไม่กี่รูป)
        faces = face_recognition.face_encodings(arr, num_jitters=5, model="large")
        if not faces:
            raise HTTPException(400, f"ไม่พบใบหน้าในไฟล์ {f.filename}")
        if len(faces) > 1:
            raise HTTPException(400, f"พบมากกว่า 1 ใบหน้าในไฟล์ {f.filename} กรุณาใช้รูปที่มีคนเดียว")
        encodings.append(faces[0].tolist())
    PROFILE_PATH.write_text(json.dumps(encodings))
    return {"enrolled_photos": len(encodings)}


@app.post("/scan")
async def scan(files: List[UploadFile] = File(...), tolerance: float = LOOSE_TOLERANCE):
    profile = load_profile()
    if not profile:
        raise HTTPException(409, "ยังไม่ได้ลงทะเบียนรูปตัวเอง กรุณากดปุ่มลงทะเบียนก่อน")
    profile_encodings = [np.array(e) for e in profile]

    matched, skipped = [], 0
    for f in files:
        raw = await f.read()
        arr = read_image_to_array(raw)

        locs = face_recognition.face_locations(arr)
        if not locs:  # ไม่เจอหน้า: ลองอีกรอบแบบละเอียดขึ้น (จับหน้าเล็กได้ดีกว่า แต่ช้ากว่า)
            locs = face_recognition.face_locations(arr, number_of_times_to_upsample=2)
        if not locs:
            skipped += 1
            continue

        best = None
        for enc in face_recognition.face_encodings(arr, locs, model="large"):
            d = float(np.min(face_recognition.face_distance(profile_encodings, enc)))
            if best is None or d < best:
                best = d

        if best is not None and best <= tolerance:
            name = f"{uuid.uuid4().hex}.jpg"
            full = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
            full.save(MATCH_DIR / name, "JPEG", quality=95)
            matched.append({
                "original_filename": f.filename,
                "download_url": f"/download/{name}",
                "distance": round(best, 3),
                "similarity_pct": round((1 - best) * 100, 1),
            })

    return {
        "scanned": len(files),
        "matched": len(matched),
        "skipped_no_face_detected": skipped,
        "results": matched,
    }


@app.post("/clear")
async def clear():
    n = 0
    for p in MATCH_DIR.iterdir():
        if p.is_file():
            p.unlink()
            n += 1
    return {"deleted": n}


@app.get("/matches")
async def list_matches():
    items = [{"filename": p.name, "download_url": f"/download/{p.name}"}
             for p in sorted(MATCH_DIR.iterdir())]
    return {"count": len(items), "items": items}


@app.get("/download/{filename}")
async def download(filename: str):
    path = MATCH_DIR / Path(filename).name
    if not path.exists():
        raise HTTPException(404, "ไม่พบไฟล์")
    return FileResponse(path, filename=path.name)


PAGE = r"""<!DOCTYPE html>
<html lang="th"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ค้นหาตัวเองในรูปเพจโรงเรียน</title>
<style>
body{margin:0;background:#11141b;color:#eee7d8;font-family:'Segoe UI',Tahoma,sans-serif}
.w{max-width:780px;margin:0 auto;padding:28px 18px 60px}
h1{font-size:26px;margin:0 0 18px}
.box{background:#171b24;border:1px solid #2a2f3b;border-radius:10px;padding:16px;margin-bottom:14px}
h2{font-size:16px;color:#c99a4a;margin:0 0 6px}
p{font-size:13.5px;color:#9aa1af;margin:0 0 10px}
button{padding:11px 18px;border:0;border-radius:8px;background:#c99a4a;color:#1a1408;font-weight:600;cursor:pointer;font-size:14px}
button:disabled{opacity:.4;cursor:not-allowed}
input[type=file]{color:#9aa1af;font-size:13px}
input[type=range]{width:220px;vertical-align:middle;accent-color:#c99a4a}
#msg1{font-size:13.5px;margin-top:10px}
.bar{display:none;margin-top:16px}
.track{height:22px;background:#2a2f3b;border-radius:11px;overflow:hidden;position:relative}
.fill{height:100%;width:0;background:#c99a4a;transition:width .2s}
.pct{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:12.5px;font-weight:700;color:#fff;text-shadow:0 0 3px #000}
#status{font-size:13.5px;color:#9aa1af;margin-top:8px;line-height:1.7}
#count{font-size:13.5px;margin-top:14px;color:#eee7d8}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px;margin-top:12px}
.card{background:#11141b;border:1px solid #2a2f3b;border-radius:8px;overflow:hidden}
.card img{width:100%;height:140px;object-fit:cover;display:block}
.card div{padding:8px;font-size:12.5px;color:#9aa1af;word-break:break-all}
.card a{display:block;margin-top:6px;padding:6px;text-align:center;background:#c99a4a;color:#1a1408;border-radius:6px;text-decoration:none;font-weight:600}
</style></head><body><div class="w">
<h1>ค้นหาตัวเองในรูปเพจโรงเรียน</h1>

<div class="box"><h2>1. ลงทะเบียนใบหน้าตัวเอง</h2>
<p>เลือกรูปหน้าตัวเอง 1-3 รูป (รูปละหนึ่งคนเท่านั้น) ยิ่งหลายรูปที่มุมและแสงต่างกัน ยิ่งแม่น</p>
<input type="file" id="me" accept="image/jpeg,image/png,image/webp" multiple>
<p style="margin-top:8px">รองรับ JPG / PNG / WEBP (ถ้าเป็น HEIC ให้แปลงเป็น JPG ก่อน)</p>
<button id="b1">ลงทะเบียน</button><div id="msg1"></div></div>

<div class="box"><h2>2. สแกนรูปจากเพจโรงเรียน</h2>
<p>เลือกรูปทั้งหมดที่โหลดมา (เลือกหลายไฟล์ได้) หมายเหตุ: การสแกนรอบใหม่จะล้างผลของรอบก่อน ดาวน์โหลดรูปที่ต้องการให้เรียบร้อยก่อน</p>
<input type="file" id="pool" accept="image/jpeg,image/png,image/webp" multiple>
<p style="margin-top:8px">รองรับ JPG / PNG / WEBP (ถ้าเป็น HEIC ให้แปลงเป็น JPG ก่อน)</p>
<button id="b2">สแกน</button>

<div class="bar" id="bar">
  <div class="track"><div class="fill" id="fill"></div><div class="pct" id="pct">0%</div></div>
  <div id="status"></div>
</div>

<div style="margin-top:18px">
  ความเข้มงวด: <input type="range" id="tol" min="0.35" max="0.65" step="0.01" value="0.55">
  <b id="tv">0.55</b>
  <p style="margin:6px 0 0">เลื่อนไปซ้าย = เข้มงวด (รูปผิดน้อยลง) / ขวา = ผ่อนปรน (เจอรูปมากขึ้น) ผลเปลี่ยนทันที ไม่ต้องสแกนใหม่</p>
</div>
<div id="count"></div>
<div class="grid" id="out"></div>
</div>
</div>

<script>
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
let candidates = [];

function render(){
  const tol = parseFloat($("tol").value);
  $("tv").textContent = tol.toFixed(2);
  const shown = candidates.filter(c => c.dist <= tol).sort((a,b) => a.dist - b.dist);
  $("count").textContent = candidates.length
    ? "แสดง " + shown.length + " จาก " + candidates.length + " รูปที่ระบบมองว่าอาจใช่ (ค่าระยะยิ่งต่ำ = ยิ่งเหมือน)"
    : "";
  $("out").innerHTML = "";
  for(const c of shown){
    const d = document.createElement("div");
    d.className = "card";
    d.innerHTML = '<img loading="lazy" src="' + c.url + '"><div>' + esc(c.name) +
      '<br>ค่าระยะ ' + c.dist.toFixed(3) +
      '<a href="' + c.url + '" download="' + esc(c.name) + '.jpg">ดาวน์โหลด</a></div>';
    $("out").appendChild(d);
  }
}
$("tol").oninput = render;

async function postFiles(url, files){
  const fd = new FormData();
  for(const f of files) fd.append("files", f);
  const r = await fetch(url, {method:"POST", body:fd});
  let j = {};
  try { j = await r.json(); } catch(e) {}
  return {ok:r.ok, status:r.status, j:j};
}

$("b1").onclick = async () => {
  const files = $("me").files;
  if(!files.length){ $("msg1").textContent = "เลือกรูปก่อน"; return; }
  $("b1").disabled = true; $("msg1").textContent = "กำลังลงทะเบียน...";
  try{
    const r = await postFiles("/enroll", files);
    $("msg1").textContent = r.ok ? "ลงทะเบียนสำเร็จ " + r.j.enrolled_photos + " รูป" : "ผิดพลาด: " + (r.j.detail || r.status);
  }catch(e){ $("msg1").textContent = "เชื่อมต่อเซิร์ฟเวอร์ไม่ได้"; }
  $("b1").disabled = false;
};

function fmtTime(sec){
  sec = Math.max(0, Math.round(sec));
  const m = Math.floor(sec / 60), s = sec % 60;
  return m ? m + " นาที " + s + " วินาที" : s + " วินาที";
}

$("b2").onclick = async () => {
  const files = Array.from($("pool").files);
  if(!files.length){ alert("เลือกรูปที่จะสแกนก่อน"); return; }
  candidates = []; render();
  $("b2").disabled = true;
  $("bar").style.display = "block";
  $("fill").style.width = "0%"; $("pct").textContent = "0%";
  $("status").textContent = "กำลังเริ่ม...";
  try{ await fetch("/clear", {method:"POST"}); }catch(e){}

  const t0 = Date.now();
  let done = 0, noface = 0, failed = 0, stopMsg = "";
  for(const f of files){
    try{
      const r = await postFiles("/scan?tolerance=0.65", [f]);
      if(r.ok){
        noface += r.j.skipped_no_face_detected || 0;
        for(const m of r.j.results) candidates.push({name:f.name, url:m.download_url, dist:m.distance});
      } else if(r.status === 409){
        stopMsg = r.j.detail || "ยังไม่ได้ลงทะเบียนรูปตัวเอง";
        break;
      } else { failed++; }
    }catch(e){
      stopMsg = "เชื่อมต่อเซิร์ฟเวอร์ไม่ได้ (เซิร์ฟเวอร์อาจหยุดทำงาน)";
      break;
    }
    done++;
    const p = Math.round(done / files.length * 100);
    $("fill").style.width = p + "%"; $("pct").textContent = p + "%";
    const el = (Date.now() - t0) / 1000;
    const left = done < files.length ? " | เหลืออีกประมาณ " + fmtTime(el / done * (files.length - done)) : "";
    $("status").textContent = "สแกนแล้ว " + done + " / " + files.length + " รูป | พบรูปที่อาจใช่ " + candidates.length + " รูป" + left;
    render();
  }

  if(stopMsg){
    $("status").textContent = "หยุดสแกน: " + stopMsg;
  } else {
    let t = "สแกนเสร็จ " + files.length + " รูป ใช้เวลา " + fmtTime((Date.now() - t0) / 1000);
    if(noface) t += " | ไม่พบใบหน้า " + noface + " รูป";
    if(failed) t += " | อ่านไฟล์ไม่ได้ " + failed + " รูป";
    $("status").textContent = t;
  }
  $("b2").disabled = false;
};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
async def home():
    return PAGE
