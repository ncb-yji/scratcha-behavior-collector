import os
import json
import gzip
import io
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv(os.getenv("BACKEND_ENV_FILE", ".env"))
except Exception:
    pass

def getenv_any(names, default=None):
    for n in names:
        v = os.getenv(n)
        if v is not None and str(v).strip() != "":
            return v
    return default

try:
    from zoneinfo import ZoneInfo
    KST = ZoneInfo("Asia/Seoul")
except Exception:
    KST = timezone(timedelta(hours=9))

ENV_NAME = getenv_any(["KS3_ENV", "ENV"], "dev")

KS3_ENDPOINT = getenv_any(["KS3_ENDPOINT", "S3_ENDPOINT_URL"], "")
KS3_REGION   = getenv_any(["KS3_REGION", "S3_REGION"], "ap-northeast-2")
KS3_BUCKET   = getenv_any(["KS3_BUCKET", "S3_BUCKET"], "")
KS3_ACCESS   = getenv_any(["KS3_ACCESS_KEY", "S3_ACCESS_KEY"], "")
KS3_SECRET   = getenv_any(["KS3_SECRET_KEY", "S3_SECRET_KEY"], "")
KS3_PREFIX   = getenv_any(["KS3_PREFIX", "S3_PREFIX"], "")
KS3_FORCE_PATH_STYLE = getenv_any(["KS3_FORCE_PATH_STYLE", "S3_FORCE_PATH_STYLE"], "1")

_ENABLE = getenv_any(["KS3_ENABLE"], None)
if _ENABLE is None:
    ENABLE_KS3 = all([KS3_BUCKET, KS3_ENDPOINT, KS3_ACCESS, KS3_SECRET])
else:
    ENABLE_KS3 = (_ENABLE == "1")

CORS_ORIGINS = [
    o.strip() for o in getenv_any(
        ["CORS_ORIGINS"],
        "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",") if o.strip()
]

def model_dump_compat(obj, *, exclude_none: bool = True):
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=exclude_none)
    if hasattr(obj, "dict"):
        return obj.dict(exclude_none=exclude_none)
    return obj

try:
    from pydantic import ConfigDict
    class BaseModelX(BaseModel):
        model_config = ConfigDict(extra="allow")
except Exception:
    class BaseModelX(BaseModel):
        class Config:
            extra = "allow"

class Rect(BaseModelX):
    left: float
    top: float
    w: float
    h: float
    version: Optional[int] = None

class SessionMeta(BaseModelX):
    device: Optional[str] = None
    viewport: Dict[str, float]
    dpr: Optional[float] = None
    roi: Optional[Rect] = None
    roi_map: Optional[Dict[str, Any]] = None
    ts_resolution_ms: Optional[int] = None
    session_id: str
    widget_version: Optional[str] = None

class PackedMoves(BaseModelX):
    base_t: int
    dts: List[int]
    xrs: List[float]
    yrs: List[float]
    xs: Optional[List[float]] = None
    ys: Optional[List[float]] = None
    oobs: Optional[List[int]] = None
    on_canvas: Optional[List[int]] = None

class EventItem(BaseModelX):
    t: Optional[int] = None
    type: str
    x: Optional[float] = None
    y: Optional[float] = None
    x_raw: Optional[float] = None
    y_raw: Optional[float] = None
    on_canvas: Optional[int] = None
    oob: Optional[int] = None
    pointerType: Optional[str] = None
    pointerId: Optional[int] = None
    is_trusted: Optional[int] = None
    target_role: Optional[str] = None
    target_answer: Optional[str] = None
    payload: Optional[PackedMoves] = None
    free: Optional[int] = None

class CollectRequest(BaseModelX):
    meta: SessionMeta
    events: List[EventItem]
    label: Optional[Dict[str, Any]] = None

class LabelPatch(BaseModelX):
    session_id: str
    label: Dict[str, Any]

# ========= KS3 업로드 유틸 =========
def _ks3_client():
    import boto3
    from botocore.config import Config
    cfg = Config(
        s3={"addressing_style": "path" if KS3_FORCE_PATH_STYLE == "1" else "virtual"},
        signature_version="s3v4",
        retries={"max_attempts": 3, "mode": "standard"},
    )
    session = boto3.session.Session(
        aws_access_key_id=KS3_ACCESS,
        aws_secret_access_key=KS3_SECRET,
        region_name=KS3_REGION or "ap-northeast-2",
    )
    return session.client("s3", endpoint_url=KS3_ENDPOINT, config=cfg)

def _make_session_key(session_id: str, gz: bool = True) -> str:
    ts = datetime.now(KST).strftime("%Y%m%d-%H%M%S")
    fname = f"{ts}_{session_id}.json" + (".gz" if gz else "")
    return f"{KS3_PREFIX.strip('/')}/{fname}".strip("/")

def _make_label_key(session_id: str) -> str:
    ts = datetime.now(KST).strftime("%Y%m%d-%H%M%S")
    return f"{KS3_PREFIX.strip('/')}/_labels/label_{ts}_{session_id}.json".strip("/")

def _serialize_jsonl_bytes(payload: CollectRequest) -> bytes:
    meta = model_dump_compat(payload.meta, exclude_none=True)
    events = [model_dump_compat(e, exclude_none=True) for e in payload.events]
    lines = [json.dumps({"type": "meta", **meta}, ensure_ascii=False)]
    for ev in events:
        lines.append(json.dumps({"type": "event", **ev}, ensure_ascii=False))
    if payload.label:
        lines.append(json.dumps({"type": "label", **payload.label}, ensure_ascii=False))
    return ("\n".join(lines) + "\n").encode("utf-8")

def _gzip_bytes(raw: bytes) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6, mtime=0) as gz:
        gz.write(raw)
    return buf.getvalue()

def upload_ks3_session(payload: CollectRequest, session_id: str):
    if not ENABLE_KS3 or not KS3_BUCKET or not KS3_ACCESS or not KS3_SECRET or not KS3_ENDPOINT:
        missing = []
        if not ENABLE_KS3:   missing.append("KS3_ENABLE(auto) == False")
        if not KS3_BUCKET:   missing.append("KS3_BUCKET")
        if not KS3_ACCESS:   missing.append("KS3_ACCESS_KEY")
        if not KS3_SECRET:   missing.append("KS3_SECRET_KEY")
        if not KS3_ENDPOINT: missing.append("KS3_ENDPOINT")
        return (None, None, f"Missing: {', '.join(missing)}")

    try:
        body = _serialize_jsonl_bytes(payload)
        gz   = _gzip_bytes(body)
        key  = _make_session_key(session_id, gz=True)

        s3 = _ks3_client()
        s3.put_object(
            Bucket=KS3_BUCKET,
            Key=key,
            Body=gz,
            ContentType="application/json",
            ContentEncoding="gzip",
        )
        return (f"s3://{KS3_BUCKET}/{key}", key, len(gz))
    except Exception as e:
        return (None, None, f"upload error: {e}")

def upload_ks3_label(session_id: str, label: Dict[str, Any]):
    try:
        key = _make_label_key(session_id)
        s3  = _ks3_client()
        s3.put_object(
            Bucket=KS3_BUCKET,
            Key=key,
            Body=json.dumps({"session_id": session_id, "label": label}, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )
        return (f"s3://{KS3_BUCKET}/{key}", key, None)
    except Exception as e:
        return (None, None, f"upload error: {e}")

# ========= FastAPI =========
app = FastAPI(title="Scratcha Collector (KS3-only)", version="1.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- 기존 엔드포인트 (작은 페이로드용, 그대로 유지) ----
@app.post("/collect")
async def collect(payload: CollectRequest):
    session_id = payload.meta.session_id
    ks3_uri, key, size_or_err = upload_ks3_session(payload, session_id)
    if ks3_uri:
        roi_keys = sorted((payload.meta.roi_map or {}).keys()) if payload.meta.roi_map else []
        print(f"[collect] session={session_id} events={len(payload.events)} roi_map={len(roi_keys)} ks3=OK key={key} size={size_or_err}")
        return {"ok": True, "ks3": ks3_uri, "key": key, "size": size_or_err}
    print(f"[collect] session={session_id} ks3=ERR {size_or_err}")
    return {"ok": False, "error": size_or_err}

# ---- 조각 업로드 구현 (메모리 조합) ----
_CHUNK_BUF: Dict[str, Dict[str, Any]] = {}  # session_id -> {"meta": dict, "events": list, "parts": int}

@app.post("/collect_chunk")
async def collect_chunk(payload: CollectRequest):
    sid = payload.meta.session_id
    rec = _CHUNK_BUF.get(sid)
    if rec is None:
        # 첫 조각: full meta 저장
        rec = {"meta": model_dump_compat(payload.meta, exclude_none=True), "events": [], "parts": 0}
        _CHUNK_BUF[sid] = rec
    else:
        # 이후 조각: meta는 최소한(session_id)만 들어왔을 수 있음 → 기존 meta 유지
        pass
    # 이벤트 이어붙이기(원본 그대로)
    for ev in payload.events:
        rec["events"].append(model_dump_compat(ev, exclude_none=True))
    rec["parts"] += 1
    return {"ok": True, "session_id": sid, "parts": rec["parts"]}

@app.post("/collect_finalize")
async def collect_finalize(payload: CollectRequest):
    # payload.meta.session_id 는 필수
    sid = payload.meta.session_id
    rec = _CHUNK_BUF.get(sid)
    if rec is None:
        # 예외: 조각 없이 finalize만 오는 경우(안전망) → 그냥 /collect 흐름으로 업로드
        ks3_uri, key, size_or_err = upload_ks3_session(payload, sid)
        if ks3_uri:
            print(f"[finalize(single)] session={sid} events={len(payload.events)} ks3=OK key={key}")
            return {"ok": True, "ks3": ks3_uri, "key": key}
        return {"ok": False, "error": size_or_err}

    # 조각 합본 → 최종 업로드 (단일 파일만 생성)
    meta = rec["meta"]
    events = rec["events"]
    label = model_dump_compat(payload.label, exclude_none=True) if payload.label else None

    # Pydantic 모델에 다시 태워주기(타입 안전)
    full = CollectRequest(
        meta=SessionMeta(**meta),
        events=[EventItem(**e) for e in events],
        label=label
    )
    ks3_uri, key, size_or_err = upload_ks3_session(full, sid)
    # 버퍼 정리
    try: del _CHUNK_BUF[sid]
    except Exception: pass

    if ks3_uri:
        print(f"[finalize] session={sid} events={len(events)} ks3=OK key={key}")
        return {"ok": True, "ks3": ks3_uri, "key": key}
    print(f"[finalize] session={sid} ks3=ERR {size_or_err}")
    return {"ok": False, "error": size_or_err}

# ---- 헬스/디버그(기존 유지) ----
@app.get("/healthz")
async def healthz():
    return {"ok": True, "env": ENV_NAME}

@app.post("/collect_raw")
async def collect_raw(req: Request):
    body = await req.json()
    return {"ok": True, "echo": body}

@app.get("/debug/env")
async def debug_env():
    return {
        "ENV_NAME": ENV_NAME,
        "ENABLE_KS3": ENABLE_KS3,
        "KS3_ENDPOINT": KS3_ENDPOINT,
        "KS3_REGION": KS3_REGION,
        "KS3_BUCKET": KS3_BUCKET,
        "KS3_PREFIX": KS3_PREFIX,
        "KS3_FORCE_PATH_STYLE": KS3_FORCE_PATH_STYLE,
        "KS3_ACCESS_KEY_len": len(KS3_ACCESS or ""),
        "KS3_SECRET_KEY_len": len(KS3_SECRET or ""),
    }

@app.get("/debug/ks3")
async def debug_ks3():
    try:
        s3 = _ks3_client()
        test_key = _make_session_key("debug", gz=False)
        s3.put_object(
            Bucket=KS3_BUCKET,
            Key=test_key,
            Body=b'{"ping":"pong"}\n',
            ContentType="application/json",
        )
        return {"enabled": True, "test_key": test_key}
    except Exception as e:
        return {"enabled": False, "error": str(e)}
