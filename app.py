from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import os
import cv2
import numpy as np
import urllib.request

import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# OPTIONAL restoration
try:
    from gfpgan import GFPGANer
    HAS_GFPGAN_LIB = True
except Exception:
    HAS_GFPGAN_LIB = False
    GFPGANer = None


# =========================
# CONFIG (env first)
# =========================

MODEL_PATH = os.environ.get("INSWAPPER_PATH", "models/inswapper_128.onnx")
INSWAPPER_URL = os.environ.get("INSWAPPER_URL", "").strip()

GFPGAN_PATH = os.environ.get("GFPGAN_PATH", "models/GFPGANv1.4.pth")
GFPGAN_URL = os.environ.get("GFPGAN_URL", "").strip()

# CORS (lock this down in prod)
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

# Providers auto
AVAILABLE = ort.get_available_providers()
providers = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in AVAILABLE
    else ["CPUExecutionProvider"]
)
USING_GPU = providers[0] == "CUDAExecutionProvider"


# =========================
# FASTAPI
# =========================

app = FastAPI(title="MorphAI FaceSwap", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS] if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# GLOBAL MODELS
# =========================

face_app = FaceAnalysis(name="buffalo_l", providers=providers)
SWAPPER = None
GFPGAN = None


# =========================
# MODEL DOWNLOAD HELPERS
# =========================

def ensure_file(path: str, url: str, label: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 1024:
        return
    if not url:
        raise RuntimeError(f"Missing model file: {path}. Set {label}_URL env var to a direct download link.")
    print(f"Downloading {label} -> {path}")
    urllib.request.urlretrieve(url, path)
    if not os.path.exists(path) or os.path.getsize(path) <= 1024:
        raise RuntimeError(f"Download failed or file too small for {label}: {path}")
    print(f"{label} download complete ({os.path.getsize(path)} bytes)")


# =========================
# STARTUP
# =========================

@app.on_event("startup")
def startup():
    global SWAPPER, GFPGAN

    # InsightFace detector/recognition models download into ~/.insightface automatically
    face_app.prepare(ctx_id=0 if USING_GPU else -1, det_size=(640, 640))

    # Ensure swapper model exists (Railway fix)
    ensure_file(MODEL_PATH, INSWAPPER_URL, "INSWAPPER")

    SWAPPER = get_model(MODEL_PATH, providers=providers)

    # Optional GFPGAN
    GFPGAN = None
    if HAS_GFPGAN_LIB:
        try:
            # download if URL provided
            if (not os.path.exists(GFPGAN_PATH) or os.path.getsize(GFPGAN_PATH) <= 1024) and GFPGAN_URL:
                ensure_file(GFPGAN_PATH, GFPGAN_URL, "GFPGAN")

            if os.path.exists(GFPGAN_PATH) and os.path.getsize(GFPGAN_PATH) > 1024:
                GFPGAN = GFPGANer(
                    model_path=GFPGAN_PATH,
                    upscale=1,
                    arch="clean",
                    channel_multiplier=2,
                    bg_upsampler=None,
                )
                print("GFPGAN loaded")
            else:
                print("GFPGAN missing — disabled")
        except Exception as e:
            print("GFPGAN init failed — disabled:", repr(e))
            GFPGAN = None
    else:
        print("GFPGAN library not installed — disabled")


# =========================
# ROUTES
# =========================

@app.get("/")
def root():
    return {
        "status": "ok",
        "engine": "inswapper_128",
        "providers_available": AVAILABLE,
        "providers_configured": providers,
        "using_gpu": USING_GPU,
        "models": {
            "inswapper_path": MODEL_PATH,
            "inswapper_exists": os.path.exists(MODEL_PATH),
            "gfpgan_enabled": GFPGAN is not None,
            "gfpgan_path": GFPGAN_PATH,
            "gfpgan_exists": os.path.exists(GFPGAN_PATH),
        },
    }

@app.get("/health")
def health():
    if SWAPPER is None:
        return JSONResponse({"ok": False, "reason": "SWAPPER not loaded"}, status_code=503)
    return {"ok": True, "gpu": USING_GPU, "gfpgan": GFPGAN is not None}


# =========================
# HELPERS
# =========================

async def read_image(file: UploadFile) -> np.ndarray:
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image")
    return img

def clamp_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w, int(x2)); y2 = min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


# ---------- AUTO QUALITY DETECTION ----------

def roi_sharpness_score(bgr: np.ndarray) -> float:
    if bgr is None or bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def should_restore_face(
    roi_bgr: np.ndarray,
    face_w: int,
    face_h: int,
    sharp_thresh: float = 85.0,
    min_face_side: int = 120,
) -> bool:
    if roi_bgr is None or roi_bgr.size == 0:
        return False
    if min(face_w, face_h) < int(min_face_side):
        return True
    return roi_sharpness_score(roi_bgr) < float(sharp_thresh)


# ---------- COLOR HARMONIZATION ----------

def lab_match(src, ref):
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)

    for i in range(3):
        s_mean, s_std = src_lab[:, :, i].mean(), src_lab[:, :, i].std()
        r_mean, r_std = ref_lab[:, :, i].mean(), ref_lab[:, :, i].std()
        src_lab[:, :, i] = (src_lab[:, :, i] - s_mean) * (r_std / (s_std + 1e-6)) + r_mean

    src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)

def harmonize(result, target, bbox, feather: float = 0.04):
    x1, y1, x2, y2 = bbox
    roi = result[y1:y2, x1:x2]
    ref = target[y1:y2, x1:x2]
    if roi.size == 0:
        return result

    matched = lab_match(roi, ref)

    mask = np.zeros((roi.shape[0], roi.shape[1]), np.float32)
    cv2.ellipse(
        mask,
        (roi.shape[1] // 2, roi.shape[0] // 2),
        (int(roi.shape[1] * 0.42), int(roi.shape[0] * 0.48)),
        0, 0, 360, 1, -1
    )
    mask = cv2.GaussianBlur(mask, (0, 0), roi.shape[1] * float(feather))
    mask = np.dstack([mask] * 3)

    blended = (matched * mask + roi * (1 - mask)).astype(np.uint8)

    out = result.copy()
    out[y1:y2, x1:x2] = blended
    return out


# ---------- SHARPEN ----------

def sharpen_roi(img, bbox, amount: float = 0.35, radius: float = 1.2):
    x1, y1, x2, y2 = bbox
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return img

    blur = cv2.GaussianBlur(roi, (0, 0), float(radius))
    sharp = cv2.addWeighted(roi, 1.0 + float(amount), blur, -float(amount), 0)

    out = img.copy()
    out[y1:y2, x1:x2] = sharp
    return out


# =========================
# SWAP ROUTE
# =========================

@app.post("/swap/single")
async def swap_single(
    source: UploadFile = File(...),
    target: UploadFile = File(...),

    # restore controls (Swagger)
    restore_sharp_thresh: float = Query(85.0, ge=10.0, le=500.0),
    restore_min_face_side: int = Query(120, ge=32, le=512),
    restore_force: bool = Query(False),
    restore_disable: bool = Query(False),

    # sharpen controls
    sharpen_amount: float = Query(0.35, ge=0.0, le=1.0),
    sharpen_radius: float = Query(1.2, ge=0.5, le=5.0),
):
    global SWAPPER, GFPGAN
    if SWAPPER is None:
        raise HTTPException(500, "Swapper not initialized (model missing or failed to load)")

    src = await read_image(source)
    tgt = await read_image(target)
    tgt_original = tgt.copy()

    src_faces = face_app.get(src)
    tgt_faces = face_app.get(tgt)

    if not src_faces:
        raise HTTPException(400, "No face found in source image")
    if not tgt_faces:
        raise HTTPException(400, "No face found in target image")

    src_face = src_faces[0]
    result = tgt.copy()

    for face in tgt_faces:
        # 1) swap
        result = SWAPPER.get(result, face, src_face, paste_back=True)

        # bbox safe
        h, w = result.shape[:2]
        bb = clamp_bbox(*[int(v) for v in face.bbox], w, h)
        if not bb:
            continue

        # 2) color harmonize
        result = harmonize(result, tgt_original, bb)

        # 3) restore (GFPGAN) auto-gated
        if GFPGAN is not None and (not restore_disable):
            x1, y1, x2, y2 = bb
            roi = result[y1:y2, x1:x2]
            if roi.size > 0:
                face_w = x2 - x1
                face_h = y2 - y1
                do_restore = restore_force or should_restore_face(
                    roi_bgr=roi,
                    face_w=face_w,
                    face_h=face_h,
                    sharp_thresh=float(restore_sharp_thresh),
                    min_face_side=int(restore_min_face_side),
                )
                if do_restore:
                    try:
                        _, _, rest = GFPGAN.enhance(roi, has_aligned=False, paste_back=True)
                        if rest is not None and rest.shape == roi.shape:
                            result[y1:y2, x1:x2] = rest
                    except Exception as e:
                        print("GFPGAN failed:", repr(e))

        # 4) sharpen
        result = sharpen_roi(result, bb, amount=sharpen_amount, radius=sharpen_radius)

    ok, buf = cv2.imencode(".png", result)
    if not ok:
        raise HTTPException(500, "Failed to encode image")

    return Response(buf.tobytes(), media_type="image/png")
