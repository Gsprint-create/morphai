from __future__ import annotations

import os
import time
import urllib.request
from typing import Optional, Tuple, List

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# =========================================================
# MorphAI ULTRA (FaceSwap + Color Harmonize + Auto Restore + Sharpen)
#
# Goals:
# - Runs on CPU (Railway) or GPU (local/RunPod) automatically.
# - Does NOT crash if GFPGAN deps are missing: restoration becomes optional.
# - Works when model files are NOT committed: can download via env URLs.
# =========================================================

# -------------------------
# Paths + Download URLs
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

INSWAPPER_PATH = os.environ.get("INSWAPPER_PATH", os.path.join(MODELS_DIR, "inswapper_128.onnx"))
INSWAPPER_URL  = os.environ.get("INSWAPPER_URL", "").strip()  # must be a direct download link

GFPGAN_PATH = os.environ.get("GFPGAN_PATH", os.path.join(MODELS_DIR, "GFPGANv1.4.pth"))
GFPGAN_URL  = os.environ.get("GFPGAN_URL", "").strip()  # must be a direct download link

# -------------------------
# Providers (GPU if available)
# -------------------------
AVAILABLE_PROVIDERS = ort.get_available_providers()
PROVIDERS = (["CUDAExecutionProvider", "CPUExecutionProvider"]
             if "CUDAExecutionProvider" in AVAILABLE_PROVIDERS
             else ["CPUExecutionProvider"])
USING_GPU = PROVIDERS[0] == "CUDAExecutionProvider"

# -------------------------
# FastAPI + CORS
# -------------------------
app = FastAPI(title="MorphAI ULTRA FaceSwap", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://morphai-frontend.vercel.app",
        "https://www.morphai.net",
        "https://morphai.net",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"^https:\/\/.*\.vercel\.app$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

# -------------------------
# Globals (cached models)
# -------------------------
face_app: Optional[FaceAnalysis] = None
SWAPPER = None

GFPGAN = None
GFPGAN_ENABLED = False
GFPGAN_IMPORT_ERROR: Optional[str] = None


# =========================================================
# Model utilities
# =========================================================

def _download(url: str, out_path: str, timeout: int = 60) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    req = urllib.request.Request(url, headers={"User-Agent": "MorphAI-ULTRA/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        f.write(r.read())
    os.replace(tmp, out_path)

def ensure_file(path: str, url: str, name: str) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 1024 * 1024:
        return
    if not url:
        raise RuntimeError(
            f"Missing {name} file: {path}. "
            f"Provide it at that path OR set {name}_URL env var to a direct download link."
        )
    print(f"[MorphAI] Downloading {name} -> {path}")
    _download(url, path)
    print(f"[MorphAI] {name} download complete ({os.path.getsize(path)} bytes)")

def try_load_gfpgan(model_path: str):
    """
    Lazy import GFPGAN. If torchvision/basicsr mismatch happens,
    we disable restoration rather than crashing the API.
    """
    global GFPGAN, GFPGAN_ENABLED, GFPGAN_IMPORT_ERROR
    try:
        from gfpgan import GFPGANer  # noqa
    except Exception as e:
        GFPGAN = None
        GFPGAN_ENABLED = False
        GFPGAN_IMPORT_ERROR = repr(e)
        print("[MorphAI] GFPGAN import failed -> disabled:", GFPGAN_IMPORT_ERROR)
        return

    try:
        if os.path.exists(model_path) and os.path.getsize(model_path) > 1024 * 1024:
            GFPGAN = GFPGANer(
                model_path=model_path,
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
            )
            GFPGAN_ENABLED = True
            GFPGAN_IMPORT_ERROR = None
            print("[MorphAI] GFPGAN loaded")
        else:
            GFPGAN = None
            GFPGAN_ENABLED = False
            print("[MorphAI] GFPGAN weights not found -> disabled")
    except Exception as e:
        GFPGAN = None
        GFPGAN_ENABLED = False
        GFPGAN_IMPORT_ERROR = repr(e)
        print("[MorphAI] GFPGAN init failed -> disabled:", GFPGAN_IMPORT_ERROR)


# =========================================================
# Startup
# =========================================================

@app.on_event("startup")
def startup():
    global face_app, SWAPPER

    # Ensure inswapper is present (do NOT commit big model to git; download instead)
    ensure_file(INSWAPPER_PATH, INSWAPPER_URL, "INSWAPPER")

    # FaceAnalysis (buffalo_l downloads into ~/.insightface automatically)
    face_app = FaceAnalysis(name="buffalo_l", providers=PROVIDERS)
    face_app.prepare(ctx_id=0 if USING_GPU else -1, det_size=(640, 640))

    # Swapper ONNX
    SWAPPER = get_model(INSWAPPER_PATH, providers=PROVIDERS)

    # GFPGAN optional
    if GFPGAN_URL and (not os.path.exists(GFPGAN_PATH)):
        try:
            ensure_file(GFPGAN_PATH, GFPGAN_URL, "GFPGAN")
        except Exception as e:
            print("[MorphAI] GFPGAN weights download failed:", repr(e))

    try_load_gfpgan(GFPGAN_PATH)


# =========================================================
# Helpers
# =========================================================

async def read_image(upload: UploadFile) -> np.ndarray:
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return img

def clamp_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

def pick_faces(tgt_faces, swap_all: bool) -> List:
    if swap_all:
        return list(tgt_faces)
    # pick largest face only
    return [max(tgt_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))]

# ---------- Quality detection ----------
def roi_sharpness_score(bgr: np.ndarray) -> float:
    if bgr is None or bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def should_restore_face(
    roi_bgr: np.ndarray,
    face_w: int,
    face_h: int,
    sharp_thresh: float,
    min_face_side: int,
) -> bool:
    if roi_bgr is None or roi_bgr.size == 0:
        return False
    if min(face_w, face_h) < int(min_face_side):
        return True
    return roi_sharpness_score(roi_bgr) < float(sharp_thresh)

# ---------- Color harmonization (LAB mean/std transfer + soft mask) ----------
def lab_match(src_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
    src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    for i in range(3):
        s_mean, s_std = float(src[:, :, i].mean()), float(src[:, :, i].std())
        r_mean, r_std = float(ref[:, :, i].mean()), float(ref[:, :, i].std())
        src[:, :, i] = (src[:, :, i] - s_mean) * (r_std / (s_std + 1e-6)) + r_mean

    src = np.clip(src, 0, 255).astype(np.uint8)
    return cv2.cvtColor(src, cv2.COLOR_LAB2BGR)

def soft_ellipse_mask(h: int, w: int, sigma: float) -> np.ndarray:
    mask = np.zeros((h, w), np.float32)
    center = (w // 2, h // 2)
    axes = (int(w * 0.44), int(h * 0.50))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.0, sigma))
    return mask

def harmonize(result_bgr: np.ndarray, target_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    roi = result_bgr[y1:y2, x1:x2]
    ref = target_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return result_bgr

    matched = lab_match(roi, ref)
    mask = soft_ellipse_mask(roi.shape[0], roi.shape[1], sigma=roi.shape[1] * 0.045)
    mask3 = np.dstack([mask, mask, mask])

    blended = (matched.astype(np.float32) * mask3 + roi.astype(np.float32) * (1.0 - mask3)).astype(np.uint8)

    out = result_bgr.copy()
    out[y1:y2, x1:x2] = blended
    return out

# ---------- Sharpen (unsharp mask + soft mask) ----------
def unsharp_mask(bgr: np.ndarray, amount: float, radius: float, threshold: int = 3) -> np.ndarray:
    blurred = cv2.GaussianBlur(bgr, (0, 0), radius)
    sharp = cv2.addWeighted(bgr, 1.0 + amount, blurred, -amount, 0)
    if threshold > 0:
        low_contrast = np.abs(bgr.astype(np.int16) - blurred.astype(np.int16)) < threshold
        sharp[low_contrast] = bgr[low_contrast]
    return sharp

def sharpen_face_roi(bgr: np.ndarray, bbox: Tuple[int, int, int, int], amount: float, radius: float) -> np.ndarray:
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = bbox

    bw, bh = (x2 - x1), (y2 - y1)
    pad = int(max(bw, bh) * 0.18)
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)

    roi = bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return bgr

    sharp = unsharp_mask(roi, amount=amount, radius=radius, threshold=3)
    mask = soft_ellipse_mask(roi.shape[0], roi.shape[1], sigma=roi.shape[1] * 0.03)
    mask3 = np.dstack([mask, mask, mask])
    blended = (sharp.astype(np.float32) * mask3 + roi.astype(np.float32) * (1.0 - mask3)).astype(np.uint8)

    out = bgr.copy()
    out[y1:y2, x1:x2] = blended
    return out


# =========================================================
# Routes
# =========================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "engine": "inswapper_128",
        "providers": AVAILABLE_PROVIDERS,
        "using": PROVIDERS,
        "gpu": USING_GPU,
        "gfpgan": bool(GFPGAN_ENABLED),
        "gfpgan_import_error": GFPGAN_IMPORT_ERROR,
    }

@app.get("/health")
def health():
    ok = bool(SWAPPER is not None and face_app is not None)
    return {
        "ok": ok,
        "using": PROVIDERS,
        "gpu": USING_GPU,
        "inswapper_exists": os.path.exists(INSWAPPER_PATH),
        "gfpgan_exists": os.path.exists(GFPGAN_PATH),
        "gfpgan_enabled": bool(GFPGAN_ENABLED),
    }

@app.post("/swap/single")
async def swap_single(
    source: UploadFile = File(...),
    target: UploadFile = File(...),

    # Selection
    swap_all: bool = Query(True, description="Swap all detected faces in target; if false, only the largest face."),

    # Restore controls
    restore_sharp_thresh: float = Query(85.0, ge=10.0, le=500.0),
    restore_min_face_side: int = Query(120, ge=32, le=512),
    restore_force: bool = Query(False),
    restore_disable: bool = Query(False),

    # Sharpen controls
    sharpen_amount: float = Query(0.35, ge=0.0, le=1.0),
    sharpen_radius: float = Query(1.2, ge=0.5, le=5.0),

    # Color harmonization toggle
    harmonize_enable: bool = Query(True),

    # Debug (optional JSON response instead of image)
    debug_json: bool = Query(False),
):
    if SWAPPER is None or face_app is None:
        raise HTTPException(status_code=500, detail="Models not initialized")

    src_img = await read_image(source)
    tgt_img = await read_image(target)

    src_faces = face_app.get(src_img)
    tgt_faces = face_app.get(tgt_img)

    if not src_faces:
        raise HTTPException(status_code=400, detail="No face found in source image")
    if not tgt_faces:
        raise HTTPException(status_code=400, detail="No face found in target image")

    src_face = src_faces[0]
    result = tgt_img.copy()
    tgt_original = tgt_img.copy()

    faces_to_swap = pick_faces(tgt_faces, swap_all=swap_all)

    restored_count = 0
    swapped_count = 0
    t0 = time.time()

    for face in faces_to_swap:
        swapped_count += 1
        result = SWAPPER.get(result, face, src_face, paste_back=True)

        h, w = result.shape[:2]
        bb = clamp_bbox(int(face.bbox[0]), int(face.bbox[1]), int(face.bbox[2]), int(face.bbox[3]), w, h)
        if not bb:
            continue

        # 1) color harmonize (helps with SD-like lighting consistency)
        if harmonize_enable:
            result = harmonize(result, tgt_original, bb)

        # 2) optional GFPGAN restore (auto-gated)
        if GFPGAN_ENABLED and (not restore_disable):
            x1, y1, x2, y2 = bb
            roi = result[y1:y2, x1:x2]
            if roi.size > 0:
                face_w, face_h = (x2 - x1), (y2 - y1)
                do_restore = restore_force or should_restore_face(
                    roi_bgr=roi,
                    face_w=face_w,
                    face_h=face_h,
                    sharp_thresh=float(restore_sharp_thresh),
                    min_face_side=int(restore_min_face_side),
                )
                if do_restore:
                    try:
                        # paste_back=False so we can safely assign only ROI
                        _, _, restored = GFPGAN.enhance(roi, has_aligned=False, paste_back=False)
                        if isinstance(restored, np.ndarray) and restored.shape == roi.shape:
                            result[y1:y2, x1:x2] = restored
                            restored_count += 1
                    except Exception as e:
                        # never crash the request due to restoration
                        print("[MorphAI] GFPGAN enhance failed:", repr(e))

        # 3) sharpen (recovers edge crispness after swap/restore)
        result = sharpen_face_roi(result, bb, amount=float(sharpen_amount), radius=float(sharpen_radius))

    dt_ms = int((time.time() - t0) * 1000)

    if debug_json:
        return JSONResponse({
            "swapped_faces": swapped_count,
            "restored_faces": restored_count,
            "gpu": USING_GPU,
            "using": PROVIDERS,
            "ms": dt_ms,
        })

    ok, buf = cv2.imencode(".png", result)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image")

    headers = {
        "X-Using-GPU": "1" if USING_GPU else "0",
        "X-Providers": ",".join(PROVIDERS),
        "X-Swapped-Faces": str(swapped_count),
        "X-Restored-Faces": str(restored_count),
        "X-Time-MS": str(dt_ms),
    }
    return Response(content=buf.tobytes(), media_type="image/png", headers=headers)
