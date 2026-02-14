from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import os
import pathlib
import urllib.request
import cv2
import numpy as np
import onnxruntime as ort

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# OPTIONAL restoration (GFPGAN)
# Keep import at top; if torchvision issues happen, you’ll see it immediately.
try:
    from gfpgan import GFPGANer
    GFPGAN_IMPORT_OK = True
except Exception as e:
    GFPGAN_IMPORT_OK = False
    GFPGAN_IMPORT_ERR = str(e)
    GFPGANer = None


# ============================================================
# PATHS + DOWNLOAD (production-grade)
# ============================================================

BASE_DIR = pathlib.Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

INSWAPPER_PATH = str(MODELS_DIR / "inswapper_128.onnx")
GFPGAN_PATH = str(MODELS_DIR / "GFPGANv1.4.pth")

# Provide URLs via env vars (required on Railway, optional locally)
INSWAPPER_URL = os.getenv("INSWAPPER_URL", "").strip()  # required if file not present
GFPGAN_URL = os.getenv("GFPGAN_URL", "").strip()        # optional

# If you're behind a tunnel with self-signed/cert quirks, this can help:
# (usually not needed; leave off)
ALLOW_INSECURE_DOWNLOAD = os.getenv("ALLOW_INSECURE_DOWNLOAD", "0") == "1"


def download_if_missing(path: str, url: str, label: str):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"[{label}] OK: {path} ({os.path.getsize(path)} bytes)")
        return

    if not url:
        raise RuntimeError(
            f"[{label}] Missing file: {path}. "
            f"Set {label}_URL env var to a direct download link."
        )

    print(f"[{label}] Downloading from {url} -> {path}")

    # Basic, robust download
    try:
        if ALLOW_INSECURE_DOWNLOAD:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context

        urllib.request.urlretrieve(url, path)

        # sanity size check
        if not os.path.exists(path) or os.path.getsize(path) < 1024 * 1024:
            raise RuntimeError(f"[{label}] Downloaded file looks too small: {path}")

        print(f"[{label}] Download complete: {path} ({os.path.getsize(path)} bytes)")
    except Exception as e:
        # remove partial file
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
        raise RuntimeError(f"[{label}] Download failed: {e}")


# ============================================================
# PROVIDERS (CPU on Railway, GPU locally if available)
# ============================================================

AVAILABLE_PROVIDERS = ort.get_available_providers()

if "CUDAExecutionProvider" in AVAILABLE_PROVIDERS:
    PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    USING_GPU = True
else:
    PROVIDERS = ["CPUExecutionProvider"]
    USING_GPU = False


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title="MorphAI FaceSwap", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://morphai-frontend.vercel.app",
        "https://morphai.net",
        "https://www.morphai.net",
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


# ============================================================
# GLOBAL MODELS
# ============================================================

face_app = FaceAnalysis(name="buffalo_l", providers=PROVIDERS)
SWAPPER = None
GFPGAN = None


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup():
    global SWAPPER, GFPGAN

    # Ensure required swapper model exists
    download_if_missing(INSWAPPER_PATH, INSWAPPER_URL, "INSWAPPER")

    # Prepare face detector/recognizer (buffalo_l auto-downloads)
    face_app.prepare(ctx_id=0 if USING_GPU else -1, det_size=(640, 640))

    # Load swapper once
    SWAPPER = get_model(INSWAPPER_PATH, providers=PROVIDERS)
    print(f"[INSWAPPER] Loaded with providers={PROVIDERS}")

    # Optional GFPGAN
    if GFPGAN_IMPORT_OK:
        if os.path.exists(GFPGAN_PATH) and os.path.getsize(GFPGAN_PATH) > 0:
            try:
                GFPGAN = GFPGANer(
                    model_path=GFPGAN_PATH,
                    upscale=1,
                    arch="clean",
                    channel_multiplier=2,
                    bg_upsampler=None
                )
                print("[GFPGAN] Loaded")
            except Exception as e:
                GFPGAN = None
                print("[GFPGAN] Failed to init:", e)
        elif GFPGAN_URL:
            # download then try load
            try:
                download_if_missing(GFPGAN_PATH, GFPGAN_URL, "GFPGAN")
                GFPGAN = GFPGANer(
                    model_path=GFPGAN_PATH,
                    upscale=1,
                    arch="clean",
                    channel_multiplier=2,
                    bg_upsampler=None
                )
                print("[GFPGAN] Downloaded+Loaded")
            except Exception as e:
                GFPGAN = None
                print("[GFPGAN] Not available:", e)
        else:
            print("[GFPGAN] Not found — disabled (set GFPGAN_URL or provide models/GFPGANv1.4.pth)")
    else:
        print("[GFPGAN] Import failed — disabled:", GFPGAN_IMPORT_ERR)


# ============================================================
# HEALTH / DEBUG
# ============================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "engine": "inswapper_128",
        "gpu": USING_GPU,
        "providers_available": AVAILABLE_PROVIDERS,
        "providers_using": PROVIDERS,
        "inswapper_present": os.path.exists(INSWAPPER_PATH),
        "gfpgan_present": os.path.exists(GFPGAN_PATH),
        "gfpgan_enabled": GFPGAN is not None,
        "gfpgan_import_ok": GFPGAN_IMPORT_OK,
    }


@app.get("/health")
def health():
    if SWAPPER is None:
        return JSONResponse({"status": "booting", "detail": "swapper not initialized yet"}, status_code=503)
    return {"status": "healthy"}


# ============================================================
# HELPERS
# ============================================================

def read_image(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    if not data:
        raise HTTPException(400, "Empty upload")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image")
    return img


def clamp_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


# ---------- AUTO QUALITY DETECTION ----------

def roi_sharpness_score(bgr: np.ndarray) -> float:
    """Higher = sharper (Laplacian variance)."""
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
    # restore if small face or blurry
    if roi_bgr is None or roi_bgr.size == 0:
        return False
    if min(face_w, face_h) < int(min_face_side):
        return True
    return roi_sharpness_score(roi_bgr) < float(sharp_thresh)


# ---------- COLOR HARMONIZATION ----------

def lab_match(src_bgr, ref_bgr):
    src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    for c in range(3):
        s_mean, s_std = float(src[:, :, c].mean()), float(src[:, :, c].std())
        r_mean, r_std = float(ref[:, :, c].mean()), float(ref[:, :, c].std())
        src[:, :, c] = (src[:, :, c] - s_mean) * (r_std / (s_std + 1e-6)) + r_mean

    src = np.clip(src, 0, 255).astype(np.uint8)
    return cv2.cvtColor(src, cv2.COLOR_LAB2BGR)


def harmonize(result_bgr, target_bgr, bbox):
    x1, y1, x2, y2 = bbox
    roi = result_bgr[y1:y2, x1:x2]
    ref = target_bgr[y1:y2, x1:x2]
    if roi.size == 0 or ref.size == 0:
        return result_bgr

    matched = lab_match(roi, ref)

    # soft elliptical mask
    mask = np.zeros((roi.shape[0], roi.shape[1]), np.float32)
    cv2.ellipse(
        mask,
        (roi.shape[1] // 2, roi.shape[0] // 2),
        (int(roi.shape[1] * 0.42), int(roi.shape[0] * 0.48)),
        0, 0, 360, 1.0, -1
    )
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.0, roi.shape[1] * 0.04))
    mask3 = np.dstack([mask] * 3)

    blended = (matched.astype(np.float32) * mask3 + roi.astype(np.float32) * (1.0 - mask3)).astype(np.uint8)

    out = result_bgr.copy()
    out[y1:y2, x1:x2] = blended
    return out


# ---------- SHARPEN ----------

def sharpen(img_bgr, bbox, amount=0.35, radius=1.2):
    x1, y1, x2, y2 = bbox
    roi = img_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return img_bgr

    blurred = cv2.GaussianBlur(roi, (0, 0), radius)
    sharp = cv2.addWeighted(roi, 1.0 + amount, blurred, -amount, 0)

    out = img_bgr.copy()
    out[y1:y2, x1:x2] = sharp
    return out


# ============================================================
# ROUTE
# ============================================================

@app.post("/swap/single")
async def swap_single(
    source: UploadFile = File(...),
    target: UploadFile = File(...),

    # restore controls (exposed)
    restore_sharp_thresh: float = Query(85.0, ge=10.0, le=500.0),
    restore_min_face_side: int = Query(120, ge=32, le=512),
    restore_force: bool = Query(False),
    restore_disable: bool = Query(False),

    # tuning knobs
    sharpen_amount: float = Query(0.35, ge=0.0, le=1.0),
    sharpen_radius: float = Query(1.2, ge=0.5, le=5.0),
):
    if SWAPPER is None:
        raise HTTPException(503, "Swapper not initialized yet")

    src = read_image(source)
    tgt = read_image(target)
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

        x1, y1, x2, y2 = bb
        face_w = x2 - x1
        face_h = y2 - y1

        # 2) color harmonization
        result = harmonize(result, tgt_original, bb)

        # 3) restore (GFPGAN) - auto gated unless forced
        if GFPGAN is not None and (not restore_disable):
            roi = result[y1:y2, x1:x2]
            if roi.size > 0:
                do_restore = restore_force or should_restore_face(
                    roi_bgr=roi,
                    face_w=face_w,
                    face_h=face_h,
                    sharp_thresh=float(restore_sharp_thresh),
                    min_face_side=int(restore_min_face_side),
                )

                if do_restore:
                    try:
                        # GFPGAN returns (cropped_faces, restored_faces, restored_img)
                        _, _, restored_roi = GFPGAN.enhance(
                            roi,
                            has_aligned=False,
                            paste_back=True
                        )
                        if restored_roi is not None and restored_roi.shape == roi.shape:
                            result[y1:y2, x1:x2] = restored_roi
                    except Exception as e:
                        # don’t fail the whole request
                        print("[GFPGAN] enhance failed:", e)

        # 4) sharpen
        result = sharpen(result, bb, amount=float(sharpen_amount), radius=float(sharpen_radius))

    ok, buf = cv2.imencode(".png", result)
    if not ok:
        raise HTTPException(500, "Failed to encode image")

    return Response(content=buf.tobytes(), media_type="image/png")
