"""
MorphAI ULTRA - production-grade FaceSwap API (FastAPI)
- InsightFace detection (buffalo_l)
- InSwapper 128 (ONNX)
- Optional GFPGAN v1.4 restoration (auto-gated + API thresholds)
- ULTRA realism pipeline:
  1) swap
  2) landmark mask + edge matting
  3) masked LAB color match
  4) masked luminance match
  5) masked shading harmonization (low-freq illumination)
  6) optional GFPGAN restore (auto-gated)
  7) Poisson (seamlessClone) blend onto original target
  8) selective feature detail boost
  9) micro film grain in face region
  10) masked sharpen (final)

Model download:
- Set env INSWAPPER_URL to a direct download link for inswapper_128.onnx
- Set env GFPGAN_URL to a direct download link for GFPGANv1.4.pth (optional)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import os
import time
import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# OPTIONAL restoration
try:
    from gfpgan import GFPGANer
except Exception:
    GFPGANer = None


# =========================================================
# PATHS / CONFIG
# =========================================================

HERE = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(HERE, "models"))
os.makedirs(MODEL_DIR, exist_ok=True)

INSWAPPER_PATH = os.path.join(MODEL_DIR, "inswapper_128.onnx")
GFPGAN_PATH = os.path.join(MODEL_DIR, "GFPGANv1.4.pth")

INSWAPPER_URL = os.getenv("INSWAPPER_URL", "").strip()  # required if not present
GFPGAN_URL = os.getenv("GFPGAN_URL", "").strip()        # optional

# If you want to force CPU on any machine:
FORCE_CPU = os.getenv("FORCE_CPU", "0").lower() in ("1", "true", "yes", "y")

# CORS
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
allow_origins = ["*"] if CORS_ORIGINS.strip() == "*" else [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]

# InsightFace det size
DET_SIZE = int(os.getenv("DET_SIZE", "640"))

# Download sanity: reject tiny/HTML files
MIN_MODEL_BYTES = int(os.getenv("MIN_MODEL_BYTES", str(5 * 1024 * 1024)))  # 5MB


# =========================================================
# PROVIDERS (GPU if available)
# =========================================================

def pick_providers():
    available = ort.get_available_providers()
    if FORCE_CPU:
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


PROVIDERS = pick_providers()
USING_GPU = (len(PROVIDERS) > 0 and PROVIDERS[0] == "CUDAExecutionProvider")


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(title="MorphAI ULTRA FaceSwap API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# GLOBAL MODELS
# =========================================================

face_app = FaceAnalysis(name="buffalo_l", providers=PROVIDERS)
SWAPPER = None
GFPGAN = None


# =========================================================
# UTIL: DOWNLOAD / ENSURE MODELS
# =========================================================

def _download_file(url: str, dst: str, timeout=60):
    import urllib.request

    tmp = dst + ".tmp"
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except:
            pass

    req = urllib.request.Request(url, headers={"User-Agent": "MorphAI-ULTRA/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        # Basic content-type sanity (some hosts return HTML 404 pages)
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "text/html" in ctype:
            raise RuntimeError(f"Download looks like HTML (Content-Type={ctype}). URL may be wrong.")
        data = r.read()

    with open(tmp, "wb") as f:
        f.write(data)

    size = os.path.getsize(tmp)
    if size < MIN_MODEL_BYTES:
        # avoid committing an HTML error page / tiny file
        # keep a hint for debugging
        preview = b""
        try:
            with open(tmp, "rb") as f:
                preview = f.read(200)
        except:
            pass
        try:
            os.remove(tmp)
        except:
            pass
        raise RuntimeError(f"Downloaded file too small ({size} bytes). Preview={preview!r}")

    os.replace(tmp, dst)


def ensure_model(path: str, url: str, friendly_name: str, required: bool):
    if os.path.exists(path) and os.path.getsize(path) >= MIN_MODEL_BYTES:
        return True

    if not url:
        if required:
            raise RuntimeError(f"Missing model file: {path}. Set {friendly_name}_URL env var to a direct download link.")
        return False

    os.makedirs(os.path.dirname(path), exist_ok=True)
    _download_file(url, path)
    return True


def ensure_models():
    # InSwapper is required
    ensure_model(INSWAPPER_PATH, INSWAPPER_URL, "INSWAPPER", required=True)

    # GFPGAN is optional
    if GFPGANer is None:
        return False
    try:
        ok = ensure_model(GFPGAN_PATH, GFPGAN_URL, "GFPGAN", required=False)
        return ok
    except Exception:
        return False


# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
def startup():
    global SWAPPER, GFPGAN, PROVIDERS, USING_GPU

    # Re-evaluate providers at boot (useful if env changed)
    PROVIDERS = pick_providers()
    USING_GPU = (len(PROVIDERS) > 0 and PROVIDERS[0] == "CUDAExecutionProvider")

    # Ensure models exist (download if needed)
    try:
        gfpgan_ok = ensure_models()
    except Exception as e:
        # Fail fast if InSwapper missing
        raise RuntimeError(str(e))

    # Prepare InsightFace
    face_app.prepare(ctx_id=0 if USING_GPU else -1, det_size=(DET_SIZE, DET_SIZE))

    # Load swapper
    if not os.path.exists(INSWAPPER_PATH):
        raise RuntimeError(f"Missing model file: {INSWAPPER_PATH}")
    SWAPPER = get_model(INSWAPPER_PATH, providers=PROVIDERS)

    # Optional GFPGAN
    GFPGAN = None
    if gfpgan_ok and os.path.exists(GFPGAN_PATH) and GFPGANer is not None:
        try:
            GFPGAN = GFPGANer(
                model_path=GFPGAN_PATH,
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
            )
            print("GFPGAN loaded")
        except Exception as e:
            GFPGAN = None
            print("GFPGAN load failed:", e)
    else:
        print("GFPGAN not found/disabled")


# =========================================================
# ROUTES: ROOT + HEALTH
# =========================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "engine": "inswapper_128",
        "providers": ort.get_available_providers(),
        "using": PROVIDERS,
        "gpu": USING_GPU,
        "models": {
            "inswapper": os.path.exists(INSWAPPER_PATH),
            "gfpgan": bool(GFPGAN is not None),
        },
        "ultra": True,
    }


@app.get("/health")
def health():
    return root()


# =========================================================
# HELPERS: IO / BBOX
# =========================================================

def read_image(file: UploadFile):
    data = file.file.read()
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


# =========================================================
# AUTO QUALITY DETECTION (restore gating)
# =========================================================

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


# =========================================================
# ULTRA: LANDMARK MASK
# =========================================================

def get_landmarks(face):
    if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
        return face.landmark_2d_106.astype(np.int32)
    if hasattr(face, "kps") and face.kps is not None:
        return face.kps.astype(np.int32)
    return None


def build_face_mask(shape_hw, landmarks, bbox, feather=22, erode=6):
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)

    if landmarks is not None and len(landmarks) >= 3:
        hull = cv2.convexHull(landmarks.reshape(-1, 1, 2))
        cv2.fillConvexPoly(mask, hull, 255)
    else:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        ax, ay = max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2)
        cv2.ellipse(mask, (cx, cy), (int(ax * 0.95), int(ay * 1.05)), 0, 0, 360, 255, -1)

    if erode > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(erode), int(erode)))
        mask = cv2.erode(mask, k, iterations=1)

    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), float(feather))

    return (mask.astype(np.float32) / 255.0)


# =========================================================
# ULTRA: MASKED COLOR/LIGHT MATCH
# =========================================================

def lab_match_masked(face_bgr, ref_bgr, mask_f):
    src = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(ref_bgr,  cv2.COLOR_BGR2LAB).astype(np.float32)

    m = mask_f[..., None]
    eps = 1e-6

    for i in range(3):
        s = src[..., i:i+1]
        r = ref[..., i:i+1]
        s_mean = float((s * m).sum() / (m.sum() + eps))
        r_mean = float((r * m).sum() / (m.sum() + eps))
        s_var  = float((((s - s_mean) ** 2) * m).sum() / (m.sum() + eps))
        r_var  = float((((r - r_mean) ** 2) * m).sum() / (m.sum() + eps))
        s_std, r_std = np.sqrt(s_var + eps), np.sqrt(r_var + eps)

        src[..., i:i+1] = ((s - s_mean) * (r_std / (s_std + eps)) + r_mean)

    out = np.clip(src, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

    out = (out.astype(np.float32) * m + face_bgr.astype(np.float32) * (1 - m)).astype(np.uint8)
    return out


def match_luminance_masked(face_bgr, ref_bgr, mask_f, strength=0.85):
    eps = 1e-6
    m = mask_f.astype(np.float32)

    fyuv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    ryuv = cv2.cvtColor(ref_bgr,  cv2.COLOR_BGR2YCrCb).astype(np.float32)

    fY = fyuv[..., 0]
    rY = ryuv[..., 0]

    f_mean = float((fY * m).sum() / (m.sum() + eps))
    r_mean = float((rY * m).sum() / (m.sum() + eps))
    f_var  = float((((fY - f_mean) ** 2) * m).sum() / (m.sum() + eps))
    r_var  = float((((rY - r_mean) ** 2) * m).sum() / (m.sum() + eps))
    f_std, r_std = np.sqrt(f_var + eps), np.sqrt(r_var + eps)

    Y = ((fY - f_mean) * (r_std / (f_std + eps)) + r_mean)
    fyuv[..., 0] = (float(strength) * Y + (1.0 - float(strength)) * fY)

    out = np.clip(fyuv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)


def shading_harmonize(face_bgr, ref_bgr, mask_f, blur_sigma=18, strength=0.7):
    eps = 1e-6
    m = mask_f[..., None].astype(np.float32)

    f = face_bgr.astype(np.float32)
    r = ref_bgr.astype(np.float32)

    f_low = cv2.GaussianBlur(f, (0, 0), float(blur_sigma))
    r_low = cv2.GaussianBlur(r, (0, 0), float(blur_sigma))

    shade = (r_low + eps) / (f_low + eps)
    shade = np.clip(shade, 0.6, 1.6)

    out = f * (float(strength) * shade + (1.0 - float(strength)))
    out = out * m + f * (1 - m)
    return np.clip(out, 0, 255).astype(np.uint8)


# =========================================================
# ULTRA: POISSON BLEND + DETAIL + GRAIN + SHARPEN
# =========================================================

def poisson_blend(src_bgr, dst_bgr, mask_f, center_xy):
    mask_u8 = np.clip(mask_f * 255.0, 0, 255).astype(np.uint8)
    return cv2.seamlessClone(src_bgr, dst_bgr, mask_u8, center_xy, cv2.NORMAL_CLONE)


def feature_detail_boost(face_bgr, landmarks, amount=0.28):
    if landmarks is None or len(landmarks) < 5 or amount <= 0:
        return face_bgr

    h, w = face_bgr.shape[:2]
    out = face_bgr.copy()

    pts = landmarks
    key = [pts[0], pts[1], pts[-2], pts[-1]]  # eyes + mouth corners

    mask = np.zeros((h, w), np.float32)
    rad = int(min(h, w) * 0.06)
    rad = max(6, rad)
    for (x, y) in key:
        cv2.circle(mask, (int(x), int(y)), rad, 1.0, -1)

    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=min(h, w) * 0.03)
    mask3 = np.dstack([mask] * 3)

    blur = cv2.GaussianBlur(out, (0, 0), 1.0)
    sharp = cv2.addWeighted(out, 1.0 + float(amount), blur, -float(amount), 0)

    out = (sharp.astype(np.float32) * mask3 + out.astype(np.float32) * (1 - mask3)).astype(np.uint8)
    return out


def add_film_grain(img_bgr, mask_f, sigma=1.8):
    if sigma <= 0:
        return img_bgr
    noise = np.random.normal(0, float(sigma), img_bgr.shape).astype(np.float32)
    out = img_bgr.astype(np.float32) + noise * (mask_f[..., None].astype(np.float32))
    return np.clip(out, 0, 255).astype(np.uint8)


def masked_sharpen(img_bgr, mask_f, amount=0.30, radius=1.2):
    if amount <= 0:
        return img_bgr
    blur = cv2.GaussianBlur(img_bgr, (0, 0), float(radius))
    sharp = cv2.addWeighted(img_bgr, 1.0 + float(amount), blur, -float(amount), 0)

    m3 = np.dstack([mask_f] * 3).astype(np.float32)
    out = sharp.astype(np.float32) * m3 + img_bgr.astype(np.float32) * (1 - m3)
    return np.clip(out, 0, 255).astype(np.uint8)


# =========================================================
# MAIN ROUTE
# =========================================================

@app.post("/swap/single")
async def swap_single(
    source: UploadFile = File(...),
    target: UploadFile = File(...),

    # --- Restore gating (exposed) ---
    restore_sharp_thresh: float = Query(85.0, ge=10.0, le=500.0),
    restore_min_face_side: int = Query(120, ge=32, le=512),
    restore_force: bool = Query(False),
    restore_disable: bool = Query(False),

    # --- ULTRA controls (exposed) ---
    mask_feather: float = Query(22.0, ge=0.0, le=80.0),
    mask_erode: int = Query(6, ge=0, le=40),

    lab_enable: bool = Query(True),
    lum_strength: float = Query(0.85, ge=0.0, le=1.0),
    shading_strength: float = Query(0.70, ge=0.0, le=1.0),
    shading_blur: float = Query(18.0, ge=2.0, le=60.0),

    poisson_enable: bool = Query(True),
    feature_detail: float = Query(0.28, ge=0.0, le=1.0),
    grain_sigma: float = Query(1.8, ge=0.0, le=12.0),

    sharpen_amount: float = Query(0.30, ge=0.0, le=1.0),
    sharpen_radius: float = Query(1.2, ge=0.5, le=6.0),
):
    if SWAPPER is None:
        raise HTTPException(500, "Server not ready: swapper not loaded")
    if not os.path.exists(INSWAPPER_PATH):
        raise HTTPException(500, f"Server missing model file: {os.path.relpath(INSWAPPER_PATH, HERE)}")

    src = read_image(source)
    tgt = read_image(target)
    tgt_original = tgt.copy()

    src_faces = face_app.get(src)
    tgt_faces = face_app.get(tgt)

    if not src_faces or not tgt_faces:
        raise HTTPException(400, "Face not detected")

    src_face = src_faces[0]
    result = tgt.copy()

    H, W = result.shape[:2]

    for face in tgt_faces:
        # 1) swap
        result = SWAPPER.get(result, face, src_face, paste_back=True)

        bb = clamp_bbox(*[int(v) for v in face.bbox], W, H)
        if not bb:
            continue

        x1, y1, x2, y2 = bb
        roi_swapped = result[y1:y2, x1:x2].copy()
        roi_ref = tgt_original[y1:y2, x1:x2].copy()
        if roi_swapped.size == 0 or roi_ref.size == 0:
            continue

        # Landmarks (shift into ROI coords)
        lm = None
        lm_full = get_landmarks(face)
        if lm_full is not None:
            lm = lm_full.copy()
            lm[:, 0] -= x1
            lm[:, 1] -= y1

        # 2) mask + matting
        mask_f = build_face_mask(
            (roi_swapped.shape[0], roi_swapped.shape[1]),
            lm,
            (0, 0, roi_swapped.shape[1], roi_swapped.shape[0]),
            feather=float(mask_feather),
            erode=int(mask_erode),
        )

        # 3) masked LAB color match
        if lab_enable:
            roi_swapped = lab_match_masked(roi_swapped, roi_ref, mask_f)

        # 4) luminance match
        if lum_strength > 0:
            roi_swapped = match_luminance_masked(roi_swapped, roi_ref, mask_f, strength=float(lum_strength))

        # 5) shading harmonize
        if shading_strength > 0:
            roi_swapped = shading_harmonize(
                roi_swapped, roi_ref, mask_f,
                blur_sigma=float(shading_blur),
                strength=float(shading_strength),
            )

        # Write ROI back pre-restore / pre-poisson
        result[y1:y2, x1:x2] = roi_swapped

        # 6) optional GFPGAN (auto gated)
        if GFPGAN is not None and (not restore_disable):
            roi_now = result[y1:y2, x1:x2]
            do_restore = restore_force or should_restore_face(
                roi_bgr=roi_now,
                face_w=(x2 - x1),
                face_h=(y2 - y1),
                sharp_thresh=float(restore_sharp_thresh),
                min_face_side=int(restore_min_face_side),
            )
            if do_restore and roi_now.size > 0:
                try:
                    # GFPGAN returns restored image
                    _, _, restored = GFPGAN.enhance(roi_now, has_aligned=False, paste_back=True)
                    if restored is not None and restored.shape == roi_now.shape:
                        # keep only within mask
                        m3 = np.dstack([mask_f] * 3).astype(np.float32)
                        blended = restored.astype(np.float32) * m3 + roi_now.astype(np.float32) * (1 - m3)
                        result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
                except Exception as e:
                    print("GFPGAN failed:", e)

        # 7) Poisson blend onto ORIGINAL target (best realism)
        # Build full-image mask for seamlessClone
        full_mask = np.zeros((H, W), np.float32)
        full_mask[y1:y2, x1:x2] = mask_f

        if poisson_enable:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            try:
                result = poisson_blend(result, tgt_original, full_mask, (int(cx), int(cy)))
            except Exception as e:
                # If seamlessClone fails, keep non-poisson result
                print("Poisson failed:", e)

        # 8) selective detail boost (eyes/lips area)
        roi_after = result[y1:y2, x1:x2].copy()
        if roi_after.size > 0 and feature_detail > 0:
            roi_after = feature_detail_boost(roi_after, lm, amount=float(feature_detail))
            # write back masked
            m3 = np.dstack([mask_f] * 3).astype(np.float32)
            base = result[y1:y2, x1:x2].astype(np.float32)
            out_roi = roi_after.astype(np.float32) * m3 + base * (1 - m3)
            result[y1:y2, x1:x2] = np.clip(out_roi, 0, 255).astype(np.uint8)

        # 9) micro grain in face region
        if grain_sigma > 0:
            result = add_film_grain(result, full_mask, sigma=float(grain_sigma))

        # 10) final masked sharpen
        if sharpen_amount > 0:
            result = masked_sharpen(result, full_mask, amount=float(sharpen_amount), radius=float(sharpen_radius))

    ok, buf = cv2.imencode(".png", result)
    if not ok:
        raise HTTPException(500, "Encode failed")

    return Response(content=buf.tobytes(), media_type="image/png")


# =========================================================
# FRIENDLY ERROR HANDLER (optional)
# =========================================================

@app.exception_handler(RuntimeError)
def runtime_error_handler(_, exc: RuntimeError):
    return JSONResponse(status_code=500, content={"error": str(exc)})
