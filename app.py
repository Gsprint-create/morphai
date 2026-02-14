from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import os
import cv2
import numpy as np
import onnxruntime as ort

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# OPTIONAL: restoration
try:
    from gfpgan import GFPGANer
    HAS_GFPGAN = True
except Exception:
    GFPGANer = None
    HAS_GFPGAN = False


# ============================================================
# CONFIG
# ============================================================

APP_NAME = "MorphAI FaceSwap (Hollywood)"
MODEL_PATH = os.environ.get("INSWAPPER_PATH", "models/inswapper_128.onnx")
GFPGAN_PATH = os.environ.get("GFPGAN_PATH", "models/GFPGANv1.4.pth")

ALLOW_ORIGINS = [
    "https://morphai-frontend.vercel.app",
    "https://www.morphai.net",
    "https://morphai.net",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Providers (GPU if available locally)
providers = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers()
    else ["CPUExecutionProvider"]
)
USING_GPU = providers[0] == "CUDAExecutionProvider"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
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

face_app = FaceAnalysis(name="buffalo_l", providers=providers)
SWAPPER = None
GFPGAN = None


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
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w, int(x2)); y2 = min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def roi_sharpness_score(bgr: np.ndarray) -> float:
    if bgr is None or bgr.size == 0:
        return 0.0
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def should_restore_face(
    roi_bgr: np.ndarray,
    face_w: int,
    face_h: int,
    sharp_thresh: float = 60.0,
    min_face_side: int = 110,
) -> bool:
    """
    HOLLYWOOD GATING:
    Restore only if face is SMALL *and* BLURRY.
    This avoids the plastic look on big sharp faces.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return False
    if min(face_w, face_h) > int(min_face_side):
        return False
    return roi_sharpness_score(roi_bgr) < float(sharp_thresh)


# ---------------------------
# HOLLYWOOD MASK (landmarks hull + feather)
# ---------------------------

def get_landmarks(face) -> np.ndarray | None:
    """
    Tries to use 106 landmarks (best). Falls back to 5 keypoints if needed.
    InsightFace Face object often has .landmark_2d_106 for buffalo_l.
    """
    lm = None
    if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
        lm = np.array(face.landmark_2d_106, dtype=np.int32)
    elif hasattr(face, "kps") and face.kps is not None:
        lm = np.array(face.kps, dtype=np.int32)
    return lm


def build_face_mask_from_landmarks(img_shape, landmarks: np.ndarray, feather: float = 0.06, grow: int = 12):
    """
    Creates a soft mask from landmark convex hull.
    feather: relative softness (0.03..0.15)
    grow: dilate pixels to cover cheeks/jawline
    """
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if landmarks is None or len(landmarks) < 3:
        return mask

    hull = cv2.convexHull(landmarks.reshape(-1, 1, 2))
    cv2.fillConvexPoly(mask, hull, 255)

    if grow > 0:
        k = max(3, int(grow) | 1)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=1)

    # Feather (Gaussian blur)
    sigma = max(1.0, float(max(h, w)) * float(feather))
    soft = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)

    soft_f = (soft.astype(np.float32) / 255.0)
    return soft_f  # float 0..1


# ---------------------------
# HOLLYWOOD COLOR MATCH (LAB mean/std inside mask)
# ---------------------------

def lab_match_masked(src_bgr, ref_bgr, mask_f, strength: float = 1.0):
    """
    Matches src to ref in LAB using mean/std only inside mask.
    strength: 0..1 (blend toward matched)
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return src_bgr

    src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    m = mask_f[..., None].astype(np.float32)
    eps = 1e-6

    out = src.copy()
    for c in range(3):
        s = src[..., c]
        r = ref[..., c]

        sw = (s * m[..., 0]).sum()
        rw = (r * m[..., 0]).sum()
        mw = m[..., 0].sum() + eps

        s_mean = sw / mw
        r_mean = rw / mw

        s_var = (((s - s_mean) ** 2) * m[..., 0]).sum() / mw
        r_var = (((r - r_mean) ** 2) * m[..., 0]).sum() / mw

        s_std = np.sqrt(s_var + eps)
        r_std = np.sqrt(r_var + eps)

        matched = (s - s_mean) * (r_std / (s_std + eps)) + r_mean
        out[..., c] = matched

    out = np.clip(out, 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

    # Blend by strength inside mask
    blended = (out_bgr.astype(np.float32) * (m * strength) +
               src_bgr.astype(np.float32) * (1.0 - (m * strength)))
    return np.clip(blended, 0, 255).astype(np.uint8)


# ---------------------------
# HOLLYWOOD LIGHT MATCH (L channel gain+bias inside mask)
# ---------------------------

def match_lighting_masked(src_bgr, ref_bgr, mask_f, strength: float = 0.7):
    """
    Adjusts src brightness/contrast to match ref using L channel statistics inside mask.
    strength: 0..1
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return src_bgr

    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    m = mask_f.astype(np.float32)
    eps = 1e-6

    sL = src_lab[..., 0]
    rL = ref_lab[..., 0]

    mw = m.sum() + eps
    s_mean = (sL * m).sum() / mw
    r_mean = (rL * m).sum() / mw

    s_var = (((sL - s_mean) ** 2) * m).sum() / mw
    r_var = (((rL - r_mean) ** 2) * m).sum() / mw

    s_std = np.sqrt(s_var + eps)
    r_std = np.sqrt(r_var + eps)

    gain = r_std / (s_std + eps)
    bias = r_mean - gain * s_mean

    newL = gain * sL + bias
    src_lab[..., 0] = np.clip((1.0 - strength) * sL + strength * newL, 0, 255)

    out = cv2.cvtColor(src_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out


# ---------------------------
# DETAIL POP (micro-contrast) inside mask
# ---------------------------

def detail_pop_masked(img_bgr, mask_f, sigma_s=8, sigma_r=0.15, strength=0.6):
    """
    Adds micro-contrast but avoids crunchy edges.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return img_bgr

    enhanced = cv2.detailEnhance(img_bgr, sigma_s=float(sigma_s), sigma_r=float(sigma_r))
    m = mask_f[..., None].astype(np.float32)
    out = enhanced.astype(np.float32) * (m * strength) + img_bgr.astype(np.float32) * (1.0 - (m * strength))
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------
# TARGETED SHARPEN (eyes + mouth) from keypoints
# ---------------------------

def unsharp(bgr, amount=0.35, radius=1.2):
    blur = cv2.GaussianBlur(bgr, (0, 0), float(radius))
    sharp = cv2.addWeighted(bgr, 1.0 + float(amount), blur, -float(amount), 0)
    return sharp


def circle_mask(h, w, cx, cy, r, feather=0.35):
    mask = np.zeros((h, w), np.float32)
    cv2.circle(mask, (int(cx), int(cy)), int(r), 1.0, -1)
    sigma = max(1.0, r * float(feather))
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return mask


def sharpen_keypoints(img_bgr, face, eye_strength=0.65, mouth_strength=0.35):
    """
    Uses 5 keypoints when available:
      kps: [left_eye, right_eye, nose, left_mouth, right_mouth]
    Applies gentle unsharp in small soft circles.
    """
    if not hasattr(face, "kps") or face.kps is None:
        return img_bgr

    kps = np.array(face.kps, dtype=np.float32)
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()
    sharp = unsharp(out, amount=0.40, radius=1.0)

    # scale radius from bbox
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    fw = max(20, x2 - x1)
    r_eye = int(fw * 0.09)
    r_mouth = int(fw * 0.10)

    le, re, _, lm, rm = kps

    m_eye = circle_mask(h, w, le[0], le[1], r_eye) + circle_mask(h, w, re[0], re[1], r_eye)
    m_eye = np.clip(m_eye, 0, 1)[..., None]

    mouth_cx = (lm[0] + rm[0]) * 0.5
    mouth_cy = (lm[1] + rm[1]) * 0.5
    m_mouth = circle_mask(h, w, mouth_cx, mouth_cy, r_mouth)[..., None]

    out = (sharp.astype(np.float32) * (m_eye * eye_strength) + out.astype(np.float32) * (1.0 - (m_eye * eye_strength)))
    out = (sharp.astype(np.float32) * (m_mouth * mouth_strength) + out.astype(np.float32) * (1.0 - (m_mouth * mouth_strength)))

    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------
# OPTIONAL seam blending
# ---------------------------

def seamless_clone_blend(result_bgr, target_bgr, mask_f, bbox):
    """
    OpenCV seamlessClone for the face region only.
    mask: float 0..1 -> uint8 0..255
    """
    x1, y1, x2, y2 = bbox
    roi_res = result_bgr[y1:y2, x1:x2]
    roi_tgt = target_bgr[y1:y2, x1:x2]
    if roi_res.size == 0:
        return result_bgr

    m = (mask_f[y1:y2, x1:x2] * 255.0).astype(np.uint8)
    if m.sum() < 10:
        return result_bgr

    center = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)

    # mask must be same size as full image for seamlessClone
    full_mask = np.zeros(target_bgr.shape[:2], np.uint8)
    full_mask[y1:y2, x1:x2] = m

    try:
        blended = cv2.seamlessClone(result_bgr, target_bgr, full_mask, center, cv2.NORMAL_CLONE)
        return blended
    except Exception:
        return result_bgr


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup():
    global SWAPPER, GFPGAN

    face_app.prepare(ctx_id=0 if USING_GPU else -1, det_size=(640, 640))

    if not os.path.exists(MODEL_PATH):
        print(f"[FATAL] inswapper missing: {MODEL_PATH}")
        # Don’t crash hard in prod; return a clear health error instead
        SWAPPER = None
        return

    SWAPPER = get_model(MODEL_PATH, providers=providers)

    if HAS_GFPGAN and os.path.exists(GFPGAN_PATH):
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
            print("GFPGAN failed to load:", e)
            GFPGAN = None
    else:
        GFPGAN = None
        print("GFPGAN not found — disabled")


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    using = providers
    return {
        "status": "ok" if SWAPPER is not None else "degraded",
        "engine": "inswapper_128",
        "providers": ort.get_available_providers(),
        "using": using,
        "gpu": USING_GPU,
        "gfpgan": GFPGAN is not None,
        "model_path": MODEL_PATH,
    }


@app.get("/health")
def health():
    if SWAPPER is None:
        return JSONResponse(
            status_code=503,
            content={"ok": False, "error": f"Missing model file: {MODEL_PATH}"},
        )
    return {"ok": True}


@app.post("/swap/single")
async def swap_single(
    source: UploadFile = File(...),
    target: UploadFile = File(...),

    # --------- RESTORE controls (GFPGAN) ----------
    restore_sharp_thresh: float = Query(60.0, ge=10.0, le=500.0),
    restore_min_face_side: int = Query(110, ge=32, le=512),
    restore_force: bool = Query(False),
    restore_disable: bool = Query(False),

    # --------- HOLLYWOOD blend controls ----------
    mask_feather: float = Query(0.07, ge=0.02, le=0.20),
    mask_grow: int = Query(12, ge=0, le=60),

    color_strength: float = Query(0.85, ge=0.0, le=1.0),
    light_strength: float = Query(0.70, ge=0.0, le=1.0),
    detail_strength: float = Query(0.55, ge=0.0, le=1.0),

    eye_detail: float = Query(0.65, ge=0.0, le=1.0),
    mouth_detail: float = Query(0.35, ge=0.0, le=1.0),

    use_seamless_clone: bool = Query(False),
):
    if SWAPPER is None:
        raise HTTPException(503, f"Server missing model file: {MODEL_PATH}")

    src = read_image(source)
    tgt = read_image(target)
    tgt_original = tgt.copy()

    src_faces = face_app.get(src)
    tgt_faces = face_app.get(tgt)

    if not src_faces:
        raise HTTPException(400, "No face found in source")
    if not tgt_faces:
        raise HTTPException(400, "No face found in target")

    src_face = src_faces[0]
    result = tgt.copy()

    for face in tgt_faces:
        # 1) Swap
        result = SWAPPER.get(result, face, src_face, paste_back=True)

        # bbox safe
        h, w = result.shape[:2]
        bb = clamp_bbox(*[int(v) for v in face.bbox], w, h)
        if not bb:
            continue

        x1, y1, x2, y2 = bb
        roi_res = result[y1:y2, x1:x2]
        roi_tgt = tgt_original[y1:y2, x1:x2]
        if roi_res.size == 0:
            continue

        # 2) Build landmark mask (best realism)
        lms = get_landmarks(face)
        mask_f = build_face_mask_from_landmarks(result.shape, lms, feather=mask_feather, grow=mask_grow)

        # 3) Hollywood color match (skin tone / temp)
        result = lab_match_masked(result, tgt_original, mask_f, strength=color_strength)

        # 4) Hollywood lighting match (shadows + exposure)
        result = match_lighting_masked(result, tgt_original, mask_f, strength=light_strength)

        # refresh ROI after adjustments
        roi_res = result[y1:y2, x1:x2]

        # 5) Restore (GFPGAN) only when SMALL + BLURRY (or forced)
        if (GFPGAN is not None) and (not restore_disable):
            do_restore = restore_force or should_restore_face(
                roi_bgr=roi_res,
                face_w=(x2 - x1),
                face_h=(y2 - y1),
                sharp_thresh=float(restore_sharp_thresh),
                min_face_side=int(restore_min_face_side),
            )
            if do_restore:
                try:
                    _, _, restored = GFPGAN.enhance(roi_res, has_aligned=False, paste_back=True)
                    if restored is not None and restored.shape == roi_res.shape:
                        result[y1:y2, x1:x2] = restored
                except Exception as e:
                    print("GFPGAN failed:", e)

        # 6) Micro-texture pop (inside mask only)
        result = detail_pop_masked(result, mask_f, sigma_s=8, sigma_r=0.15, strength=detail_strength)

        # 7) Eyes + mouth only (prevents crunchy skin)
        result = sharpen_keypoints(result, face, eye_strength=eye_detail, mouth_strength=mouth_detail)

        # 8) Optional seamless clone (great for hard lighting changes)
        if use_seamless_clone:
            result = seamless_clone_blend(result, tgt_original, mask_f, bb)

    ok, buf = cv2.imencode(".png", result)
    if not ok:
        raise HTTPException(500, "Failed to encode image")

    return Response(content=buf.tobytes(), media_type="image/png")
