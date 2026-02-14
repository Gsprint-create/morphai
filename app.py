from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

import os, cv2, numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

def ensure_torchvision_functional_tensor():
    """
    BasicSR (used by GFPGAN) may import torchvision.transforms.functional_tensor
    which is missing in some torchvision builds. We shim it.
    """
    try:
        from torchvision.transforms.functional_tensor import rgb_to_grayscale  # noqa: F401
        return
    except Exception:
        import sys, types
        from torchvision.transforms.functional import rgb_to_grayscale

        m = types.ModuleType("torchvision.transforms.functional_tensor")
        m.rgb_to_grayscale = rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = m




# =========================
# CONFIG
# =========================

import os, pathlib, urllib.request

BASE_DIR = pathlib.Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = str(MODELS_DIR / "inswapper_128.onnx")
MODEL_URL  = os.getenv("INSWAPPER_URL", "")  # set on Railway

GFPGAN_PATH = str(MODELS_DIR / "GFPGANv1.4.pth")
GFPGAN_URL  = os.getenv("GFPGAN_URL", "")    # optional

def ensure_file(path: str, url: str, label: str):
    if os.path.exists(path):
        return
    if not url:
        raise RuntimeError(f"Missing {label}: {path}. Set {label}_URL env var.")
    print(f"Downloading {label} -> {path}")
    urllib.request.urlretrieve(url, path)
    print(f"{label} downloaded.")


providers = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers()
    else ["CPUExecutionProvider"]
)

USING_GPU = providers[0] == "CUDAExecutionProvider"


# =========================
# FASTAPI
# =========================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://morphai-frontend.vercel.app",
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


# =========================
# GLOBAL MODELS
# =========================

face_app = FaceAnalysis(name="buffalo_l", providers=providers)
SWAPPER = None
GFPGAN = None


# =========================
# STARTUP
# =========================

@app.on_event("startup")
def startup():
    global SWAPPER, GFPGAN

    face_app.prepare(ctx_id=0 if USING_GPU else -1, det_size=(640, 640))
    SWAPPER = get_model(MODEL_PATH, providers=providers)

    # ✅ prevent GFPGAN/BasicSR import crash
    ensure_torchvision_functional_tensor()

    if os.path.exists(GFPGAN_PATH):
        from gfpgan import GFPGANer  # import AFTER shim
        GFPGAN = GFPGANer(
            model_path=GFPGAN_PATH,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None
        )
        print("GFPGAN loaded")
    else:
        print("GFPGAN not found — disabled")



# =========================
# HELPERS
# =========================

def read_image(file: UploadFile):
    data = file.file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image")
    return img


def clamp_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


# ---------- AUTO QUALITY DETECTION ----------

def roi_sharpness_score(bgr: np.ndarray) -> float:
    """Higher = sharper. Uses Laplacian variance."""
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
    """
    Restore if:
    - face is small, OR
    - ROI is soft/blurry (low Laplacian variance)
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return False

    if min(face_w, face_h) < int(min_face_side):
        return True

    sharp = roi_sharpness_score(roi_bgr)
    return sharp < float(sharp_thresh)


# ---------- COLOR HARMONIZATION ----------

def lab_match(src, ref):
    src = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)

    for i in range(3):
        s_mean, s_std = src[:, :, i].mean(), src[:, :, i].std()
        r_mean, r_std = ref[:, :, i].mean(), ref[:, :, i].std()
        src[:, :, i] = (src[:, :, i] - s_mean) * (r_std / (s_std + 1e-6)) + r_mean

    src = np.clip(src, 0, 255).astype(np.uint8)
    return cv2.cvtColor(src, cv2.COLOR_LAB2BGR)


def harmonize(result, target, bbox):
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
    mask = cv2.GaussianBlur(mask, (0, 0), roi.shape[1] * 0.04)
    mask = np.dstack([mask] * 3)

    blended = (matched * mask + roi * (1 - mask)).astype(np.uint8)

    out = result.copy()
    out[y1:y2, x1:x2] = blended
    return out


# ---------- SHARPEN ----------

def sharpen(img, bbox):
    x1, y1, x2, y2 = bbox
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return img

    blur = cv2.GaussianBlur(roi, (0, 0), 1.2)
    sharp = cv2.addWeighted(roi, 1.35, blur, -0.35, 0)

    out = img.copy()
    out[y1:y2, x1:x2] = sharp
    return out


# =========================
# ROUTE
# =========================

@app.post("/swap/single")
async def swap_single(
    source: UploadFile = File(...),
    target: UploadFile = File(...),

    # ✅ exposed controls
    restore_sharp_thresh: float = Query(85.0, ge=10.0, le=500.0),
    restore_min_face_side: int = Query(120, ge=32, le=512),
    restore_force: bool = Query(False),
    restore_disable: bool = Query(False),
):
    src = read_image(source)
    tgt = read_image(target)
    tgt_original = tgt.copy()

    src_faces = face_app.get(src)
    tgt_faces = face_app.get(tgt)

    if not src_faces or not tgt_faces:
        raise HTTPException(400, "Face not detected")

    src_face = src_faces[0]
    result = tgt.copy()

    for face in tgt_faces:
        # 1 swap
        result = SWAPPER.get(result, face, src_face, paste_back=True)

        # bbox safe
        h, w = result.shape[:2]
        bb = clamp_bbox(*[int(v) for v in face.bbox], w, h)
        if not bb:
            continue

        # 2 harmonize color
        result = harmonize(result, tgt_original, bb)

        # 3 restore (auto-gated)
        if GFPGAN and (not restore_disable):
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
                        _, _, rest = GFPGAN.enhance(
                            roi,
                            has_aligned=False,
                            paste_back=True
                        )
                        if rest.shape == roi.shape:
                            result[y1:y2, x1:x2] = rest
                    except Exception as e:
                        print("GFPGAN failed:", e)

        # 4 sharpen
        result = sharpen(result, bb)

    ok, buf = cv2.imencode(".png", result)
    if not ok:
        raise HTTPException(500, "Encode failed")

    return Response(buf.tobytes(), media_type="image/png")
