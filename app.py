from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import os
import cv2
import numpy as np
import urllib.request

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

app = FastAPI(title="MorphAI FaceSwap")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://morphai-frontend.vercel.app",  # ✅ add this explicitly
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


MODEL_PATH = os.environ.get("INSWAPPER_PATH", "inswapper_128.onnx")
MODEL_URL = os.environ.get("INSWAPPER_URL", "")

def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    if not MODEL_URL:
        raise RuntimeError(
            f"Missing model file: {MODEL_PATH}. "
            f"Set INSWAPPER_URL env var to a direct download link."
        )
    print(f"Downloading model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model download complete.")

PROVIDERS = ["CPUExecutionProvider"]  # Railway CPU
face_app = FaceAnalysis(name="buffalo_l", providers=PROVIDERS)

SWAPPER = None  # ✅ cache swapper globally

@app.on_event("startup")
def startup():
    global SWAPPER
    ensure_model()
    face_app.prepare(ctx_id=-1, det_size=(640, 640))
    SWAPPER = get_model(MODEL_PATH, providers=PROVIDERS)  # ✅ load once

async def read_image(upload: UploadFile) -> np.ndarray:
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return img

def unsharp_mask(img, amount=0.35, radius=1.2, threshold=3):
    """Mild sharpening that won’t destroy skin tones."""
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    sharp = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    if threshold > 0:
        low_contrast = np.abs(img.astype(np.int16) - blurred.astype(np.int16)) < threshold
        sharp[low_contrast] = img[low_contrast]
    return sharp

def sharpen_face_roi(img, bbox, amount=0.35, radius=1.2):
    """
    Sharpen only around the face bbox using a soft elliptical mask
    to avoid halos at the edges.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # expand bbox slightly (helps keep details around eyes/cheeks)
    bw = x2 - x1
    bh = y2 - y1
    pad = int(max(bw, bh) * 0.18)

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return img

    sharp = unsharp_mask(roi, amount=amount, radius=radius, threshold=3)

    # soft elliptical mask
    mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.float32)
    center = (roi.shape[1] // 2, roi.shape[0] // 2)
    axes = (int(roi.shape[1] * 0.42), int(roi.shape[0] * 0.48))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=roi.shape[1] * 0.03)

    mask3 = np.dstack([mask, mask, mask])
    blended = (sharp.astype(np.float32) * mask3 + roi.astype(np.float32) * (1 - mask3)).astype(np.uint8)

    out = img.copy()
    out[y1:y2, x1:x2] = blended
    return out

@app.get("/")
def root():
    return {"status": "ok", "engine": "inswapper_128", "cpu": True}

@app.post("/swap/single")
async def swap_single(source: UploadFile = File(...), target: UploadFile = File(...)):
    global SWAPPER
    if SWAPPER is None:
        raise HTTPException(status_code=500, detail="Swapper not initialized")

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

    for face in tgt_faces:
        result = SWAPPER.get(result, face, src_face, paste_back=True)

        # ✅ Recover sharpness (tune amount/radius if needed)
        result = sharpen_face_roi(result, face.bbox, amount=0.35, radius=1.2)

    ok, buf = cv2.imencode(".png", result)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image")

    return Response(content=buf.tobytes(), media_type="image/png")
