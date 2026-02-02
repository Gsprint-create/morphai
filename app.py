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

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Model download settings ----
MODEL_PATH = os.environ.get("INSWAPPER_PATH", "inswapper_128.onnx")
MODEL_URL = os.environ.get("INSWAPPER_URL", "")  # set in Railway Variables

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

# ---- InsightFace init ----
PROVIDERS = ["CPUExecutionProvider"]  # Railway CPU
face_app = FaceAnalysis(name="buffalo_l", providers=PROVIDERS)

@app.on_event("startup")
def startup():
    ensure_model()
    # CPU => ctx_id=-1
    face_app.prepare(ctx_id=-1, det_size=(640, 640))

# Load swapper after model exists
def get_swapper():
    ensure_model()
    return get_model(MODEL_PATH, providers=PROVIDERS)

# ---- Helpers ----
async def read_image(upload: UploadFile) -> np.ndarray:
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return img

# ---- Routes ----
@app.get("/")
def root():
    return {"status": "ok", "engine": "inswapper_128", "cpu": True}

@app.post("/swap/single")
async def swap_single(
    source: UploadFile = File(...),
    target: UploadFile = File(...),
):
    """
    IMPORTANT:
    source = face to insert
    target = image to receive the face
    """

    src_img = await read_image(source)
    tgt_img = await read_image(target)

    src_faces = face_app.get(src_img)
    tgt_faces = face_app.get(tgt_img)

    if not src_faces:
        raise HTTPException(status_code=400, detail="No face found in source image")
    if not tgt_faces:
        raise HTTPException(status_code=400, detail="No face found in target image")

    swapper = get_swapper()

    src_face = src_faces[0]
    result = tgt_img.copy()

    # swap on all target faces
    for face in tgt_faces:
        result = swapper.get(result, face, src_face, paste_back=True)

    ok, buf = cv2.imencode(".png", result)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image")

    return Response(content=buf.tobytes(), media_type="image/png")
