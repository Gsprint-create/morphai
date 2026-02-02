from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import cv2
import numpy as np
import os
import urllib.request

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

app = FastAPI(title="MorphAI FaceSwap")

# ✅ CORS: wildcard like "https://*.vercel.app" does NOT work in allow_origins.
# Use allow_origin_regex for Vercel preview deployments.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        # add your custom domain here if you have one:
        # "https://morphai.yourdomain.com",
    ],
    allow_origin_regex=r"^https:\/\/.*\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Model download (Railway)
# -------------------------
MODEL_PATH = "inswapper_128.onnx"
MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx"

if not os.path.exists(MODEL_PATH):
    print("Downloading inswapper_128.onnx...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# -------------------------
# Load InsightFace
# -------------------------
PROVIDERS = ["CPUExecutionProvider"]  # CPU on Railway by default
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "inswapper_128.onnx"

face_app = FaceAnalysis(name="buffalo_l", providers=PROVIDERS)
face_app.prepare(ctx_id=-1, det_size=(640, 640))

swapper = get_model(MODEL_PATH, providers=PROVIDERS)

# -------------------------
# Helpers
# -------------------------
async def read_image(upload: UploadFile) -> np.ndarray:
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return img

# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    return {"status": "ok", "engine": "inswapper_128"}

@app.post("/swap/single")
async def swap_single(
    source: UploadFile = File(...),
    target: UploadFile = File(...),
):
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
        result = swapper.get(result, face, src_face, paste_back=True)

    ok, buf = cv2.imencode(".png", result)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image")

    return Response(content=buf.tobytes(), media_type="image/png")

