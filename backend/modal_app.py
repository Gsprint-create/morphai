# modal_app.py
# Fast Face Swap service on Modal (FastAPI + InsightFace/Inswapper)
# - GPU-only ONNX Runtime for speed
# - Keep-warm container to reduce cold starts
# - OpenCV imdecode + optional downscale for big photos
# - 3.9-safe typing (Union[...] instead of |)

import io
from typing import List, Union

import modal
import numpy as np

# --------------------------
# Modal App & Container Image
# --------------------------
app = modal.App("face-swap-app")

# Cache model files between runs to cut cold-start time
models_volume = modal.Volume.from_name("insightface-cache", create_if_missing=True)

# Build runtime image
modal_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        # web server
        "fastapi",
        "uvicorn[standard]",
        "python-multipart",
        # image + math
        "pillow",            # (kept for compatibility; we encode via cv2)
        "numpy",
        "opencv-python",
        # runtime + models
        "onnxruntime-gpu==1.17.1",
        "insightface==0.7.3",
    )
    .run_commands("mkdir -p /root/.insightface/models")
)

# --------------------------
# Globals (per container)
# --------------------------
face_analyzer = None
face_swapper = None

# Prefer faster pools & fall back if busy
GPU: Union[str, List[str]] = ["L4", "A10", "T4"]

# Downscale very large images to speed up detection/compositing
MAX_LONG_SIDE = 1600  # px (try 1280/1024 for more speed, slight quality tradeoff)


def _imdecode_bgr(buf: bytes):
    """Fast bytes -> BGR ndarray using OpenCV."""
    import cv2
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise ValueError("Failed to decode image")
    return img


def _resize_if_needed(bgr, max_side: int = MAX_LONG_SIDE):
    import cv2
    h, w = bgr.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return bgr
    scale = max_side / float(long_side)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _bgr_to_png_bytes(img_bgr) -> bytes:
    """Encode BGR ndarray -> PNG bytes (no PIL roundtrip)."""
    import cv2
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def _load_models():
    """Load InsightFace detector and Inswapper once per container."""
    global face_analyzer, face_swapper
    if face_analyzer is not None and face_swapper is not None:
        return

    import insightface
    from insightface.app import FaceAnalysis

    # GPU only for best throughput; if a machine has no GPU, Modal will schedule accordingly
    providers = ["CUDAExecutionProvider"]

    # Detector + landmarks
    face_analyzer = FaceAnalysis(
        name="buffalo_l",
        providers=providers,
        root="/root/.insightface",
    )
    # 640x640 is a solid balance; raise if you miss tiny faces
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

    # Face swapper
    face_swapper = insightface.model_zoo.get_model(
        "inswapper_128.onnx",
        root="/root/.insightface",
        download=True,
        providers=providers,
    )


@app.function(
    image=modal_image,
    gpu=GPU,
    timeout=600,
    keep_warm=1,  # keep one worker hot to reduce cold starts
    volumes={"/root/.insightface": models_volume},
    # Optional extra CPU/RAM for preprocessing if needed:
    # cpu=2.0, memory=4096,
)
@modal.asgi_app()
def fastapi_app():
    """ASGI entrypoint exposing POST /swap"""
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response
    import cv2  # noqa: F401

    # Make cv2 visible to helper functions defined at module scope
    globals()["cv2"] = cv2

    _load_models()

    api = FastAPI(title="Face Swap Service (Modal)")

    # During testing it's fine to allow "*".
    # In production, restrict to your Vercel/custom domain(s).
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # e.g. ["https://your-app.vercel.app", "https://your-domain.com"]
        allow_credentials=False,
        allow_methods=["POST", "OPTIONS"],
        allow_headers=["*"],
        max_age=3600,
    )

    @api.post("/swap", summary="Swap the face from `source` onto `target`")
    async def swap_face(
        source: UploadFile = File(..., description="Source (donor) face image"),
        target: UploadFile = File(..., description="Target (destination) image"),
    ):
        try:
            src_bytes = await source.read()
            dst_bytes = await target.read()

            # Fast decode + optional downscale of target
            src_bgr = _imdecode_bgr(src_bytes)
            dst_bgr = _imdecode_bgr(dst_bytes)
            dst_bgr = _resize_if_needed(dst_bgr, MAX_LONG_SIDE)

            # Detect
            src_faces = face_analyzer.get(src_bgr)
            if not src_faces:
                raise HTTPException(status_code=422, detail="No face detected in source image.")
            dst_faces = face_analyzer.get(dst_bgr)
            if not dst_faces:
                raise HTTPException(status_code=422, detail="No face detected in target image.")

            src_face = src_faces[0]
            dst_face = dst_faces[0]

            # Swap (paste back into original target frame)
            result_bgr = face_swapper.get(dst_bgr, dst_face, src_face, paste_back=True)

            return Response(content=_bgr_to_png_bytes(result_bgr), media_type="image/png")

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Swap failed: {e}")

    return api
