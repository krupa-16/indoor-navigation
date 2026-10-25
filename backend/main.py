
from pathlib import Path
from io import BytesIO

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

app = FastAPI(title="Indoor Navigation API")
from localization import router as localization_router
app.include_router(localization_router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).parent

MODELS = {
    "kaggle": YOLO(str(BASE_DIR / "best.pt")),
    "original": YOLO(str(BASE_DIR / "best_old.pt")),
    "general": YOLO(str(BASE_DIR / "yolov8n.pt")),
}


@app.get("/")
def home():
    return {"message": "Indoor Navigation API running"}


@app.get("/api/models")
def list_models():
    return {
        key: {
            "classes": len(model.names),
            "names": model.names,
        }
        for key, model in MODELS.items()
    }


@app.post("/api/detect")
async def detect(
    file: UploadFile = File(...),
    model_name: str = Form("kaggle"),
):
    if model_name not in MODELS:
        raise HTTPException(400, "Unknown model selected.")

    if file.content_type not in {
        "image/jpeg", "image/png", "image/webp"
    }:
        raise HTTPException(400, "Upload a JPG, PNG, or WebP image.")

    raw = await file.read(8 * 1024 * 1024 + 1)

    if len(raw) > 8 * 1024 * 1024:
        raise HTTPException(413, "Image must be 8 MB or smaller.")

    try:
        image = Image.open(BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Invalid image file.")

    model = MODELS[model_name]
    results = model.predict(image, conf=0.10, verbose=False)
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())

            detections.append({
                "label": model.names[class_id],
                "confidence": round(float(box.conf[0].item()), 3),
                "box": [
                    round(x1, 2), round(y1, 2),
                    round(x2, 2), round(y2, 2),
                ],
            })

    return {
        "model_used": model_name,
        "image_width": image.width,
        "image_height": image.height,
        "detections": detections,
    }
