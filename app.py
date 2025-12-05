# ---------------------------------------------------------------
# UPDATED app.py FOR NEW MODEL + NEW JSON FORMAT (FINAL)
# ---------------------------------------------------------------

import os
import io
import json
import logging
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm

# -------- Configuration --------
MODEL_ARCH = os.environ.get("MODEL_ARCH", "mobilenetv3_small_100")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "best_quick.pth")
SPLITS_PATH = os.environ.get("SPLITS_PATH", "splits.json")
PESTICIDE_JSON = os.environ.get("PESTICIDE_JSON", "pest_pesticide_map_final.json")
IMG_SIZE = int(os.environ.get("IMG_SIZE", 128))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Logging --------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pest-api")

# -------- FastAPI app --------
app = FastAPI(
    title="Rice Pest Predictor",
    description="Predict rice pest + pesticide dosage + field/mini-farm dose",
    version="2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- New response model --------
class PredictionResponse(BaseModel):
    predicted_pest: str
    confidence: float
    pesticide: str | None
    acre_dose: Dict[str, Any]
    mini_farm_dose: Dict[str, Any]
    water_liters_per_acre: float
    water_liters_mini_farm: float

# -------- Global variables --------
model = None
classes: List[str] = []
recommendations: Dict[str, Any] = {}
water_liters_per_acre: float = 0
water_liters_mini_farm: float = 0

# -------- Utilities --------
def load_splits(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing splits.json at {path}")
    with open(path, "r") as f:
        return json.load(f)["classes"]

def load_recommendation_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing pesticide JSON at {path}")
    with open(path, "r") as f:
        data = json.load(f)

    rec = data.get("recommendations", {})
    water = data.get("water_info", {})

    return rec, water.get("water_liters_per_acre", 200), water.get("water_liters_mini_farm", 0.0025)

def build_model(arch: str, num_classes: int, checkpoint: str):
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Model checkpoint missing: {checkpoint}")

    model = timm.create_model(arch, pretrained=False, num_classes=num_classes)

    state = torch.load(checkpoint, map_location=DEVICE)
    model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()
    return model

def preprocess_img(img_bytes: bytes):
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return tf(img).unsqueeze(0)

# -------- Startup: Load everything --------
@app.on_event("startup")
def startup_event():
    global model, classes, recommendations
    global water_liters_per_acre, water_liters_mini_farm

    logger.info("Loading splits...")
    classes = load_splits(SPLITS_PATH)

    logger.info("Loading pesticide + dosage data...")
    recommendations, water_liters_per_acre, water_liters_mini_farm = load_recommendation_json(PESTICIDE_JSON)

    logger.info("Loading model...")
    model = build_model(MODEL_ARCH, len(classes), CHECKPOINT_PATH)

    logger.info("Ready.")

# -------- Routes --------
@app.get("/")
def root():
    return {"message": "Rice Pest Predictor API v2.0 Ready"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": model is not None,
        "num_classes": len(classes)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    global model, recommendations

    if not file.content_type.startswith("image"):
        raise HTTPException(status_code=400, detail="Upload a valid image.")

    img_bytes = await file.read()
    img_tensor = preprocess_img(img_bytes).to(DEVICE)

    with torch.no_grad():
        out = model(img_tensor)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]

    pred_idx = probs.argmax()
    confidence = float(probs[pred_idx])
    pest_name = classes[pred_idx]

    # Lookup recommendation
    rec = recommendations.get(pest_name, {})
    pesticide = rec.get("pesticide", None)
    acre_dose = rec.get("acre_dose", {})
    mini_dose = rec.get("mini_farm_dose", {})

    response = PredictionResponse(
        predicted_pest=pest_name,
        confidence=round(confidence, 6),
        pesticide=pesticide,
        acre_dose=acre_dose,
        mini_farm_dose=mini_dose,
        water_liters_per_acre=water_liters_per_acre,
        water_liters_mini_farm=water_liters_mini_farm
    )

    return JSONResponse(status_code=200, content=response.dict())

# -------- Local run --------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
