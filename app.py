# app.py
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

# -------- Configuration (change via env vars if needed) --------
MODEL_ARCH = os.environ.get("MODEL_ARCH", "mobilenetv3_large_100")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "best_fast.pth")
SPLITS_PATH = os.environ.get("SPLITS_PATH", "splits.json")
PESTICIDE_JSON = os.environ.get("PESTICIDE_JSON", "pest_pesticide_map.json")
IMG_SIZE = int(os.environ.get("IMG_SIZE", 160))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Logging --------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pest-api")

# -------- FastAPI app --------
app = FastAPI(
    title="Rice Pest Predictor",
    description="Upload rice leaf image → returns predicted pest, confidence, and recommended pesticides",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Response model --------
class PredictionResponse(BaseModel):
    predicted_pest: str
    confidence: float
    recommended_pesticides: List[str]

# -------- Global state --------
model = None
classes: List[str] = []
pesticide_map: Dict[str, List[str]] = {}

# -------- Utilities --------
def load_splits(splits_path: str) -> List[str]:
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"splits.json not found at: {splits_path}")
    with open(splits_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "classes" not in data or not isinstance(data["classes"], list):
        raise ValueError("splits.json must contain a 'classes' list")
    return data["classes"]

def load_pesticide_map(path: str) -> Dict[str, List[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pesticide JSON not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("pesticide mapping must be a JSON object mapping pest->list")
    # ensure list values
    for k, v in data.items():
        if not isinstance(v, list):
            raise ValueError(f"Value for '{k}' in pesticide map must be a list")
    return data

def build_model(arch: str, num_classes: int, checkpoint_path: str, device: torch.device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    # create model architecture (no pretrained weights here for inference; checkpoint has trained weights)
    logger.info(f"Creating model architecture: {arch} (num_classes={num_classes})")
    try:
        m = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    except Exception as e:
        raise RuntimeError(f"Failed to create model '{arch}': {e}")

    # load state_dict (your training used torch.save(model.state_dict(), CHECKPOINT))
    logger.info(f"Loading state_dict from {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        # strip 'module.' if DataParallel used while saving
        new_state = {}
        for k, v in state.items():
            new_k = k.replace("module.", "") if k.startswith("module.") else k
            new_state[new_k] = v
        state = new_state
    try:
        m.load_state_dict(state)
    except Exception as e:
        raise RuntimeError(f"Failed to load state_dict into model: {e}")

    m.to(device)
    m.eval()
    return m

# Preprocessing similar to your training code
def get_preprocess(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

def preprocess_image_bytes(image_bytes: bytes, preprocess_tf):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError("Invalid image file") from e
    return preprocess_tf(img).unsqueeze(0)  # add batch dim

# -------- Startup event: load classes, mapping, model --------
@app.on_event("startup")
def startup():
    global model, classes, pesticide_map
    logger.info(f"Starting app. Device={DEVICE}, IMG_SIZE={IMG_SIZE}, ARCH={MODEL_ARCH}")
    # load classes
    try:
        classes = load_splits(SPLITS_PATH)
        logger.info(f"Loaded {len(classes)} classes from {SPLITS_PATH}")
    except Exception as e:
        logger.exception("Failed to load splits.json")
        raise

    # load pesticide mapping
    try:
        pesticide_map = load_pesticide_map(PESTICIDE_JSON)
        logger.info(f"Loaded pesticide mapping with {len(pesticide_map)} keys")
    except Exception as e:
        logger.exception("Failed to load pesticide mapping")
        raise

    # build model and load weights (state_dict)
    try:
        model = build_model(MODEL_ARCH, num_classes=len(classes), checkpoint_path=CHECKPOINT_PATH, device=DEVICE)
        logger.info("Model loaded and ready")
    except Exception as e:
        logger.exception("Failed to load model")
        raise

# -------- Routes --------
@app.get("/", tags=["root"])
def root():
    return {"message": "Rice Pest Predictor API. POST images to /predict"}

@app.get("/health", tags=["health"])
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": model is not None,
        "num_classes": len(classes)
    }

@app.post("/predict", response_model=PredictionResponse, tags=["predict"])
async def predict(file: UploadFile = File(...)):
    """
    Accepts form-data file (image). Returns:
    - predicted_pest: str
    - confidence: float (0..1)
    - recommended_pesticides: list[str]
    """
    global model, classes, pesticide_map

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not (file.content_type and file.content_type.startswith("image")):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    preprocess_tf = get_preprocess(IMG_SIZE)
    try:
        input_tensor = preprocess_image_bytes(contents, preprocess_tf)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    input_tensor = input_tensor.to(DEVICE)
    with torch.no_grad():
        try:
            outputs = model(input_tensor)  # shape [1, num_classes]
        except Exception as e:
            logger.exception("Model forward error")
            raise HTTPException(status_code=500, detail=f"Model inference error: {e}")

        # outputs -> probabilities
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        if not torch.is_tensor(outputs):
            try:
                outputs = torch.tensor(outputs, device=DEVICE, dtype=torch.float32)
            except Exception:
                raise HTTPException(status_code=500, detail="Unexpected model output type")

        # convert logits to probabilities across dim=1
        if outputs.dim() == 2:
            probs = F.softmax(outputs, dim=1).cpu().squeeze(0).numpy()
        elif outputs.dim() == 1:
            probs = F.softmax(outputs, dim=0).cpu().numpy()
        else:
            raise HTTPException(status_code=500, detail=f"Unexpected model output shape: {outputs.shape}")

    pred_index = int(probs.argmax())
    confidence = float(probs[pred_index])
    predicted_pest = classes[pred_index] if pred_index < len(classes) else "unknown"
    recommended = pesticide_map.get(predicted_pest, ["No data available"])

    resp = PredictionResponse(
        predicted_pest=predicted_pest,
        confidence=round(confidence, 6),
        recommended_pesticides=recommended
    )
    return JSONResponse(status_code=200, content=resp.dict())

# -------- Run (for local dev) --------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
