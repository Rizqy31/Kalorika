
import os, io, json, math
from typing import Optional, Dict, Any, List
from PIL import Image
from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# -----------------------------
# Config via env / defaults
# -----------------------------
MODEL_PATH     = os.getenv("MODEL_PATH", "models/best.pt")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD  = float(os.getenv("IOU_THRESHOLD", "0.45"))
DEVICE         = os.getenv("DEVICE", "cpu")
NUTRITION_PATH = os.getenv("NUTRITION_PATH", "nutrition.json")
DEFAULT_PLATE_DIAMETER_CM = float(os.getenv("PLATE_DIAMETER_CM", "26.0"))

# -----------------------------
# App init
# -----------------------------
app = FastAPI(title="FoodCal YOLO API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load model and nutrition
# -----------------------------
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Gagal load model dari {MODEL_PATH}: {e}")

try:
    with open(NUTRITION_PATH, "r", encoding="utf-8") as f:
        NUTR = json.load(f)
except Exception as e:
    raise RuntimeError(f"Gagal load nutrition config {NUTRITION_PATH}: {e}")

# Ultralytics model class mapping
model_names: Dict[int, str] = model.names if hasattr(model, "names") else model.model.names  # type: ignore

# Validate nutrition entries
missing = [name for name in model_names.values() if name not in NUTR.get("classes", {})]
if missing:
    # Don't crash, but warn loudly on startup; per-request we'll fallback to defaults.
    print(f"[WARN] Nutrition entries missing for classes: {missing}")

# -----------------------------
# Helpers
# -----------------------------
def plate_area_cm2(plate_diam_cm: float) -> float:
    r = plate_diam_cm / 2.0
    return math.pi * r * r

def get_spec(name: str) -> Dict[str, float]:
    dfl = NUTR.get("defaults", {})
    spec = {**dfl, **NUTR.get("classes", {}).get(name, {})}
    # Ensure needed keys
    for k in ["kcal_per_g", "density_g_per_cm3", "thickness_cm", "bbox_fill", "packing_factor"]:
        if k not in spec:
            spec[k] = dfl.get(k, 1.0)
    return spec  # type: ignore

def estimate_grams(name: str, bbox, img_w: int, img_h: int, plate_cm: float) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    box_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    img_area = float(img_w * img_h)
    if img_area <= 0.0:
        return 0.0
    rel_area = box_area / img_area  # fraction of image covered by bbox
    spec = get_spec(name)
    rel_area_effective = rel_area * float(spec["bbox_fill"])
    cm2 = rel_area_effective * plate_area_cm2(plate_cm)
    volume_cm3 = cm2 * float(spec["thickness_cm"])
    grams = volume_cm3 * float(spec["density_g_per_cm3"]) * float(spec["packing_factor"])
    return max(0.0, grams)

def est_calories_kcal(name: str, grams: float) -> float:
    spec = get_spec(name)
    kcal_per_g = float(spec["kcal_per_g"])
    return max(0.0, grams * kcal_per_g)

# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model": MODEL_PATH, "conf": CONF_THRESHOLD, "iou": IOU_THRESHOLD, "device": DEVICE}

@app.get("/labels")
def labels() -> Dict[str, Any]:
    return {"model_classes": model_names}

@app.get("/nutrition")
def nutrition() -> Dict[str, Any]:
    return {"nutrition": NUTR}

@app.post("/predict")
async def predict(file: UploadFile, plate_diameter_cm: Optional[float] = Form(None)) -> Dict[str, Any]:
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="File harus jpg/png/webp")
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = img.size
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal baca gambar: {e}")

    plate_cm = float(plate_diameter_cm or DEFAULT_PLATE_DIAMETER_CM)

    try:
        results = model.predict(img, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, device=DEVICE, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inferensi gagal: {e}")

    detections: List[Dict[str, Any]] = []
    aggregate: Dict[str, Dict[str, float]] = {}

    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        xyxy = boxes.xyxy.cpu().numpy().tolist()
        cls  = boxes.cls.cpu().numpy().tolist()
        conf = boxes.conf.cpu().numpy().tolist()

        for (bb, c, p) in zip(xyxy, cls, conf):
            name = model_names.get(int(c), str(int(c)))
            grams = estimate_grams(name, bb, w, h, plate_cm)
            kcal  = est_calories_kcal(name, grams)

            detections.append({
                "class": name,
                "confidence": float(p),
                "bbox_xyxy": [float(v) for v in bb],
                "estimate": {"grams": grams, "kcal": kcal}
            })

            if name not in aggregate:
                aggregate[name] = {"grams": 0.0, "kcal": 0.0, "count": 0}
            aggregate[name]["grams"] += grams
            aggregate[name]["kcal"]  += kcal
            aggregate[name]["count"] += 1

    total_kcal = float(sum(v["kcal"] for v in aggregate.values()))
    total_grams = float(sum(v["grams"] for v in aggregate.values()))

    return {
        "image_size": {"width": w, "height": h},
        "plate_diameter_cm": plate_cm,
        "detections": detections,
        "aggregate": aggregate,
        "total": {"grams": total_grams, "kcal": total_kcal},
        "estimation_method": "bbox_area_ratio * plate_area * thickness * density * packing_factor; kcal = grams * kcal_per_g"
    }
