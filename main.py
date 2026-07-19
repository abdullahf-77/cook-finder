"""
Cook Finder Backend — last.pt (87 classes)
Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
import base64, json, os
from typing import List
import cv2, numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO

MODEL_PATH   = os.getenv("MODEL_PATH",   "last.pt")
RECIPES_PATH = os.getenv("RECIPES_PATH", "recipes.json")
CONF_THRESH  = float(os.getenv("CONF_THRESH", "0.25"))
MAX_RECIPES  = int(os.getenv("MAX_RECIPES",   "6"))

app = FastAPI(title="Cook Finder API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

print("Loading model:", MODEL_PATH)
model = YOLO(MODEL_PATH)
print(f"  {len(model.names)} classes")

print("Loading recipes:", RECIPES_PATH)
with open(RECIPES_PATH, encoding="utf-8") as f:
    RAW = json.load(f)

# Pre-build lowercase ingredient sets for fast matching
RECIPES = [{**r, "_ing_lower": {i.lower() for i in r.get("ingredients", [])}} for r in RAW]
print(f"  {len(RECIPES)} recipes loaded")

# Map YOLO class names → all ingredient name variants they should match
# YOLO returns e.g. "Chili Pepper (Khursani)" but recipe says "Chili Pepper (Khursani)"
# This also handles partial matches and aliases
ALIASES = {
    # Lentils
    "red lentils":              {"red lentils"},
    "black lentils":            {"black lentils", "lentils"},
    "green lentils":            {"green lentils", "lentils"},
    # Rice variants
    "rice (chamal)":            {"rice (chamal)", "rice"},
    "beaten rice (chiura)":     {"beaten rice (chiura)", "beaten rice"},
    # Chili / pepper
    "chili pepper (khursani)":  {"chili pepper (khursani)", "chili pepper", "chili", "pepper"},
    "capsicum":                 {"capsicum", "bell pepper", "green pepper"},
    # Onion variants
    "onion leaves":             {"onion leaves", "spring onion", "green onion"},
    "onion":                    {"onion"},
    # Spinach variants
    "palak (indian spinach)":   {"palak (indian spinach)", "palak", "spinach"},
    "palungo (nepali spinach)": {"palungo (nepali spinach)", "palungo", "spinach"},
    # Gourds
    "ash gourd (kubhindo)":     {"ash gourd (kubhindo)", "ash gourd"},
    "bottle gourd (lauka)":     {"bottle gourd (lauka)", "bottle gourd"},
    "snake gourd (chichindo)":  {"snake gourd (chichindo)", "snake gourd"},
    "sponge gourd (ghiraula)":  {"sponge gourd (ghiraula)", "sponge gourd"},
    "chayote (iskus)":          {"chayote (iskus)", "chayote"},
    "pointed gourd (chuche karela)": {"pointed gourd (chuche karela)", "pointed gourd"},
    "pumpkin (farsi)":          {"pumpkin (farsi)", "pumpkin"},
    # Legumes
    "broad beans (bakullo)":    {"broad beans (bakullo)", "broad beans"},
    "long beans (bodi)":        {"long beans (bodi)", "long beans", "green beans"},
    "green soyabean (hariyo bhatmas)": {"green soyabean (hariyo bhatmas)", "soyabean", "edamame"},
    "soyabean (bhatmas)":       {"soyabean (bhatmas)", "soyabean"},
    "nutrela (soya chunks)":    {"nutrela (soya chunks)", "nutrela", "soya chunks"},
    "red beans":                {"red beans", "kidney beans", "beans"},
    # Herbs / greens
    "coriander (dhaniya)":      {"coriander (dhaniya)", "coriander", "cilantro"},
    "green mint (pudina)":      {"green mint (pudina)", "green mint", "mint"},
    "garden cress (chamsur ko saag)": {"garden cress (chamsur ko saag)", "garden cress"},
    "fiddlehead ferns (niguro)": {"fiddlehead ferns (niguro)", "fiddlehead ferns"},
    "stinging nettle (sisnu)":  {"stinging nettle (sisnu)", "stinging nettle"},
    "moringa leaves (sajyun ko munta)": {"moringa leaves (sajyun ko munta)", "moringa leaves", "moringa"},
    "sajjyun (moringa drumsticks)": {"sajjyun (moringa drumsticks)", "moringa drumsticks"},
    "taro leaves (karkalo)":    {"taro leaves (karkalo)", "taro leaves"},
    "taro root (pidalu)":       {"taro root (pidalu)", "taro root", "taro"},
    # Citrus
    "lemon (nimbu)":            {"lemon (nimbu)", "lemon"},
    "lime (kagati)":            {"lime (kagati)", "lime", "lemon"},
    # Root veg
    "sweet potato (suthuni)":   {"sweet potato (suthuni)", "sweet potato"},
    "cassava (ghar tarul)":     {"cassava (ghar tarul)", "cassava"},
    # Fruit
    "lapsi (nepali hog plum)":  {"lapsi (nepali hog plum)", "lapsi"},
    "tree tomato (rukh tamatar)": {"tree tomato (rukh tamatar)", "tree tomato", "tomato"},
    # Meat
    "minced meat":              {"minced meat", "ground meat", "mince"},
    "crab meat":                {"crab meat", "crab"},
    # Noodles
    "chowmein noodles":         {"chowmein noodles", "chowmein", "noodles"},
    "thukpa noodles":           {"thukpa noodles", "thukpa", "noodles"},
    # Others
    "okra (bhindi)":            {"okra (bhindi)", "okra", "bhindi"},
    "bamboo shoots (tama)":     {"bamboo shoots (tama)", "bamboo shoots"},
    "jack fruit":               {"jack fruit", "jackfruit"},
    "brinjal":                  {"brinjal", "eggplant", "aubergine"},
    "paneer":                   {"paneer", "cottage cheese", "cheese"},
}

def normalize(name: str) -> set:
    """Return set of all ingredient names this detection should match."""
    k = name.lower().strip()
    if k in ALIASES:
        return ALIASES[k]
    # also check if any alias key is contained in k
    for key, vals in ALIASES.items():
        if key in k or k in key:
            return vals | {k}
    return {k}

def run_yolo(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    res   = model(img, conf=CONF_THRESH)[0]
    detected, seen = [], set()
    for idx in res.boxes.cls.cpu().numpy().astype(int):
        name = model.names[idx]
        if name not in seen:
            seen.add(name); detected.append(name)
    ann = res.plot()
    _, buf = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return detected, base64.b64encode(buf.tobytes()).decode()

def match_recipes(detected: List[str]):
    if not detected: return []
    # expand all detections through alias map
    expanded = set()
    for d in detected:
        expanded.update(normalize(d))
    out = []
    for r in RECIPES:
        ing   = r["_ing_lower"]
        mtch  = {i for i in ing if i in expanded or any(i in e or e in i for e in expanded)}
        if not mtch: continue
        disp  = [i for i in r["ingredients"] if i.lower() in mtch]
        score = round(len(mtch) / len(ing) * 100)
        out.append({
            "name":        r["name"],
            "cuisine":     r["cuisine"],
            "category":    r["category"],
            "total_time":  r["total_time"],
            "servings":    r["servings"],
            "description": r["description"],
            "ingredients": r["ingredients"],
            "steps":       r.get("steps", []),
            "matched":     disp,
            "match_count": len(mtch),
            "score":       score,
        })
    out.sort(key=lambda x: (x["score"], x["match_count"]), reverse=True)
    return out[:MAX_RECIPES]

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if not (image.content_type or "").startswith("image/"):
        raise HTTPException(400, "Image required")
    data = await image.read()
    if len(data) > 16 * 1024 * 1024:
        raise HTTPException(413, "Max 16MB")
    detected, b64 = run_yolo(data)
    return JSONResponse({
        "detected":        detected,
        "annotated_image": b64,
        "recipes":         match_recipes(detected),
    })

@app.get("/health")
def health():
    return {"status": "ok", "recipes": len(RECIPES), "classes": len(model.names)}

@app.get("/classes")
def get_classes():
    return {"classes": list(model.names.values())}
