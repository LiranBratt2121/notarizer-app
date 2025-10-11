# server.py
# FastAPI + MongoDB Image Validation Server (CV-based, Gemini removed)
#
# Dependencies:
# pip install fastapi uvicorn pillow python-dotenv motor imagehash numpy

import io
import os
import json
import imagehash
import motor.motor_asyncio
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
MONGO_DETAILS = os.getenv(
    "MONGO_URL",
    "mongodb+srv://liran:liran2121@cluster0.zrt3vl8.mongodb.net/"
)

DATABASE_NAME = "image_validation_db"
COLLECTION_NAME = "valid_images"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

HASH_SIZE = 8

# --- Initialize MongoDB ---
mongo_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)
db = mongo_client[DATABASE_NAME]
image_collection = db[COLLECTION_NAME]

# --- FastAPI Setup ---
app = FastAPI(title="Image Validation API (CV-based)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Computer Vision Function ---
def get_document_bounds_cv(image: Image.Image, black_threshold=15) -> tuple:
    """Uses classic CV to find the largest non-black area in the center."""
    
    img_array = np.array(image.convert("L"))  # Convert to grayscale
    h, w = img_array.shape

    # Ignore top 15% and bottom 20% of image (likely UI)
    top_ignore = int(h * 0.15)
    bottom_ignore = int(h * 0.20)

    # Find vertical boundaries
    y1 = top_ignore
    for y in range(top_ignore, h - bottom_ignore):
        if np.mean(img_array[y, :]) > black_threshold:
            y1 = y
            break

    y2 = h - bottom_ignore
    for y in range(h - bottom_ignore, top_ignore, -1):
        if np.mean(img_array[y, :]) > black_threshold:
            y2 = y
            break

    # Find horizontal boundaries
    content_area = img_array[y1:y2, :]
    x1 = 0
    for x in range(w // 2):
        if np.mean(content_area[:, x]) > black_threshold:
            x1 = x
            break

    x2 = w
    for x in range(w - 1, w // 2, -1):
        if np.mean(content_area[:, x]) > black_threshold:
            x2 = x
            break

    # Fallback if detection fails
    if y2 <= y1 or x2 <= x1:
        return (0, 0, w, h)

    return (x1, y1, x2, y2)


# --- Validation Route ---
@app.post("/validate-image")
async def validate_image(file: UploadFile = File(...)):
    """Check if uploaded image matches any previously stored valid image."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        bounds = get_document_bounds_cv(image)
        cropped = image.crop(bounds) if bounds else image

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(UPLOAD_DIR, f"cropped_{timestamp}.jpg")
        cropped.save(output_path)

        resized = cropped.resize((HASH_SIZE * 2, HASH_SIZE * 2), Image.Resampling.LANCZOS)
        hash_to_check = imagehash.phash(resized, hash_size=HASH_SIZE)

        is_valid = False
        async for valid_doc in image_collection.find({}, {"phash": 1}):
            known_hash = imagehash.hex_to_hash(valid_doc["phash"])
            if (hash_to_check - known_hash) <= 5:
                is_valid = True
                break

        return JSONResponse(content={"is_valid": is_valid})

    except Exception as e:
        print(f"❌ Validation error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# --- Add Valid Image Route ---
@app.post("/add-valid-image")
async def add_valid_image(file: UploadFile = File(...)):
    """Store an image's perceptual hash (pHash) in MongoDB as a valid reference."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        resized = image.resize((HASH_SIZE * 2, HASH_SIZE * 2), Image.Resampling.LANCZOS)
        p_hash = str(imagehash.phash(resized, hash_size=HASH_SIZE))

        if await image_collection.find_one({"phash": p_hash}):
            return JSONResponse(content={"message": "Image hash already exists."})

        await image_collection.insert_one({"phash": p_hash, "filename": file.filename})
        return JSONResponse(content={"success": True, "added_phash": p_hash})

    except Exception as e:
        print(f"❌ Add valid image error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# --- Run Command ---
# uvicorn server:app --host 0.0.0.0 --port 8080 --workers 4
