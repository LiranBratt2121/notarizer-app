# server.py
# FastAPI + Gemini 2.5 + MongoDB Image Validation Server (Fixed and Optimized)
#
# Dependencies:
# pip install fastapi uvicorn pillow python-dotenv motor imagehash google-genai

import io
import os
import json
import asyncio
import base64
import imagehash
import motor.motor_asyncio
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
MONGO_DETAILS = os.getenv("MONGO_URL", "mongodb+srv://liran:liran2121@cluster0.zrt3vl8.mongodb.net/")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DATABASE_NAME = "image_validation_db"
COLLECTION_NAME = "valid_images"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

HASH_SIZE = 8
MAX_IMAGE_SIZE = 1600  # limit to avoid huge Gemini payloads
GEMINI_MODEL = "gemini-2.0-flash"  # latest production model

# --- Initialize Gemini and MongoDB ---
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in environment variables!")

mongo_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)
db = mongo_client[DATABASE_NAME]
image_collection = db[COLLECTION_NAME]

genai_client = genai.Client(api_key=GEMINI_API_KEY)

# --- FastAPI Setup ---
app = FastAPI(title="Image Validation API (Gemini 2.5)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Gemini Vision Function ---
async def get_document_bounds_from_gemini(image: Image.Image) -> tuple | None:
    """
    Ask Gemini 2.5 to detect the bounding box of the main content in an image.
    Returns x1, y1, x2, y2 coordinates or None if not found.
    """
    prompt = """
    Analyze this mobile screenshot which displays a photograph.
    Your task is to identify the bounding box for the **FULL RECTANGULAR AREA OF THE PHOTOGRAPH'S VISIBLE CONTENT**.
    You must strictly exclude the following from the bounding box:
    1.  **ALL APP USER INTERFACE (UI) ELEMENTS** (status bar, toolbars, buttons, etc.).
    2.  **ANY SOLID BLACK OR WHITE PADDING/BARS** surrounding the photograph itself (letterboxing/pillarboxing).

    The output coordinates must define the tightest possible rectangle around the subject photo content.
    Respond ONLY as raw JSON: {"x1":int,"y1":int,"x2":int,"y2":int}.
    If no content is found, use null values.
    """

    # Preprocess image
    image = image.convert("RGB")
    w, h = image.size
    if max(w, h) > MAX_IMAGE_SIZE:
        image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
        print(f"🔹 Resized image to {image.size} before sending to Gemini")

    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    image_bytes = buf.getvalue()

    # Gemini content payload
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(text=prompt.strip()),
                types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=image_bytes)),
            ],
        ),
    ]

    try:
        # ✅ Gemini 2.5 async call (new syntax)
        response = await asyncio.to_thread(
            genai_client.models.generate_content,
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=256,
            ),
        )

        raw_text = (response.text or "").strip()
        print("🔸 Gemini Response:")
        print(raw_text)

        cleaned = raw_text.strip().lstrip("```json").rstrip("```").strip()
        data = json.loads(cleaned)

        x1, y1, x2, y2 = data.get("x1"), data.get("y1"), data.get("x2"), data.get("y2")

        if not all(isinstance(v, (int, float)) for v in [x1, y1, x2, y2]):
            return None

        # Clamp bounds
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        w, h = image.size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 > x1 and y2 > y1:
            print(f"✅ Cropping bounds: {(x1, y1, x2, y2)} | Image size: {image.size}")
            return (x1, y1, x2, y2)
        else:
            print("⚠️ Invalid bounding box from Gemini.")
            return None

    except Exception as e:
        print(f"❌ Error in Gemini Vision: {e}")
        return None


# --- Recommended CV Function (Alternative to Gemini) ---
def get_document_bounds_cv(image: Image.Image, black_threshold=15) -> tuple:
    """Uses classic CV to find the largest non-black area in the center."""
    import numpy as np
    
    # Convert PIL Image to an array for processing
    img_array = np.array(image.convert("L")) # Convert to Grayscale
    h, w = img_array.shape

    # 1. Mask out likely UI areas (Heuristic based on your screenshots)
    # Assume top 15% and bottom 20% are UI and are ignored initially
    top_ignore = int(h * 0.15)
    bottom_ignore = int(h * 0.20)
    
    # 2. Find vertical boundaries (y1 and y2)
    
    # Iterate from top_ignore down to find the photo's top edge (y1)
    y1 = top_ignore
    for y in range(top_ignore, h - bottom_ignore):
        # Check if the average pixel value in this row is above the black threshold
        if np.mean(img_array[y, :]) > black_threshold:
            y1 = y
            break
            
    # Iterate from bottom_ignore up to find the photo's bottom edge (y2)
    y2 = h - bottom_ignore
    for y in range(h - bottom_ignore, top_ignore, -1):
        if np.mean(img_array[y, :]) > black_threshold:
            y2 = y
            break
            
    # 3. Find horizontal boundaries (x1 and x2)
    # Use the area between y1 and y2 for horizontal scanning
    content_area = img_array[y1:y2, :]
    
    # Find x1 (first column that is not mostly black)
    x1 = 0
    for x in range(w // 2): # Scan from left to center
        # Check if the average pixel value in this column is above threshold
        if np.mean(content_area[:, x]) > black_threshold:
            x1 = x
            break

    # Find x2 (last column that is not mostly black)
    x2 = w
    for x in range(w - 1, w // 2, -1): # Scan from right to center
        if np.mean(content_area[:, x]) > black_threshold:
            x2 = x
            break

    # Ensure y2 is larger than y1 and x2 is larger than x1
    if y2 <= y1 or x2 <= x1:
        # Fallback to full image if detection fails
        return (0, 0, w, h)
        
    return (x1, y1, x2, y2)

# NOTE: This requires installing numpy and replacing the Gemini call with:
# bounds = get_document_bounds_cv(image)
# --- Validation Route ---
@app.post("/validate-image")
async def validate_image(file: UploadFile = File(...)):
    """Check if uploaded image matches any previously stored valid image."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # bounds = await get_document_bounds_from_gemini(image)
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


# --- Run Command ---
# uvicorn server:app --host 0.0.0.0 --port 8080 --workers 4
