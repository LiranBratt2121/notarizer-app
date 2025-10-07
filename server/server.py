# server.py
import io
import os
import imagehash
import motor.motor_asyncio
import cv2
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

UPLOAD_DIR = "uploads/"
MONGO_DETAILS = "mongodb+srv://liran:liran2121@cluster0.zrt3vl8.mongodb.net/"
DATABASE_NAME = "image_validation_db"
COLLECTION_NAME = "valid_images"

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)
db = client[DATABASE_NAME]
image_collection = db[COLLECTION_NAME]

app = FastAPI(title="Image Validation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

HASH_SIZE = 8 # Standard size for phash

def crop_to_document(image: Image.Image) -> Image.Image:
    """Finds the largest rectangular object in an image and crops to it."""
    open_cv_image = np.array(image.convert('RGB'))
    img_gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, 10, 50)
    contours, _ = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image # Return original if no object is found

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return image.crop((x, y, x + w, y + h))

@app.post("/add-valid-image")
async def add_valid_image(file: UploadFile = File(...)):
    """
    Endpoint for the React Native app.
    NO cropping. Just resizes, calculates phash, and saves to MongoDB.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        resized_image = image.resize((HASH_SIZE * 2, HASH_SIZE * 2), Image.Resampling.LANCZOS)
        
        p_hash = str(imagehash.phash(resized_image, hash_size=HASH_SIZE))

        if await image_collection.find_one({"phash": p_hash}):
            return JSONResponse(status_code=200, content={"message": "Image hash already exists."})

        await image_collection.insert_one({"phash": p_hash, "filename": file.filename})
        return JSONResponse(content={"success": True, "added_phash": p_hash})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")


@app.post("/validate-image")
async def validate_image(file: UploadFile = File(...)):
    """
    Endpoint for the Android bubble.
    Crops, resizes, calculates phash, and checks against the DB.
    NOW ALSO SAVES THE UPLOADED FILE FOR DEBUGGING.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        
        # --- NEW: Save the incoming file for debugging ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_filename = f"validation_attempt_{timestamp}.jpg"
        with open(os.path.join(UPLOAD_DIR, debug_filename), "wb") as f:
            f.write(contents)
        # --- End of new code ---
            
        image = Image.open(io.BytesIO(contents))
        
        cropped_image = crop_to_document(image)
        resized_image = cropped_image.resize((HASH_SIZE * 2, HASH_SIZE * 2), Image.Resampling.LANCZOS)
        hash_to_check = imagehash.phash(resized_image, hash_size=HASH_SIZE)
        
        is_valid = False
        async for valid_doc in image_collection.find({}, {"phash": 1}):
            known_hash = imagehash.hex_to_hash(valid_doc["phash"])
            if (hash_to_check - known_hash) <= 5: 
                is_valid = True
                break
        
        return JSONResponse(content={"is_valid": is_valid})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")