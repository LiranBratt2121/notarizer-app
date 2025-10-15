# server.py
# FastAPI + MongoDB Image Validation Server with Difference Visualization
# Uses ONLY PIL/Pillow - NO OpenCV required!

import io
import os
import base64
import imagehash
import motor.motor_asyncio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from dotenv import load_dotenv
import logging

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
MONGO_DETAILS = os.getenv(
    "MONGO_URL", "mongodb+srv://liran:liran2121@cluster0.zrt3vl8.mongodb.net/"
)

DATABASE_NAME = "image_validation_db"
COLLECTION_NAME = "valid_images"
HASH_SIZE = 16
# **FIX 1 CONFIG:** Calculate the correct maximum possible Hamming distance.
MAX_DISTANCE = HASH_SIZE * HASH_SIZE # 16 * 16 = 256

# --- Initialize MongoDB ---
logger.info("🔌 Connecting to MongoDB...")
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

logger.info("✅ Server initialized successfully")


# --- Computer Vision Function (Unchanged) ---
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
        logger.warning(f"⚠️ Document bounds detection failed, using full image")
        return (0, 0, w, h)

    logger.info(f"📐 Document bounds: ({x1}, {y1}, {x2}, {y2})")
    return (x1, y1, x2, y2)


def create_difference_visualization(
    img1_pil: Image.Image, img2_pil: Image.Image
) -> str:
    """
    Create a side-by-side comparison with difference overlay using ONLY PIL.
    Note: img1_pil and img2_pil MUST have the exact same dimensions.
    Returns base64 encoded comparison image.
    """
    logger.info("🎨 Creating difference visualization (Green/Red Heatmap)...")

    # --- 1. Preparation and Resizing ---
    # **NOTE:** img1_pil and img2_pil must already be size-aligned before calling this function.
    target_width = 600
    if img1_pil.width > 0:
        aspect_ratio = img1_pil.height / img1_pil.width
        target_height = int(target_width * aspect_ratio)
    else:
        logger.error("Error: Input image has zero width.")
        return ""

    img1_resized = img1_pil.resize(
        (target_width, target_height), Image.Resampling.LANCZOS
    )
    img2_resized = img2_pil.resize(
        (target_width, target_height), Image.Resampling.LANCZOS
    )

    # Convert to numpy arrays for difference calculation
    img1_np = np.array(img1_resized)
    img2_np = np.array(img2_resized)

    # --- 2. Difference Calculation and Masking ---
    # Calculate absolute difference
    diff = np.abs(img1_np.astype(int) - img2_np.astype(int))

    # Convert difference to grayscale (max channel difference)
    diff_gray = np.max(diff, axis=2)

    # Threshold to find significant differences (Difference greater than 15 is considered a mismatch)
    THRESHOLD = 15
    diff_mask = diff_gray > THRESHOLD

    # --- 3. Creating the Green/Red Heatmap Image (Third Panel) ---

    # Initialize the heatmap with a base color (e.g., green for 'valid')
    heatmap_np = np.full(
        img1_np.shape, [0, 200, 0], dtype=np.uint8
    )  # Dark Green for valid parts

    # Apply Red to the areas that are different (where diff_mask is True)
    heatmap_np[diff_mask] = [255, 0, 0]  # Red for non-valid parts

    # Apply a light blend with the original image to keep context
    # Blending 20% original image with 80% heatmap mask
    blended_heatmap = (img1_np * 0.2 + heatmap_np * 0.8).astype(np.uint8)

    # --- 4. Create Side-by-Side Comparison Image ---
    comparison_width = target_width * 3
    comparison_height = target_height
    comparison = Image.new(
        "RGB", (comparison_width, comparison_height), (255, 255, 255)
    )

    # Paste images
    comparison.paste(img1_resized, (0, 0))
    comparison.paste(img2_resized, (target_width, 0))
    comparison.paste(
        Image.fromarray(blended_heatmap), (target_width * 2, 0)
    )

    # --- 5. Add Text Labels ---
    draw = ImageDraw.Draw(comparison)

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    labels = [
        ("Your Screenshot", 10),
        ("Database Image", target_width + 10),
        ("Match/Diff Heatmap", target_width * 2 + 10),
    ]

    for label_text, x_pos in labels:
        bbox = draw.textbbox((x_pos, 10), label_text, font=font)

        draw.rectangle(
            [(bbox[0] - 5, bbox[1] - 5), (bbox[2] + 5, bbox[3] + 5)],
            fill=(255, 255, 255),
            outline=(0, 0, 0),
            width=2,
        )

        draw.text((x_pos, 10), label_text, fill=(0, 0, 0), font=font)

    # --- 6. Convert to base64 ---
    buffered = io.BytesIO()
    comparison.save(buffered, format="JPEG", quality=80)
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    logger.info(f"✅ Difference visualization created - Size: {len(img_base64)} chars")
    return img_base64


# --- Health Check Route (Unchanged) ---
@app.get("/")
async def root():
    """Health check endpoint"""
    logger.info("🏥 Health check requested")
    return {
        "status": "running",
        "service": "Notarizer Image Validation API",
        "timestamp": datetime.now().isoformat(),
    }


# --- Validation Route (Updated for Fixes) ---
@app.post("/validate-image")
async def validate_image(file: UploadFile = File(...)):
    """Check if uploaded image matches any previously stored valid image."""
    logger.info(
        f"📥 Validation request received - File: {file.filename}, Type: {file.content_type}"
    )

    if not file.content_type.startswith("image/"):
        logger.error(f"❌ Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        logger.info(f"📦 File size: {len(contents)} bytes")

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        logger.info(f"🖼️ Image dimensions: {image.size}")

        # Crop to document bounds
        bounds = get_document_bounds_cv(image)
        cropped = image.crop(bounds) if bounds else image
        logger.info(f"✂️ Cropped to: {cropped.size}")

        # Calculate hash
        resized = cropped.resize(
            (HASH_SIZE * 2, HASH_SIZE * 2), Image.Resampling.LANCZOS
        )
        hash_to_check = imagehash.phash(resized, hash_size=HASH_SIZE)
        logger.info(f"🔑 Calculated hash: {hash_to_check}")

        # Check against database
        matched_doc = None
        min_distance = float("inf")
        checked_count = 0

        logger.info("🔍 Checking against database...")
        async for valid_doc in image_collection.find(
            {}, {"phash": 1, "filename": 1, "uploaded_at": 1, "image_data": 1}
        ):
            checked_count += 1
            known_hash = imagehash.hex_to_hash(valid_doc["phash"])
            distance = hash_to_check - known_hash

            logger.info(
                f"  Comparing with {valid_doc.get('filename', 'Unknown')} - Distance: {distance}"
            )

            if distance <= 60 and distance < min_distance:
                min_distance = distance
                matched_doc = valid_doc
                logger.info(f"  ✅ Match found! Distance: {distance}")

        logger.info(f"🔢 Total documents checked: {checked_count}")

        if matched_doc:
            logger.info(
                f"✅ VALIDATION SUCCESS - Matched: {matched_doc.get('filename', 'Unknown')}"
            )

            # Create difference visualization
            diff_image_base64 = None
            try:
                # Decode stored image from base64
                stored_image_data = base64.b64decode(matched_doc.get("image_data", ""))
                stored_image = Image.open(io.BytesIO(stored_image_data)).convert("RGB")

                # **FIX 2:** Align stored image size to the cropped input size for accurate comparison
                if stored_image.size != cropped.size:
                    stored_image_aligned = stored_image.resize(cropped.size, Image.Resampling.LANCZOS)
                else:
                    stored_image_aligned = stored_image
                
                # Create comparison visualization
                diff_image_base64 = create_difference_visualization(
                    cropped, stored_image_aligned
                )

            except Exception as e:
                logger.error(f"❌ Error creating diff visualization: {str(e)}")
                diff_image_base64 = None
            
            # **FIX 1:** Corrected Confidence Calculation using MAX_DISTANCE (256)
            confidence_value = round((1 - min_distance / MAX_DISTANCE) * 100, 2)
            
            return JSONResponse(
                content={
                    "is_valid": True,
                    "approved": True,
                    "match_found": True,
                    "matched_filename": matched_doc.get("filename", "Unknown"),
                    "confidence": confidence_value, # Use the corrected value
                    "message": "✅ Image verified! This matches an approved document.",
                    "matched_at": datetime.now().isoformat(),
                    "distance": int(min_distance),
                    "comparison_image": diff_image_base64,
                    # Removed unnecessary base64 return fields (original_screenshot, database_image)
                }
            )
        else:
            logger.warning(f"❌ VALIDATION FAILED - No match found")
            return JSONResponse(
                content={
                    "is_valid": False,
                    "approved": False,
                    "match_found": False,
                    "message": "❌ Image NOT verified. No matching approved document found.",
                    "checked_at": datetime.now().isoformat(),
                    "checked_count": checked_count,
                }
            )

    except Exception as e:
        logger.error(f"❌ Validation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# --- Add Valid Image Route (Unchanged) ---
@app.post("/add-valid-image")
async def add_valid_image(file: UploadFile = File(...)):
    """Store an image (as base64) and its perceptual hash in MongoDB as a valid reference."""
    logger.info(
        f"📤 Upload request received - File: {file.filename}, Type: {file.content_type}"
    )

    if not file.content_type.startswith("image/"):
        logger.error(f"❌ Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        logger.info(f"📦 File size: {len(contents)} bytes")

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        logger.info(f"🖼️ Image dimensions: {image.size}")

        # Calculate hash
        resized = image.resize((HASH_SIZE * 2, HASH_SIZE * 2), Image.Resampling.LANCZOS)
        p_hash = str(imagehash.phash(resized, hash_size=HASH_SIZE))
        logger.info(f"🔑 Calculated hash: {p_hash}")

        # Check if already exists
        existing = await image_collection.find_one({"phash": p_hash})
        if existing:
            logger.warning(f"⚠️ Duplicate detected - Hash already exists")
            return JSONResponse(
                content={
                    "success": False,
                    "message": "Image hash already exists in database.",
                    "phash": p_hash,
                }
            )

        # Convert image to base64 for storage
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        logger.info(f"💾 Compressed to base64 - Size: {len(img_base64)} chars")

        # Store in MongoDB
        document = {
            "phash": p_hash,
            "filename": file.filename,
            "image_data": img_base64,
            "content_type": "image/jpeg",
            "uploaded_at": datetime.now().isoformat(),
            "file_size_kb": round(len(img_base64) / 1024, 2),
        }

        result = await image_collection.insert_one(document)
        logger.info(f"✅ UPLOAD SUCCESS - Document ID: {result.inserted_id}")

        return JSONResponse(
            content={
                "success": True,
                "message": "✅ Image successfully added to approved list",
                "added_phash": p_hash,
                "document_id": str(result.inserted_id),
                "filename": file.filename,
                "uploaded_at": document["uploaded_at"],
            }
        )

    except Exception as e:
        logger.error(f"❌ Upload error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# --- Get All Valid Images Route (Unchanged) ---
@app.get("/valid-images")
async def get_valid_images():
    """Retrieve all valid images from database (without full image data for performance)"""
    logger.info("📋 Request to list all valid images")

    try:
        images = []
        async for doc in image_collection.find({}, {"image_data": 0}):
            doc["_id"] = str(doc["_id"])
            images.append(doc)

        logger.info(f"✅ Retrieved {len(images)} images")
        return JSONResponse(
            content={"success": True, "count": len(images), "images": images}
        )
    except Exception as e:
        logger.error(f"❌ Get valid images error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# --- Startup Event (Unchanged) ---
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Server starting up...")
    try:
        # Test MongoDB connection
        await db.command("ping")
        logger.info("✅ MongoDB connection successful")

        # Count existing documents
        count = await image_collection.count_documents({})
        logger.info(f"📊 Current approved images in database: {count}")

    except Exception as e:
        logger.error(f"❌ Startup error: {str(e)}", exc_info=True)