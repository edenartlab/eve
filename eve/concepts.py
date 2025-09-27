import logging
import os
import tempfile
from bson import ObjectId
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, Depends, BackgroundTasks, Request
from PIL import Image
from io import BytesIO
from eve.mongo import Document, Collection
from eve.api.errors import handle_errors, APIError
from eve.api.api_requests import (
    CreateConceptRequest,
    UpdateConceptRequest,
)
from eve.utils.media_utils import upload_media, download_image_to_PIL, PIL_to_bytes

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@Collection("concepts2")
class Concept(Document):
    name: str
    user: ObjectId
    agent: Optional[ObjectId] = None
    thumbnail: Optional[str] = None
    # public: bool = False
    deleted: bool = False
    usage_instructions: Optional[str] = None
    images: Optional[List[Any]] = None

    # args: Optional[Dict[str, Any]] = None
    # sdxl_lora: str
    # flux_lora: str
    
    creationCount: int = 0


def center_crop_resize(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """
    Resize and center crop an image to exact dimensions while preserving aspect ratio.
    """
    original_width, original_height = image.size
    target_ratio = target_width / target_height
    original_ratio = original_width / original_height

    if original_ratio > target_ratio:
        # Image is wider than target - crop width
        new_height = original_height
        new_width = int(new_height * target_ratio)
        left = (original_width - new_width) // 2
        top = 0
        right = left + new_width
        bottom = original_height
    else:
        # Image is taller than target - crop height
        new_width = original_width
        new_height = int(new_width / target_ratio)
        left = 0
        top = (original_height - new_height) // 2
        right = original_width
        bottom = top + new_height

    # Crop to the calculated dimensions
    cropped = image.crop((left, top, right, bottom))

    # Resize to exact target dimensions
    return cropped.resize((target_width, target_height), Image.LANCZOS)


def create_thumbnail(images: List[str]) -> str:
    """
    Create a 1024x1024px webp thumbnail from a list of image URLs.

    - 1 image: copy that image
    - 2 images: split (1024x512 + 1024x512 or 512x1024 + 512x1024)
    - 3 images: one split + subdivide one side
    - 4+ images: 2x2 grid (first 4 images only)
    """
    if not images:
        return None

    # Take first 4 images max
    images_to_use = images[:4]
    num_images = len(images_to_use)

    # Create 1024x1024 canvas
    canvas = Image.new('RGB', (1024, 1024), 'white')

    if num_images == 1:
        # Single image - center crop and resize to 1024x1024
        img = download_image_to_PIL(images_to_use[0])
        canvas = center_crop_resize(img, 1024, 1024)

    elif num_images == 2:
        # Two images - determine layout based on aspect ratios
        img1 = download_image_to_PIL(images_to_use[0])
        img2 = download_image_to_PIL(images_to_use[1])

        # Calculate average aspect ratio
        avg_aspect = (img1.width / img1.height + img2.width / img2.height) / 2

        if avg_aspect > 1.0:  # Wider images - use horizontal split (two rows)
            img1_resized = center_crop_resize(img1, 1024, 512)
            img2_resized = center_crop_resize(img2, 1024, 512)
            canvas.paste(img1_resized, (0, 0))
            canvas.paste(img2_resized, (0, 512))
        else:  # Taller images - use vertical split (two columns)
            img1_resized = center_crop_resize(img1, 512, 1024)
            img2_resized = center_crop_resize(img2, 512, 1024)
            canvas.paste(img1_resized, (0, 0))
            canvas.paste(img2_resized, (512, 0))

    elif num_images == 3:
        # Three images - one main + two subdivided
        img1 = download_image_to_PIL(images_to_use[0])
        img2 = download_image_to_PIL(images_to_use[1])
        img3 = download_image_to_PIL(images_to_use[2])

        # Calculate average aspect ratio
        avg_aspect = (img1.width / img1.height + img2.width / img2.height + img3.width / img3.height) / 3

        if avg_aspect > 1.0:  # Wider images - horizontal main split, then vertical subdivision
            img1_resized = center_crop_resize(img1, 1024, 512)
            img2_resized = center_crop_resize(img2, 512, 512)
            img3_resized = center_crop_resize(img3, 512, 512)
            canvas.paste(img1_resized, (0, 0))
            canvas.paste(img2_resized, (0, 512))
            canvas.paste(img3_resized, (512, 512))
        else:  # Taller images - vertical main split, then horizontal subdivision
            img1_resized = center_crop_resize(img1, 512, 1024)
            img2_resized = center_crop_resize(img2, 512, 512)
            img3_resized = center_crop_resize(img3, 512, 512)
            canvas.paste(img1_resized, (0, 0))
            canvas.paste(img2_resized, (512, 0))
            canvas.paste(img3_resized, (512, 512))

    else:  # 4+ images - 2x2 grid
        positions = [(0, 0), (512, 0), (0, 512), (512, 512)]
        for i in range(4):
            img = download_image_to_PIL(images_to_use[i])
            img_resized = center_crop_resize(img, 512, 512)
            canvas.paste(img_resized, positions[i])

    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.webp', delete=False)
    canvas.save(temp_file.name, 'WEBP', quality=95)

    # Upload the thumbnail
    result = upload_media(temp_file.name, save_thumbnails=False)

    # Clean up temp file
    os.unlink(temp_file.name)

    return result['filename']


def create_concept_thumbnail(concept: Concept) -> str:
    try:
        if not concept.images:
            return
        images = [i["image"] for i in concept.images]
        thumbnail_url = create_thumbnail(images)
        if thumbnail_url:
            concept.update(thumbnail=thumbnail_url)
            logger.info(f"Generated thumbnail: {thumbnail_url}")
    except Exception as e:
        logger.error(f"Error generating thumbnail: {e}")


@handle_errors
async def handle_concept_create(
    request: CreateConceptRequest,
    background_tasks: BackgroundTasks
):
    # Log the incoming request to check for name field
    logger.info(f"Creating concept with request: {request}")

    concept = Concept.from_mongo(request.concept)
    if not concept:
        raise APIError(f"Concept not found: {request.concept}", status_code=404)

    # Generate thumbnail if images exist
    if concept.images:
        from eve.api.api import create_concept_thumbnail_fn
        create_concept_thumbnail_fn.spawn(concept)

    return {
        "id": str(concept.id),
    }



@handle_errors
async def handle_concept_update(
    request: UpdateConceptRequest,
    background_tasks: BackgroundTasks
):
    # Log the incoming request to check for name field
    logger.info(f"Updating concept with request: {request}")

    concept = Concept.from_mongo(request.concept)
    if not concept:
        raise APIError(f"Concept not found: {request.concept}", status_code=404)

    # Generate thumbnail if images exist
    if concept.images:
        from eve.api.api import create_concept_thumbnail_fn
        create_concept_thumbnail_fn.spawn(concept)

    return {
        "id": str(concept.id),
    }

