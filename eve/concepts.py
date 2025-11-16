import logging
import os
from typing import Any, List, Optional

import modal
from bson import ObjectId
from fastapi import BackgroundTasks

from eve.api.api_requests import (
    CreateConceptRequest,
    UpdateConceptRequest,
)
from eve.api.errors import APIError, handle_errors
from eve.mongo import Collection, Document
from eve.utils.media_utils import create_thumbnail

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


@handle_errors
def create_concept_thumbnail(concept: Concept) -> str:
    try:
        if not concept.images:
            raise APIError("No images found", status_code=400)
        images = [i["image"] for i in concept.images]
        thumbnail_url = create_thumbnail(images)
        if thumbnail_url:
            concept.update(thumbnail=thumbnail_url)
            logger.info(f"Generated thumbnail: {thumbnail_url}")
            return thumbnail_url
        else:
            raise APIError("No thumbnail generated", status_code=400)
    except Exception as e:
        logger.error(f"Error generating thumbnail: {e}")
        raise e


@handle_errors
async def handle_concept_create(
    request: CreateConceptRequest, background_tasks: BackgroundTasks
):
    # Log the incoming request to check for name field
    logger.info(f"Creating concept with request: {request}")

    concept = Concept.from_mongo(request.concept)
    if not concept:
        raise APIError(f"Concept not found: {request.concept}", status_code=404)

    # Generate thumbnail if images exist
    if concept.images:
        # from eve.api.api import create_concept_thumbnail_fn
        # create_concept_thumbnail_fn.spawn(concept)
        db = os.getenv("DB", "STAGE").upper()
        func = modal.Function.from_name(
            f"api-{db.lower()}", "create_concept_thumbnail", environment_name="main"
        )
        func.spawn(concept)

    return {
        "id": str(concept.id),
    }


@handle_errors
async def handle_concept_update(
    request: UpdateConceptRequest, background_tasks: BackgroundTasks
):
    # Log the incoming request to check for name field
    logger.info(f"Creating concept with request: {request}")

    concept = Concept.from_mongo(request.concept)
    if not concept:
        raise APIError(f"Concept not found: {request.concept}", status_code=404)

    # Generate thumbnail if images exist
    if concept.images:
        # from eve.api.api import create_concept_thumbnail_fn
        # create_concept_thumbnail_fn.spawn(concept)
        db = os.getenv("DB", "STAGE").upper()
        func = modal.Function.from_name(
            f"api-{db.lower()}", "create_concept_thumbnail", environment_name="main"
        )
        func.spawn(concept)

    return {
        "id": str(concept.id),
    }
