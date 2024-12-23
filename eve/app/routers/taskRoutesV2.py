from fastapi import APIRouter, HTTPException, Depends
from eve.app.database.mongo import serialize_document
from eve.app.auth import auth
from eve.app.controllers.taskControllerV2 import TaskRequest, handle_task

router = APIRouter()

@router.post("/create")
async def task_admin(request: TaskRequest, _: dict = Depends(auth.authenticate_admin)):
    """API endpoint to create a task."""
    result = await handle_task(request.tool, request.user_id, request.args)
    return serialize_document(result.model_dump())
