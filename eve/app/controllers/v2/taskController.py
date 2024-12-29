import os
from pydantic import BaseModel
from eve.tool import Tool

# Config setup
db = os.getenv("DB", "STAGE").upper()
if db not in ["PROD", "STAGE"]:
    raise Exception(f"Invalid environment: {db}. Must be PROD or STAGE")

class TaskRequest(BaseModel):
    tool: str
    args: dict
    user_id: str

async def handle_task(tool: str, user_id: str, args: dict = {}) -> dict:
    tool = Tool.load(key=tool, db=db)
    return await tool.async_start_task(requester_id=user_id, user_id=user_id, args=args, db=db)