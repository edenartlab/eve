import os
from bson import ObjectId
from pydantic import BaseModel
from typing import Optional, Tuple, Dict
from eve.tool import Tool
from eve.app.schemas.user import Manna, User
from eve.app.database.mongo import get_collection

# Config setup
db = os.getenv("DB", "STAGE").upper()
if db not in ["PROD", "STAGE"]:
    raise ValueError(f"Invalid environment: {db}. Must be PROD or STAGE")

users = get_collection("users", db=db)
mannas = get_collection("mannas", db=db)

class GetUserResponse(BaseModel):
    balance: float
    subscriptionBalance: float
    foreverBalance: float
    creator: dict

def getUser(user_id: str) -> Optional[Tuple[Dict, Dict]]:
    try:
        user_data = users.find_one({"_id": ObjectId(user_id)})
        if not user_data:
            return None
        
        manna_data = mannas.find_one({"user": ObjectId(user_id)})
        return user_data, manna_data
    except Exception as e:
        raise RuntimeError(f"Error fetching user data: {str(e)}")
