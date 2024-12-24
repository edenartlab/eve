from eve.tool import Tool
import os
from bson import ObjectId
from pydantic import BaseModel
from eve.app.schemas.user import Manna, User
from eve.app.database.mongo import get_collection




# Config setup
db = os.getenv("DB", "STAGE").upper()
users = get_collection("users", db="PROD")
mannas = get_collection("mannas", db="PROD")
if db not in ["PROD", "STAGE"]:
    raise Exception(f"Invalid environment: {db}. Must be PROD or STAGE")

class GetUserResponse(BaseModel):
    balance: float
    subscriptionBalance: float
    foreverBalance: float
    creator: dict

def getUser(user_id : str):
    user_data = users.find_one({"_id": ObjectId(user_id)})
    manna_data = mannas.find_one({"user": ObjectId(user_id)})

    return user_data, manna_data