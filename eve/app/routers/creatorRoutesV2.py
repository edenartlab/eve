from fastapi import APIRouter, HTTPException, Depends, Query
from eve.app.database.mongo import serialize_document
from eve.app.auth import auth
from eve.app.controllers.creatorControllerV2 import getUser
from eve.app.database.mongo import serialize_document
from eve.app.controllers.creatorControllerV2 import GetUserResponse

router = APIRouter()

@router.get("/v2/creator/me")
async def getCreatorMe(userData: dict = Depends(auth.authenticate)):
    res = getUser(userData.user_id)
    user, manna = res
    balance = manna["balance"]
    subscriptionBalance = manna["subscriptionBalance"]
    totalBalance = subscriptionBalance + balance
    user = serialize_document(user)
    return GetUserResponse(
        balance=totalBalance,
        subscriptionBalance=subscriptionBalance,
        foreverBalance=balance,
        creator=user
    )