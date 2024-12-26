from fastapi import APIRouter, HTTPException, Depends, Query
from eve.app.auth import auth
from eve.app.controllers.v2.creatorController import getUser
from eve.app.database.mongo import serialize_document
from eve.app.controllers.v2.creatorController import GetUserResponse

router = APIRouter()

@router.get("/v2/creator/me", response_model=GetUserResponse)
async def get_creator_me(user_data: dict = Depends(auth.authenticate)):
    try:
        user_id = getattr(user_data, "user_id", None)
        print("0000000000", user_id)
        res = getUser(user_id)
        if not res:
            raise HTTPException(status_code=404, detail="User not found")
        
        user, manna = res
        if not manna:
            raise HTTPException(status_code=404, detail="Manna data not found")
        
        balance = manna.get("balance", 0.0)
        subscription_balance = manna.get("subscriptionBalance", 0.0)
        total_balance = subscription_balance + balance

        return GetUserResponse(
            balance=total_balance,
            subscriptionBalance=subscription_balance,
            foreverBalance=balance,
            creator=serialize_document(user)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
