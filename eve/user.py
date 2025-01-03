from bson import ObjectId
from pydantic import Field
from typing import Optional, Literal, List

from .mongo import Document, Collection, get_collection, MongoDocumentNotFound


@Collection("mannas")
class Manna(Document):
    user: ObjectId
    balance: float = 0
    subscriptionBalance: float = 0

    @classmethod
    def load(cls, user: ObjectId, db=None):
        try:
            return super().load(user=user, db=db)
        except MongoDocumentNotFound as e:
            # if mannas not found, check if user exists, and create a new manna document
            user = User.from_mongo(user, db=db)
            if not user:
                raise Exception(f"User {user} not found")
            manna = Manna(user=user.id, db=db)
            manna.save()
            return manna
        except Exception as e:
            print(e)
            raise e

    def spend(self, amount: float):
        subscription_spend = min(self.subscriptionBalance, amount)
        self.subscriptionBalance -= subscription_spend
        self.balance -= (amount - subscription_spend)
        if self.balance < 0:
            raise Exception(f"Insufficient manna balance. Need {amount} but only have {self.balance + self.subscriptionBalance}")
        self.save()

    def refund(self, amount: float):
        # todo: make it refund to subscription balance first if it spent from there
        self.balance += amount
        self.save()


@Collection("transactions")
class Transaction(Document):
    manna: ObjectId
    task: ObjectId
    amount: float
    type: Literal["spend", "refund"]


@Collection("users3")
class User(Document):
    # todo underscore
    # type of user
    type: Optional[Literal["user", "agent"]] = "user"
    isAdmin: Optional[bool] = False
    deleted: Optional[bool] = False

    # auth settings
    userId: Optional[str] = None
    isWeb2: Optional[bool] = False
    email: Optional[str] = None
    normalizedEmail: Optional[str] = None

    # agent settings
    agent: Optional[ObjectId] = None
    owner: Optional[ObjectId] = None

    # permissions
    featureFlags: Optional[List[str]] = []
    subscriptionTier: Optional[int] = None
    highestMonthlySubscriptionTier: Optional[int] = None

    # profile
    username: str
    userImage: Optional[str] = None

    # origins
    discordId: Optional[str] = None
    discordUsername: Optional[str] = None
    telegramId: Optional[str] = None
    telegramUsername: Optional[str] = None
    farcasterId: Optional[str] = None
    farcasterUsername: Optional[str] = None

    def check_manna(self, amount: float):
        manna = Manna.load(self.id, db=self.db)
        total_balance = manna.balance + manna.subscriptionBalance
        if total_balance < amount:
            raise Exception(
                f"Insufficient manna balance. Need {amount} but only have {total_balance}"
            )

    @classmethod
    def from_discord(cls, discord_id, discord_username, db):
        discord_id = str(discord_id)
        discord_username = str(discord_username)
        users = get_collection(cls.collection_name, db=db)
        user = users.find_one({"discordId": discord_id})
        if not user:
            username = cls._get_unique_username(f"discord_{discord_username}", db=db)
            new_user = cls(
                db=db,
                discordId=discord_id,
                discordUsername=discord_username,
                username=username,
            )
            new_user.save()
            return new_user
        return cls(**user, db=db)

    @classmethod
    def from_farcaster(cls, farcaster_id, farcaster_username, db):
        farcaster_id = str(farcaster_id)
        farcaster_username = str(farcaster_username)
        users = get_collection(cls.collection_name, db=db)
        user = users.find_one({"farcasterId": farcaster_id})
        if not user:
            username = cls._get_unique_username(f"farcaster_{farcaster_username}", db=db)
            new_user = cls(
                db=db,
                farcasterId=farcaster_id,
                farcasterUsername=farcaster_username,
                username=username,
            )
            new_user.save()
            return new_user
        return cls(**user, db=db)

    @classmethod
    def from_telegram(cls, telegram_id, telegram_username, db):
        telegram_id = str(telegram_id)
        telegram_username = str(telegram_username)
        users = get_collection(cls.collection_name, db=db)        
        user = users.find_one({"telegramId": telegram_id})
        if not user:
            username = cls._get_unique_username(f"telegram_{telegram_username}", db=db)
            new_user = cls(
                db=db,
                telegramId=telegram_id,
                telegramUsername=telegram_username,
                username=username,
            )
            new_user.save()
            return new_user
        return cls(**user, db=db)

    @classmethod
    def _get_unique_username(cls, base_username, db):
        users = get_collection(cls.collection_name, db=db)
        username = base_username
        counter = 2
        while users.find_one({"username": username}):
            username = f"{base_username}{counter}"
            counter += 1
        return username
