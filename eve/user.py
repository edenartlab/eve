import re
from bson import ObjectId
from pydantic import BaseModel
from typing import Optional, Literal, List, Dict, Any, Iterable

from .mongo import (
    Document,
    Collection,
    MongoDocumentNotFound,
    get_collection,
)
from loguru import logger


@Collection("mannas")
class Manna(Document):
    user: ObjectId
    balance: float = 0
    subscriptionBalance: float = 0

    @classmethod
    def load(cls, user: ObjectId):
        try:
            return super().load(user=user)
        except MongoDocumentNotFound:
            # if mannas not found, check if user exists, and create a new manna document
            user = User.from_mongo(user)
            if not user:
                raise Exception(f"User {user} not found")
            manna = Manna(user=user.id)
            manna.save()
            return manna
        except Exception as e:
            logger.error(e)
            raise e

    def spend(self, amount: float):
        subscription_spend = min(self.subscriptionBalance, amount)
        self.subscriptionBalance -= subscription_spend
        self.balance -= amount - subscription_spend
        if self.balance < 0:
            raise Exception(
                f"Insufficient manna balance. Need {amount} but only have {self.balance + self.subscriptionBalance}"
            )
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


# todo: add more stats
class UserStats(BaseModel):
    messageCount: int = 0


@Collection("users3")
class User(Document):
    type: Optional[Literal["user", "agent"]] = "user"
    isAdmin: Optional[bool] = False
    deleted: Optional[bool] = False
    eden_user_id: Optional[ObjectId] = None

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
    stats: Optional[UserStats] = UserStats()

    # preferences
    preferences: Optional[Dict] = {"agent_spend_threshold": 50}

    # origins
    discordId: Optional[str] = None
    discordUsername: Optional[str] = None
    telegramId: Optional[str] = None
    telegramUsername: Optional[str] = None
    farcasterId: Optional[str] = None
    farcasterUsername: Optional[str] = None
    twitterId: Optional[str] = None
    twitterUsername: Optional[str] = None

    def _ensure_ids(self):
        if not self.id:
            self.id = ObjectId()
        if not self.eden_user_id:
            self.eden_user_id = self.id

    @property
    def canonical_user_id(self) -> ObjectId:
        self._ensure_ids()
        return self.eden_user_id or self.id

    def save(self, upsert_filter=None, **kwargs):
        self._ensure_ids()
        super().save(upsert_filter=upsert_filter, **kwargs)

    @classmethod
    def load(cls, username, cache=False):
        return super().load(username=username)

    def check_manna(self, amount: float):
        if "free_tools" in (self.featureFlags or []):
            return
        manna = Manna.load(self.id)
        total_balance = manna.balance + manna.subscriptionBalance
        if total_balance < amount:
            raise Exception(
                f"Insufficient manna balance. Need {amount} but only have {total_balance}"
            )

    @classmethod
    def from_discord(cls, discord_id, discord_username):
        discord_id = str(discord_id)
        discord_username = str(discord_username)
        users = get_collection(cls.collection_name)
        matching_users = list(users.find({"discordId": discord_id}))

        if not matching_users:
            username = cls._get_unique_username(f"discord_{discord_username}")
            new_user = cls(
                discordId=discord_id,
                discordUsername=discord_username,
                eden_user_id=None,
                username=username,
            )
            new_user.save()
            return new_user

        # Find user with userId if any exist
        user_with_id = next((u for u in matching_users if u.get("userId")), None)
        return cls(**(user_with_id or matching_users[0]))

    @classmethod
    def from_farcaster(cls, farcaster_id, farcaster_username):
        farcaster_id = str(farcaster_id)
        farcaster_username = str(farcaster_username)
        users = get_collection(cls.collection_name)
        user = users.find_one({"farcasterId": farcaster_id})
        if not user:
            username = cls._get_unique_username(f"farcaster_{farcaster_username}")
            new_user = cls(
                farcasterId=farcaster_id,
                farcasterUsername=farcaster_username,
                eden_user_id=None,
                username=username,
            )
            new_user.save()
            return new_user
        return cls(**user)

    @classmethod
    def from_twitter(cls, twitter_id, twitter_username):
        twitter_id = str(twitter_id)
        twitter_username = str(twitter_username)
        users = get_collection(cls.collection_name)
        user = users.find_one({"twitterId": twitter_id})
        if not user:
            username = cls._get_unique_username(f"twitter_{twitter_username}")
            new_user = cls(
                twitterId=twitter_id,
                twitterUsername=twitter_username,
                eden_user_id=None,
                username=username,
            )
            new_user.save()
            return new_user
        return cls(**user)

    @classmethod
    def from_telegram(cls, telegram_id, telegram_username):
        telegram_id = str(telegram_id)
        telegram_username = str(telegram_username)
        users = get_collection(cls.collection_name)
        user = users.find_one({"telegramId": telegram_id})
        if not user:
            username = cls._get_unique_username(f"telegram_{telegram_username}")
            new_user = cls(
                telegramId=telegram_id,
                telegramUsername=telegram_username,
                eden_user_id=None,
                username=username,
            )
            new_user.save()
            return new_user
        return cls(**user)

    @classmethod
    def from_email(cls, email: str):
        normalized = email.strip().lower()
        users = get_collection(cls.collection_name)
        user = users.find_one({"normalizedEmail": normalized})
        if not user:
            local_part = normalized.split("@")[0] if "@" in normalized else normalized
            safe_local = re.sub(r"[^a-z0-9_]", "_", local_part)
            base_username = f"email_{safe_local or 'user'}"
            username = cls._get_unique_username(base_username)
            new_user = cls(
                email=email.strip(),
                normalizedEmail=normalized,
                eden_user_id=None,
                username=username,
            )
            new_user.save()
            return new_user
        return cls(**user)

    @classmethod
    def from_gmail(cls, email_address: str, fallback_username: Optional[str] = None):
        if not email_address:
            raise ValueError("email_address is required")

        normalized_email = email_address.strip().lower()
        users = get_collection(cls.collection_name)

        # Prefer normalized email, but also fall back to raw casing if needed
        user_doc = users.find_one({"normalizedEmail": normalized_email})
        if not user_doc:
            user_doc = users.find_one({"email": normalized_email}) or users.find_one(
                {"email": email_address}
            )

        if user_doc:
            return cls(**user_doc)

        base_username = fallback_username or normalized_email.split("@")[0]
        base_username = (base_username or "").strip().lower().replace(" ", "_")
        if not base_username:
            base_username = normalized_email.split("@")[0]
        username = cls._get_unique_username(f"email_{base_username}")
        new_user = cls(
            email=email_address,
            normalizedEmail=normalized_email,
            eden_user_id=None,
            username=username,
        )
        new_user.save()
        return new_user

    @classmethod
    def _get_unique_username(cls, base_username):
        users = get_collection(cls.collection_name)
        username = base_username
        counter = 2
        while users.find_one({"username": username}):
            username = f"{base_username}{counter}"
            counter += 1
        return username

    @staticmethod
    def _normalize_object_id(value: Optional[ObjectId]) -> Optional[ObjectId]:
        if value is None:
            return None
        if isinstance(value, ObjectId):
            return value
        return ObjectId(value)

    @classmethod
    def get_canonical_id_map(
        cls, user_ids: Iterable[Optional[ObjectId]]
    ) -> Dict[ObjectId, ObjectId]:
        normalized_ids = [
            cls._normalize_object_id(user_id) for user_id in user_ids if user_id
        ]
        if not normalized_ids:
            return {}

        users = get_collection(cls.collection_name)
        cursor = users.find(
            {"_id": {"$in": normalized_ids}}, {"eden_user_id": 1},
        )

        canonical_lookup: Dict[ObjectId, ObjectId] = {}
        for doc in cursor:
            original_id: ObjectId = doc["_id"]
            canonical_lookup[original_id] = doc.get("eden_user_id") or original_id

        result: Dict[ObjectId, ObjectId] = {}
        for original in normalized_ids:
            result[original] = canonical_lookup.get(original, original)
        return result

    @classmethod
    def get_canonical_user_id(
        cls, user_id: Optional[ObjectId]
    ) -> Optional[ObjectId]:
        normalized = cls._normalize_object_id(user_id)
        if not normalized:
            return None
        return cls.get_canonical_id_map([normalized]).get(normalized, normalized)


@Collection("user_identities")
class UserIdentity(Document):
    provider: str
    provider_user_id: str
    eden_user_id: Optional[ObjectId] = None
    status: Optional[Literal["pending", "linked", "unlinked"]] = "pending"
    linked_at: Optional[str] = None
    unlinked_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@Collection("user_identity_history")
class UserIdentityHistory(Document):
    user_identity_id: ObjectId
    eden_user_id: Optional[ObjectId] = None
    actor_eden_user_id: Optional[ObjectId] = None
    event_type: Literal["link", "unlink", "transfer"]
    metadata: Optional[Dict[str, Any]] = None
