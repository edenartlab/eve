import re
from typing import Any, Dict, Iterable, List, Literal, Optional

from bson import ObjectId
from loguru import logger
from pydantic import BaseModel

from .mongo import Collection, Document, MongoDocumentNotFound, get_collection
from .mongo_async import get_async_collection


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
    platformUserImage: Optional[str] = (
        None  # Original platform avatar URL (Discord CDN, Farcaster, Twitter)
    )
    stats: Optional[UserStats] = UserStats()
    social_accounts: Optional[Dict[str, Any]] = {}

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

    def is_admin(self) -> bool:
        return "eden_admin" in (self.featureFlags or [])

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
    def from_discord(
        cls, discord_id, discord_username, discord_avatar: Optional[str] = None
    ):
        discord_id = str(discord_id)
        discord_username = str(discord_username)
        users = get_collection(cls.collection_name)
        matching_users = list(users.find({"discordId": discord_id}))

        # Build the platform avatar URL if we have an avatar hash
        platform_avatar_url = None
        if discord_avatar:
            ext = "gif" if discord_avatar.startswith("a_") else "png"
            platform_avatar_url = f"https://cdn.discordapp.com/avatars/{discord_id}/{discord_avatar}.{ext}"

        if not matching_users:
            username = cls._get_unique_username(f"discord_{discord_username}")
            new_user = cls(
                discordId=discord_id,
                discordUsername=discord_username,
                eden_user_id=None,
                username=username,
            )
            # Upload avatar for new user
            if platform_avatar_url:
                try:
                    uploaded_url = cls._upload_platform_avatar(platform_avatar_url)
                    if uploaded_url:
                        new_user.platformUserImage = platform_avatar_url
                        new_user.userImage = uploaded_url
                except Exception as e:
                    logger.warning(
                        f"Failed to upload Discord avatar for new user {discord_id}: {e}"
                    )
            new_user.save()
            return new_user

        # Find user with userId if any exist
        user_with_id = next((u for u in matching_users if u.get("userId")), None)
        user = cls(**(user_with_id or matching_users[0]))

        # Check if avatar has changed (or is new)
        if platform_avatar_url and platform_avatar_url != user.platformUserImage:
            try:
                uploaded_url = cls._upload_platform_avatar(platform_avatar_url)
                if uploaded_url:
                    user.update(
                        platformUserImage=platform_avatar_url, userImage=uploaded_url
                    )
                    logger.info(f"Updated Discord avatar for user {discord_id}")
            except Exception as e:
                logger.warning(
                    f"Failed to upload Discord avatar for user {discord_id}: {e}"
                )

        return user

    @classmethod
    def _upload_platform_avatar(cls, avatar_url: str) -> Optional[str]:
        """
        Upload a platform avatar to Eden S3.

        Args:
            avatar_url: The full URL to the avatar image

        Returns:
            The S3 URL of the uploaded avatar, or None if upload failed
        """
        from .s3 import upload_file_from_url

        try:
            s3_url, _ = upload_file_from_url(avatar_url)
            logger.info(f"Uploaded platform avatar: {s3_url}")
            return s3_url
        except Exception as e:
            logger.warning(f"Failed to upload platform avatar from {avatar_url}: {e}")
            return None

    @classmethod
    def from_farcaster(
        cls,
        farcaster_id,
        farcaster_username,
        farcaster_avatar_url: Optional[str] = None,
    ):
        farcaster_id = str(farcaster_id)
        farcaster_username = str(farcaster_username)
        users = get_collection(cls.collection_name)
        user_doc = users.find_one({"farcasterId": farcaster_id})

        if not user_doc:
            username = cls._get_unique_username(f"farcaster_{farcaster_username}")
            new_user = cls(
                farcasterId=farcaster_id,
                farcasterUsername=farcaster_username,
                eden_user_id=None,
                username=username,
            )
            # Upload avatar for new user
            if farcaster_avatar_url:
                try:
                    uploaded_url = cls._upload_platform_avatar(farcaster_avatar_url)
                    if uploaded_url:
                        new_user.platformUserImage = farcaster_avatar_url
                        new_user.userImage = uploaded_url
                except Exception as e:
                    logger.warning(
                        f"Failed to upload Farcaster avatar for new user {farcaster_id}: {e}"
                    )
            new_user.save()
            return new_user

        user = cls(**user_doc)

        # Check if avatar has changed (or is new)
        if farcaster_avatar_url and farcaster_avatar_url != user.platformUserImage:
            try:
                uploaded_url = cls._upload_platform_avatar(farcaster_avatar_url)
                if uploaded_url:
                    user.update(
                        platformUserImage=farcaster_avatar_url, userImage=uploaded_url
                    )
                    logger.info(f"Updated Farcaster avatar for user {farcaster_id}")
            except Exception as e:
                logger.warning(
                    f"Failed to upload Farcaster avatar for user {farcaster_id}: {e}"
                )

        return user

    @classmethod
    def from_twitter(
        cls, twitter_id, twitter_username, twitter_avatar_url: Optional[str] = None
    ):
        twitter_id = str(twitter_id)
        twitter_username = str(twitter_username)
        users = get_collection(cls.collection_name)
        user_doc = users.find_one({"twitterId": twitter_id})

        if not user_doc:
            username = cls._get_unique_username(f"twitter_{twitter_username}")
            new_user = cls(
                twitterId=twitter_id,
                twitterUsername=twitter_username,
                eden_user_id=None,
                username=username,
            )
            # Upload avatar for new user
            if twitter_avatar_url:
                try:
                    uploaded_url = cls._upload_platform_avatar(twitter_avatar_url)
                    if uploaded_url:
                        new_user.platformUserImage = twitter_avatar_url
                        new_user.userImage = uploaded_url
                except Exception as e:
                    logger.warning(
                        f"Failed to upload Twitter avatar for new user {twitter_id}: {e}"
                    )
            new_user.save()
            return new_user

        user = cls(**user_doc)

        # Check if avatar has changed (or is new)
        if twitter_avatar_url and twitter_avatar_url != user.platformUserImage:
            try:
                uploaded_url = cls._upload_platform_avatar(twitter_avatar_url)
                if uploaded_url:
                    user.update(
                        platformUserImage=twitter_avatar_url, userImage=uploaded_url
                    )
                    logger.info(f"Updated Twitter avatar for user {twitter_id}")
            except Exception as e:
                logger.warning(
                    f"Failed to upload Twitter avatar for user {twitter_id}: {e}"
                )

        return user

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
            {"_id": {"$in": normalized_ids}},
            {"eden_user_id": 1},
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
    def get_canonical_user_id(cls, user_id: Optional[ObjectId]) -> Optional[ObjectId]:
        normalized = cls._normalize_object_id(user_id)
        if not normalized:
            return None
        return cls.get_canonical_id_map([normalized]).get(normalized, normalized)


def increment_message_count(user_id: ObjectId) -> None:
    """
    Efficiently increment stats.messageCount for a user or agent.
    Uses MongoDB $inc for atomic update without loading the document.
    """
    if not user_id:
        return

    users = get_collection(User.collection_name)
    users.update_one(
        {"_id": user_id},
        {"$inc": {"stats.messageCount": 1}},
    )


async def async_increment_message_count(user_id: ObjectId) -> None:
    """
    Async version of increment_message_count using Motor.
    """
    if not user_id:
        return

    users = get_async_collection(User.collection_name)
    await users.update_one(
        {"_id": user_id},
        {"$inc": {"stats.messageCount": 1}},
    )


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


class DiscordGuild(BaseModel):
    """Discord guild info from OAuth."""

    id: str
    name: str
    permissions: str  # bitfield
    owner: Optional[bool] = False
    icon: Optional[str] = None
    has_eden_bot: Optional[bool] = False


@Collection("discord_connections")
class DiscordConnection(Document):
    """User's Discord OAuth connection."""

    user_id: ObjectId
    discord_user_id: str
    discord_username: str
    access_token: str
    refresh_token: str
    expires_at: Optional[str] = None

    # Cached guild data
    guilds: Optional[List[DiscordGuild]] = []

    @classmethod
    def get_for_user(cls, user_id: ObjectId):
        """Get Discord connection for a user."""
        try:
            return cls.find_one({"user_id": user_id})
        except MongoDocumentNotFound:
            return None


@Collection("discord_guild_access")
class DiscordGuildAccess(Document):
    """Role-based access configuration for Discord guilds."""

    guild_id: str
    allowed_role_ids: List[str] = []
    updated_by_user_id: Optional[ObjectId] = None
    updated_by_discord_id: Optional[str] = None

    @classmethod
    def get_for_guild(cls, guild_id: str):
        return cls.find_one({"guild_id": guild_id})

    @classmethod
    def set_roles(
        cls,
        guild_id: str,
        role_ids: List[str],
        updated_by_discord_id: Optional[str] = None,
        updated_by_user_id: Optional[ObjectId] = None,
    ):
        unique_roles = list({role_id for role_id in role_ids if role_id})
        doc = cls.get_for_guild(guild_id) or cls(guild_id=guild_id)
        doc.allowed_role_ids = unique_roles
        doc.updated_by_discord_id = updated_by_discord_id
        doc.updated_by_user_id = updated_by_user_id
        doc.save(upsert_filter={"guild_id": guild_id})
        return doc


@Collection("guild_webhooks")
class GuildWebhook(Document):
    """Shared webhook for a Discord channel."""

    guild_id: str
    channel_id: str
    webhook_id: str
    webhook_token: str
    webhook_url: str
    created_by: ObjectId

    @classmethod
    def get_for_channel(cls, guild_id: str, channel_id: str):
        """Get webhook for a channel."""
        try:
            return cls.find_one({"guild_id": guild_id, "channel_id": channel_id})
        except MongoDocumentNotFound:
            return None
