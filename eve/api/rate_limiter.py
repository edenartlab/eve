import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from bson import ObjectId

from eve.api.errors import APIError
from eve.task import Task
from eve.user import User


@dataclass
class RateLimit:
    count: int
    period: int


@dataclass
class MannaSpendRateLimit:
    spend: int
    period: int


# Define tier rate limits
SUBSCRIPTION_TIER_MANNA_LIMITS = {
    0: [
        MannaSpendRateLimit(spend=250, period=24 * 60 * 60),
    ],
    1: [
        MannaSpendRateLimit(spend=500, period=3 * 60 * 60),
        MannaSpendRateLimit(spend=250, period=60),
    ],
    2: [
        MannaSpendRateLimit(spend=1500, period=3 * 60 * 60),
        MannaSpendRateLimit(spend=500, period=60),
    ],
    3: [
        MannaSpendRateLimit(spend=5000, period=3 * 60 * 60),
        MannaSpendRateLimit(spend=750, period=60),
    ],
}

FEATURE_FLAG_MANNA_LIMITS = {
    "free_tools": [
        MannaSpendRateLimit(spend=10**5, period=60),
        MannaSpendRateLimit(spend=10**5, period=3 * 60 * 60),
        MannaSpendRateLimit(spend=10**6, period=24 * 60 * 60),
    ],
    "limits_Admin": [
        MannaSpendRateLimit(spend=10**5, period=60),
        MannaSpendRateLimit(spend=10**5, period=3 * 60 * 60),
        MannaSpendRateLimit(spend=10**6, period=24 * 60 * 60),
    ],
    "test_rate_limit": [
        MannaSpendRateLimit(spend=2, period=60),
        MannaSpendRateLimit(spend=4, period=3 * 60 * 60),
    ],
}

AGENT_RATE_LIMITS = {
    "basic_limits": [
        MannaSpendRateLimit(spend=100, period=60),
        MannaSpendRateLimit(spend=500, period=3 * 60 * 60),
        MannaSpendRateLimit(spend=3000, period=24 * 60 * 60),
    ],
    "premium_limits": [
        MannaSpendRateLimit(spend=200, period=60),
        MannaSpendRateLimit(spend=1500, period=3 * 60 * 60),
        MannaSpendRateLimit(spend=7500, period=30 * 24 * 60 * 60),
    ],
    "unlimited": [
        MannaSpendRateLimit(spend=10**5, period=60),
        MannaSpendRateLimit(spend=10**5, period=3 * 60 * 60),
        MannaSpendRateLimit(spend=10**6, period=30 * 24 * 60 * 60),
    ],
    "test_agent_rate_limit": [
        MannaSpendRateLimit(spend=2, period=60),
        MannaSpendRateLimit(spend=4, period=3 * 60 * 60),
    ],
}

SUBSCRIPTION_TIER_MESSAGE_LIMITS = {
    0: [
        RateLimit(count=5, period=60),
        RateLimit(count=60, period=3 * 60 * 60),
        RateLimit(count=300, period=24 * 60 * 60),
    ],
    1: [
        RateLimit(count=10, period=60),
        RateLimit(count=180, period=3 * 60 * 60),
        RateLimit(count=900, period=24 * 60 * 60),
    ],
    2: [
        RateLimit(count=20, period=60),
        RateLimit(count=360, period=3 * 60 * 60),
        RateLimit(count=2000, period=24 * 60 * 60),
    ],
    3: [
        RateLimit(count=20, period=60),
        RateLimit(count=720, period=3 * 60 * 60),
        RateLimit(count=4000, period=24 * 60 * 60),
    ],
}

FEATURE_FLAG_MESSAGE_LIMITS = {
    "free_tools": [
        RateLimit(count=10**1, period=60),
        RateLimit(count=10**3, period=3 * 60 * 60),
        RateLimit(count=10**4, period=24 * 60 * 60),
    ],
    "limits_Admin": [
        RateLimit(count=10**1, period=60),
        RateLimit(count=10**3, period=3 * 60 * 60),
        RateLimit(count=10**4, period=24 * 60 * 60),
    ],
    "test_message_rate_limit": [
        RateLimit(count=2, period=60),
        RateLimit(count=4, period=3 * 60 * 60),
    ],
}


class RateLimiter:
    def __init__(self):
        self._lock = asyncio.Lock()

    def _find_highest_limits(self, all_applicable_limits):
        """
        Find the highest spend limit for each time period from the applicable limits.
        Returns a list of the highest limit for each period.
        """
        # Group limits by period
        period_to_limits = {}
        for limit in all_applicable_limits:
            if limit.period not in period_to_limits:
                period_to_limits[limit.period] = []
            period_to_limits[limit.period].append(limit)

        # Find the highest spend limit for each period
        highest_limits = []
        for period, limits in period_to_limits.items():
            highest_limit = max(
                limits,
                key=lambda x: getattr(x, "spend", getattr(x, "count", 0)),
            )
            highest_limits.append(highest_limit)

        return highest_limits

    async def _check_against_limits(self, highest_limits, pipeline):
        """
        Check if total spend exceeds any of the limits using the provided pipeline.
        Raises APIError if any limit is exceeded.
        """
        for limit in highest_limits:
            result = list(Task.get_collection().aggregate(pipeline(limit)))
            total_spend = result[0]["total_spend"] if result else 0
            if total_spend >= limit.spend:
                period_minutes = limit.period // 60
                period_display = (
                    f"{period_minutes} minutes"
                    if period_minutes < 60
                    else f"{period_minutes // 60} hours"
                )

                raise APIError(
                    f"Rate limit of {limit.spend} manna per {period_display} exceeded",
                    status_code=429,
                )

        return True

    async def check_manna_spend_rate_limit(self, user: User) -> bool:
        """
        Check if user has exceeded their manna spend rate limit based on their tier.
        Uses the highest applicable limit for the user.
        """
        async with self._lock:
            # Collect all applicable limits for the user
            all_applicable_limits = []

            # Check feature flag limits first - they override subscription tier limits
            feature_flag_applied = False
            for flag in user.featureFlags or []:
                if flag in FEATURE_FLAG_MANNA_LIMITS.keys():
                    all_applicable_limits.extend(FEATURE_FLAG_MANNA_LIMITS[flag])
                    feature_flag_applied = True

            # Only check subscription tier limits if no feature flags were applied
            if not feature_flag_applied:
                if (
                    user.subscriptionTier is not None
                    and user.subscriptionTier in SUBSCRIPTION_TIER_MANNA_LIMITS
                ):
                    all_applicable_limits.extend(
                        SUBSCRIPTION_TIER_MANNA_LIMITS[user.subscriptionTier]
                    )

            # If no limits apply, allow access
            if not all_applicable_limits:
                return True

            # Find the highest limit for each time period
            highest_limits = self._find_highest_limits(all_applicable_limits)

            # Define pipeline generator for manna spend limits
            def generate_pipeline(limit):
                cutoff = datetime.now(timezone.utc) - timedelta(seconds=limit.period)
                return [
                    {
                        "$match": {
                            "user": ObjectId(user.id),
                            "status": {"$ne": "failed"},
                            "createdAt": {"$gte": cutoff},
                        }
                    },
                    {"$group": {"_id": None, "total_spend": {"$sum": "$cost"}}},
                ]

            # Check against each highest limit
            limits = await self._check_against_limits(highest_limits, generate_pipeline)
            return limits

    async def check_agent_rate_limit(self, user: User, agent_id: str) -> bool:
        """
        Check if user has exceeded their agent rate limit based on their tier.
        Uses the highest applicable limit for the user.
        """
        async with self._lock:
            # Determine which rate limit applies based on user tier or feature flags
            applicable_limits = []

            # Check if user has free_agents feature flag
            if user.featureFlags and any(
                flag in user.featureFlags
                for flag in ["free_agents", "free_tools", "limits_Admin"]
            ):
                applicable_limits.extend(AGENT_RATE_LIMITS["unlimited"])

            if user.featureFlags and "test_agent_rate_limit" in user.featureFlags:
                applicable_limits.extend(AGENT_RATE_LIMITS["test_agent_rate_limit"])

            if not user.subscriptionTier:
                applicable_limits.extend(AGENT_RATE_LIMITS["basic_limits"])
            # Otherwise use tier-based limits
            elif user.subscriptionTier == 3:
                applicable_limits.extend(AGENT_RATE_LIMITS["premium_limits"])
            elif user.subscriptionTier <= 2:
                applicable_limits.extend(AGENT_RATE_LIMITS["basic_limits"])

            # Find the highest limit for each time period (if there are multiple with same period)
            highest_limits = self._find_highest_limits(applicable_limits)

            # Define pipeline generator for agent rate limits
            def generate_pipeline(limit):
                cutoff = datetime.now(timezone.utc) - timedelta(seconds=limit.period)
                return [
                    {
                        "$match": {
                            "agent": ObjectId(agent_id),
                            "status": {"$ne": "failed"},
                            "createdAt": {"$gte": cutoff},
                        }
                    },
                    {"$group": {"_id": None, "total_spend": {"$sum": "$cost"}}},
                ]

            # Check against each limit
            limits = await self._check_against_limits(highest_limits, generate_pipeline)

            return limits

    async def check_message_rate_limit(self, billed_user: User) -> bool:
        """
        Check if the billed user has exceeded their assistant message rate limit.
        Limits are based on subscription tier unless overridden by feature flags.
        """
        if not billed_user:
            return True

        async with self._lock:
            applicable_limits = []

            feature_flag_applied = False
            for flag in billed_user.featureFlags or []:
                if flag in FEATURE_FLAG_MESSAGE_LIMITS:
                    applicable_limits.extend(FEATURE_FLAG_MESSAGE_LIMITS[flag])
                    feature_flag_applied = True

            if not feature_flag_applied:
                tier = billed_user.subscriptionTier
                if tier is not None and tier in SUBSCRIPTION_TIER_MESSAGE_LIMITS:
                    applicable_limits.extend(SUBSCRIPTION_TIER_MESSAGE_LIMITS[tier])

            if not applicable_limits:
                return True

            highest_limits = self._find_highest_limits(applicable_limits)

            from eve.agent.session.models import ChatMessage

            messages_collection = ChatMessage.get_collection()

            def generate_pipeline(limit):
                cutoff = datetime.now(timezone.utc) - timedelta(seconds=limit.period)
                return [
                    {
                        "$match": {
                            "billed_user": ObjectId(str(billed_user.id)),
                            "role": "assistant",
                            "createdAt": {"$gte": cutoff},
                        }
                    },
                    {"$group": {"_id": None, "message_count": {"$sum": 1}}},
                ]

            for limit in highest_limits:
                result = list(messages_collection.aggregate(generate_pipeline(limit)))
                total_messages = result[0]["message_count"] if result else 0
                if total_messages >= limit.count:
                    period_minutes = limit.period // 60
                    period_display = (
                        f"{period_minutes} minutes"
                        if period_minutes < 60
                        else f"{period_minutes // 60} hours"
                    )
                    raise APIError(
                        f"Message limit of {limit.count} responses per {period_display} exceeded",
                        status_code=429,
                    )

            return True
