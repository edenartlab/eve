from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import asyncio
from bson import ObjectId

from eve.api.errors import APIError
from eve.user import User
from eve.task import Task


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
        MannaSpendRateLimit(spend=250, period=24 * 60 * 60),  # 0m / 24 hours
    ],
    1: [
        MannaSpendRateLimit(spend=500, period=3 * 60 * 60),  # 500m / 3 hours
        MannaSpendRateLimit(spend=250, period=60),  # 250m / minute
    ],
    2: [
        MannaSpendRateLimit(spend=1500, period=3 * 60 * 60),  # 1500m / 3 hours
        MannaSpendRateLimit(spend=500, period=60),  # 500m / minute
    ],
    3: [
        MannaSpendRateLimit(spend=5000, period=3 * 60 * 60),  # 5000m / 3 hours
        MannaSpendRateLimit(spend=750, period=60),  # 750m / minute
    ],
}

FEATURE_FLAG_MANNA_LIMITS = {
    "free_tools": [
        MannaSpendRateLimit(spend=10**5, period=60),  # unlimited / minute
        MannaSpendRateLimit(spend=10**5, period=3 * 60 * 60),  # unlimited / 3 hours
        MannaSpendRateLimit(spend=10**5, period=24 * 60 * 60),  # unlimited / 24 hours
    ],
    "test_rate_limit": [
        MannaSpendRateLimit(spend=2, period=60),  # 2m / minute
        MannaSpendRateLimit(spend=4, period=3 * 60 * 60),  # 2m / 3 hours
    ],
}


class RateLimiter:
    def __init__(self):
        self._lock = asyncio.Lock()

    async def check_manna_spend_rate_limit(self, user: User) -> bool:
        """
        Check if user has exceeded their manna spend rate limit based on their tier.
        Uses the highest applicable limit for the user.
        """
        async with self._lock:
            # Collect all applicable limits for the user
            all_applicable_limits = []

            # Check subscription tier limits
            if (
                user.subscriptionTier is not None
                and user.subscriptionTier in SUBSCRIPTION_TIER_MANNA_LIMITS
            ):
                all_applicable_limits.extend(
                    SUBSCRIPTION_TIER_MANNA_LIMITS[user.subscriptionTier]
                )

            # Check feature flag limits
            for flag in user.featureFlags or []:
                if flag in FEATURE_FLAG_MANNA_LIMITS:
                    all_applicable_limits.extend(FEATURE_FLAG_MANNA_LIMITS[flag])

            # If no limits apply, allow access
            if not all_applicable_limits:
                return True

            # Find the highest limit for each time period
            # Group limits by period
            period_to_limits = {}
            for limit in all_applicable_limits:
                if limit.period not in period_to_limits:
                    period_to_limits[limit.period] = []
                period_to_limits[limit.period].append(limit)

            # Find the highest spend limit for each period
            highest_limits = []
            for period, limits in period_to_limits.items():
                highest_limit = max(limits, key=lambda x: x.spend)
                highest_limits.append(highest_limit)
            print("XXX highest_limits", highest_limits)

            # Check against each highest limit
            for limit in highest_limits:
                cutoff = datetime.now(timezone.utc) - timedelta(seconds=limit.period)

                # Aggregate manna spend from completed tasks in the period
                pipeline = [
                    {
                        "$match": {
                            "user": ObjectId(user.id),
                            "status": {"$ne": "failed"},
                            "createdAt": {"$gte": cutoff},
                        }
                    },
                    {"$group": {"_id": None, "total_spend": {"$sum": "$cost"}}},
                ]
                result = list(Task.get_collection().aggregate(pipeline))
                total_spend = result[0]["total_spend"] if result else 0

                if total_spend >= limit.spend:
                    period_minutes = limit.period // 60
                    period_display = (
                        f"{period_minutes} minutes"
                        if period_minutes < 60
                        else f"{period_minutes // 60} hours"
                    )

                    raise APIError(
                        f"Manna spend rate limit of {limit.spend} manna per {period_display} exceeded",
                        status_code=429,
                    )

            return True
