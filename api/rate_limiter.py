from datetime import datetime, timedelta, timezone
from bson.objectid import ObjectId
from eve.api.errors import APIError
from eve.models.task import Task


class RateLimiter:
    async def check_agent_rate_limit(self, user: User, agent_id: str) -> bool:
        """
        Check if user has exceeded their agent rate limit based on their tier.
        Uses the highest applicable limit for the user.
        """
        async with self._lock:
            # Determine which rate limit applies based on user tier or feature flags
            rate_limit_key = "basic_limits"  # Default

            # Check if user has free_agents feature flag
            if user.featureFlags and "free_agents" in user.featureFlags:
                rate_limit_key = "unlimited"
            # Otherwise use tier-based limits
            elif user.subscriptionTier == 3:
                rate_limit_key = "premium_limits"
            elif user.subscriptionTier == 2:
                rate_limit_key = "basic_limits"

            # Get the applicable limits
            applicable_limits = AGENT_RATE_LIMITS.get(
                rate_limit_key, AGENT_RATE_LIMITS["basic_limits"]
            )

            # Check against each limit
            for limit in applicable_limits:
                cutoff = datetime.now(timezone.utc) - timedelta(seconds=limit.period)

                # Aggregate manna spend from completed tasks in the period with the specific agent
                pipeline = [
                    {
                        "$match": {
                            "user": ObjectId(user.id),
                            "agent": ObjectId(agent_id),
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
                        f"Agent rate limit of {limit.spend} manna per {period_display} exceeded",
                        status_code=429,
                    )

            return True
