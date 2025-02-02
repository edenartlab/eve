from datetime import datetime, timedelta
from typing import Optional
import asyncio
from bson import ObjectId

from eve.api.errors import APIError
from eve.user import User
from eve.tool import Tool
from eve.task import Task


class RateLimiter:
    def __init__(self):
        self._lock = asyncio.Lock()

    async def check_create_rate_limit(self, user: User, tool: Tool) -> bool:
        async with self._lock:
            # Check if user has unlimited access
            if "freeTools" in (user.featureFlags or []):
                return True

            # Check if tool has rate limits
            if not tool.rate_limits:
                return True

            # Find applicable rate limits from user's feature flags
            applicable_limits = []
            all_restricted = True

            for flag in user.featureFlags or []:
                if flag in tool.rate_limits.keys():
                    limit = tool.rate_limits[flag]
                    if (
                        limit.count > 0
                    ):  # If any limit allows usage, tool isn't restricted
                        all_restricted = False
                    applicable_limits.append(limit)

            # Check if tool is completely restricted
            if applicable_limits and all_restricted:
                raise APIError("Tool not available for your tier", status_code=403)

            # If no limits apply to user, allow access
            if not applicable_limits:
                return True

            # Use highest rate limit
            highest_limit = max(applicable_limits, key=lambda x: x.period)

            # Count recent usage from Task collection
            cutoff = datetime.now() - timedelta(seconds=highest_limit.period)
            total_calls = Task.collection.count_documents(
                {
                    "user": ObjectId(user.id),
                    "tool": tool.key,
                    "createdAt": {"$gte": cutoff},
                }
            )

            if total_calls >= highest_limit.count:
                raise APIError(f"Rate limit exceeded for {tool.key}", status_code=429)

            return True

    async def check_chat_rate_limit(
        self, user_id: str, tool_name: Optional[str] = None, count: int = 1
    ) -> bool:
        return True
