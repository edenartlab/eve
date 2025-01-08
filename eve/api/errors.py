import functools
import logging
import sentry_sdk
from fastapi import HTTPException
from typing import Callable, Any

logger = logging.getLogger(__name__)


class APIError(HTTPException):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(status_code=status_code, detail=message)


def handle_errors(func: Callable) -> Callable:
    """
    Decorator that handles errors consistently across API endpoints.
    Logs errors, reports to Sentry, and returns appropriate responses.
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        try:
            result = await func(*args, **kwargs)
            return result

        except APIError as e:
            # Log API errors but don't send to Sentry as these are expected
            logger.error(f"API Error in {func.__name__}: {str(e)}", exc_info=True)
            raise

        except Exception as e:
            # Log unexpected errors and send to Sentry
            logger.error(f"Unexpected error in {func.__name__}", exc_info=True)
            sentry_sdk.capture_exception(e)
            raise APIError(f"Operation failed: {str(e)}")

    return wrapper
