import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from pydantic import SecretStr

home_dir = str(Path.home())
db = os.getenv("DB", "STAGE").upper()
EDEN_API_KEY = None


def configure_logging():
    # Get log level from environment, default to INFO
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    # Allow external debug logging with separate flag
    external_debug = os.getenv("EXTERNAL_DEBUG", "false").lower() == "true"

    class InfoFilter(logging.Filter):
        def filter(self, record):
            return record.levelno < logging.WARNING

    # Configure root logger to suppress third-party debug logs
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if external_debug else logging.INFO)

    # Configure the handler for stdout (for INFO and DEBUG)
    handler_stdout = logging.StreamHandler(sys.stdout)
    info_handler_level = min(logging.INFO, log_level)
    handler_stdout.setLevel(info_handler_level)
    handler_stdout.addFilter(InfoFilter())

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler_stdout.setFormatter(formatter)

    # Configure the handler for stderr (for WARNING and above)
    handler_stderr = logging.StreamHandler(sys.stderr)
    handler_stderr.setLevel(logging.WARNING)
    handler_stderr.setFormatter(formatter)

    # Clear any existing handlers and add the new ones
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.addHandler(handler_stdout)
    root_logger.addHandler(handler_stderr)

    # Set eve app logger to the desired log level
    eve_logger = logging.getLogger("eve")
    eve_logger.setLevel(log_level)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def setup_eve():
    def setup_sentry():
        sentry_dsn = os.getenv("SENTRY_DSN")
        if not sentry_dsn:
            return

        import sentry_sdk

        # Determine environment
        sentry_env = os.getenv(
            "SENTRY_ENV", "production" if db == "PROD" else "staging"
        )

        # Set sampling rates
        base_sample_rate = 1.0 if os.getenv("SENTRY_ENV") else 0.01
        profiles_sample_rate = 1.0 if os.getenv("SENTRY_ENV") else 0.01

        def before_send(event, hint):
            """Filter out certain errors before sending to Sentry"""
            # Check if there's an exception in the hint
            if "exc_info" in hint:
                error = hint["exc_info"][1]
                error_message = str(error)

                # Filter out "Document not found" errors
                if (
                    "not found" in error_message.lower()
                    and "document" in error_message.lower()
                ):
                    return None  # Don't send to Sentry

                # Filter out specific trigger not found errors
                if "not found in triggers" in error_message:
                    return None  # Don't send to Sentry

            # Check the event message as well
            if "message" in event:
                message = event["message"]
                if "not found" in message.lower() and "document" in message.lower():
                    return None
                if "not found in triggers" in message:
                    return None

            return event  # Send everything else to Sentry

        # Only import integrations needed for ERROR tracking, not performance tracing
        from sentry_sdk.integrations.argv import ArgvIntegration
        from sentry_sdk.integrations.atexit import AtexitIntegration
        from sentry_sdk.integrations.dedupe import DedupeIntegration
        from sentry_sdk.integrations.excepthook import ExcepthookIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration
        from sentry_sdk.integrations.modules import ModulesIntegration
        from sentry_sdk.integrations.stdlib import StdlibIntegration
        from sentry_sdk.integrations.threading import ThreadingIntegration

        sentry_sdk.init(
            dsn=sentry_dsn,
            traces_sample_rate=base_sample_rate,  # Simple sampling, we control transactions manually
            profiles_sample_rate=profiles_sample_rate,
            environment=sentry_env,
            debug=True if os.getenv("SENTRY_ENV") == "jmill-dev" else False,
            before_send=before_send,
            # Disable ALL auto-instrumentation - we manually instrument what we need
            default_integrations=False,
            # Only enable basic error tracking integrations (NO performance/tracing integrations)
            integrations=[
                LoggingIntegration(level=None, event_level=None),
                StdlibIntegration(),
                ExcepthookIntegration(),
                DedupeIntegration(),
                AtexitIntegration(),
                ModulesIntegration(),
                ArgvIntegration(),
                ThreadingIntegration(),
            ],
        )

    if os.getenv("SETUP_SENTRY") == "no":
        logger.debug("Skipping sentry setup because SETUP_SENTRY is no")
        pass
    else:
        setup_sentry()


def verify_env():
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION_NAME = os.getenv("AWS_REGION_NAME")
    MONGO_URI = os.getenv("MONGO_URI")

    if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME]):
        logger.warning(
            "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION_NAME must be set in the environment"
        )

    if not MONGO_URI:
        logger.warning("MONGO_URI must be set in the environment")


if db not in ["STAGE", "PROD", "WEB3-STAGE", "WEB3-PROD"]:
    raise Exception(
        f"Invalid environment: {db}. Must be STAGE, PROD, WEB3-STAGE, or WEB3-PROD"
    )


def load_env(db):
    global EDEN_API_KEY

    db = db.upper()
    if db not in ["STAGE", "PROD", "WEB3-STAGE", "WEB3-PROD"]:
        raise ValueError(f"Invalid database: {db}")

    # First try ~/.eve
    stage = db == "STAGE"
    env_file = ".env.STAGE" if stage else ".env"
    eve_file = ".eve.STAGE" if stage else ".eve"
    eve_path = os.path.join(home_dir, eve_file)

    if os.path.exists(eve_path):
        load_dotenv(eve_path, override=True)

    # Then try ENV_PATH from environment or .env
    env_path_override = os.getenv("ENV_PATH")
    if env_path_override and os.path.exists(env_path_override):
        load_dotenv(env_path_override, override=True)
    elif os.path.exists(env_file):
        load_dotenv(env_file, override=True)

    # load api keys
    EDEN_API_KEY = str(os.getenv("EDEN_API_KEY", ""))

    if not EDEN_API_KEY:
        logger.warning("EDEN_API_KEY is not set")
    else:
        EDEN_API_KEY = SecretStr(EDEN_API_KEY)

    verify_env()


configure_logging()
load_env(db)
setup_eve()
