import logging
from dotenv import load_dotenv
from pathlib import Path
from pydantic import SecretStr
import os
import sentry_sdk
from langfuse.decorators import langfuse_context


home_dir = str(Path.home())

EDEN_API_KEY = None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
db = os.getenv("DB", "STAGE").upper()

LANGFUSE_ENV = os.getenv("LANGFUSE_ENV", "production" if db == "PROD" else "staging")


def setup_langfuse():
    langfuse_private_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not langfuse_private_key:
        return

    print(f"Setting up langfuse for {LANGFUSE_ENV}")
    langfuse_context.configure(environment=LANGFUSE_ENV)


def setup_sentry():
    sentry_dsn = os.getenv("SENTRY_DSN")
    if not sentry_dsn:
        return

    # Determine environment
    sentry_env = os.getenv("SENTRY_ENV", "production" if db == "PROD" else "staging")
    print(f"Setting up sentry for {sentry_env}")

    # Set sampling rates
    traces_sample_rate = 1.0 if os.getenv("SENTRY_ENV") else 0.01
    profiles_sample_rate = 1.0 if os.getenv("SENTRY_ENV") else 0.01
    print(f"Traces sample rate: {traces_sample_rate}")
    print(f"Profiles sample rate: {profiles_sample_rate}")

    sentry_sdk.init(
        dsn=sentry_dsn,
        traces_sample_rate=traces_sample_rate,
        profiles_sample_rate=profiles_sample_rate,
        environment=sentry_env,
        debug=True if os.getenv("SENTRY_ENV") == "jmill-dev" else False,
        _experiments={
            "continuous_profiling_auto_start": True
            if os.getenv("SENTRY_ENV")
            else False,
        },
    )


def load_env(db):
    global EDEN_API_KEY

    db = db.upper()
    if db not in ["STAGE", "PROD", "WEB3-STAGE", "WEB3-PROD"]:
        raise ValueError(f"Invalid database: {db}")

    os.environ["DB"] = db

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

    # start sentry and langfuse
    setup_sentry()
    setup_langfuse()

    # load api keys
    EDEN_API_KEY = SecretStr(os.getenv("EDEN_API_KEY", ""))

    if not EDEN_API_KEY:
        print("WARNING: EDEN_API_KEY is not set")

    verify_env()


def verify_env():
    MONGO_URI = os.getenv("MONGO_URI")
    
    if not MONGO_URI:
        print("WARNING: MONGO_URI must be set in the environment")


if db not in ["STAGE", "PROD", "WEB3-STAGE", "WEB3-PROD"]:
    raise Exception(
        f"Invalid environment: {db}. Must be LOCAL, STAGE, PROD, WEB3-STAGE, or WEB3-PROD"
    )

load_env(db)
