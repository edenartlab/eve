import logging
from pathlib import Path
import os
from dotenv import load_dotenv
from pydantic import SecretStr

home_dir = str(Path.home())
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
db = os.getenv("DB", "STAGE").upper()
EDEN_API_KEY = None


def setup_eve():
    def setup_langfuse():
        LANGFUSE_ENV = os.getenv(
            "LANGFUSE_ENV", "production" if db == "PROD" else "staging"
        )

        langfuse_private_key = os.getenv("LANGFUSE_SECRET_KEY")
        if not langfuse_private_key:
            # print("Skipping langfuse setup because LANGFUSE_SECRET_KEY is not set")
            return

        from langfuse.decorators import langfuse_context

        print(f"Setting up langfuse for {LANGFUSE_ENV}")
        langfuse_context.configure(environment=LANGFUSE_ENV)

    def setup_sentry():
        sentry_dsn = os.getenv("SENTRY_DSN")
        if not sentry_dsn:
            # print("Skipping sentry setup because SENTRY_DSN is not set")
            return

        import sentry_sdk

        # Determine environment
        sentry_env = os.getenv(
            "SENTRY_ENV", "production" if db == "PROD" else "staging"
        )
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

    if os.getenv("SETUP_SENTRY") == "no":
        # print("Skipping sentry setup because SETUP_SENTRY is no")
        pass
    else:
        setup_sentry()

    if os.getenv("SETUP_LANGFUSE") == "no":
        # print("Skipping langfuse setup because SETUP_LANGFUSE is no")
        pass
    else:
        setup_langfuse()


def verify_env():
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION_NAME = os.getenv("AWS_REGION_NAME")
    MONGO_URI = os.getenv("MONGO_URI")

    if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME]):
        print(
            "WARNING: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION_NAME must be set in the environment"
        )

    if not MONGO_URI:
        print("WARNING: MONGO_URI must be set in the environment")


if db not in ["STAGE", "PROD", "WEB3-STAGE", "WEB3-PROD"]:
    raise Exception(
        f"Invalid environment: {db}. Must be STAGE, PROD, WEB3-STAGE, or WEB3-PROD"
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

    # load api keys
    EDEN_API_KEY = str(os.getenv("EDEN_API_KEY", ""))

    if not EDEN_API_KEY:
        print("WARNING: EDEN_API_KEY is not set")
    else:
        EDEN_API_KEY = SecretStr(EDEN_API_KEY)

    verify_env()


load_env(db)
setup_eve()
