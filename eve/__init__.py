import sentry_sdk
from dotenv import load_dotenv
from pathlib import Path
from pydantic import SecretStr
import os

home_dir = str(Path.home())


EDEN_API_KEY = None


def load_env(db):
    global EDEN_API_KEY

    db = db.upper()
    if db not in ["STAGE", "PROD"]:
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

    # start sentry
    sentry_dsn = os.getenv("SENTRY_DSN")
    sentry_sdk.init(dsn=sentry_dsn, traces_sample_rate=1.0, profiles_sample_rate=1.0)

    # load api keys
    EDEN_API_KEY = SecretStr(os.getenv("EDEN_API_KEY", ""))

    if not EDEN_API_KEY:
        print("WARNING: EDEN_API_KEY is not set")


db = os.getenv("DB", "STAGE")
load_env(db)
