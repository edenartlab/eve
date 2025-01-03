import sentry_sdk
from dotenv import load_dotenv
from pathlib import Path
from pydantic import SecretStr
import os

home_dir = str(Path.home())
eve_path = os.path.join(home_dir, ".eve")
env_path = ".env"

# First try ENV_PATH from environment
env_path_override = os.getenv("ENV_PATH")
if env_path_override and os.path.exists(env_path_override):
    load_dotenv(env_path_override)

# Then try ~/.eve
if os.path.exists(eve_path):
    load_dotenv(eve_path, override=True)

# Finally fall back to .env
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)

# start sentry
sentry_dsn = os.getenv("SENTRY_DSN")
sentry_sdk.init(dsn=sentry_dsn, traces_sample_rate=1.0, profiles_sample_rate=1.0)

# load api keys
EDEN_API_KEY = SecretStr(os.getenv("EDEN_API_KEY", ""))

if not EDEN_API_KEY:
    print("WARNING: EDEN_API_KEY is not set")
