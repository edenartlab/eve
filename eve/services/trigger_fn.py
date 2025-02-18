import os
import modal

db = os.getenv("DB", "STAGE").upper()


trigger_app = modal.App()

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"DB": db})
    .pip_install("requests")
    .pip_install("pydantic")
    .pip_install("python-dotenv")
    .pip_install("sentry-sdk")
)


def trigger_fn():
    print("Trigger function executed")
