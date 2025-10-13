import requests
import os
import logging
from urllib.parse import urlparse
from typing import Union, Mapping, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Initialize logger
logger = logging.getLogger(__name__)

# Get IPFS config from environment variables
IPFS_PREFIX = os.getenv("IPFS_PREFIX", "ipfs://")
IPFS_BASE_URL = os.getenv("IPFS_BASE_URL", "https://api.pinata.cloud")
PINATA_JWT = os.getenv("PINATA_JWT")


class IPFSError(Exception):
    """Error during IPFS operations."""
    pass


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=32),
    retry=retry_if_exception_type((IPFSError, requests.RequestException)),
    before_sleep=lambda retry_state: logger.info(f"Retrying IPFS upload (attempt {retry_state.attempt_number}/5)...")
)
def pin(data: Union[str, Mapping[str, Any]]) -> str:
    """
    Upload to Pinata and return an IPFS hash.

    Accepts:
      - URL string (http/https): downloads and uploads as a file
      - Local file path string: uploads the file
      - JSON blob (dict-like Mapping): uses pinJSONToIPFS

    Returns:
      - IPFS hash (without ipfs:// prefix)
    """

    if not PINATA_JWT:
        raise IPFSError("PINATA_JWT not configured")

    file_endpoint = f"{IPFS_BASE_URL}/pinning/pinFileToIPFS"
    auth_headers = {"Authorization": f"Bearer {PINATA_JWT}"}

    # JSON blob
    if isinstance(data, Mapping):
        url = f"{IPFS_BASE_URL}/pinning/pinJSONToIPFS"
        payload = {"pinataContent": dict(data)}
        logger.info("Uploading JSON to IPFS...")
        r = requests.post(url, headers=auth_headers, json=payload, timeout=60)

    # URL
    elif data.startswith(("http://", "https://")):
        logger.info(f"Downloading file from {data}...")
        dl = requests.get(data, timeout=60)
        if dl.status_code != 200:
            raise IPFSError(f"Failed to download file {data}: {dl.status_code}")
        filename = os.path.basename(urlparse(data).path) or "file"
        files = {"file": (filename, dl.content)}
        logger.info("Uploading downloaded file to IPFS...")
        r = requests.post(file_endpoint, files=files, headers=auth_headers, timeout=60)

    # Local file path
    elif os.path.isfile(data):
        filename = os.path.basename(data)
        logger.info(f"Uploading local file to IPFS: {filename}...")
        with open(data, "rb") as f:
            files = {"file": (filename, f)}
            r = requests.post(file_endpoint, files=files, headers=auth_headers, timeout=60)

    else:
        raise ValueError("Data must be json, URL, or local file path")

    # Check response
    if r.status_code != 200:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise IPFSError(f"IPFS upload failed: {r.status_code} {detail}")

    ipfs_hash = r.json().get("IpfsHash")
    if not ipfs_hash:
        raise IPFSError(f"Malformed response from IPFS: {r.text}")

    logger.info(f"Uploaded to IPFS: {IPFS_PREFIX}{ipfs_hash}")
    return ipfs_hash
