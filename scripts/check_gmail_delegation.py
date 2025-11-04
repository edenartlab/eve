#!/usr/bin/env python
"""
Quick sanity check that the Gmail service-account key can impersonate the expected user.

Usage:
    python scripts/check_gmail_delegation.py --user team@solienne.ai
    python scripts/check_gmail_delegation.py --user team@solienne.ai --history-id 1234

The script looks for a service-account JSON key in the sibling directory `../gcp`.
Pass --key-file if you want to point at a specific file instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import logging
from google.oauth2 import service_account
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)

SCOPES: Iterable[str] = (
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
)


def resolve_key_path(explicit_path: Optional[str]) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Key file not found: {path}")
        return path

    base_dir = Path(__file__).resolve().parent / "gcp"
    if not base_dir.is_dir():
        raise FileNotFoundError(
            f"Default key directory {base_dir} does not exist. "
            "Pass --key-file to point at an explicit key."
        )

    candidates = list(base_dir.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No *.json files found in {base_dir}. Pass --key-file explicitly."
        )
    if len(candidates) > 1:
        names = ", ".join(candidate.name for candidate in candidates)
        raise RuntimeError(
            f"Multiple key files found in {base_dir}: {names}. "
            "Pass --key-file to disambiguate."
        )

    return candidates[0]


def build_gmail_client(key_path: Path, delegated_user: str):
    creds = service_account.Credentials.from_service_account_file(
        key_path, scopes=SCOPES, subject=delegated_user
    )
    return build("gmail", "v1", credentials=creds)


def main():
    parser = argparse.ArgumentParser(
        description="Validate Gmail service-account delegation and optionally inspect history."
    )
    parser.add_argument(
        "--user",
        required=True,
        help="The primary Gmail mailbox to impersonate (e.g. team@solienne.ai)",
    )
    parser.add_argument(
        "--key-file",
        help="Path to the service-account JSON key (defaults to ../gcp/*.json)",
    )
    parser.add_argument(
        "--history-id",
        help="Optional startHistoryId to fetch gmail.users.history.list results",
    )

    args = parser.parse_args()

    key_path = resolve_key_path(args.key_file)
    logger.info(f"Using key file: {key_path}")

    gmail = build_gmail_client(key_path, args.user)

    profile = gmail.users().getProfile(userId="me").execute()
    logger.info("\n=== Gmail Profile ===")
    logger.info(json.dumps(profile, indent=2))

    if args.history_id:
        history = (
            gmail.users()
            .history()
            .list(userId="me", startHistoryId=args.history_id, maxResults=50)
            .execute()
        )
        logger.info(f"\n=== History Since startHistoryId {args.history_id} ===")
        logger.info(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
