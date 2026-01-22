#!/usr/bin/env python3
"""
Process account deletion requests from the accountdeletionrequests collection.

Dry-run is the default. Use --apply to perform updates.

Example:
  DB=PROD MONGO_DB_NAME=eden-prod python scripts/process_account_deletions.py
  DB=PROD MONGO_DB_NAME=eden-prod python scripts/process_account_deletions.py --apply
  DB=PROD MONGO_DB_NAME=eden-prod python scripts/process_account_deletions.py --request-id 69063db27eb71abf99b2c973
"""

import argparse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bson import ObjectId
from loguru import logger

from eve.auth import get_clerk
from eve.mongo import get_collection

ACCOUNT_DELETION_COLLECTION = "accountdeletionrequests"

# Reusable definition of user-owned collections to soft delete.
OWNABLES: List[Dict[str, Any]] = [
    {"name": "user", "collection": "users3", "filters": [{"_id": "$user_id"}]},
    {
        "name": "agents",
        "collection": "users3",
        "filters": [{"type": "agent", "owner": "$user_id"}],
    },
    {
        "name": "creations",
        "collection": "creations3",
        "filters": [{"user": "$user_id"}],
    },
    {
        "name": "tasks",
        "collection": "tasks3",
        "filters": [{"user": "$user_id"}, {"paying_user": "$user_id"}],
    },
    {"name": "models", "collection": "models3", "filters": [{"user": "$user_id"}]},
    {
        "name": "collections",
        "collection": "collections3",
        "filters": [{"user": "$user_id"}],
    },
    {"name": "sessions", "collection": "sessions", "filters": [{"owner": "$user_id"}]},
    {
        "name": "messages",
        "collection": "messages",
        "filters": [
            {"sender": "$user_id"},
            {"triggering_user": "$user_id"},
            {"billed_user": "$user_id"},
            {"agent_owner": "$user_id"},
        ],
    },
    {"name": "triggers", "collection": "triggers2", "filters": [{"user": "$user_id"}]},
    {
        "name": "deployments",
        "collection": "deployments2",
        "filters": [{"user": "$user_id"}],
    },
    {"name": "apikeys", "collection": "apikeys", "filters": [{"user": "$user_id"}]},
]


def _maybe_object_id(value: Optional[str]) -> Optional[ObjectId]:
    if not value:
        return None
    if isinstance(value, ObjectId):
        return value
    try:
        return ObjectId(str(value))
    except Exception:
        return None


def _materialize_filters(
    filters: List[Dict[str, Any]], user_id: ObjectId
) -> List[Dict[str, Any]]:
    materialized: List[Dict[str, Any]] = []
    for entry in filters:
        resolved = {}
        for key, value in entry.items():
            resolved[key] = user_id if value == "$user_id" else value
        materialized.append(resolved)
    return materialized


def _build_query(ownable: Dict[str, Any], user_id: ObjectId) -> Dict[str, Any]:
    filters = _materialize_filters(ownable["filters"], user_id)
    if not filters:
        return {}
    if len(filters) == 1:
        return filters[0]
    return {"$or": filters}


def _and_query(query: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    if not query:
        return extra
    return {"$and": [query, extra]}


def _resolve_user(request_doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    users = get_collection("users3")
    user_id = request_doc.get("user")
    user_doc = None
    if user_id:
        user_id = _maybe_object_id(user_id)
        if user_id:
            user_doc = users.find_one({"_id": user_id})
    if not user_doc and request_doc.get("userId"):
        user_doc = users.find_one({"userId": request_doc["userId"]})
    if not user_doc and request_doc.get("email"):
        user_doc = users.find_one({"email": request_doc["email"]})
    if not user_doc and request_doc.get("normalizedEmail"):
        user_doc = users.find_one({"normalizedEmail": request_doc["normalizedEmail"]})
    return user_doc


def _log_sample(collection, query: Dict[str, Any], limit: int = 3) -> None:
    sample = list(collection.find(query, {"_id": 1}).limit(limit))
    if not sample:
        return
    for doc in sample:
        logger.info(f"    sample _id={doc.get('_id')}")


def _soft_delete_collection(
    ownable: Dict[str, Any],
    user_id: ObjectId,
    dry_run: bool,
) -> Dict[str, int]:
    collection = get_collection(ownable["collection"])
    base_query = _build_query(ownable, user_id)
    update_query = _and_query(base_query, {"deleted": {"$ne": True}})

    total = collection.count_documents(base_query)
    pending = collection.count_documents(update_query)

    logger.info(
        f"[{ownable['name']}] collection={ownable['collection']} total={total} pending={pending}"
    )
    if dry_run and pending:
        _log_sample(collection, update_query)
        return {"total": total, "updated": 0}

    if pending:
        result = collection.update_many(
            update_query,
            {"$set": {"deleted": True}, "$currentDate": {"updatedAt": True}},
        )
        return {"total": total, "updated": result.modified_count}

    return {"total": total, "updated": 0}


def _delete_clerk_user(clerk_user_id: Optional[str], dry_run: bool, skip: bool) -> bool:
    if skip:
        logger.info("Skipping Clerk deletion (--skip-clerk)")
        return True
    if not clerk_user_id:
        logger.info("No Clerk userId on user record")
        return True
    if dry_run:
        logger.info(f"[DRY RUN] Would delete Clerk user {clerk_user_id}")
        return True
    try:
        clerk = get_clerk()
        clerk.users.delete(clerk_user_id)
        logger.info(f"Deleted Clerk user {clerk_user_id}")
        return True
    except Exception as exc:
        logger.error(f"Failed to delete Clerk user {clerk_user_id}: {exc}")
        return False


def _fetch_requests(args) -> List[Dict[str, Any]]:
    requests = get_collection(ACCOUNT_DELETION_COLLECTION)
    query: Dict[str, Any] = {}
    if args.request_id:
        request_oid = _maybe_object_id(args.request_id)
        query = {"_id": request_oid} if request_oid else {"_id": args.request_id}
    elif args.user_id:
        user_oid = _maybe_object_id(args.user_id)
        if user_oid:
            query = {"user": user_oid}
        else:
            query = {"userId": args.user_id}
    elif args.status:
        query = {"status": args.status}

    cursor = requests.find(query).sort("createdAt", 1)
    if args.limit:
        cursor = cursor.limit(args.limit)
    return list(cursor)


def process_requests(args) -> None:
    requests = _fetch_requests(args)
    if not requests:
        logger.info("No account deletion requests found for the given criteria.")
        return

    dry_run = not args.apply
    logger.info(f"Found {len(requests)} account deletion request(s). dry_run={dry_run}")

    for request_doc in requests:
        request_id = request_doc.get("_id")
        logger.info("-" * 72)
        logger.info(f"Processing request {request_id}")

        user_doc = _resolve_user(request_doc)
        if not user_doc:
            logger.error("No user found for request; skipping.")
            continue

        user_id = user_doc["_id"]
        logger.info(
            f"User _id={user_id} userId={user_doc.get('userId')} email={user_doc.get('email')}"
        )

        totals: Dict[str, int] = {}
        updated: Dict[str, int] = {}

        for ownable in OWNABLES:
            result = _soft_delete_collection(ownable, user_id, dry_run=dry_run)
            totals[ownable["name"]] = result["total"]
            updated[ownable["name"]] = result["updated"]

        clerk_ok = _delete_clerk_user(
            user_doc.get("userId"),
            dry_run=dry_run,
            skip=args.skip_clerk,
        )

        if dry_run:
            continue

        if not clerk_ok:
            logger.error(
                "Skipping request status update due to Clerk deletion failure."
            )
            continue

        requests_collection = get_collection(ACCOUNT_DELETION_COLLECTION)
        requests_collection.update_one(
            {"_id": request_id},
            {
                "$set": {
                    "status": args.status_update,
                    "processedAt": datetime.now(timezone.utc),
                    "results": {"totals": totals, "updated": updated},
                },
                "$currentDate": {"updatedAt": True},
            },
        )
        logger.info(f"Marked request {request_id} as {args.status_update}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process account deletion requests",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply updates (default is dry run).",
    )
    parser.add_argument(
        "--status",
        default="pending",
        help="Request status to process (default: pending).",
    )
    parser.add_argument(
        "--status-update",
        default="processed",
        help="Status to set after successful processing (default: processed).",
    )
    parser.add_argument("--limit", type=int, help="Limit number of requests.")
    parser.add_argument("--request-id", help="Process a specific request _id.")
    parser.add_argument(
        "--user-id",
        help="Process requests for a specific user ObjectId or userId.",
    )
    parser.add_argument(
        "--skip-clerk",
        action="store_true",
        help="Skip deleting the Clerk user.",
    )

    args = parser.parse_args()
    process_requests(args)


if __name__ == "__main__":
    main()
