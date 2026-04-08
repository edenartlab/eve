#!/usr/bin/env python3
"""
Migrate legacy creation reactions (praise) into likes3.

Usage:
  ENV_PATH=.env.DUMMY python scripts/migrate_reactions_to_likes.py --dry-run
  ENV_PATH=.env.DUMMY python scripts/migrate_reactions_to_likes.py --apply
"""

import argparse
import os
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne


def load_env() -> None:
    env_path = os.getenv("ENV_PATH")
    if env_path and os.path.exists(env_path):
        load_dotenv(env_path, override=True)
    else:
        load_dotenv(override=True)


def as_object_id(value: Any) -> Optional[ObjectId]:
    if isinstance(value, ObjectId):
        return value
    if isinstance(value, str) and ObjectId.is_valid(value):
        return ObjectId(value)
    return None


def log(message: str, data: Optional[Dict[str, Any]] = None) -> None:
    if data:
        print(f"{message} {data}")
    else:
        print(message)


def batched(
    iterable: Iterable[Dict[str, Any]], size: int
) -> Iterable[List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate reactions to likes3")
    parser.add_argument("--apply", action="store_true", help="Apply writes to likes3")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (default)")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dry_run = not args.apply
    if args.dry_run:
        dry_run = True

    load_env()

    mongo_uri = os.getenv("MONGO_URI") or os.getenv("MONGODB_URI")
    mongo_db_name = os.getenv("MONGO_DB_NAME")
    if not mongo_uri or not mongo_db_name:
        raise SystemExit("MONGO_URI/MONGODB_URI and MONGO_DB_NAME are required")

    client = MongoClient(mongo_uri)
    db = client[mongo_db_name]
    reactions = db.reactions
    creations = db.creations3
    likes = db.likes3

    log("Reactions -> Likes3 migration", {"dry_run": dry_run})

    total_praise = reactions.count_documents({"reaction": "praise"})
    log("Found praise reactions", {"total": total_praise})

    processed = 0
    skipped_invalid_id = 0
    skipped_missing_user = 0
    skipped_missing_creation = 0
    skipped_existing_like = 0
    inserted = 0
    affected_creation_ids: Set[str] = set()

    cursor = reactions.find(
        {"reaction": "praise"},
        {"user": 1, "creation": 1, "reaction": 1, "createdAt": 1, "updatedAt": 1},
    ).batch_size(args.batch_size)

    if args.limit:
        cursor = cursor.limit(args.limit)

    for batch in batched(cursor, args.batch_size):
        user_ids: List[ObjectId] = []
        creation_ids: List[ObjectId] = []
        normalized: List[Tuple[ObjectId, ObjectId, Dict[str, Any]]] = []

        for doc in batch:
            processed += 1
            user_id = as_object_id(doc.get("user"))
            creation_id = as_object_id(doc.get("creation"))

            if not user_id or not creation_id:
                skipped_invalid_id += 1
                continue

            normalized.append((user_id, creation_id, doc))
            user_ids.append(user_id)
            creation_ids.append(creation_id)

        if not normalized:
            continue

        existing_creations = creations.find({"_id": {"$in": creation_ids}}, {"_id": 1})
        existing_creation_set = {c["_id"].__str__() for c in existing_creations}

        existing_likes = likes.find(
            {
                "entityType": "creation",
                "entityId": {"$in": creation_ids},
                "user": {"$in": user_ids},
            },
            {"user": 1, "entityId": 1},
        )
        existing_like_set = {
            f"{like['user']!s}:{like['entityId']!s}" for like in existing_likes
        }

        ops: List[UpdateOne] = []
        for user_id, creation_id, doc in normalized:
            if str(creation_id) not in existing_creation_set:
                skipped_missing_creation += 1
                continue
            if not user_id:
                skipped_missing_user += 1
                continue

            affected_creation_ids.add(str(creation_id))
            key = f"{user_id!s}:{creation_id!s}"
            if key in existing_like_set:
                skipped_existing_like += 1
                continue

            ops.append(
                UpdateOne(
                    {
                        "user": user_id,
                        "entityType": "creation",
                        "entityId": creation_id,
                    },
                    {
                        "$setOnInsert": {
                            "user": user_id,
                            "entityType": "creation",
                            "entityId": creation_id,
                            "createdAt": doc.get("createdAt"),
                            "updatedAt": doc.get("updatedAt") or doc.get("createdAt"),
                        }
                    },
                    upsert=True,
                )
            )

        if not ops:
            continue

        if dry_run:
            inserted += len(ops)
        else:
            result = likes.bulk_write(ops, ordered=False)
            inserted += result.upserted_count or 0

    like_count_updates = 0
    if not dry_run and affected_creation_ids:
        affected_ids = [ObjectId(cid) for cid in affected_creation_ids]
        counts = list(
            likes.aggregate(
                [
                    {
                        "$match": {
                            "entityType": "creation",
                            "entityId": {"$in": affected_ids},
                        }
                    },
                    {"$group": {"_id": "$entityId", "count": {"$sum": 1}}},
                ]
            )
        )
        count_map = {str(item["_id"]): item["count"] for item in counts}
        updates = [
            UpdateOne(
                {"_id": cid},
                {"$set": {"likeCount": count_map.get(str(cid), 0)}},
            )
            for cid in affected_ids
        ]
        if updates:
            update_result = creations.bulk_write(updates, ordered=False)
            like_count_updates = update_result.modified_count or 0

    log(
        "Migration summary",
        {
            "total_praise": total_praise,
            "processed": processed,
            "skipped_invalid_id": skipped_invalid_id,
            "skipped_missing_user": skipped_missing_user,
            "skipped_missing_creation": skipped_missing_creation,
            "skipped_existing_like": skipped_existing_like,
            "would_insert" if dry_run else "inserted": inserted,
            "affected_creations": len(affected_creation_ids),
            "like_count_updates": like_count_updates,
        },
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
