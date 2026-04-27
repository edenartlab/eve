"""
Disable creations3 vector search to free Atlas Search RAM.

Three steps, run in order:
  1. Drop the `img_vec_idx` Atlas Vector Search index on creations3
     (was 4.92 GB on prod, dominating M20's 4 GB RAM budget).
  2. $unset the `embedding` field from all creations3 documents to reclaim
     disk + working-set memory. Done in batches.
  3. Same on eden-stg (smaller index, ~83 MB; included so the schema is
     consistent across environments).

Pre-req: code in eve and eden1 has the VECTOR_SEARCH_DISABLED guards in
place (otherwise dropping the index will start producing 500s for
in-flight $vectorSearch queries).

Run with a writable Mongo URI (atlasAdmin or equivalent).

To restore the feature later:
  - Re-create the Atlas Vector Search index in the Atlas UI:
      db: eden-prod, collection: creations3, name: img_vec_idx,
      type: vectorSearch, definition matching the original.
  - Set env VECTOR_SEARCH_ENABLED=true on the API and Modal services.
  - Run the embed_recent_creations cron to backfill embeddings.
"""

import os
import sys
import time

from pymongo import MongoClient
from pymongo.errors import OperationFailure

TARGETS = [
    ("eden-prod", "creations3", "img_vec_idx"),
    ("eden-stg", "creations3", "img_vec_idx"),
]

UNSET_BATCH_SIZE = 1000


def drop_search_index(
    client: MongoClient, db_name: str, coll_name: str, idx_name: str
) -> None:
    coll = client[db_name][coll_name]
    try:
        existing = list(coll.aggregate([{"$listSearchIndexes": {}}]))
    except OperationFailure as e:
        print(f"  ✗ {db_name}.{coll_name}: cannot list search indexes ({e}); skipping")
        return

    matching = [i for i in existing if i.get("name") == idx_name]
    if not matching:
        print(
            f"  ⊙ {db_name}.{coll_name}: search index '{idx_name}' not found (already dropped?)"
        )
        return

    print(
        f"  → {db_name}.{coll_name}: dropping search index '{idx_name}' "
        f"(type={matching[0].get('type','?')}, status={matching[0].get('status','?')}) ..."
    )
    coll.drop_search_index(idx_name)
    print(
        f"  ✓ {db_name}.{coll_name}: drop_search_index('{idx_name}') accepted (async)"
    )


def unset_embeddings(client: MongoClient, db_name: str, coll_name: str) -> None:
    coll = client[db_name][coll_name]
    total = coll.count_documents({"embedding": {"$exists": True}})
    if total == 0:
        print(f"  ⊙ {db_name}.{coll_name}: no documents with embedding field")
        return

    print(
        f"  → {db_name}.{coll_name}: {total:,} docs have an `embedding` field; "
        f"unsetting in batches of {UNSET_BATCH_SIZE} ..."
    )
    done = 0
    t0 = time.time()
    while True:
        ids = [
            d["_id"]
            for d in coll.find({"embedding": {"$exists": True}}, {"_id": 1}).limit(
                UNSET_BATCH_SIZE
            )
        ]
        if not ids:
            break
        result = coll.update_many({"_id": {"$in": ids}}, {"$unset": {"embedding": ""}})
        done += result.modified_count
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        print(f"     {done:,}/{total:,}  ({rate:.0f}/s)")
    print(f"  ✓ {db_name}.{coll_name}: unset complete in {time.time()-t0:.1f}s")


def main() -> int:
    uri = os.environ.get("MONGO_URI")
    if not uri:
        print("ERROR: MONGO_URI not set", file=sys.stderr)
        return 1

    do_drop = os.environ.get("DROP_INDEX", "true").lower() == "true"
    do_unset = os.environ.get("UNSET_EMBEDDING", "true").lower() == "true"

    client = MongoClient(uri, serverSelectionTimeoutMS=10000, retryWrites=True)
    client.admin.command("ping")

    if do_drop:
        print("\n=== Step 1: drop Atlas Vector Search index ===")
        for db_name, coll_name, idx_name in TARGETS:
            drop_search_index(client, db_name, coll_name, idx_name)

    if do_unset:
        print("\n=== Step 2: $unset `embedding` field from all docs ===")
        for db_name, coll_name, _ in TARGETS:
            unset_embeddings(client, db_name, coll_name)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
