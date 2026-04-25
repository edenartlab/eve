"""
Add the indexes flagged by Atlas Performance Advisor to cut cluster cost.

Reads MONGO_URI from env. Targets both eden-prod and eden-stg explicitly,
so this is a one-off script — not part of the model ensure_indexes flow.

Each create_index is idempotent: if an equivalent index already exists,
pymongo raises OperationFailure with codeName=IndexOptionsConflict or
returns the existing name. We swallow the conflict and continue.

Run with the eden-prod-db-rw (or atlasAdmin) user — readAnyDatabase is not
enough for createIndex.
"""

import os
import sys
import time

from pymongo import MongoClient
from pymongo.errors import OperationFailure

# (db, collection, key_spec, name, options)
# key_spec is a list of (field, direction) tuples
INDEXES = [
    # ---- eden-prod.sessions ----
    # Slow query: find by session_key — currently scans 62K docs to return 1.
    ("eden-prod", "sessions", [("session_key", 1)], "session_key_1", {}),
    # Slow query #8 (1,251 occurrences): {deleted, $or:[{owner},{users}], platform, ...}
    # Two indexes — one rooted at owner, one at users — so the planner can use
    # an index-or-merge for the $or branches.
    (
        "eden-prod",
        "sessions",
        [
            ("users", 1),
            ("platform", 1),
            ("updatedAt", -1),
            ("createdAt", -1),
            ("deleted", 1),
        ],
        "users_platform_updatedAt_createdAt_deleted_idx",
        {},
    ),
    (
        "eden-prod",
        "sessions",
        [
            ("owner", 1),
            ("platform", 1),
            ("updatedAt", -1),
            ("createdAt", -1),
            ("deleted", 1),
        ],
        "owner_platform_updatedAt_createdAt_deleted_idx",
        {},
    ),
    # ---- eden-prod.messages ----
    # Slow query #7: {session, pinned} — 26K-doc scan, 0 returned, 34 occurrences.
    (
        "eden-prod",
        "messages",
        [("session", 1), ("pinned", 1)],
        "session_1_pinned_1",
        {},
    ),
    # ---- eden-prod.transactions ----
    # Slow query #5: {task, type} — 41K-doc scan, 0 returned, 5 occurrences.
    ("eden-prod", "transactions", [("task", 1), ("type", 1)], "task_1_type_1", {}),
    # ---- eden-prod.memory2_facts ----
    # Slow queries #10, #11: scope+formed_at filters not covered by existing
    # (agent_id, scope) — adding formed_at sort key avoids the in-memory sort.
    (
        "eden-prod",
        "memory2_facts",
        [("agent_id", 1), ("scope", 1), ("formed_at", -1)],
        "agent_id_scope_formed_at_idx",
        {},
    ),
    # ---- eden-stg.messages ----
    # Slow query #1 (worst inefficiency score, 105K-doc scan).
    # Sparse — channel may be absent on most messages.
    (
        "eden-stg",
        "messages",
        [("channel.key", 1), ("channel.type", 1)],
        "channel_key_type_idx",
        {"sparse": True},
    ),
]


def main() -> int:
    uri = os.environ.get("MONGO_URI")
    if not uri:
        print("ERROR: MONGO_URI not set", file=sys.stderr)
        return 1

    client = MongoClient(uri, serverSelectionTimeoutMS=10000, retryWrites=True)
    client.admin.command("ping")

    created = skipped = failed = 0
    for db_name, coll_name, keys, name, opts in INDEXES:
        coll = client[db_name][coll_name]

        existing_keys = {
            tuple(idx["key"].items()): idx["name"] for idx in coll.list_indexes()
        }
        if tuple(keys) in existing_keys:
            print(
                f"  ⊙ {db_name}.{coll_name}: {name} — equivalent index "
                f"'{existing_keys[tuple(keys)]}' already exists"
            )
            skipped += 1
            continue

        t0 = time.time()
        print(f"  → {db_name}.{coll_name}: creating {name} {keys} ...", flush=True)
        try:
            coll.create_index(keys, name=name, background=True, **opts)
            print(f"  ✓ {db_name}.{coll_name}: {name} created in {time.time()-t0:.1f}s")
            created += 1
        except OperationFailure as e:
            if e.code == 85 or "IndexOptionsConflict" in str(e):
                print(
                    f"  ⊙ {db_name}.{coll_name}: {name} — conflict (already exists "
                    f"with different options): {e}"
                )
                skipped += 1
            else:
                print(f"  ✗ {db_name}.{coll_name}: {name} FAILED: {e}")
                failed += 1

    print(f"\nSummary: {created} created, {skipped} skipped, {failed} failed")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
