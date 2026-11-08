"""
Disqualify Abraham sessions from the Covenant, or reinstate them.

Disqualification is the `abraham_disqualified` flag on the session document.
abraham_daily and abraham_remember skip flagged sessions, and abraham_covenant
refuses to publish from one. No deploy is needed.

    DB=PROD python scripts/abraham_disqualify_session.py <session_id> [...]
    DB=PROD python scripts/abraham_disqualify_session.py --undo <session_id>
    DB=PROD python scripts/abraham_disqualify_session.py --dry-run <session_id>

DB defaults to STAGE, as everywhere in eve.
"""

import argparse
import sys

from bson import ObjectId

from eve import db
from eve.agent.session.models import Session
from eve.tools.abraham.guardrails import DISQUALIFIED_FIELD


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("session_ids", nargs="+")
    parser.add_argument(
        "--undo", action="store_true", help="reinstate instead of disqualify"
    )
    parser.add_argument("--dry-run", action="store_true", help="show, don't write")
    args = parser.parse_args()

    try:
        ids = [ObjectId(s) for s in args.session_ids]
    except Exception as e:
        print(f"Invalid session id: {e}", file=sys.stderr)
        return 2

    collection = Session.get_collection()
    print(f"DB={db} collection={collection.full_name}")

    found = {
        d["_id"]: d
        for d in collection.find(
            {"_id": {"$in": ids}}, {"title": 1, DISQUALIFIED_FIELD: 1}
        )
    }
    missing = [i for i in ids if i not in found]
    if missing:
        print(f"Not found, nothing written: {', '.join(map(str, missing))}")
        return 1

    for i in ids:
        d = found[i]
        print(
            f"  {i}  {DISQUALIFIED_FIELD}={d.get(DISQUALIFIED_FIELD, False)}  {d.get('title')!r}"
        )

    if args.dry_run:
        return 0

    update = (
        {"$unset": {DISQUALIFIED_FIELD: ""}}
        if args.undo
        else {"$set": {DISQUALIFIED_FIELD: True}}
    )
    result = collection.update_many({"_id": {"$in": ids}}, update)
    verb = "Reinstated" if args.undo else "Disqualified"
    print(f"{verb}: matched {result.matched_count}, modified {result.modified_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
