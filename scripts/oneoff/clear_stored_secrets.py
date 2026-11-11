"""One-off: remove stored third-party credentials from the database.

Clears, in the database named by --db:

  deployments2         secrets -> null, encrypted flag removed, valid -> false,
                       config.discord.channel_configs[].webhook_token -> null
  discord_connections  access_token, refresh_token -> ""
  guild_webhooks       webhook_token, webhook_url -> ""

Fields are blanked rather than removed where the models declare them as
required, so existing documents still load.

DRY RUN BY DEFAULT: without --apply this only counts what would change.
It never prints a secret value, only counts.

Usage (connection string is read from the MONGO_URI environment variable):

    python scripts/oneoff/clear_stored_secrets.py --db <database>            # dry run
    python scripts/oneoff/clear_stored_secrets.py --db <database> --apply

Take a database snapshot before running with --apply. The owners of the
cleared credentials still have to revoke them at the issuing platform.
"""

import argparse
import os
import sys

from pymongo import MongoClient

HAS_VALUE = {"$exists": True, "$nin": [None, ""]}

# (label, collection, filter matching documents that still hold a credential, update)
STEPS = [
    (
        "deployments2: stored secrets",
        "deployments2",
        {"secrets": {"$exists": True, "$ne": None}},
        {"$set": {"secrets": None, "valid": False}, "$unset": {"encrypted": ""}},
        None,
    ),
    (
        "deployments2: per-channel Discord webhook tokens",
        "deployments2",
        {
            "config.discord.channel_configs": {
                "$elemMatch": {"webhook_token": HAS_VALUE}
            }
        },
        {"$set": {"config.discord.channel_configs.$[c].webhook_token": None}},
        [{"c.webhook_token": HAS_VALUE}],
    ),
    (
        "discord_connections: OAuth tokens",
        "discord_connections",
        {"$or": [{"access_token": HAS_VALUE}, {"refresh_token": HAS_VALUE}]},
        {"$set": {"access_token": "", "refresh_token": ""}},
        None,
    ),
    (
        "guild_webhooks: webhook credentials",
        "guild_webhooks",
        {"$or": [{"webhook_token": HAS_VALUE}, {"webhook_url": HAS_VALUE}]},
        {"$set": {"webhook_token": "", "webhook_url": ""}},
        None,
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", required=True, help="database name (no default)")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually modify documents (default: dry run, counts only)",
    )
    args = parser.parse_args()

    uri = os.environ.get("MONGO_URI")
    if not uri:
        print("MONGO_URI is not set", file=sys.stderr)
        return 2

    db = MongoClient(uri)[args.db]
    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"[{mode}] database: {args.db}")

    remaining_total = 0
    for label, collection, query, update, array_filters in STEPS:
        coll = db[collection]
        total = coll.count_documents({})
        before = coll.count_documents(query)
        print(f"- {label}: {before} of {total} documents hold a value")
        if not args.apply:
            continue
        kwargs = {"array_filters": array_filters} if array_filters else {}
        result = coll.update_many(query, update, **kwargs)
        after = coll.count_documents(query)
        remaining_total += after
        print(f"  modified {result.modified_count}; {after} still hold a value")

    if not args.apply:
        print("Dry run only. Re-run with --apply to clear these values.")
        return 0
    if remaining_total:
        print(
            f"WARNING: {remaining_total} documents still hold a value", file=sys.stderr
        )
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
