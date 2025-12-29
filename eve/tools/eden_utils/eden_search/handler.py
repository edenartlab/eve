"""
Eden Search Tool - Universal search across Eden database collections.

Supports searching: creations, collections, models, concepts, agents
with fuzzy text matching, privacy controls, and agent username lookup.
"""

import re
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import List, Optional

from bson import ObjectId

from eve.mongo import get_collection
from eve.tool import ToolContext

# ============================================================
# CONFIGURATION
# ============================================================

FUZZY_THRESHOLD = 0.7  # Balanced threshold for typo tolerance vs accuracy

COLLECTION_CONFIG = {
    "creations": {
        "collection_name": "creations3",
        "owner_field": "user",
        "public_field": "public",
        "search_fields": ["name"],
        "allowed_filters": ["agent", "tool"],
        "extra_base_filter": {},
        "output_fields": [
            "_id",
            "name",
            "tool",
            "filename",
            "mediaAttributes",
            "createdAt",
            "user",
            "agent",
        ],
    },
    "collections": {
        "collection_name": "collections3",
        "owner_field": "user",
        "public_field": "public",
        "search_fields": ["name", "description"],
        "allowed_filters": [],
        "extra_base_filter": {},
        "output_fields": [
            "_id",
            "name",
            "description",
            "coverCreation",
            "createdAt",
            "user",
        ],
    },
    "models": {
        "collection_name": "models3",
        "owner_field": "user",
        "public_field": "public",
        "search_fields": ["name", "lora_trigger_text"],
        "allowed_filters": ["agent", "base_model"],
        "extra_base_filter": {},
        "output_fields": [
            "_id",
            "name",
            "base_model",
            "lora_trigger_text",
            "thumbnail",
            "creationCount",
            "createdAt",
            "user",
            "agent",
        ],
    },
    "concepts": {
        "collection_name": "concepts2",
        "owner_field": "user",
        "public_field": None,  # Owner-only access - no public field
        "search_fields": ["name", "usage_instructions"],
        "allowed_filters": ["agent"],
        "extra_base_filter": {},
        "output_fields": [
            "_id",
            "name",
            "usage_instructions",
            "thumbnail",
            "creationCount",
            "createdAt",
            "user",
            "agent",
        ],
    },
    "agents": {
        "collection_name": "users3",
        "owner_field": "owner",  # Note: different from others!
        "public_field": "public",
        "search_fields": ["name", "username", "description"],
        "allowed_filters": [],
        "extra_base_filter": {"type": "agent"},
        "output_fields": [
            "_id",
            "username",
            "name",
            "description",
            "userImage",
            "createdAt",
            "owner",
        ],
    },
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def parse_date_filter(value: str) -> Optional[datetime]:
    """
    Parse date filter value.
    Supports:
    - Relative hours: "24h", "2h", "168h"
    - ISO datetime: "2024-01-01T00:00:00Z", "2024-01-01"
    """
    if not value:
        return None

    value = value.strip()

    # Check for hours format (e.g., "24h", "2h")
    hours_match = re.match(r"^(\d+)h$", value.lower())
    if hours_match:
        hours = int(hours_match.group(1))
        return datetime.now(timezone.utc) - timedelta(hours=hours)

    # Try parsing as ISO datetime
    for fmt in [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ]:
        try:
            dt = datetime.strptime(value, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    return None


def resolve_agent_filter(value: str) -> Optional[ObjectId]:
    """
    Resolve agent filter value to ObjectId.

    1. Try parsing as ObjectId directly
    2. Try exact case-insensitive username match
    3. Fall back to fuzzy username matching (0.7 threshold)
    """
    if not value:
        return None

    # Try parsing as ObjectId first
    try:
        return ObjectId(value)
    except Exception:
        pass

    # Search by username (case-insensitive)
    collection = get_collection("users3")

    # Try exact case-insensitive match first
    agent = collection.find_one(
        {
            "type": "agent",
            "username": {"$regex": f"^{re.escape(value)}$", "$options": "i"},
            "deleted": {"$ne": True},
        }
    )

    if agent:
        return agent["_id"]

    # Fall back to fuzzy matching
    agents = list(
        collection.find(
            {"type": "agent", "deleted": {"$ne": True}}, {"_id": 1, "username": 1}
        )
    )

    best_match = None
    best_ratio = 0
    query_lower = value.lower()

    for agent in agents:
        username = agent.get("username", "").lower()
        ratio = SequenceMatcher(None, query_lower, username).ratio()
        if ratio > best_ratio and ratio >= FUZZY_THRESHOLD:
            best_ratio = ratio
            best_match = agent["_id"]

    return best_match


def fuzzy_score(query: str, text: str) -> float:
    """
    Calculate fuzzy match score between query and text.

    Returns a score between 0 and 1 where:
    - 1.0 = exact substring match
    - 0.7+ = good fuzzy match
    - < 0.7 = poor match
    """
    if not query or not text:
        return 0.0

    query_lower = query.lower()
    text_lower = text.lower()

    # Exact substring match gets highest score
    if query_lower in text_lower:
        return 1.0

    # Word-level matching
    query_words = query_lower.split()
    text_words = text_lower.split()

    if not query_words:
        return 0.0

    # Score based on how many query words fuzzy-match text words
    matched_words = 0
    for qw in query_words:
        for tw in text_words:
            if SequenceMatcher(None, qw, tw).ratio() >= FUZZY_THRESHOLD:
                matched_words += 1
                break

    word_score = matched_words / len(query_words)

    # Overall sequence similarity
    seq_score = SequenceMatcher(None, query_lower, text_lower).ratio()

    # Combine scores with weighting
    return max(word_score * 0.7 + seq_score * 0.3, seq_score)


def calculate_document_score(
    doc: dict, query: str, search_fields: List[str], is_owned: bool
) -> float:
    """
    Calculate relevance score for a document.

    Owned documents get a boost to prioritize user's own items.
    """
    base_score = 0.0

    if query:
        for field in search_fields:
            text = doc.get(field, "")
            if text:
                field_score = fuzzy_score(query, str(text))
                base_score = max(base_score, field_score)
    else:
        base_score = 1.0  # No query means all docs are equally relevant

    # Boost owned documents to prioritize them
    if is_owned:
        base_score += 0.5

    return base_score


def sanitize_filter(
    filter_dict: Optional[dict], allowed_filters: List[str], user_id: Optional[ObjectId]
) -> dict:
    """
    Sanitize and validate user-provided filter.

    Security measures:
    - Only allow keys in allowed_filters list
    - Convert agent username to ObjectId
    - Reject user filter unless it's for self
    - Strip MongoDB operators from values
    - Reject nested objects/arrays
    """
    if not filter_dict:
        return {}

    sanitized = {}

    for key, value in filter_dict.items():
        # Skip disallowed keys (except 'user' which has special handling)
        if key not in allowed_filters and key != "user":
            continue

        # Handle user filter - only allow filtering by self
        if key == "user":
            if not user_id:
                continue  # Can't filter by user without user context
            if isinstance(value, str):
                try:
                    filter_user_id = ObjectId(value)
                    if filter_user_id != user_id:
                        continue  # Silently skip - can't filter by other users
                    sanitized["user"] = filter_user_id
                except Exception:
                    continue
            continue

        # Handle agent filter - resolve username/ID to ObjectId
        if key == "agent":
            if isinstance(value, str):
                agent_id = resolve_agent_filter(value)
                if agent_id:
                    sanitized["agent"] = agent_id
            continue

        # Handle other allowed filters - only simple values for security
        if isinstance(value, str):
            sanitized[key] = value
        elif isinstance(value, (int, float, bool)):
            sanitized[key] = value
        # Skip complex values (dicts with operators, arrays) for security

    return sanitized


def build_privacy_filter(
    config: dict, user_id: Optional[ObjectId], include_own_only: bool
) -> dict:
    """
    Build MongoDB filter that enforces privacy rules.

    Rules:
    - Always exclude deleted items
    - If no user_id: only return public items
    - If include_own_only: only return owned items
    - Otherwise: return owned OR public items
    - For Concepts (no public field): owner-only access (requires user)
    """
    owner_field = config["owner_field"]
    public_field = config["public_field"]

    base_filter = {"deleted": {"$ne": True}}
    base_filter.update(config.get("extra_base_filter", {}))

    if not user_id:
        # No user context - only return public items
        if public_field:
            base_filter[public_field] = True
        else:
            # No public field (Concepts) - can't access without user
            base_filter["_id"] = None  # Will match nothing
        return base_filter

    if include_own_only:
        # Only return user's own items
        base_filter[owner_field] = user_id
    else:
        if public_field:
            # Return: (owned) OR (public)
            base_filter["$or"] = [{owner_field: user_id}, {public_field: True}]
        else:
            # No public field (Concepts) - owner-only access
            base_filter[owner_field] = user_id

    return base_filter


def format_output(docs: List[dict], config: dict) -> List[dict]:
    """
    Format documents for output based on collection type.

    Converts ObjectIds and datetimes to strings.
    Resolves user/agent/owner IDs to usernames.
    """
    output_fields = config["output_fields"]

    # Collect all user/agent IDs that need username lookup
    user_ids_to_lookup = set()
    for doc in docs:
        for field in ["user", "agent", "owner"]:
            if field in output_fields and field in doc and doc[field]:
                user_ids_to_lookup.add(doc[field])

    # Batch lookup usernames
    username_map = {}
    if user_ids_to_lookup:
        users_collection = get_collection("users3")
        users = users_collection.find(
            {"_id": {"$in": list(user_ids_to_lookup)}}, {"_id": 1, "username": 1}
        )
        for u in users:
            username_map[u["_id"]] = u.get("username", str(u["_id"]))

    formatted = []
    for doc in docs:
        item = {}
        for field in output_fields:
            if field == "_id":
                item["id"] = str(doc.get("_id", ""))
            elif field in doc:
                value = doc[field]
                # Resolve user/agent/owner to username
                if field in ["user", "agent", "owner"] and isinstance(value, ObjectId):
                    item[field] = username_map.get(value, str(value))
                elif isinstance(value, ObjectId):
                    item[field] = str(value)
                elif isinstance(value, datetime):
                    item[field] = value.isoformat()
                else:
                    item[field] = value
        formatted.append(item)

    return formatted


# ============================================================
# MAIN HANDLER
# ============================================================


async def handler(context: ToolContext):
    """
    Main handler for eden_search tool.

    Searches across Eden database collections with:
    - Fuzzy text matching
    - Privacy controls (no private items from others)
    - Agent username/ID lookup
    - Date range filtering
    - Prioritization of user-owned items
    """
    # Extract arguments
    collection_type = context.args.get("collection")
    query = context.args.get("query", "")
    filter_arg = context.args.get("filter", {})
    created_after = context.args.get("created_after")
    created_before = context.args.get("created_before")
    limit = context.args.get("limit", 20)
    sort_by = context.args.get("sort_by", "createdAt")
    sort_order = context.args.get("sort_order", "desc")
    include_own_only = context.args.get("include_own_only", False)

    # Get user ID from context (optional - if None, only public items returned)
    user_id = ObjectId(context.user) if context.user else None

    # Validate include_own_only requires user context
    if include_own_only and not user_id:
        return {
            "output": {
                "error": "include_own_only=True requires user context. Pass user_id in args.",
                "results": [],
            }
        }

    # Validate collection type
    if collection_type not in COLLECTION_CONFIG:
        return {
            "output": {
                "error": f"Invalid collection: {collection_type}. Must be one of: {list(COLLECTION_CONFIG.keys())}",
                "results": [],
            }
        }

    config = COLLECTION_CONFIG[collection_type]
    collection = get_collection(config["collection_name"])

    # Build base query with privacy filters (hardcoded security)
    mongo_filter = build_privacy_filter(config, user_id, include_own_only)

    # Apply sanitized user filters
    sanitized_filter = sanitize_filter(filter_arg, config["allowed_filters"], user_id)
    mongo_filter.update(sanitized_filter)

    # Apply date filters
    if created_after or created_before:
        date_filter = {}
        if created_after:
            after_dt = parse_date_filter(created_after)
            if after_dt:
                date_filter["$gte"] = after_dt
        if created_before:
            before_dt = parse_date_filter(created_before)
            if before_dt:
                date_filter["$lte"] = before_dt
        if date_filter:
            mongo_filter["createdAt"] = date_filter

    # Fetch documents
    sort_direction = -1 if sort_order == "desc" else 1

    if query:
        # Use regex to pre-filter documents containing query words
        # This ensures we find matches regardless of sort order
        query_words = query.lower().split()
        search_fields = config["search_fields"]

        # Build regex pattern for each word (case-insensitive partial match)
        regex_conditions = []
        for word in query_words:
            if len(word) >= 2:  # Skip very short words
                word_pattern = re.escape(word)
                field_conditions = [
                    {field: {"$regex": word_pattern, "$options": "i"}}
                    for field in search_fields
                ]
                regex_conditions.append({"$or": field_conditions})

        # Add regex filter to query (match any word in any field)
        if regex_conditions:
            if "$and" in mongo_filter:
                mongo_filter["$and"].extend(regex_conditions)
            else:
                # We need to restructure the filter to include both privacy and regex
                existing_filter = mongo_filter.copy()
                # Remove $or if present (privacy filter) and handle separately
                privacy_or = existing_filter.pop("$or", None)

                if privacy_or:
                    # Combine: (privacy $or) AND (any regex word matches)
                    mongo_filter = {
                        "$and": [
                            {"$or": privacy_or},
                            {"$or": regex_conditions[0]["$or"]}
                            if len(regex_conditions) == 1
                            else {
                                "$or": [
                                    rc["$or"][0] for rc in regex_conditions for _ in [1]
                                ]
                            },
                        ]
                    }
                    mongo_filter["$and"][0].update(existing_filter)
                else:
                    # Simpler case: just add regex conditions
                    mongo_filter = existing_filter
                    if len(regex_conditions) == 1:
                        mongo_filter["$or"] = regex_conditions[0]["$or"]
                    else:
                        # Match any word in any field
                        all_field_conditions = []
                        for rc in regex_conditions:
                            all_field_conditions.extend(rc["$or"])
                        mongo_filter["$or"] = all_field_conditions

        # Fetch more candidates for fuzzy scoring
        fetch_limit = limit * 10
        cursor = collection.find(mongo_filter).limit(fetch_limit)
        docs = list(cursor)

        # Score and sort by fuzzy match quality
        scored_docs = []
        for doc in docs:
            is_owned = doc.get(config["owner_field"]) == user_id
            score = calculate_document_score(
                doc, query, config["search_fields"], is_owned
            )
            scored_docs.append((score, doc))

        # Sort by score descending (owned items get boosted)
        scored_docs.sort(key=lambda x: -x[0])
        docs = [doc for _, doc in scored_docs[:limit]]
    else:
        # No query - just fetch with sort, prioritizing owned items
        cursor = (
            collection.find(mongo_filter).sort(sort_by, sort_direction).limit(limit * 2)
        )
        docs = list(cursor)
        # Re-sort to prioritize owned items
        owned = [d for d in docs if d.get(config["owner_field"]) == user_id]
        public = [d for d in docs if d.get(config["owner_field"]) != user_id]
        docs = (owned + public)[:limit]

    # Format output
    results = format_output(docs, config)

    return {
        "output": {
            "collection": collection_type,
            "count": len(results),
            "results": results,
        }
    }
