"""
Dump all memories for a given agent to separate .txt files per scope.

Exports facts, reflections, and consolidated memories split by scope
(agent, user, session). Outputs files into memory2/scripts/dumps/<agent_id>/.

Usage:
    python -m eve.agent.memory2.scripts.dump_memories --agent-id <agent_id>
    DB=PROD python -m eve.agent.memory2.scripts.dump_memories --agent-id 67f8af96f2cc4291ee840cc5  
"""

import argparse
from collections import defaultdict
from pathlib import Path

from bson import ObjectId
from loguru import logger

from eve.agent.memory2.models import ConsolidatedMemory, Fact, Reflection

SCRIPT_DIR = Path(__file__).parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "dumps"


def _scope_key(doc: dict) -> str:
    return doc.get("scope", "unknown")


def _short_dt(dt) -> str:
    """Format datetime as compact 'YYYY-MM-DD HH:MM' or empty string."""
    if not dt:
        return ""
    try:
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(dt)[:16]


def _user_prefix(doc: dict, scope: str) -> str:
    """Return 'user: <id> | ' prefix for user-scoped entries, else ''."""
    if scope == "user":
        user_id = doc.get("user_id", "?")
        return f"user: {user_id} | "
    return ""


def _clean_content(text: str) -> str:
    """Collapse runs of 3+ newlines down to 2, strip trailing whitespace."""
    import re
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def dump_facts(agent_id: ObjectId, output_dir: Path) -> int:
    """Dump facts split by scope, sorted chronologically."""
    collection = Fact.get_collection()
    facts = list(collection.find({"agent_id": agent_id}).sort("formed_at", 1))

    by_scope = defaultdict(list)
    for fact in facts:
        by_scope[_scope_key(fact)].append(fact)

    for scope, items in by_scope.items():
        with open(output_dir / f"facts_{scope}.txt", "w") as f:
            for fact in items:
                ts = _short_dt(fact.get("formed_at"))
                up = _user_prefix(fact, scope)
                content = fact.get("content", "").strip()
                if not content:
                    continue
                f.write(f"[{ts}] {up}{content}\n")
                prev = fact.get("previous_content")
                if prev:
                    f.write(f"  [revised from: {prev}]\n")
                f.write("\n")

    return len(facts)


def dump_reflections(agent_id: ObjectId, output_dir: Path) -> int:
    """Dump reflections split by scope, sorted chronologically."""
    collection = Reflection.get_collection()
    reflections = list(collection.find({"agent_id": agent_id}).sort("formed_at", 1))

    by_scope = defaultdict(list)
    for ref in reflections:
        by_scope[_scope_key(ref)].append(ref)

    for scope, items in by_scope.items():
        with open(output_dir / f"reflections_{scope}.txt", "w") as f:
            for ref in items:
                ts = _short_dt(ref.get("formed_at"))
                up = _user_prefix(ref, scope)
                content = ref.get("content", "").strip()
                if not content:
                    continue
                absorbed = ref.get("absorbed", False)
                prefix = "[absorbed] " if absorbed else ""
                f.write(f"[{ts}] {up}{prefix}{content}\n\n")

    return len(reflections)


def dump_consolidated(agent_id: ObjectId, output_dir: Path) -> int:
    """Dump consolidated memories split by scope, sorted chronologically."""
    collection = ConsolidatedMemory.get_collection()
    consolidated = list(
        collection.find({"agent_id": agent_id}).sort("last_consolidated_at", 1)
    )

    by_scope = defaultdict(list)
    for cons in consolidated:
        by_scope[_scope_key(cons)].append(cons)

    for scope, items in by_scope.items():
        with open(output_dir / f"consolidated_{scope}.txt", "w") as f:
            for cons in items:
                ts = _short_dt(cons.get("last_consolidated_at"))
                up = _user_prefix(cons, scope)
                content = _clean_content(cons.get("consolidated_content", ""))
                if not content:
                    continue
                f.write(f"[{ts}] {up}\n{content}\n\n---\n\n")

    return len(consolidated)


def main():
    parser = argparse.ArgumentParser(
        description="Dump all memories for an agent to .txt files (split by scope)",
    )
    parser.add_argument("--agent-id", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    try:
        agent_id = ObjectId(args.agent_id)
    except Exception as e:
        logger.error(f"Invalid agent_id: {args.agent_id}. Error: {e}")
        return

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT / args.agent_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Dumping memories for agent {args.agent_id}")

    n_facts = dump_facts(agent_id, output_dir)
    logger.info(f"Facts: {n_facts}")

    n_reflections = dump_reflections(agent_id, output_dir)
    logger.info(f"Reflections: {n_reflections}")

    n_consolidated = dump_consolidated(agent_id, output_dir)
    logger.info(f"Consolidated: {n_consolidated}")

    logger.info(f"Files written to {output_dir}/")


if __name__ == "__main__":
    main()
