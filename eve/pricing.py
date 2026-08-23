"""Single source of truth for tool pricing.

A tool's price is a ``cost_estimate`` expression (see ``eve.utils.cost_utils``).
It is authored in ``api.yaml`` in a repo and *projected* into the ``tools3``
collection in Mongo, which is what production reads at runtime.

This module is the one place that knows how to answer:

  * what does the repo say a tool costs?
  * what does the database say a tool costs?
  * do they agree?

Three properties of the real system shape this code, and all three are
load-bearing:

1. **Prices are inherited.** A tool with ``parent_tool`` and no
   ``cost_estimate`` of its own inherits its parent's price. ``Tool.from_yaml``
   resolves that inheritance by reading the parent *out of Mongo*
   (``Tool._get_schema(parent, from_yaml=False)``), which makes "what the repo
   says" depend on the database and on the order tools happen to be written in.
   For pricing we resolve inheritance **inside the YAML corpus only**, so the
   repo answer is a pure function of the repo.

2. **Not every tool has a YAML file.** Some tools exist only in ``tools3``
   (retired ComfyUI workflows, tools whose source has been deleted). The repo
   is authoritative for the tools it *can see*; it is never authoritative about
   deletion. Legitimately DB-only tools are enumerated in
   ``eve/db_only_tools.yaml`` so that a *new* one -- i.e. someone hand-editing
   Mongo -- is still caught.

3. **Tools live in three repos.** ``eve``, ``workflows`` and
   ``private_workflows``. CI does not always have all three checked out. A repo
   that is absent is reported as UNCHECKED; its tools are never reported as
   drifted, unpriced, or deleted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml

from eve.utils.cost_utils import eval_cost

# Repos that can contain tool definitions, in the same order and at the same
# locations that eve.tool.get_api_files() searches.
TOOL_REPOS = ("eve", "workflows", "private_workflows")

DB_ONLY_REGISTRY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db_only_tools.yaml")


@dataclass
class YamlPrice:
    """A price as authored in the repo."""

    key: str
    cost_estimate: Optional[str]
    path: str
    repo: str
    # Set when cost_estimate came from a parent tool rather than this file.
    inherited_from: Optional[str] = None

    @property
    def source(self) -> str:
        if self.inherited_from:
            return f"{self.path} (inherited from {self.inherited_from})"
        return self.path


@dataclass
class DriftReport:
    """The result of comparing repo prices against database prices."""

    # Hard failures.
    drifted: List[Tuple[str, Optional[str], Optional[str]]] = field(default_factory=list)
    invalid: List[Tuple[str, str, str]] = field(default_factory=list)
    unregistered_db_only: List[Tuple[str, Optional[str]]] = field(default_factory=list)
    unresolved_parent: List[Tuple[str, str]] = field(default_factory=list)
    collisions: List[Tuple[str, str, str]] = field(default_factory=list)

    # Informational.
    not_yet_in_db: List[str] = field(default_factory=list)
    registered_db_only: List[str] = field(default_factory=list)
    db_only_untracked_info: List[str] = field(default_factory=list)
    unchecked_repos: List[str] = field(default_factory=list)
    checked_repos: List[str] = field(default_factory=list)
    compared: int = 0

    @property
    def ok(self) -> bool:
        return not (
            self.drifted
            or self.invalid
            or self.unregistered_db_only
            or self.unresolved_parent
            or self.collisions
        )


def repo_roots(eve_root: Optional[str] = None) -> Dict[str, str]:
    """Map repo name -> filesystem root, matching get_api_files()'s layout."""
    eve_root = eve_root or os.path.dirname(os.path.abspath(__file__))
    return {
        "eve": os.path.join(eve_root, "tools"),
        "workflows": os.path.join(eve_root, "..", "..", "workflows"),
        "private_workflows": os.path.join(eve_root, "..", "..", "private_workflows"),
    }


def discover_repos(eve_root: Optional[str] = None) -> Tuple[Dict[str, str], List[str]]:
    """Split the tool repos into those present on disk and those missing."""
    present, missing = {}, []
    for name, root in repo_roots(eve_root).items():
        if os.path.isdir(root):
            present[name] = os.path.normpath(root)
        else:
            missing.append(name)
    return present, missing


def _tool_key(api_file: str) -> str:
    """Derive a tool key from an api.yaml path, matching get_api_files()."""
    directory = os.path.dirname(api_file)
    key = os.path.basename(directory)
    # Substring test, deliberately identical to get_api_files() so that the
    # keys this module reports are the keys `eve tool update` accepts.
    if "legacy" in directory:
        key = f"legacy_{key}"
    return key


def collect_yaml_prices(
    repos: Optional[Dict[str, str]] = None,
    eve_root: Optional[str] = None,
) -> Tuple[Dict[str, YamlPrice], List[Tuple[str, str, str]], List[Tuple[str, str]]]:
    """Read every api.yaml in the given repos and resolve effective prices.

    Returns (prices, collisions, unresolved_parents). Inheritance is resolved
    strictly within the collected corpus -- never against the database.
    """
    if repos is None:
        repos, _ = discover_repos(eve_root)

    raw: Dict[str, Dict[str, Any]] = {}
    collisions: List[Tuple[str, str, str]] = []

    for repo, root in repos.items():
        for dirpath, _dirnames, filenames in os.walk(root):
            if "inactive_workflows" in dirpath.split(os.sep):
                continue
            if "api.yaml" not in filenames:
                continue
            api_file = os.path.join(dirpath, "api.yaml")
            try:
                with open(api_file, "r") as f:
                    schema = yaml.safe_load(f) or {}
            except Exception as e:  # malformed YAML is a pricing problem too
                schema = {"__error__": str(e)}

            key = _tool_key(api_file)
            if key in raw:
                prev = raw[key]
                # Two files claiming one key. Only a real problem if they
                # disagree about price -- otherwise it is merely untidy.
                if prev.get("cost_estimate") != schema.get("cost_estimate"):
                    collisions.append((key, prev["__path__"], api_file))
                continue
            schema["__path__"] = api_file
            schema["__repo__"] = repo
            raw[key] = schema

    prices: Dict[str, YamlPrice] = {}
    unresolved: List[Tuple[str, str]] = []

    for key, schema in raw.items():
        cost = schema.get("cost_estimate")
        inherited_from = None

        # Walk up the parent chain within the corpus.
        seen = {key}
        parent = schema.get("parent_tool")
        while cost is None and parent:
            if parent in seen:  # cycle
                unresolved.append((key, f"circular parent_tool chain via {parent}"))
                break
            seen.add(parent)
            parent_schema = raw.get(parent)
            if parent_schema is None:
                unresolved.append((key, f"parent_tool '{parent}' not found in repos"))
                break
            cost = parent_schema.get("cost_estimate")
            inherited_from = parent
            parent = parent_schema.get("parent_tool")

        prices[key] = YamlPrice(
            key=key,
            cost_estimate=None if cost is None else str(cost),
            path=schema.get("__path__", "?"),
            repo=schema.get("__repo__", "?"),
            inherited_from=inherited_from if cost is not None else None,
        )

    return prices, collisions, unresolved


def collect_db_prices() -> Dict[str, Optional[str]]:
    """Read every tool's cost_estimate out of tools3. Requires load_env()."""
    from eve.mongo import get_collection

    collection = get_collection("tools3")
    out: Dict[str, Optional[str]] = {}
    for doc in collection.find({}, {"key": 1, "cost_estimate": 1}):
        key = doc.get("key")
        if not key:
            continue
        cost = doc.get("cost_estimate")
        out[key] = None if cost is None else str(cost)
    return out


def load_db_only_registry(path: str = DB_ONLY_REGISTRY) -> List[str]:
    """Tools that legitimately have no api.yaml anywhere."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    return list(data.get("db_only_tools") or [])


class _Probe(float):
    """A stand-in value that satisfies every operation the grammar can perform.

    Validation must not need real arguments, but an empty environment binds
    free variables to None, and None breaks the moment an expression indexes,
    iterates or takes the length of a parameter -- so perfectly good
    expressions like `20 * sumLengths(segments, "text")` would look broken.
    Binding every free identifier to this object instead exercises the whole
    expression structurally: what survives is a real syntax/shape error.

    It is a float (of value 1, so it is safe as a divisor) that also answers to
    len(), iteration, indexing and attribute access.
    """

    __slots__ = ()

    def __new__(cls):
        return super().__new__(cls, 1.0)

    def __len__(self):
        return 1

    def __iter__(self):
        return iter([self])

    def get(self, _key, _default=None):
        return self

    def __getattr__(self, _name):
        return self

    def __getitem__(self, _key):
        return self


# Words that are part of the language, not tool parameters.
_RESERVED = {"true", "false", "null", "none", "not", "length", "sumLengths"}
_IDENT_RE = None


def free_identifiers(expression: str) -> List[str]:
    global _IDENT_RE
    if _IDENT_RE is None:
        import re

        _IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    # Drop string literals first so their contents are not mistaken for names.
    import re

    stripped = re.sub(r"'[^']*'|\"[^\"]*\"", " ", expression)
    return [
        name
        for name in dict.fromkeys(_IDENT_RE.findall(stripped))
        if name not in _RESERVED and name.lower() not in _RESERVED
    ]


def validate_expression(expression: Optional[str]) -> Optional[str]:
    """Return an error string if the expression is not evaluable, else None.

    This turns a pricing typo into a PR-time failure instead of a runtime one:
    production evaluates these strings on every generation, and a bad one
    currently only surfaces when a user tries to run the tool.
    """
    if expression is None:
        return None
    expression = str(expression)
    variables = {name: _Probe() for name in free_identifiers(expression)}
    try:
        result = eval_cost(expression, **variables)
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        return f"does not evaluate to a number (got {type(result).__name__}: {result!r})"
    return None


def is_priced(value: Any) -> bool:
    """True if this cost_estimate actually charges for anything."""
    return value is not None and str(value).strip() not in ("", "0", "0.0")


def would_erase_price(old: Any, new: Any) -> bool:
    """True if writing ``new`` over ``old`` would make a paid tool free.

    The guard on `eve tool update`. A tool whose api.yaml lost its
    cost_estimate -- or inherited nothing because its parent could not be
    resolved -- must never silently zero out a live price.
    """
    return is_priced(old) and not is_priced(new)


def normalize(expression: Optional[str]) -> Optional[str]:
    """Compare prices as trimmed strings; whitespace is not a price change."""
    if expression is None:
        return None
    return " ".join(str(expression).split())


def compute_drift(
    yaml_prices: Dict[str, YamlPrice],
    db_prices: Dict[str, Optional[str]],
    db_only: Optional[List[str]] = None,
    checked_repos: Optional[List[str]] = None,
    unchecked_repos: Optional[List[str]] = None,
    collisions: Optional[List[Tuple[str, str, str]]] = None,
    unresolved_parent: Optional[List[Tuple[str, str]]] = None,
    flag_db_only: bool = True,
) -> DriftReport:
    """Compare repo prices against database prices.

    ``flag_db_only`` controls whether a database tool with no api.yaml is an
    error. It should be True for PROD -- an unexplained tool there means
    someone hand-edited Mongo -- and False for STAGE, which legitimately
    carries in-development tools that were never meant to reach a repo.
    """
    db_only = set(db_only or [])
    report = DriftReport(
        checked_repos=sorted(checked_repos or []),
        unchecked_repos=sorted(unchecked_repos or []),
        collisions=list(collisions or []),
        unresolved_parent=list(unresolved_parent or []),
    )

    for key in sorted(yaml_prices):
        price = yaml_prices[key]
        error = validate_expression(price.cost_estimate)
        if error:
            report.invalid.append((key, price.cost_estimate or "", error))

        if key not in db_prices:
            report.not_yet_in_db.append(key)
            continue

        report.compared += 1
        if normalize(price.cost_estimate) != normalize(db_prices[key]):
            report.drifted.append((key, price.cost_estimate, db_prices[key]))

    # Tools in the database with no api.yaml in any *checked* repo. If some
    # repos are unchecked we cannot tell "deleted" from "not looked at", so we
    # only flag these when every repo was available.
    if not report.unchecked_repos:
        for key in sorted(db_prices):
            if key in yaml_prices:
                continue
            if key in db_only:
                report.registered_db_only.append(key)
            elif flag_db_only:
                report.unregistered_db_only.append((key, db_prices[key]))
            else:
                report.db_only_untracked_info.append(key)

    return report


def format_report(report: DriftReport, db_label: str) -> str:
    """Render a drift report as readable text."""
    lines: List[str] = []
    add = lines.append

    add(f"Pricing check against {db_label}")
    add(f"  repos checked   : {', '.join(report.checked_repos) or '(none)'}")
    if report.unchecked_repos:
        add(f"  repos UNCHECKED : {', '.join(report.unchecked_repos)}")
        add("    (tools defined only in these repos were not compared)")
    add(f"  tools compared  : {report.compared}")

    if report.drifted:
        add("")
        add(f"PRICE DRIFT ({len(report.drifted)}):")
        for key, repo_cost, db_cost in report.drifted:
            add(f"  {key}")
            add(f"    repo: {repo_cost!r}")
            add(f"    db  : {db_cost!r}")

    if report.invalid:
        add("")
        add(f"INVALID COST EXPRESSIONS ({len(report.invalid)}):")
        for key, expression, error in report.invalid:
            add(f"  {key}: {expression!r}")
            add(f"    {error}")

    if report.unresolved_parent:
        add("")
        add(f"UNRESOLVED PRICE INHERITANCE ({len(report.unresolved_parent)}):")
        for key, reason in report.unresolved_parent:
            add(f"  {key}: {reason}")

    if report.collisions:
        add("")
        add(f"DUPLICATE TOOL KEYS WITH DIFFERENT PRICES ({len(report.collisions)}):")
        for key, first, second in report.collisions:
            add(f"  {key}:")
            add(f"    {first}")
            add(f"    {second}")

    if report.unregistered_db_only:
        add("")
        add(f"IN DATABASE BUT NOT IN ANY REPO ({len(report.unregistered_db_only)}):")
        add("  These have no api.yaml. If that is intentional, add them to")
        add(f"  {os.path.relpath(DB_ONLY_REGISTRY)}; otherwise restore the file.")
        for key, cost in report.unregistered_db_only:
            add(f"  {key}: {cost!r}")

    if report.not_yet_in_db:
        add("")
        add(f"in repo, not yet in database ({len(report.not_yet_in_db)}): "
            f"{', '.join(report.not_yet_in_db)}")
        add("  (new tools; they are created on the next sync)")

    if report.registered_db_only:
        add("")
        add(f"database-only, registered ({len(report.registered_db_only)}): "
            f"{', '.join(report.registered_db_only)}")

    if report.db_only_untracked_info:
        add("")
        add(f"database-only, not in any repo ({len(report.db_only_untracked_info)}): "
            f"{', '.join(report.db_only_untracked_info)}")
        add("  (not an error on this database)")

    add("")
    add("OK - repo and database agree." if report.ok else "FAILED - see above.")
    return "\n".join(lines)


def check(
    eve_root: Optional[str] = None,
    db_only_path: str = DB_ONLY_REGISTRY,
    flag_db_only: bool = True,
) -> DriftReport:
    """Convenience: collect both sides and diff them. Requires load_env()."""
    repos, missing = discover_repos(eve_root)
    yaml_prices, collisions, unresolved = collect_yaml_prices(repos, eve_root)
    return compute_drift(
        yaml_prices=yaml_prices,
        db_prices=collect_db_prices(),
        db_only=load_db_only_registry(db_only_path),
        checked_repos=list(repos),
        unchecked_repos=missing,
        collisions=collisions,
        unresolved_parent=unresolved,
        flag_db_only=flag_db_only,
    )
