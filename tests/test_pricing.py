"""Tests for single-source-of-truth tool pricing.

These are offline: they build small YAML corpora on disk and fake database
dictionaries, so nothing here touches Mongo.
"""

import os
import textwrap

import pytest

from eve import pricing


def write_tool(root, name, body, legacy=False):
    """Create <root>[/legacy]/<name>/api.yaml and return its path."""
    parts = [root, "legacy", name] if legacy else [root, name]
    directory = os.path.join(*parts)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "api.yaml")
    with open(path, "w") as f:
        f.write(textwrap.dedent(body))
    return path


# --------------------------------------------------------------------------
# Collecting prices from the repo
# --------------------------------------------------------------------------


def test_collects_cost_estimate_from_yaml(tmp_path):
    root = str(tmp_path / "tools")
    write_tool(root, "widget", "name: Widget\ncost_estimate: 5 * n_samples\n")

    prices, collisions, unresolved = pricing.collect_yaml_prices({"eve": root})

    assert collisions == []
    assert unresolved == []
    assert prices["widget"].cost_estimate == "5 * n_samples"
    assert prices["widget"].inherited_from is None


def test_legacy_tools_get_the_legacy_prefix(tmp_path):
    """The key must match get_api_files(), or `eve tool update` misses it."""
    root = str(tmp_path / "tools")
    write_tool(root, "controlnet", "name: CN\ncost_estimate: '10'\n", legacy=True)

    prices, _, _ = pricing.collect_yaml_prices({"eve": root})

    assert "legacy_controlnet" in prices
    assert "controlnet" not in prices


def test_price_is_inherited_from_parent_tool_within_the_repo(tmp_path):
    """A child with no cost_estimate inherits its parent's -- from the repo.

    Tool.convert_from_yaml() resolves parent_tool out of Mongo, which makes the
    repo's answer depend on the database. Pricing must not do that.
    """
    root = str(tmp_path / "tools")
    write_tool(root, "base", "name: Base\ncost_estimate: n_samples * 10\n")
    write_tool(root, "child", "name: Child\nparent_tool: base\n")

    prices, _, unresolved = pricing.collect_yaml_prices({"eve": root})

    assert unresolved == []
    assert prices["child"].cost_estimate == "n_samples * 10"
    assert prices["child"].inherited_from == "base"


def test_inheritance_walks_more_than_one_level(tmp_path):
    root = str(tmp_path / "tools")
    write_tool(root, "grand", "name: G\ncost_estimate: '7'\n")
    write_tool(root, "mid", "name: M\nparent_tool: grand\n")
    write_tool(root, "leaf", "name: L\nparent_tool: mid\n")

    prices, _, _ = pricing.collect_yaml_prices({"eve": root})

    assert prices["leaf"].cost_estimate == "7"


def test_child_overrides_parent_price(tmp_path):
    root = str(tmp_path / "tools")
    write_tool(root, "base", "name: Base\ncost_estimate: '10'\n")
    write_tool(root, "child", "name: Child\nparent_tool: base\ncost_estimate: '3'\n")

    prices, _, _ = pricing.collect_yaml_prices({"eve": root})

    assert prices["child"].cost_estimate == "3"
    assert prices["child"].inherited_from is None


def test_missing_parent_is_reported_not_silently_unpriced(tmp_path):
    root = str(tmp_path / "tools")
    write_tool(root, "orphan", "name: O\nparent_tool: nowhere\n")

    prices, _, unresolved = pricing.collect_yaml_prices({"eve": root})

    assert prices["orphan"].cost_estimate is None
    assert unresolved and unresolved[0][0] == "orphan"
    assert "nowhere" in unresolved[0][1]


def test_circular_parent_chain_terminates(tmp_path):
    root = str(tmp_path / "tools")
    write_tool(root, "a", "name: A\nparent_tool: b\n")
    write_tool(root, "b", "name: B\nparent_tool: a\n")

    _prices, _, unresolved = pricing.collect_yaml_prices({"eve": root})

    assert any("circular" in reason for _key, reason in unresolved)


def test_duplicate_keys_only_collide_when_prices_disagree(tmp_path):
    """Two files, one key. Harmless if they agree on price; a bug if not."""
    agree_root = str(tmp_path / "agree")
    write_tool(os.path.join(agree_root, "x"), "dup", "name: D\ncost_estimate: '5'\n")
    write_tool(os.path.join(agree_root, "y"), "dup", "name: D\ncost_estimate: '5'\n")
    _prices, collisions, _ = pricing.collect_yaml_prices({"eve": agree_root})
    assert collisions == []

    clash_root = str(tmp_path / "clash")
    write_tool(os.path.join(clash_root, "x"), "dup", "name: D\ncost_estimate: '5'\n")
    write_tool(os.path.join(clash_root, "y"), "dup", "name: D\ncost_estimate: '9'\n")
    _prices, collisions, _ = pricing.collect_yaml_prices({"eve": clash_root})
    assert collisions and collisions[0][0] == "dup"


# --------------------------------------------------------------------------
# The drift check
# --------------------------------------------------------------------------


def yaml_price(key, cost):
    return pricing.YamlPrice(key=key, cost_estimate=cost, path=f"{key}/api.yaml", repo="eve")


def test_no_drift_when_repo_and_db_agree():
    report = pricing.compute_drift(
        yaml_prices={"a": yaml_price("a", "5 * n_samples")},
        db_prices={"a": "5 * n_samples"},
        checked_repos=["eve", "workflows", "private_workflows"],
    )
    assert report.ok
    assert report.drifted == []
    assert report.compared == 1


def test_drift_is_detected_and_reported_both_ways():
    """This is the check that would have caught the 17 drifted tools."""
    report = pricing.compute_drift(
        yaml_prices={"a": yaml_price("a", "2 * n_samples")},
        db_prices={"a": "5 * n_samples"},
        checked_repos=["eve", "workflows", "private_workflows"],
    )
    assert not report.ok
    assert report.drifted == [("a", "2 * n_samples", "5 * n_samples")]
    text = pricing.format_report(report, "eden-prod")
    assert "PRICE DRIFT" in text
    assert "2 * n_samples" in text and "5 * n_samples" in text


def test_whitespace_is_not_a_price_change():
    report = pricing.compute_drift(
        yaml_prices={"a": yaml_price("a", "5  *   n_samples")},
        db_prices={"a": "5 * n_samples"},
        checked_repos=["eve", "workflows", "private_workflows"],
    )
    assert report.ok


def test_a_yaml_tool_missing_a_price_drifts_against_a_priced_db_tool():
    """The five-tools-become-free scenario, seen from the checker's side."""
    report = pricing.compute_drift(
        yaml_prices={"a": yaml_price("a", None)},
        db_prices={"a": "n_samples * 10"},
        checked_repos=["eve", "workflows", "private_workflows"],
    )
    assert not report.ok
    assert report.drifted == [("a", None, "n_samples * 10")]


def test_new_tool_in_repo_is_informational_not_a_failure():
    report = pricing.compute_drift(
        yaml_prices={"brand_new": yaml_price("brand_new", "1")},
        db_prices={},
        checked_repos=["eve", "workflows", "private_workflows"],
    )
    assert report.ok
    assert report.not_yet_in_db == ["brand_new"]


def test_registered_db_only_tool_passes_but_unregistered_one_fails():
    report = pricing.compute_drift(
        yaml_prices={},
        db_prices={"retired": "5", "mystery": "9"},
        db_only=["retired"],
        checked_repos=["eve", "workflows", "private_workflows"],
    )
    assert not report.ok
    assert report.registered_db_only == ["retired"]
    assert report.unregistered_db_only == [("mystery", "9")]


def test_db_only_tools_are_not_flagged_when_flag_db_only_is_off():
    """STAGE legitimately carries tools that were never meant to reach a repo."""
    report = pricing.compute_drift(
        yaml_prices={},
        db_prices={"experiment": "0"},
        checked_repos=["eve", "workflows", "private_workflows"],
        flag_db_only=False,
    )
    assert report.ok
    assert report.db_only_untracked_info == ["experiment"]


def test_missing_repo_never_makes_its_tools_look_deleted():
    """The three-repo rule: an unchecked repo must not imply deletion."""
    report = pricing.compute_drift(
        yaml_prices={},
        db_prices={"lives_in_private_workflows": "20"},
        db_only=[],
        checked_repos=["eve", "workflows"],
        unchecked_repos=["private_workflows"],
    )
    assert report.ok
    assert report.unregistered_db_only == []
    assert "UNCHECKED" in pricing.format_report(report, "eden-prod")


def test_collisions_and_unresolved_parents_fail_the_check():
    assert not pricing.compute_drift(
        yaml_prices={}, db_prices={}, collisions=[("d", "a.yaml", "b.yaml")]
    ).ok
    assert not pricing.compute_drift(
        yaml_prices={}, db_prices={}, unresolved_parent=[("o", "missing parent")]
    ).ok


# --------------------------------------------------------------------------
# Expression validation -- a syntax error must never reach production
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression",
    [
        "6 + 4 * n_samples * (width / 1024)",
        "50",
        "0.8 * duration",
        'output == "video" ? 85 * duration : 13 * n_samples',
        "5.5 + 0.05 * image_input.length",
        '20 * sumLengths(segments, "text")',
        "high_quality ? 12 * n_samples : 6 * n_samples",
        '(quality == "high" ? 30 : quality == "medium" ? 9 : 2) * n_samples',
        "1.0 * n_frames * (width + height)/(2*512) * (steps/25)",
    ],
)
def test_real_cost_expressions_validate(expression):
    assert pricing.validate_expression(expression) is None


@pytest.mark.parametrize(
    "expression",
    ["5 * * n_samples", "(5 * n_samples", "5 +", '"just a string"'],
)
def test_broken_cost_expressions_are_rejected(expression):
    assert pricing.validate_expression(expression) is not None


def test_invalid_expression_fails_the_drift_check():
    report = pricing.compute_drift(
        yaml_prices={"a": yaml_price("a", "5 * * n_samples")},
        db_prices={"a": "5 * * n_samples"},
        checked_repos=["eve", "workflows", "private_workflows"],
    )
    assert not report.ok
    assert report.invalid and report.invalid[0][0] == "a"
    assert "INVALID COST EXPRESSIONS" in pricing.format_report(report, "eden-prod")


def test_validate_expression_ignores_missing_arguments():
    """Validation must not need real args, or it flags healthy tools."""
    assert pricing.validate_expression("n_samples * width / height") is None


# --------------------------------------------------------------------------
# The guard that stops a live price being erased
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "old,new,should_block",
    [
        ("n_samples * 10", None, True),      # yaml lost its cost_estimate
        ("n_samples * 10", "", True),        # blank
        ("n_samples * 10", "0", True),       # zeroed -> tool becomes free
        ("n_samples * 10", "0.0", True),
        ("n_samples * 10", "n_samples * 5", False),  # a real, intentional change
        (None, "5", False),                  # newly priced tool
        ("0", "0", False),                   # deliberately free tool stays free
        (None, None, False),
    ],
)
def test_price_erasure_guard(old, new, should_block):
    """Refuse to write an absent/zero price over a live one.

    This single rule is what turns "five tools are now free" into an error.
    """
    assert pricing.would_erase_price(old, new) is should_block


def test_normalize_treats_none_and_values_distinctly():
    assert pricing.normalize(None) is None
    assert pricing.normalize("  5 *  n ") == "5 * n"
    assert pricing.normalize(50) == "50"


# --------------------------------------------------------------------------
# The shipped db-only registry must describe the real world
# --------------------------------------------------------------------------


def test_db_only_registry_loads_and_is_a_list_of_names():
    keys = pricing.load_db_only_registry()
    assert isinstance(keys, list)
    assert keys, "registry should not be empty"
    assert all(isinstance(k, str) and k for k in keys)
    assert len(keys) == len(set(keys)), "registry has duplicates"


def test_repo_roots_match_get_api_files_layout():
    from eve.tool import get_api_files  # noqa: F401  (import proves it exists)

    roots = pricing.repo_roots()
    assert set(roots) == {"eve", "workflows", "private_workflows"}
    assert roots["eve"].endswith(os.path.join("eve", "tools"))
