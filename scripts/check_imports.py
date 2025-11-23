"""Utility to import Eve modules and report failures."""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = "eve"


@dataclass
class ImportFailure:
    module: str
    traceback: str


def ensure_repo_on_path() -> None:
    repo_str = str(REPO_ROOT)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def walk_package_modules(package_name: str) -> Iterable[str]:
    package = importlib.import_module(package_name)
    pkg_path = getattr(package, "__path__", None)
    if pkg_path is None:
        yield package_name
        return
    for module in pkgutil.walk_packages(pkg_path, prefix=f"{package_name}."):
        yield module.name


def run_import_checks(
    packages: Sequence[str], fail_fast: bool = False
) -> List[ImportFailure]:
    failures: List[ImportFailure] = []
    for package in packages:
        for module_name in walk_package_modules(package):
            try:
                importlib.import_module(module_name)
            except (ModuleNotFoundError, ImportError):
                failure = ImportFailure(
                    module=module_name, traceback=traceback.format_exc()
                )
                failures.append(failure)
                print(f"\n--- {module_name} ---\n{failure.traceback}")
                if fail_fast:
                    return failures
    return failures


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import eve.* modules to surface missing dependencies early."
    )
    parser.add_argument(
        "packages",
        nargs="*",
        default=[DEFAULT_PACKAGE],
        help="Packages to inspect (defaults to 'eve').",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first import failure.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    ensure_repo_on_path()
    args = parse_args(argv)
    failures = run_import_checks(args.packages, fail_fast=args.fail_fast)
    if failures:
        print(f"{len(failures)} modules failed to import.")
        return 1
    print("All modules imported cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
