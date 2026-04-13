#!/usr/bin/env python3
"""Compare the original thesis results JSON with the safe same-structure mock template.

This script checks:
1. key-path parity between the original results JSON and the safe mock template
2. whether numeric leaves in the original are nulled in the mock template
3. whether each major row in the mock template carries `_mock_info`

Usage:
  python scripts/compare_results_template_and_mock.py
  python scripts/compare_results_template_and_mock.py \
      --original thesis-medsam/data/results_update_template.json \
      --mock thesis-medsam/data/results_mock_same_structure_template.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


DEFAULT_ORIGINAL = Path("thesis-medsam/data/results_update_template.json")
DEFAULT_MOCK = Path("thesis-medsam/data/results_mock_same_structure_template.json")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def join_path(parts: Sequence[str]) -> str:
    out: List[str] = []
    for p in parts:
        if p.startswith("["):
            if out:
                out[-1] = out[-1] + p
            else:
                out.append(p)
        else:
            out.append(p)
    return ".".join(out)


def collect_leaf_paths(obj: Any, prefix: Tuple[str, ...] = ()) -> Dict[str, Any]:
    leaves: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            leaves.update(collect_leaf_paths(v, prefix + (k,)))
        return leaves
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            leaves.update(collect_leaf_paths(v, prefix + (f"[{i}]",)))
        return leaves
    leaves[join_path(prefix)] = obj
    return leaves


def walk_rows(obj: Any, prefix: Tuple[str, ...] = ()) -> Iterable[Tuple[str, dict]]:
    if isinstance(obj, dict):
        if "_mock_info" in obj:
            yield join_path(prefix), obj
        for k, v in obj.items():
            yield from walk_rows(v, prefix + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_rows(v, prefix + (f"[{i}]",))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare original thesis results JSON with the safe same-structure mock template.")
    parser.add_argument("--original", default=str(DEFAULT_ORIGINAL), help="Path to the original results JSON.")
    parser.add_argument("--mock", default=str(DEFAULT_MOCK), help="Path to the safe same-structure mock JSON.")
    args = parser.parse_args()

    original = load_json(Path(args.original))
    mock = load_json(Path(args.mock))

    original_leaves = collect_leaf_paths(original)
    mock_leaves = collect_leaf_paths(mock)

    missing_in_mock = sorted(set(original_leaves) - set(mock_leaves))
    extra_in_mock = sorted(set(mock_leaves) - set(original_leaves))

    nullified_numeric: List[str] = []
    mismatched_numeric: List[str] = []
    for path, value in sorted(original_leaves.items()):
        if isinstance(value, (int, float)):
            mock_value = mock_leaves.get(path, "__missing__")
            if mock_value is None:
                nullified_numeric.append(path)
            elif mock_value == "__missing__":
                mismatched_numeric.append(path + " (missing)")
            else:
                mismatched_numeric.append(path + f" (expected null, found {mock_value!r})")

    row_infos = list(walk_rows(mock))

    print("Original vs Safe Mock Template")
    print("==============================")
    print(f"Original: {args.original}")
    print(f"Mock:     {args.mock}")
    print()

    print("Path Summary")
    print("------------")
    print(f"- original leaf count: {len(original_leaves)}")
    print(f"- mock leaf count:     {len(mock_leaves)}")
    print(f"- row blocks with _mock_info: {len(row_infos)}")
    print()

    if missing_in_mock:
        print("Missing Paths In Mock")
        print("---------------------")
        for path in missing_in_mock[:50]:
            print(f"- {path}")
        if len(missing_in_mock) > 50:
            print(f"- ... ({len(missing_in_mock) - 50} more)")
        print()

    if extra_in_mock:
        print("Extra Paths In Mock")
        print("-------------------")
        for path in extra_in_mock[:80]:
            print(f"- {path}")
        if len(extra_in_mock) > 80:
            print(f"- ... ({len(extra_in_mock) - 80} more)")
        print()

    print("Numeric Nulling Check")
    print("---------------------")
    print(f"- numeric leaves nulled in mock: {len(nullified_numeric)}")
    print(f"- numeric leaves not nulled:     {len(mismatched_numeric)}")
    if mismatched_numeric:
        for path in mismatched_numeric[:50]:
            print(f"- {path}")
        if len(mismatched_numeric) > 50:
            print(f"- ... ({len(mismatched_numeric) - 50} more)")
    print()

    print("Sample Mock Guidance")
    print("--------------------")
    for path, row in row_infos[:20]:
        info = row["_mock_info"]
        role = info.get("row_role", "n/a")
        pattern = info.get("expected_pattern") or info.get("expected_local_shape") or "n/a"
        print(f"- {path}")
        print(f"  role: {role}")
        print(f"  shape: {pattern}")
    if len(row_infos) > 20:
        print(f"- ... ({len(row_infos) - 20} more row blocks)")

    if missing_in_mock or mismatched_numeric:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
