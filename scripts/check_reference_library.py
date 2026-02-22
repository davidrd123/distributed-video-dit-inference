#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_manifest(manifest_path: Path) -> dict[str, dict]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required to parse refs/manifest.yaml. Install it with `uv pip install pyyaml`."
        ) from exc

    data = yaml.safe_load(_read_text(manifest_path))
    if not isinstance(data, dict) or "resources" not in data:
        raise ValueError(f"Unexpected manifest format: {manifest_path}")

    resources = data["resources"]
    if not isinstance(resources, list):
        raise ValueError(f"Unexpected manifest format: {manifest_path} (resources is not a list)")

    by_id: dict[str, dict] = {}
    for r in resources:
        if not isinstance(r, dict):
            continue
        rid = r.get("id")
        if not isinstance(rid, str) or not rid.strip():
            continue
        if rid in by_id:
            raise ValueError(f"Duplicate resource id in manifest: {rid}")
        by_id[rid] = r
    return by_id


def _iter_numbered_topic_files(repo_root: Path) -> list[Path]:
    topics_dir = repo_root / "refs" / "topics"
    return sorted(topics_dir.glob("[0-9][0-9]-*.md"))


def _extract_impl_context_section(md: str) -> str | None:
    m = re.search(r"^## Implementation context\s*$", md, flags=re.MULTILINE)
    if not m:
        return None
    start = m.end()
    m2 = re.search(r"^##\s+\S", md[start:], flags=re.MULTILINE)
    end = start + (m2.start() if m2 else len(md[start:]))
    return md[start:end]


def _parse_status_field(card_md: str) -> str | None:
    m = re.search(r"^\|\s*Status\s*\|\s*([^|]+?)\s*\|\s*$", card_md, flags=re.MULTILINE)
    if not m:
        return None
    return m.group(1).strip()


def _extract_resource_ids_from_impl_context(md: str) -> set[str]:
    """
    Extract referenced resource IDs from refs/implementation-context.md without
    accidentally grabbing code identifiers.

    Heuristics:
    - For markdown tables, treat the first cell as the resource ID cell and take the
      first backticked token in that cell (allows annotations like "(medium priority)").
    - For the ordered "Resource priority ..." list, capture **`id`** patterns.
    """

    referenced: set[str] = set()
    for line in md.splitlines():
        if line.startswith("|"):
            parts = line.split("|")
            if len(parts) < 3:
                continue
            first_cell = parts[1]
            m = re.search(r"`([^`]+)`", first_cell)
            if m:
                referenced.add(m.group(1).strip())

        m2 = re.match(r"^\s*\d+\.\s+\*\*`([^`]+)`\*\*", line)
        if m2:
            referenced.add(m2.group(1).strip())

    return {r for r in referenced if r}


def main() -> int:
    parser = argparse.ArgumentParser(description="Consistency checks for the reference library.")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Path to repo root (default: current directory).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (useful for CI-style gating).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    manifest_path = repo_root / "refs" / "manifest.yaml"
    impl_context_path = repo_root / "refs" / "implementation-context.md"

    errors: list[str] = []
    warnings: list[str] = []

    def error(msg: str) -> None:
        errors.append(msg)

    def warn(msg: str) -> None:
        warnings.append(msg)

    # ---- Load manifest ----
    try:
        manifest_by_id = _load_manifest(manifest_path)
    except Exception as exc:
        error(f"Failed to parse {manifest_path}: {exc}")
        manifest_by_id = {}

    # ---- Topic stubs: implementation context presence and citations ----
    for topic_path in _iter_numbered_topic_files(repo_root):
        md = _read_text(topic_path)
        section = _extract_impl_context_section(md)
        if section is None:
            error(f"{topic_path}: missing '## Implementation context' section")
            continue

        if not re.search(r"^See:\s+", section, flags=re.MULTILINE):
            error(f"{topic_path}: Implementation context missing 'See:' reference line")

        if "refs/implementation-context.md" not in section:
            warn(f"{topic_path}: 'See:' line does not reference refs/implementation-context.md")

        if not re.search(r"(scope-drd/notes/|\bRun\s+\d+)", section):
            warn(f"{topic_path}: Implementation context may lack a concrete run/notes citation")

    # ---- Bridge doc: referenced resource IDs exist in manifest ----
    if impl_context_path.exists() and manifest_by_id:
        impl_md = _read_text(impl_context_path)
        referenced_ids = _extract_resource_ids_from_impl_context(impl_md)
        missing = sorted(rid for rid in referenced_ids if rid not in manifest_by_id)
        for rid in missing:
            warn(f"{impl_context_path}: references unknown resource id `{rid}` (not in refs/manifest.yaml)")

    # ---- Resource cards: status alignment with manifest ----
    if manifest_by_id:
        for rid, res in sorted(manifest_by_id.items(), key=lambda kv: kv[0]):
            manifest_status = str(res.get("status", "")).strip()
            card_path = repo_root / "refs" / "resources" / f"{rid}.md"
            full_md_path = repo_root / "sources" / rid / "full.md"

            if manifest_status in {"converted", "condensed"} and not full_md_path.exists():
                error(f"{rid}: manifest status={manifest_status} but missing {full_md_path}")

            if manifest_status == "condensed" and not card_path.exists():
                error(f"{rid}: manifest status=condensed but missing {card_path}")
                continue

            if not card_path.exists():
                continue

            card_md = _read_text(card_path)
            card_status = _parse_status_field(card_md)
            if card_status is None:
                warn(f"{card_path}: missing Status field")
                continue

            card_status_norm = card_status.strip().lower()
            if manifest_status == "condensed":
                if card_status_norm == "stub":
                    error(f"{card_path}: Status=stub but manifest says condensed")
            else:
                if card_status_norm != "stub":
                    warn(
                        f"{card_path}: Status={card_status!r} but manifest status={manifest_status!r} (expected stub unless condensed)"
                    )

    if args.strict and warnings:
        errors.extend([f"(strict) {w}" for w in warnings])
        warnings = []

    if errors:
        print(f"Errors ({len(errors)}):")
        for msg in errors:
            print(f"- {msg}")
    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for msg in warnings:
            print(f"- {msg}")
    if not errors and not warnings:
        print("OK: no issues found.")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
