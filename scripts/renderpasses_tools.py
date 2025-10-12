"""
Utilities to validate and (conservatively) backfill alias_to_param in renderPasses.json.

Usage:
    python scripts/renderpasses_tools.py --validate
    python scripts/renderpasses_tools.py --backfill [--write]

The backfill operation is conservative: it only creates mappings when a setting provides a
`name` or `alias`. It will not invent parameter names from labels alone.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_config(path: str, config: Dict[str, Any]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def backfill_alias_to_param(json_path: str, write: bool = False) -> Tuple[int, Dict[str, Dict[str, str]]]:
    """Conservatively backfill `alias_to_param` for passes that lack it.

    Returns (count_written, generated_maps)
    """
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(json_path)
    config = load_config(json_path)
    generated = {}
    count = 0
    for pass_name, entry in config.items():
        if not isinstance(entry, dict):
            continue
        existing = entry.get('alias_to_param')
        if isinstance(existing, dict) and existing:
            continue
        settings = entry.get('settings')
        if not isinstance(settings, list):
            continue
        mapping: Dict[str, str] = {}
        for s in settings:
                if not isinstance(s, dict):
                    continue
                # prefer original_name, then name, then alias
                orig = s.get('original_name') or s.get('name')
                alias = s.get('alias')
                label = s.get('label')
                if orig:
                    if alias:
                        mapping[alias] = orig
                    if label and label != alias:
                        mapping[label] = orig
                elif alias:
                    # No explicit original name but alias is provided; map alias->alias and label->alias
                    mapping[alias] = alias
                    if label and label != alias:
                        mapping[label] = alias
                else:
                    # Conservative: don't create mapping for settings with only label
                    continue
        if mapping:
            generated[pass_name] = mapping
            entry['alias_to_param'] = mapping
            count += 1
    if write and count > 0:
        save_config(json_path, config)
    return count, generated


def validate_renderpasses(json_path: str) -> Dict[str, Any]:
    """Validate renderPasses.json structure and report potential issues.

    Returns a report dict with keys: ok (bool), issues (list[str])
    """
    p = Path(json_path)
    if not p.exists():
        return {'ok': False, 'issues': [f'File not found: {json_path}']}
    config = load_config(json_path)
    issues = []
    for pass_name, entry in config.items():
        if not isinstance(entry, dict):
            issues.append(f"Pass '{pass_name}' is not an object")
            continue
        settings = entry.get('settings')
        if not isinstance(settings, list):
            issues.append(f"Pass '{pass_name}': missing or invalid 'settings' list")
            continue
        alias_map = entry.get('alias_to_param')
        # If any setting contains a 'name' or 'alias', prefer alias_to_param to exist
        need_map = any(isinstance(s, dict) and (s.get('name') or s.get('alias')) for s in settings)
        if need_map and not isinstance(alias_map, dict):
            issues.append(f"Pass '{pass_name}': recommended to include 'alias_to_param' mapping")
        if isinstance(alias_map, dict):
            # sanity checks
            for k, v in alias_map.items():
                if not k or not isinstance(k, str):
                    issues.append(f"Pass '{pass_name}': alias_to_param contains a non-string key: {k}")
                if not v or not isinstance(v, str):
                    issues.append(f"Pass '{pass_name}': alias_to_param contains a non-string target for '{k}': {v}")
    return {'ok': len(issues) == 0, 'issues': issues}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='renderPasses.json tools: validate and backfill alias_to_param')
    parser.add_argument('--file', '-f', default=None, help='Path to renderPasses.json (defaults to project root)')
    parser.add_argument('--backfill', action='store_true', help='Conservatively backfill alias_to_param for passes that lack it')
    parser.add_argument('--write', action='store_true', help='Write changes when using --backfill')
    parser.add_argument('--validate', action='store_true', help='Validate renderPasses.json and report issues')

    args = parser.parse_args()
    repo_root = Path(__file__).parent.parent
    json_path = args.file or str(repo_root / 'renderPasses.json')

    if args.backfill:
        count, generated = backfill_alias_to_param(json_path, write=args.write)
        print(f"Backfill completed: {count} passes updated")
        if generated:
            for p, m in generated.items():
                print(f"- {p}: {len(m)} mappings")
        if args.write:
            print(f"Wrote changes to {json_path}")
        else:
            print("Dry-run (use --write to persist changes)")

    if args.validate:
        report = validate_renderpasses(json_path)
        if report['ok']:
            print("renderPasses.json OK")
        else:
            print("Validation issues found:")
            for i in report['issues']:
                print(" - ", i)

    if not args.backfill and not args.validate:
        parser.print_help()
