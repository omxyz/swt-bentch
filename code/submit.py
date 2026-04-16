"""SWT-Bench leaderboard JSONL packager and validator.

Three modes:
  --input <path> --output <path>   Convert harness output to canonical JSONL
  --validate <path>                Assert all fields present and track-complete
  --merge <prior.jsonl> <phase2.jsonl>  Additive merge; NEVER overwrite prior predictions

Floor protection rule (plan §3 P2.0):
  The 175 prior patches from submission/predictions.jsonl are NEVER overwritten.
  Phase 2 outputs are additive-only for failing100 instances.

Canonical SWT-Bench JSONL schema (verified against submission/predictions.jsonl:1):
  {"instance_id": str, "model_patch": str, "model_name_or_path": str}
  Optional: "full_output": str

See plan §3 (submit.py spec) and §6 AC A19.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo / submission paths
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
# parents: [0]=swt_bench_v1, [1]=benchmarks, [2]=evaluation, [3]=openhands-fork, [4]=swt-bench
REPO_ROOT = _THIS_FILE.parents[4]
PRIOR_PREDICTIONS = REPO_ROOT / "submission" / "predictions.jsonl"
PRIOR_REPORT = REPO_ROOT / "submission" / "report.json"

# Expected instance counts per track (plan §3 P0.1)
TRACK_INSTANCE_COUNTS = {
    "lite": 275,
    "verified": 433,
}

CANONICAL_REQUIRED_FIELDS = {"instance_id", "model_patch", "model_name_or_path"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno} — invalid JSON: {e}") from e
    return records


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _load_prior_predictions() -> dict[str, dict]:
    """Load the 63.6% baseline predictions keyed by instance_id."""
    if not PRIOR_PREDICTIONS.exists():
        return {}
    records = _read_jsonl(PRIOR_PREDICTIONS)
    return {r["instance_id"]: r for r in records if "instance_id" in r}


def _to_canonical(record: dict) -> dict:
    """Normalise a harness record to the canonical SWT-Bench submission schema."""
    canonical = {
        "instance_id": record["instance_id"],
        "model_patch": record.get("model_patch") or record.get("diff") or "",
        "model_name_or_path": record.get("model_name_or_path") or record.get("model") or "unknown",
    }
    if "full_output" in record:
        canonical["full_output"] = record["full_output"]
    return canonical


# ---------------------------------------------------------------------------
# --input / --output mode
# ---------------------------------------------------------------------------


def convert_input(input_path: Path, output_path: Path) -> int:
    """Read harness predictions, emit canonical JSONL."""
    records = _read_jsonl(input_path)
    canonical = []
    errors = []
    for i, rec in enumerate(records):
        if "instance_id" not in rec:
            errors.append(f"Record {i}: missing instance_id")
            continue
        canonical.append(_to_canonical(rec))

    if errors:
        for e in errors:
            print(f"WARNING: {e}", file=sys.stderr)

    _write_jsonl(canonical, output_path)
    print(f"Wrote {len(canonical)} canonical predictions to {output_path}")
    return 0


# ---------------------------------------------------------------------------
# --validate mode
# ---------------------------------------------------------------------------


def validate(path: Path, track: str = "lite") -> int:
    """Validate a submission JSONL for completeness and correctness."""
    records = _read_jsonl(path)
    errors: list[str] = []
    seen_ids: set[str] = set()

    for i, rec in enumerate(records):
        # Check required fields
        missing = CANONICAL_REQUIRED_FIELDS - set(rec.keys())
        if missing:
            errors.append(f"Record {i} ({rec.get('instance_id', '?')}): missing fields {missing}")

        iid = rec.get("instance_id", "")
        if not iid:
            errors.append(f"Record {i}: empty instance_id")
            continue

        if iid in seen_ids:
            errors.append(f"Duplicate instance_id: {iid}")
        seen_ids.add(iid)

        # model_patch may be empty string (no-change prediction) but must be present
        if "model_patch" not in rec:
            errors.append(f"Record {i} ({iid}): missing model_patch field")

        if not rec.get("model_name_or_path"):
            errors.append(f"Record {i} ({iid}): empty model_name_or_path")

    # Check track completeness
    expected_count = TRACK_INSTANCE_COUNTS.get(track)
    if expected_count is not None and len(seen_ids) != expected_count:
        errors.append(
            f"Track '{track}' expects {expected_count} instances, "
            f"got {len(seen_ids)}"
        )

    if errors:
        print(f"VALIDATION FAILED: {len(errors)} error(s)", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    print(
        f"OK: {path} — {len(records)} records, all required fields present, "
        f"track={track} count={len(seen_ids)}/{expected_count or '?'}"
    )
    return 0


# ---------------------------------------------------------------------------
# --merge mode  (floor protection)
# ---------------------------------------------------------------------------


def merge(prior_path: Path, phase2_path: Path, output_path: Path) -> int:
    """Merge prior and phase2 predictions with floor protection.

    Floor protection rule: prior predictions for ANY instance_id are NEVER
    overwritten by phase2 predictions. Phase 2 is additive-only for instances
    NOT already in the prior.

    This preserves the 175 already-resolved instances from the 63.6% baseline.
    """
    prior_records = _read_jsonl(prior_path)
    phase2_records = _read_jsonl(phase2_path)

    # Build prior set keyed by instance_id
    merged: dict[str, dict] = {}
    for rec in prior_records:
        iid = rec.get("instance_id")
        if iid:
            merged[iid] = _to_canonical(rec)

    # Add phase2 predictions only for NEW instances (no overwrite)
    added = 0
    skipped = 0
    for rec in phase2_records:
        iid = rec.get("instance_id")
        if not iid:
            continue
        if iid in merged:
            # Floor protection: prior prediction stands
            skipped += 1
        else:
            merged[iid] = _to_canonical(rec)
            added += 1

    output = list(merged.values())
    _write_jsonl(output, output_path)
    print(
        f"Merged: {len(prior_records)} prior + {added} new phase2 predictions "
        f"({skipped} phase2 records skipped — floor protection). "
        f"Total: {len(output)} written to {output_path}"
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="submit",
        description="SWT-Bench leaderboard JSONL packager",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input",
        type=Path,
        metavar="PATH",
        help="Input harness prediction JSONL to convert to canonical format",
    )
    group.add_argument(
        "--validate",
        type=Path,
        metavar="PATH",
        help="Validate an existing submission JSONL",
    )
    group.add_argument(
        "--merge",
        nargs=2,
        type=Path,
        metavar=("PRIOR", "PHASE2"),
        help="Merge prior.jsonl and phase2.jsonl with floor protection",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for --input or --merge modes",
    )
    parser.add_argument(
        "--track",
        choices=["lite", "verified"],
        default="lite",
        help="Track for --validate completeness check (default: lite)",
    )

    args = parser.parse_args(argv)

    if args.input is not None:
        if args.output is None:
            parser.error("--input requires --output")
        return convert_input(args.input, args.output)

    if args.validate is not None:
        return validate(args.validate, track=args.track)

    if args.merge is not None:
        prior_path, phase2_path = args.merge
        if args.output is None:
            parser.error("--merge requires --output")
        return merge(prior_path, phase2_path, args.output)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
