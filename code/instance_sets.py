"""Named instance-set loaders for SWT-Bench v1.

Provides the canonical instance-ID sets used across all phases.

Source of truth for Lite sets: submission/report.json
  - resolved_ids at :259  (175 entries)
  - unresolved_ids at :436 (72 entries)
  - error_ids at :510     (28 entries)

Verified set is loaded directly from the HF dataset at runtime.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()


def _find_repo_root() -> Path:
    """Walk up to find the repo root (has submission/ or data/baseline/)."""
    env_override = os.environ.get("SWT_BENCH_REPO_ROOT")
    if env_override:
        return Path(env_override).resolve()
    for parent in [_THIS_FILE.parent, *_THIS_FILE.parents]:
        if (parent / "submission" / "report.json").exists():
            return parent
        if (parent / "data" / "baseline" / "report.json").exists():
            return parent
    # Fallback to legacy layout
    if len(_THIS_FILE.parents) >= 5:
        return _THIS_FILE.parents[4]
    return _THIS_FILE.parents[-1]


import os  # noqa: E402
REPO_ROOT = _find_repo_root()

# Try both canonical layouts for the baseline report
_REPORT_CANDIDATES = [
    REPO_ROOT / "submission" / "report.json",
    REPO_ROOT / "data" / "baseline" / "report.json",
    REPO_ROOT / "submission_baseline" / "report.json",
]
SUBMISSION_REPORT: Path = next(
    (p for p in _REPORT_CANDIDATES if p.exists()), _REPORT_CANDIDATES[0]
)

_JINA_PATH = str(REPO_ROOT / "jina-harness")
if _JINA_PATH not in sys.path:
    sys.path.insert(0, _JINA_PATH)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal report cache
# ---------------------------------------------------------------------------
_report_cache: dict | None = None


def _report() -> dict:
    global _report_cache
    if _report_cache is None:
        _report_cache = json.loads(SUBMISSION_REPORT.read_text())
    return _report_cache


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def load_resolved175() -> list[str]:
    """Return the 175 Lite instance IDs that resolved in the 63.6% baseline.

    Source: submission/report.json::resolved_ids (line 259, 175 entries).
    These are the WIN set — do not regress on them.
    """
    ids = list(_report()["resolved_ids"])
    assert len(ids) == 175, f"Expected 175 resolved_ids, got {len(ids)}"
    return ids


def load_failing100() -> list[str]:
    """Return the 100 Lite instance IDs that failed in the 63.6% baseline.

    = unresolved_ids (72, line 436) + error_ids (28, line 510) = 100 total.
    These are the TARGET set for Phase 2.
    No overlap with resolved_ids is enforced.
    """
    report = _report()
    unresolved = list(report["unresolved_ids"])
    errors = list(report["error_ids"])
    assert len(unresolved) == 72, f"Expected 72 unresolved_ids, got {len(unresolved)}"
    assert len(errors) == 28, f"Expected 28 error_ids, got {len(errors)}"

    failing = unresolved + errors
    assert len(failing) == 100, f"Expected 100 failing total, got {len(failing)}"

    # Sanity: no overlap with resolved
    resolved_set = set(load_resolved175())
    overlap = set(failing) & resolved_set
    assert not overlap, f"failing100 overlaps resolved175: {overlap}"

    return failing


def load_ratelimited76() -> list[str]:
    """Return the 76 failing100 instances that got empty patches due to rate limiting.

    These are instances from failing100 where Phase 2 Option D produced
    empty model_patch (fix generation failed, mostly due to API rate limits).
    Source: final_results/predictions_100.jsonl — instances with empty model_patch.
    """
    # Try multiple candidate locations for the phase 2 predictions file
    candidates = [
        REPO_ROOT / "data" / "phase2_run1" / "predictions_100.jsonl",
        REPO_ROOT / "final_results" / "predictions_100.jsonl",
        REPO_ROOT / "submission_jina_test" / "predictions_100.jsonl",
    ]
    predictions_path = next((p for p in candidates if p.exists()), None)
    if predictions_path is None:
        raise FileNotFoundError(
            f"phase 2 predictions file not found. Tried: {[str(p) for p in candidates]}"
        )

    empty_ids = []
    with predictions_path.open() as f:
        for line in f:
            rec = json.loads(line)
            if not rec.get("model_patch", "").strip():
                empty_ids.append(rec["instance_id"])

    logger.info("Loaded %d rate-limited instances from %s", len(empty_ids), predictions_path)

    # Sanity: all should be in failing100
    failing_set = set(load_failing100())
    not_failing = set(empty_ids) - failing_set
    assert not not_failing, f"ratelimited76 has IDs not in failing100: {not_failing}"

    return empty_ids


def load_lite275() -> list[str]:
    """Return all 275 Lite instance IDs (resolved + unresolved + errors).

    Source: submission/report.json::completed_ids (247) + any uncompleted.
    Uses resolved + failing = 175 + 100 = 275.
    """
    return load_resolved175() + load_failing100()


def load_verified433() -> list[str]:
    """Return all 433 Verified instance IDs loaded from the HF dataset.

    Dataset: eth-sri/SWT-bench_Verified_bm25_27k_zsp
    Count verified against public SWT-Bench Verified leaderboard (433 instances).
    """
    from evaluation.benchmarks.swt_bench_v1.config import SWT_BENCH_VERIFIED_HF
    from datasets import load_dataset  # lazy import

    logger.info("Loading Verified433 instance IDs from %s", SWT_BENCH_VERIFIED_HF)
    ds = load_dataset(SWT_BENCH_VERIFIED_HF, split="test")
    ids = [row["instance_id"] for row in ds]
    logger.info("Loaded %d Verified instance IDs", len(ids))
    return ids


def load_lite30_calib() -> list[str]:
    """Return 30 Lite instances for calibration (stratified sample of failing100).

    Uses the first 30 entries from failing100 as a deterministic calibration set.
    For phase 0/1 calibration gates G0.6, G1.1, G2.1.
    """
    return load_failing100()[:30]


def load_verified30_calib() -> list[str]:
    """Return 30 Verified instances for calibration.

    Uses the first 30 entries from verified433 as a deterministic calibration set.
    For phase 0 Verified baseline gate G0.6.
    """
    return load_verified433()[:30]


def load_instance_set(name: str) -> list[str]:
    """Load a named instance set by string key.

    Valid names: failing100, resolved175, lite275, verified433,
                 lite30_calib, verified30_calib.
    """
    _loaders = {
        "failing100": load_failing100,
        "ratelimited76": load_ratelimited76,
        "resolved175": load_resolved175,
        "lite275": load_lite275,
        "verified433": load_verified433,
        "lite30_calib": load_lite30_calib,
        "verified30_calib": load_verified30_calib,
    }
    if name not in _loaders:
        raise ValueError(f"Unknown instance set: {name!r}. Valid: {sorted(_loaders)}")
    return _loaders[name]()
