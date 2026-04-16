"""cxv.py — K×J CXV (Cross-Validation) matrix builder.

For each (test_k, fix_j) pair, spin up a fresh sandbox at base_commit,
apply fix_j, apply test_k, run pytest, classify output via verdict.py.

Cells run in parallel via asyncio.gather with a concurrency semaphore.

Reference: e-Otter++ §4.3 (arXiv 2508.06365), S1 in strategies.md.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from .verdict import Verdict, classify_pytest_output, verdict_kills

logger = logging.getLogger(__name__)

# Default concurrency limit for CXV cell execution
_DEFAULT_CONCURRENCY = 8


@dataclass
class CXVCell:
    """Result for a single (test_k, fix_j) cell in the CXV matrix."""

    test_candidate_idx: int  # 0..K-1
    fix_candidate_idx: int   # 0..J-1
    verdict: Verdict
    killed: bool             # True if GOOD_FAIL — test catches a problem in this fix
    pytest_output: str = field(default="", repr=False)
    error: str | None = None


# ---------------------------------------------------------------------------
# Sandbox protocol
# ---------------------------------------------------------------------------
# sandbox_factory is a callable: (instance_id, base_commit) -> async context manager
# The context manager yields an object with:
#   apply_patch(patch: str) -> None  (apply a unified diff)
#   run_tests(test_files: list[str]) -> str  (returns pytest stdout)
#   reset() -> None  (reset to base_commit — optional, used when reusing sandboxes)


async def _run_cell(
    test_idx: int,
    fix_idx: int,
    test_candidate: dict[str, Any],
    fix_candidate: dict[str, Any],
    sandbox_factory: Callable[..., Any],
    instance_id: str,
    base_commit: str,
    repo_url: str,
    semaphore: asyncio.Semaphore,
) -> CXVCell:
    """Execute a single CXV cell: apply fix_j + test_k, run pytest, classify."""
    async with semaphore:
        try:
            async with sandbox_factory(instance_id, base_commit, repo_url) as sandbox:
                # Step 1: Apply the fix patch
                fix_patch = fix_candidate.get("patch", "")
                if fix_patch:
                    await sandbox.apply_patch(fix_patch)

                # Step 2: Apply the test patch
                test_patch = test_candidate.get("patch", "")
                if test_patch:
                    await sandbox.apply_patch(test_patch)

                # Step 3: Run pytest on the test files
                test_files = test_candidate.get("test_files", [])
                pytest_output = await sandbox.run_tests(test_files)

                # Step 4: Classify
                verdict = classify_pytest_output(pytest_output, test_patch=test_patch)
                killed = verdict_kills(verdict)

                logger.info(
                    "[cxv] cell(%d,%d) verdict=%s killed=%s pytest_tail=%r",
                    test_idx,
                    fix_idx,
                    verdict.value,
                    killed,
                    pytest_output[-600:] if pytest_output else "",
                )
                return CXVCell(
                    test_candidate_idx=test_idx,
                    fix_candidate_idx=fix_idx,
                    verdict=verdict,
                    killed=killed,
                    pytest_output=pytest_output,
                )
        except Exception as e:
            logger.warning(
                "[cxv] cell(%d,%d) error: %s",
                test_idx,
                fix_idx,
                e,
            )
            return CXVCell(
                test_candidate_idx=test_idx,
                fix_candidate_idx=fix_idx,
                verdict=Verdict.INFRA_ERROR,
                killed=False,
                pytest_output="",
                error=str(e),
            )


async def build_cxv_matrix(
    instance_id: str,
    base_commit: str,
    repo_url: str,
    test_candidates: list[dict[str, Any]],
    fix_candidates: list[dict[str, Any]],
    sandbox_factory: Callable[..., Any],
    *,
    concurrency: int = _DEFAULT_CONCURRENCY,
) -> list[CXVCell]:
    """Build the K×J CXV matrix.

    For each (test_k, fix_j) pair:
      1. Spin up fresh sandbox at base_commit via sandbox_factory
      2. Apply fix_j to the repo
      3. Apply test_k to the repo
      4. Run pytest (only the new test files from test_k)
      5. Classify output via verdict.py
      6. Record CXVCell (killed=True iff GOOD_FAIL)

    Cells run in parallel via asyncio.gather with a semaphore limiting
    concurrent sandbox executions to ``concurrency``.

    Args:
        instance_id: SWT-bench instance identifier.
        base_commit: Git commit hash for the buggy baseline.
        repo_url: Repository URL (passed to sandbox_factory).
        test_candidates: K test candidate dicts (from bon_runner).
            Each dict: {patch: str, test_files: list[str], ...}
        fix_candidates: J fix candidate dicts (from fix_gen).
            Each dict: {patch: str, ...}
        sandbox_factory: Async context manager factory callable.
            Signature: (instance_id, base_commit, repo_url) -> AsyncContextManager
            The yielded sandbox must implement:
              - apply_patch(patch: str) -> Awaitable[None]
              - run_tests(test_files: list[str]) -> Awaitable[str]
        concurrency: Max simultaneous sandbox executions. Default 8.

    Returns:
        Flat list of K*J CXVCell objects, ordered by (test_idx, fix_idx).
    """
    k = len(test_candidates)
    j = len(fix_candidates)
    logger.info(
        "[cxv] Building %d×%d matrix for %s (concurrency=%d)",
        k,
        j,
        instance_id,
        concurrency,
    )

    semaphore = asyncio.Semaphore(concurrency)

    tasks = [
        _run_cell(
            test_idx=ti,
            fix_idx=fi,
            test_candidate=test_candidates[ti],
            fix_candidate=fix_candidates[fi],
            sandbox_factory=sandbox_factory,
            instance_id=instance_id,
            base_commit=base_commit,
            repo_url=repo_url,
            semaphore=semaphore,
        )
        for ti in range(k)
        for fi in range(j)
    ]

    cells: list[CXVCell] = await asyncio.gather(*tasks)

    total_killed = sum(1 for c in cells if c.killed)
    logger.info(
        "[cxv] Done: %d/%d cells killed (GOOD_FAIL)",
        total_killed,
        len(cells),
    )
    return cells


def score_candidates_by_frequency(
    cells: list[CXVCell],
    k: int,
    j: int,
) -> list[float]:
    """Compute frequency-rank score per test candidate.

    A test k 'kills' fix j if the cell(k, j) has verdict GOOD_FAIL.
    Score(test_k) = number of fixes killed / J.

    Higher score = test catches more candidate fixes = better proxy for
    a real bug-revealing test.

    Args:
        cells: Flat list of CXVCell from build_cxv_matrix.
        k: Number of test candidates (K).
        j: Number of fix candidates (J).

    Returns:
        List of K floats in [0, 1]. Index i = score for test_candidates[i].
    """
    if j == 0:
        return [0.0] * k

    # Count kills per test candidate
    kill_counts = [0] * k
    for cell in cells:
        if cell.killed and 0 <= cell.test_candidate_idx < k:
            kill_counts[cell.test_candidate_idx] += 1

    return [count / j for count in kill_counts]


def select_best_test(
    cells: list[CXVCell],
    test_candidates: list[dict[str, Any]],
    k: int,
    j: int,
) -> dict[str, Any] | None:
    """Select the best test candidate using frequency-rank scoring.

    Returns the test candidate with the highest kill score, or None if no
    test killed any fix (all scores are 0).

    Tiebreak by smaller patch size (LOC).
    """
    scores = score_candidates_by_frequency(cells, k, j)

    if not scores or max(scores) == 0.0:
        return None

    # Find best: highest score, tiebreak by smallest patch
    best_idx = max(
        range(k),
        key=lambda i: (
            scores[i],
            -len(test_candidates[i].get("patch", "").splitlines()),
        ),
    )
    result = dict(test_candidates[best_idx])
    result["cxv_score"] = scores[best_idx]
    result["cxv_rank"] = best_idx
    return result
