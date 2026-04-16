"""Stage 4: Winner selection for Option E pipeline.

Given a list of scored test candidates, pick the best one using a priority
ordering:
  1. CXV frequency rank (how many proxy fixes each test confirms, more = better)
  2. Score (weighted quality)
  3. Diff length (shorter patch = simpler test, preferred)
  4. Path name (stable alphabetical tiebreaker)

A candidate with no diff at all is never selected as the winner unless there
are zero alternatives.
"""

from __future__ import annotations

from typing import Any


def pick_winner(
    scores: list[float],
    candidates: list[dict[str, Any]],
    *,
    cxv_counts: list[int] | None = None,
    tiebreaker_order: tuple[str, ...] = ("cxv_count", "score", "diff_len", "path_name"),
) -> int:
    """Return the index of the best test candidate.

    Parameters
    ----------
    scores:
        Parallel list of float scores (0-1) for each candidate.
    candidates:
        List of candidate dicts, each with at minimum:
          - ``diff`` (str)          : the unified diff of the test
          - ``path_name`` (str)     : prompt path name (e.g. "direct")
        Optional keys used by tiebreakers:
          - ``is_viable`` (bool)    : False means infra error / no fail on buggy
    cxv_counts:
        Parallel list of int counts — how many proxy fixes the test confirmed.
        If None, CXV tiebreaker is skipped and only score/diff/path are used.
    tiebreaker_order:
        Ordered tuple of tiebreaker keys.  Valid keys: ``cxv_count``,
        ``score``, ``diff_len``, ``path_name``.

    Returns
    -------
    int
        Index into ``candidates`` of the chosen winner. Returns -1 only when
        ``candidates`` is empty (callers should guard against this).
    """
    if not candidates:
        return -1

    n = len(candidates)
    if len(scores) != n:
        raise ValueError(
            f"len(scores)={len(scores)} != len(candidates)={n}"
        )
    if cxv_counts is not None and len(cxv_counts) != n:
        raise ValueError(
            f"len(cxv_counts)={len(cxv_counts)} != len(candidates)={n}"
        )

    # Build sort keys per candidate
    def _sort_key(idx: int) -> tuple:
        c = candidates[idx]
        s = scores[idx]
        cxv = cxv_counts[idx] if cxv_counts is not None else 0
        diff = c.get("diff", "")
        path = c.get("path_name", "")
        parts: list[Any] = []
        for key in tiebreaker_order:
            if key == "cxv_count":
                parts.append(-cxv)           # higher count = better → negate
            elif key == "score":
                parts.append(-s)             # higher score = better → negate
            elif key == "diff_len":
                parts.append(len(diff))      # shorter = better
            elif key == "path_name":
                parts.append(path)           # alphabetical, stable
        return tuple(parts)

    # Prefer viable candidates (fail on buggy code, non-infra error)
    viable_indices = [
        i for i in range(n) if candidates[i].get("is_viable", True) and candidates[i].get("diff", "").strip()
    ]
    non_empty_indices = [i for i in range(n) if candidates[i].get("diff", "").strip()]

    if viable_indices:
        pool = viable_indices
    elif non_empty_indices:
        pool = non_empty_indices
    else:
        # All candidates are empty diffs — return first
        return 0

    return min(pool, key=_sort_key)


def build_cxv_counts(
    cxv_matrix: list[list[bool]],
) -> list[int]:
    """Convert a K×J boolean matrix to a per-test CXV count vector.

    ``cxv_matrix[i][j]`` is True if test candidate i passes when fix j is applied.
    Returns a list of length K where entry i is the number of fixes confirmed by test i.
    """
    return [sum(row) for row in cxv_matrix]
