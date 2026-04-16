"""SWT-Bench v1 runner — Option E full pipeline orchestrator.

Pipeline per instance:
  Stage 1: BoN 5 paths × K=6 temperatures = 30 test candidates via bon_runner
  Stage 2: J=5 fix candidates via fix_gen.generate_fix_candidates
  Stage 3: K×J=150 CXV matrix via cxv.build_cxv_matrix
  Stage 4: Score + select winner via selector.pick_winner

CLI:
    python -m evaluation.benchmarks.swt_bench_v1.runner \\
        --phase 0 \\
        --track lite \\
        --instance-set failing100 \\
        --smoke \\          # 1 instance only, $20 budget cap
        --num-workers 2

P0.0: Resolve and record model IDs at startup.
P0.0.5: Assert litellm versions match between jina-harness and fork envs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# sys.path surgery — must happen before any jina_test import
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
# parents: [0]=swt_bench_v1, [1]=benchmarks, [2]=evaluation, [3]=openhands-fork, [4]=swt-bench
REPO_ROOT = _THIS_FILE.parents[4]
sys.path.insert(0, str(REPO_ROOT / "jina-harness"))

# ---------------------------------------------------------------------------
# Local config — no SDK imports here so --help works without openhands
# ---------------------------------------------------------------------------
from evaluation.benchmarks.swt_bench_v1.config import (  # noqa: E402
    EVAL_OUTPUT_ROOT,
    KILL_SWITCH_USD,
    MAX_CONCURRENT_SANDBOXES,
    PHASE1_AGENT_MODEL,
    PHASE2_FIX_MODEL,
)
from evaluation.benchmarks.swt_bench_v1.cxv import (  # noqa: E402
    build_cxv_matrix,
    score_candidates_by_frequency,
)
from evaluation.benchmarks.swt_bench_v1.fix_gen import (  # noqa: E402
    generate_fix_candidates,
)
from evaluation.benchmarks.swt_bench_v1.selector import (  # noqa: E402
    build_cxv_counts,
    pick_winner,
)
from evaluation.benchmarks.swt_bench_v1.instance_sets import load_instance_set  # noqa: E402

# bon_runner is lazy-imported inside _stage1_generate_tests to avoid
# import-time failures when openhands SDK is not installed (e.g. --help,
# unit tests with mocks).

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PROMPTS_DIR = _THIS_FILE.parent / "prompts"
_PROMPT_FILENAMES = [
    "direct.j2",
    "snippet.j2",
    "assertflip.j2",
    "contract.j2",
    "exception.j2",
]
_BOK_TEMPERATURES = [0.0, 0.6]                          # K=2 (option D: reduced for speed)
_FIX_TEMPERATURES = (0.0, 0.5, 1.0)                     # J=3 (option D: reduced for speed)

# Smoke config: full BoN + KJ feature test over 2 instances.
# 5 paths × 6 temps = 30 test candidates, 5 fix candidates, 150 CXV cells.
# Expected per-instance: ~30 min, ~$7. Total for 2 instances: ~60 min, ~$14.
_PROMPT_FILENAMES_SMOKE = [
    "direct.j2",
    "snippet.j2",
    "assertflip.j2",
    "contract.j2",
    "exception.j2",
]
_BOK_TEMPERATURES_SMOKE = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
_FIX_TEMPERATURES_SMOKE = (0.0, 0.3, 0.6, 0.9, 1.2)
_SMOKE_INSTANCE_COUNT = 2  # Number of instances to run in smoke mode
_SMOKE_BUDGET_USD = 30.0  # 2 instances × full BoN+CXV config


# ---------------------------------------------------------------------------
# P0.0 — Model ID resolution
# ---------------------------------------------------------------------------


def resolve_model_ids() -> tuple[str, str]:
    """Return (phase1_model, phase2_model)."""
    phase1 = PHASE1_AGENT_MODEL
    phase2 = PHASE2_FIX_MODEL
    logger.info("P0.0 model IDs: phase1=%s  phase2=%s", phase1, phase2)
    return phase1, phase2


# ---------------------------------------------------------------------------
# P0.0.5 — litellm version assertion
# ---------------------------------------------------------------------------


def _assert_litellm_versions_match() -> str:
    """Assert jina-harness and fork envs resolve the same litellm version."""
    jina_out = subprocess.check_output(
        ["poetry", "show", "litellm", "--no-ansi"],
        cwd=str(REPO_ROOT / "jina-harness"),
    ).decode()
    fork_out = subprocess.check_output(
        ["poetry", "show", "litellm", "--no-ansi"],
        cwd=str(REPO_ROOT / "openhands-fork"),
    ).decode()

    jina_ver = jina_out.split()[1]
    fork_ver = fork_out.split()[1]

    assert jina_ver == fork_ver, (
        f"litellm version mismatch: jina-harness={jina_ver}, fork={fork_ver}. "
        "Pin both to the same version before running."
    )
    logger.info("P0.0.5 litellm version check passed: %s", jina_ver)
    return jina_ver


# ---------------------------------------------------------------------------
# Results header writer
# ---------------------------------------------------------------------------


def _write_phase0_header(
    phase1_model: str,
    phase2_model: str,
    litellm_version: str | None,
) -> None:
    EVAL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    header_path = EVAL_OUTPUT_ROOT / "phase0.jsonl"
    record = {
        "type": "header",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase1_agent_model_resolved": phase1_model,
        "phase2_fix_model_resolved": phase2_model,
        "litellm_version": litellm_version,
        "max_concurrent_sandboxes": MAX_CONCURRENT_SANDBOXES,
        "kill_switch_usd": KILL_SWITCH_USD,
    }
    with header_path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    logger.info("P0 header written to %s", header_path)


# ---------------------------------------------------------------------------
# Budget guard
# ---------------------------------------------------------------------------


def _sum_costs(output_dir: Path, run_id: str) -> float:
    total = 0.0
    run_dir = output_dir / run_id
    if not run_dir.exists():
        return total
    for jsonl_path in sorted(run_dir.glob("*.jsonl")):
        with jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    total += float(rec.get("cost_usd") or rec.get("total_cost_usd") or 0.0)
                except Exception:
                    continue
    return total


def _check_budget(output_dir: Path, run_id: str, max_usd: float) -> float:
    total = _sum_costs(output_dir, run_id)
    if total > max_usd:
        raise RuntimeError(
            f"Budget exceeded: ${total:.2f} > ${max_usd:.2f}. Stopping."
        )
    return total


# ---------------------------------------------------------------------------
# Stage 1: BoN test generation  (5 paths × K=6 temperatures = 30 candidates)
#
# bon_runner.run_single_candidate() is the primary async API.
# Each call spins up a fresh OpenHands Agent + Docker workspace.
# Lazy-imported to avoid openhands SDK at import time.
# ---------------------------------------------------------------------------


async def _stage1_generate_tests(
    instance: dict[str, Any],
    llm: Any,         # jina_test.llm.LLMClient
    agent_cfg: Any,   # jina_test.config.AgentConfig
) -> list[dict[str, Any]]:
    """Generate 30 test candidates: 5 prompt paths × 6 temperatures.

    Each candidate dict has:
      instance_id, path_name, temperature, patch, test_files, is_viable, error
    """
    # Lazy import — openhands SDK only required at runtime, not import time
    from evaluation.benchmarks.swt_bench_v1 import bon_runner  # noqa: PLC0415

    instance_id = instance["instance_id"]
    logger.info(
        "[%s] Stage 1: %d paths × %d temps = %d candidates",
        instance_id,
        len(_PROMPT_FILENAMES),
        len(_BOK_TEMPERATURES),
        len(_PROMPT_FILENAMES) * len(_BOK_TEMPERATURES),
    )

    # Throttle concurrent Docker workspaces to avoid OOM / disk exhaustion.
    # Each bon_runner call creates its own Docker workspace (~2GB each).
    _BON_CONCURRENCY = 6
    bon_sem = asyncio.Semaphore(_BON_CONCURRENCY)

    async def _throttled_bon(fn: str, temp: float) -> dict:
        async with bon_sem:
            return await bon_runner.run_single_candidate(
                instance=instance,
                prompt_path=_PROMPTS_DIR / fn,
                temperature=temp,
                llm=llm,
                agent_cfg=agent_cfg,
            )

    tasks = [
        _throttled_bon(fn, temp)
        for fn in _PROMPT_FILENAMES
        for temp in _BOK_TEMPERATURES
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    candidates: list[dict[str, Any]] = []
    for i, res in enumerate(results):
        fn = _PROMPT_FILENAMES[i // len(_BOK_TEMPERATURES)]
        temp = _BOK_TEMPERATURES[i % len(_BOK_TEMPERATURES)]
        path_name = fn.replace(".j2", "")

        if isinstance(res, Exception):
            logger.warning(
                "[%s] Stage 1 path=%s temp=%.1f raised: %s",
                instance_id, path_name, temp, res,
            )
            candidates.append({
                "instance_id": instance_id,
                "path_name": path_name,
                "temperature": temp,
                "patch": "",
                "test_files": [],
                "is_viable": False,
                "error": str(res),
                "cost_usd": 0.0,
            })
        else:
            candidates.append(res)

    viable = sum(1 for c in candidates if c.get("is_viable"))
    rate_limit_errors = sum(
        1 for c in candidates
        if c.get("error") and "RateLimitError" in c.get("error", "")
    )
    logger.info(
        "[%s] Stage 1 done: %d candidates (%d viable, %d rate_limited)",
        instance_id, len(candidates), viable, rate_limit_errors,
    )
    if rate_limit_errors > len(candidates) * 0.5:
        logger.error(
            "[%s] ALERT: >50%% of Stage 1 candidates hit rate limits (%d/%d). "
            "API quota may be exhausted. Consider pausing or reducing concurrency.",
            instance_id, rate_limit_errors, len(candidates),
        )
    if viable == 0:
        logger.error(
            "[%s] ALERT: 0 viable test candidates from Stage 1. "
            "This instance will produce an empty prediction.",
            instance_id,
        )
    return candidates


# ---------------------------------------------------------------------------
# Stage 2: Fix generation  (J=5 candidate fixes via LLM completions)
# ---------------------------------------------------------------------------


async def _stage2_generate_fixes(
    instance: dict[str, Any],
    fix_llm: Any,       # jina_test.llm.LLMClient configured with phase2 model
    agent_cfg: Any,     # jina_test.config.AgentConfig
) -> list[dict[str, Any]]:
    """Generate J candidate fix patches via OpenHands Agent with fix_agent.j2.

    Bug 1 fix: the previous raw-litellm fix_gen.py produced corrupt diffs
    because it asked the LLM to write unified diff text directly (line numbers
    are hard to hallucinate). Instead we reuse bon_runner.run_single_candidate
    with a fix-focused prompt — the OpenHands Agent's file_editor tool produces
    valid diffs. Each fix candidate is one full agent run (slower but reliable).
    """
    from evaluation.benchmarks.swt_bench_v1 import bon_runner  # noqa: PLC0415

    instance_id = instance["instance_id"]
    fix_prompt_path = _PROMPTS_DIR / "fix_agent.j2"
    logger.info(
        "[%s] Stage 2: %d fix candidates via OpenHands Agent (fix_agent.j2)",
        instance_id,
        len(_FIX_TEMPERATURES),
    )

    _FIX_CONCURRENCY = 4
    fix_sem = asyncio.Semaphore(_FIX_CONCURRENCY)

    async def _throttled_fix(temp: float) -> dict:
        async with fix_sem:
            return await bon_runner.run_single_candidate(
                instance=instance,
                prompt_path=fix_prompt_path,
                temperature=temp,
                llm=fix_llm,
                agent_cfg=agent_cfg,
            )

    tasks = [_throttled_fix(temp) for temp in _FIX_TEMPERATURES]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    fix_candidates: list[dict[str, Any]] = []
    for i, res in enumerate(results):
        temp = _FIX_TEMPERATURES[i]
        if isinstance(res, Exception):
            logger.warning(
                "[%s] Stage 2 fix temp=%.1f raised: %s",
                instance_id, temp, res,
            )
            fix_candidates.append({
                "instance_id": instance_id,
                "temperature": temp,
                "patch": "",
                "is_viable": False,
                "error": str(res),
                "cost_usd": 0.0,
            })
        else:
            fix_candidates.append(res)

    viable = sum(1 for c in fix_candidates if c.get("is_viable"))
    rate_limit_errors = sum(
        1 for c in fix_candidates
        if c.get("error") and "RateLimitError" in c.get("error", "")
    )
    logger.info(
        "[%s] Stage 2 done: %d/%d fix candidates (%d viable, %d rate_limited)",
        instance_id, len(fix_candidates), len(_FIX_TEMPERATURES), viable, rate_limit_errors,
    )
    if rate_limit_errors > 0:
        logger.error(
            "[%s] ALERT: %d/%d fix candidates hit rate limits. "
            "Fix patches may be degraded or empty.",
            instance_id, rate_limit_errors, len(fix_candidates),
        )
    if viable == 0:
        logger.error(
            "[%s] ALERT: 0 viable fix candidates from Stage 2. "
            "This instance will produce an empty prediction.",
            instance_id,
        )
    return fix_candidates


# ---------------------------------------------------------------------------
# Stage 3 + 4: CXV matrix + selection
# ---------------------------------------------------------------------------


async def _stage3_cxv_and_select(
    instance: dict[str, Any],
    test_candidates: list[dict[str, Any]],
    fix_candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[float], list]:
    """Build K×J CXV matrix and select the winning test candidate."""
    instance_id = instance["instance_id"]
    k = len(test_candidates)
    j = len(fix_candidates)
    logger.info("[%s] Stage 3: CXV %d×%d = %d cells", instance_id, k, j, k * j)

    cells = await build_cxv_matrix(
        instance_id=instance_id,
        base_commit=instance.get("base_commit", "HEAD"),
        repo_url=instance.get("repo", ""),
        test_candidates=test_candidates,
        fix_candidates=fix_candidates,
        sandbox_factory=_make_sandbox_factory(),
        concurrency=MAX_CONCURRENT_SANDBOXES,
    )

    scores = score_candidates_by_frequency(cells, k, j)
    logger.info(
        "[%s] Stage 3 done: top score=%.2f",
        instance_id, max(scores) if scores else 0.0,
    )

    # Stage 4: select winner
    cxv_matrix_per_test: list[list[bool]] = [
        [c.killed for c in cells if c.test_candidate_idx == ti]
        for ti in range(k)
    ]
    cxv_counts = build_cxv_counts(cxv_matrix_per_test)
    winner_idx = pick_winner(scores, test_candidates, cxv_counts=cxv_counts)
    winner = test_candidates[winner_idx] if winner_idx >= 0 else None

    logger.info(
        "[%s] Stage 4: winner_idx=%d path=%s score=%.2f",
        instance_id, winner_idx,
        winner.get("path_name", "?") if winner else "none",
        scores[winner_idx] if winner_idx >= 0 and scores else 0.0,
    )
    return winner, scores, cells


# ---------------------------------------------------------------------------
# CXV sandbox factory — wraps DockerSandbox to cxv.py protocol
# ---------------------------------------------------------------------------


class _SandboxWrapper:
    def __init__(self, sandbox: Any, instance_id: str = "") -> None:
        self._sandbox = sandbox
        self._instance_id = instance_id

    async def apply_patch(self, patch: str) -> None:
        if not patch.strip():
            return
        import base64
        b64 = base64.b64encode(patch.encode()).decode("ascii")
        # Use --check first to see if patch applies cleanly, then apply.
        # Log the result so we can diagnose silent failures.
        cmd = (
            "TMPF=$(mktemp /tmp/cxv-XXXXXX.patch) && "
            f"echo {b64} | base64 -d > $TMPF && "
            "cd /testbed && "
            "(git apply --recount --whitespace=nowarn $TMPF 2>&1 && echo PATCH_OK) || "
            "(git apply --recount --whitespace=nowarn -3 $TMPF 2>&1 && echo PATCH_OK_3WAY) || "
            "(git apply --recount --whitespace=nowarn --reject $TMPF 2>&1; echo PATCH_REJECTED)"
        )
        result = await self._sandbox.exec(cmd)
        output = getattr(result, "combined", "") or ""
        if "PATCH_REJECTED" in output or ("error:" in output.lower() and "PATCH_OK" not in output):
            logger.warning(
                "[cxv] apply_patch failed in %s: %s",
                self._instance_id,
                output[-400:],
            )
            # Bug 2 fix: raise on failure so cxv._run_cell marks the cell
            # INFRA_ERROR instead of silently proceeding with no patch applied.
            raise RuntimeError(
                f"apply_patch failed in {self._instance_id}: {output[-200:]}"
            )
        logger.info(
            "[cxv] apply_patch OK in %s (%d bytes patch)",
            self._instance_id,
            len(patch),
        )

    async def run_tests(self, test_files: list[str]) -> str:
        """Run the new test files inside the testbed conda env.

        SWE-bench images don't ship pytest on PATH — each repo has its own
        test command. We dispatch based on repo portion of instance_id and
        activate the testbed conda env first.
        """
        if not test_files:
            return "(no test files)"

        # Extract repo from instance_id (format: "<org>__<repo>-<id>")
        repo = ""
        if "__" in self._instance_id:
            repo_part = self._instance_id.split("__", 1)[1]
            repo = repo_part.split("-", 1)[0] if "-" in repo_part else repo_part

        # Normalize test file paths (strip leading / or /testbed/)
        rel_paths = []
        for p in test_files:
            p = p.lstrip("/")
            if p.startswith("testbed/"):
                p = p[len("testbed/"):]
            rel_paths.append(p)

        # Build the test command per repo
        if repo == "django":
            # Django: convert tests/forms_tests/.../test_urlfield.py → forms_tests.field_tests.test_urlfield
            modules = []
            for p in rel_paths:
                if p.startswith("tests/"):
                    p = p[len("tests/"):]
                mod = p.replace("/", ".").removesuffix(".py")
                modules.append(mod)
            test_args = " ".join(modules)
            test_cmd = f"./tests/runtests.py --verbosity 2 {test_args}"
        elif repo == "sympy":
            test_args = " ".join(rel_paths)
            test_cmd = f"bin/test -C {test_args}"
        elif repo == "sphinx":
            test_args = " ".join(rel_paths)
            test_cmd = f"python -m pytest -rA {test_args}"
        else:
            # Default: use python -m pytest (works if pytest installed as module)
            test_args = " ".join(rel_paths)
            test_cmd = f"python -m pytest -rA --tb=long -p no:cacheprovider {test_args}"

        activate = "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed"
        cmd = f"cd /testbed && ({activate}) && ({test_cmd} 2>&1 || true)"
        result = await self._sandbox.exec(cmd, timeout=600)
        return result.combined


class _SandboxFactoryContext:
    def __init__(self, instance_id: str, base_commit: str, repo_url: str) -> None:
        self._instance_id = instance_id
        self._sandbox: Any = None

    async def __aenter__(self) -> "_SandboxWrapper":
        from jina_test.config import SandboxConfig
        from jina_test.sandbox import DockerSandbox
        self._sandbox = DockerSandbox(self._instance_id, SandboxConfig())
        await self._sandbox.__aenter__()
        await self._sandbox.exec(
            "git -C /testbed reset --hard HEAD && git -C /testbed clean -fdx"
        )
        return _SandboxWrapper(self._sandbox, instance_id=self._instance_id)

    async def __aexit__(self, *exc: Any) -> None:
        if self._sandbox is not None:
            await self._sandbox.__aexit__(*exc)


def _make_sandbox_factory() -> Any:
    def factory(instance_id: str, base_commit: str, repo_url: str) -> _SandboxFactoryContext:
        return _SandboxFactoryContext(instance_id, base_commit, repo_url)
    return factory


# ---------------------------------------------------------------------------
# Full pipeline for one instance
# ---------------------------------------------------------------------------


async def run_instance_full_pipeline(
    instance: dict[str, Any],
    llm: Any,           # jina_test.llm.LLMClient (for Stage 1 via bon_runner)
    agent_cfg: Any,     # jina_test.config.AgentConfig (for Stage 1)
    phase2_model: str,  # model string for Stage 2 fix_gen
    output_dir: Path,
    run_id: str,
    *,
    budget_usd: float = _SMOKE_BUDGET_USD,
    base_llm: Any = None,
) -> dict[str, Any]:
    """Run the full Option E pipeline on a single instance.

    Returns SWT-Bench JSONL prediction:
      {"instance_id": ..., "model_patch": ..., "model_name_or_path": ...}
    """
    instance_id = instance["instance_id"]
    t_start = time.monotonic()

    # Stage 1: 30 test candidates (async, each spins own Docker workspace)
    test_candidates = await _stage1_generate_tests(instance, llm, agent_cfg)

    try:
        _check_budget(output_dir, run_id, budget_usd)
    except RuntimeError as e:
        logger.error("[%s] %s", instance_id, e)
        return {"instance_id": instance_id, "model_patch": "", "model_name_or_path": run_id, "error": str(e)}

    # Stage 2: J=5 fix candidates via OpenHands Agent (fix_agent.j2 prompt)
    # Build a fix-specific LLMClient configured with phase2_model so bon_runner
    # uses the cheaper model for fix generation.
    from jina_test.config import LLMConfig as _LLMConfig, load_api_key as _load_api_key  # noqa: PLC0415
    from jina_test.llm import LLMClient as _LLMClient  # noqa: PLC0415
    fix_llm = _LLMClient(
        _LLMConfig(generator_model=phase2_model),
        _load_api_key(),
    )
    fix_candidates = await _stage2_generate_fixes(instance, fix_llm, agent_cfg)

    try:
        _check_budget(output_dir, run_id, budget_usd)
    except RuntimeError as e:
        logger.error("[%s] %s", instance_id, e)
        return {"instance_id": instance_id, "model_patch": "", "model_name_or_path": run_id, "error": str(e)}

    # Stages 3 + 4: CXV matrix + selection
    winner, scores, cells = await _stage3_cxv_and_select(instance, test_candidates, fix_candidates)

    # Select the best FIX candidate using the winning test as discriminator.
    # The winning test has the highest kill rate. A fix that SURVIVES the
    # winning test (not killed) is the best candidate — it passes the most
    # discriminating test, meaning it likely addresses the bug correctly.
    best_fix_idx = -1
    fix_patch = ""
    if winner and fix_candidates:
        winner_idx = test_candidates.index(winner) if winner in test_candidates else -1
        if winner_idx >= 0:
            # Find fixes that survive (are NOT killed by) the winning test
            surviving = [
                c.fix_candidate_idx for c in cells
                if c.test_candidate_idx == winner_idx and not c.killed
            ]
            if surviving:
                best_fix_idx = surviving[0]
            else:
                # No fix survived the winning test — pick fix with lowest
                # overall kill count (least-killed = most robust)
                fix_kills = [0] * len(fix_candidates)
                for c in cells:
                    if c.killed:
                        fix_kills[c.fix_candidate_idx] += 1
                best_fix_idx = fix_kills.index(min(fix_kills))
        else:
            best_fix_idx = 0
        fix_patch = fix_candidates[best_fix_idx].get("patch", "")
    elif fix_candidates:
        # No winning test (CXV failed) — fall back to first viable fix
        for fi, fc in enumerate(fix_candidates):
            if fc.get("is_viable"):
                best_fix_idx = fi
                fix_patch = fc.get("patch", "")
                break

    elapsed = time.monotonic() - t_start
    patch = fix_patch

    # SANITY CHECK: model_patch must be a SOURCE fix, never a test patch.
    # This catches the bug where test_candidates[winner_idx].patch was
    # written as model_patch instead of fix_candidates[best_fix_idx].patch.
    if patch and ("/test" in patch or "test_" in patch.split("\n")[0]):
        _first_diff = patch.split("\n")[0] if patch else ""
        if "/test" in _first_diff or "test_" in _first_diff:
            logger.error(
                "[%s] SANITY FAIL: model_patch appears to be a TEST patch, not a fix! "
                "First line: %s. This is likely a selector bug — writing test patch "
                "instead of fix patch. Setting model_patch to empty.",
                instance_id, _first_diff[:200],
            )
            patch = ""

    logger.info(
        "[%s] Pipeline done in %.0fs. best_fix_idx=%d fix_patch_lines=%d test_path=%s",
        instance_id, elapsed, best_fix_idx,
        patch.count("\n") if patch else 0,
        winner.get("path_name", "?") if winner else "none",
    )

    total_cost = sum(float(c.get("cost_usd", 0)) for c in test_candidates)
    total_cost += sum(float(f.get("cost_usd", 0)) for f in fix_candidates)

    run_record = {
        "instance_id": instance_id,
        "winner_path": winner.get("path_name") if winner else None,
        "winner_temperature": winner.get("temperature") if winner else None,
        "winner_score": float(max(scores)) if scores else 0.0,
        "best_fix_idx": best_fix_idx,
        "num_test_candidates": len(test_candidates),
        "num_fix_candidates": len(fix_candidates),
        "num_cxv_cells": len(cells),
        "elapsed_s": round(elapsed, 1),
        "cost_usd": round(total_cost, 6),
    }
    run_path = output_dir / run_id / "runs.jsonl"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    with run_path.open("a") as f:
        f.write(json.dumps(run_record) + "\n")

    return {
        "instance_id": instance_id,
        "model_patch": patch,
        "model_name_or_path": run_id,
    }


# ---------------------------------------------------------------------------
# Parallel multi-instance runner
# ---------------------------------------------------------------------------


async def _run_all(
    instances: list[dict[str, Any]],
    phase1_model: str,
    phase2_model: str,
    output_dir: Path,
    run_id: str,
    num_workers: int,
    budget_usd: float,
) -> list[dict[str, Any]]:
    from jina_test.config import AgentConfig, LLMConfig
    from jina_test.llm import LLMClient
    from jina_test.config import load_api_key

    semaphore = asyncio.Semaphore(num_workers)
    predictions_path = output_dir / run_id / "predictions.jsonl"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip already-completed IDs
    completed_ids: set[str] = set()
    if predictions_path.exists():
        with predictions_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    completed_ids.add(json.loads(line)["instance_id"])
                except Exception:
                    continue
    if completed_ids:
        logger.info("Resuming: %d already done, skipping", len(completed_ids))
        instances = [i for i in instances if i["instance_id"] not in completed_ids]

    api_key = load_api_key()
    llm_cfg = LLMConfig(generator_model=phase1_model)
    llm = LLMClient(llm_cfg, api_key)
    agent_cfg = AgentConfig(best_of_k=6, max_iterations=80)

    results: list[dict[str, Any]] = []
    lock = asyncio.Lock()

    # Circuit breaker: track consecutive empty patches to detect rate limiting
    _consecutive_empty = 0
    _CIRCUIT_BREAKER_THRESHOLD = 8  # stop if 8 consecutive empties

    async def _process(instance: dict[str, Any]) -> None:
        nonlocal _consecutive_empty
        async with semaphore:
            # Check circuit breaker before starting
            if _consecutive_empty >= _CIRCUIT_BREAKER_THRESHOLD:
                logger.error(
                    "[CIRCUIT BREAKER] %d consecutive empty patches — likely rate limited. "
                    "Skipping %s. Fix: wait for rate limit window to reset, then re-run.",
                    _consecutive_empty, instance["instance_id"],
                )
                pred = {
                    "instance_id": instance["instance_id"],
                    "model_patch": "",
                    "model_name_or_path": run_id,
                    "error": f"circuit_breaker: {_consecutive_empty} consecutive empties",
                }
            else:
                pred = await run_instance_full_pipeline(
                    instance=instance,
                    llm=llm,
                    agent_cfg=agent_cfg,
                    phase2_model=phase2_model,
                    output_dir=output_dir,
                    run_id=run_id,
                    budget_usd=budget_usd,
                )

            # Track consecutive empties for circuit breaker
            async with lock:
                if not pred.get("model_patch", "").strip():
                    _consecutive_empty += 1
                    if _consecutive_empty >= _CIRCUIT_BREAKER_THRESHOLD:
                        logger.error(
                            "[CIRCUIT BREAKER] Hit %d consecutive empty patches after %s. "
                            "Halting new instance starts. %d predictions so far.",
                            _consecutive_empty, instance["instance_id"], len(results),
                        )
                else:
                    _consecutive_empty = 0  # reset on success

                results.append(pred)
                with predictions_path.open("a") as f:
                    f.write(json.dumps(pred) + "\n")

    # Stagger worker starts: launch instances with 30s delay between each
    # to avoid burst API requests that trigger rate limiting.
    _STAGGER_DELAY_S = 30
    tasks = []
    for i, inst in enumerate(instances):
        if i > 0 and i % num_workers == 0:
            # Wait between batches to spread API load
            logger.info(
                "Staggering: waiting %ds before next batch (after %d/%d instances)",
                _STAGGER_DELAY_S, i, len(instances),
            )
            await asyncio.sleep(_STAGGER_DELAY_S)
        tasks.append(asyncio.create_task(_process(inst)))

    await asyncio.gather(*tasks)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="runner",
        description="SWT-Bench v1 runner — Option E full pipeline (BoN + CXV)",
    )
    p.add_argument("--phase", type=int, choices=[0, 1, 2], default=0,
                   help="Execution phase: 0=smoke/calibration, 1=full Lite, 2=CXV on failing-100")
    p.add_argument("--track", choices=["lite", "verified"], default="lite",
                   help="Dataset track")
    p.add_argument("--instance-set",
                   choices=["failing100", "ratelimited76", "resolved175", "lite275",
                            "verified433", "lite30_calib", "verified30_calib"],
                   default=None, help="Named instance subset to run")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke-test mode: run 1 instance only with $20 budget cap")
    p.add_argument("--skip-litellm-check", action="store_true",
                   help="Skip P0.0.5 litellm version assertion")
    p.add_argument("--num-workers", type=int, default=4,
                   help="Outer parallelism (instances); CXV has inner concurrency")
    p.add_argument("--budget-usd", type=float, default=_SMOKE_BUDGET_USD,
                   help=f"Per-run budget cap in USD (default: {_SMOKE_BUDGET_USD})")
    p.add_argument("--output-dir", type=Path, default=EVAL_OUTPUT_ROOT,
                   help="Directory to write predictions.jsonl and runs.jsonl")
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    phase1_model, phase2_model = resolve_model_ids()

    litellm_version: str | None = None
    if not args.skip_litellm_check:
        try:
            litellm_version = _assert_litellm_versions_match()
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            logger.warning("P0.0.5 litellm check skipped: %s", exc)
    else:
        logger.info("P0.0.5 litellm check skipped via --skip-litellm-check flag")

    _write_phase0_header(phase1_model, phase2_model, litellm_version)

    run_id = f"swt_bench_v1_option_e_phase{args.phase}_{args.track}"
    if args.instance_set:
        run_id += f"_{args.instance_set}"
    if args.smoke:
        run_id += "_smoke"

    from jina_test.dataset import load_instances
    dataset_name = (
        "eth-sri/SWT-bench_Verified_bm25_27k_zsp"
        if args.track == "verified"
        else "eth-sri/SWT-bench_Lite_bm25_27k_zsp"
    )
    raw_instances = load_instances(dataset_name, "test", n_limit=0)

    if args.instance_set:
        wanted_ids = set(load_instance_set(args.instance_set))
        instances = [i for i in raw_instances if i["instance_id"] in wanted_ids]
    else:
        instances = raw_instances

    if args.smoke:
        instances = instances[:_SMOKE_INSTANCE_COUNT]
        budget = _SMOKE_BUDGET_USD
        # Full BoN + KJ feature test: 5 paths × 6 temps = 30 test candidates,
        # 5 fix candidates, 150 CXV cells per instance.
        global _PROMPT_FILENAMES, _BOK_TEMPERATURES, _FIX_TEMPERATURES
        _PROMPT_FILENAMES = _PROMPT_FILENAMES_SMOKE
        _BOK_TEMPERATURES = _BOK_TEMPERATURES_SMOKE
        _FIX_TEMPERATURES = _FIX_TEMPERATURES_SMOKE
        logger.info(
            "SMOKE MODE: %d instances, $%.0f budget cap, paths=%s, BoK=%s, fixes=%s",
            len(instances), budget, _PROMPT_FILENAMES, _BOK_TEMPERATURES, _FIX_TEMPERATURES,
        )
    else:
        budget = args.budget_usd

    logger.info(
        "Runner startup: phase=%d track=%s instance_set=%s instances=%d smoke=%s",
        args.phase, args.track, args.instance_set, len(instances), args.smoke,
    )

    if not instances:
        logger.warning("No instances to run. Exiting.")
        return

    predictions = asyncio.run(
        _run_all(
            instances=instances,
            phase1_model=phase1_model,
            phase2_model=phase2_model,
            output_dir=args.output_dir,
            run_id=run_id,
            num_workers=args.num_workers,
            budget_usd=budget,
        )
    )
    logger.info("Done. %d predictions written to %s/%s/", len(predictions), args.output_dir, run_id)


if __name__ == "__main__":
    main()
