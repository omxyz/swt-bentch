"""Configuration constants for the SWT-Bench v1 evaluation harness.

Dataset suffix is `_bm25_27k_zsp` (NOT `_zsb`).
Verified 2026-04-08 via submission/metadata.json:54.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory roots
# ---------------------------------------------------------------------------

# This file lives at: openhands-fork/evaluation/benchmarks/swt_bench_v1/config.py
# parents[0] = swt_bench_v1/
# parents[1] = benchmarks/
# parents[2] = evaluation/
# parents[3] = openhands-fork/
# parents[4] = swt-bench/   ← REPO_ROOT
REPO_ROOT: Path = Path(__file__).resolve().parents[4]

EVAL_OUTPUT_ROOT: Path = REPO_ROOT / "openhands-fork" / "results"

# ---------------------------------------------------------------------------
# Model IDs (P0.0)
# Verified 2026-04-08 via developers.openai.com/api/docs/models — see plan §3 P0.0
# The 63.6% baseline used the undated `openai/gpt-5.4` (submission/metadata.json:3).
# Using dated aliases is an iter-3 hardening, not a model swap.
# Fallback: undated aliases if dated resolution fails (handled in runner.py).
# ---------------------------------------------------------------------------

PHASE1_AGENT_MODEL: str = "openai/gpt-5.4-2026-03-05"
PHASE2_FIX_MODEL: str = "openai/gpt-5.4-mini-2026-03-17"

# Populated at runtime after model ID resolution; None until then.
PHASE1_AGENT_MODEL_RESOLVED: str | None = None

# ---------------------------------------------------------------------------
# Execution limits
# ---------------------------------------------------------------------------

MAX_CONCURRENT_SANDBOXES: int = 8
KILL_SWITCH_USD: float = 40_000.0

# ---------------------------------------------------------------------------
# Dataset names (P0.1)
# Suffix `_bm25_27k_zsp` = BM25-retrieved context, 27k token cap, ZeroShotPlus.
# Verified 2026-04-08 via submission/metadata.json:54 and
# data/eval_outputs/eth-sri__SWT-bench_Verified_bm25_27k_zsp-test/.../metadata.json:54
# SWT-Bench is maintained by LogicStar AI (logic-star-ai/swt-bench), NeurIPS 2024.
# eth-sri publishes the canonical pre-formatted HF datasets.
# ---------------------------------------------------------------------------

SWT_BENCH_LITE_HF: str = "eth-sri/SWT-bench_Lite_bm25_27k_zsp"
SWT_BENCH_VERIFIED_HF: str = "eth-sri/SWT-bench_Verified_bm25_27k_zsp"
SWT_BENCH_DATASET_SOURCE: str = (
    "eth-sri/SWT-bench (HF) <- logic-star-ai/swt-bench (GitHub upstream)"
)

# Pinned SWT-bench vendor commit (cloned by scripts/setup_vendor.sh)
SWT_BENCH_PIN_COMMIT: str = "main"  # Will be pinned to exact SHA after first clone
VENDOR_DIR: Path = Path(__file__).resolve().parent / ".vendor" / "SWT-bench"
