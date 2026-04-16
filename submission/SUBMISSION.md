# SWT-Bench Lite Submission: Jina Test + GPT-5.4

## Method Name

Jina Test + GPT-5.4 (BoN×K=10 + CXV×J=3 + frequency-rank selector)

## Approach

Jina Test is a custom agentic harness built on top of OpenHands 1.16.1 with a
cross-validation proxy oracle (CXV) pipeline for automated patch selection.

### Pipeline

1. **BoN Test Generation**: Generate K=10 test candidates using 5 heterogeneous
   prompt strategies (direct, snippet, assertflip, contract, exception) × 2 temperatures
   (0.0, 0.6) via Jina Test Agent + GPT-5.4 (`openai/gpt-5.4-2026-03-05`).

2. **Fix Generation**: Generate J=3 fix candidates at temperatures (0.0, 0.5, 1.0)
   using a dedicated fix-agent prompt via Jina Test Agent + GPT-5.4.

3. **CXV Matrix**: Build a K×J = 30-cell cross-validation matrix. Each cell applies
   test_k + fix_j to a Docker sandbox and classifies the outcome using a 10-verdict
   regex classifier (GOOD_FAIL, UNEXPECTED_PASS, WRONG_FAIL, TIMEOUT, COMPILE_ERROR,
   DESELECTED, XFAIL, XPASS, NO_TEST, and OTHER). A test "kills" a fix when both
   applied produces GOOD_FAIL (the test catches a regression in the fix).

4. **Selection**: The test with the highest kill rate (most discriminating) is
   selected as the proxy oracle. The fix that survives this test is selected as the
   best candidate.

### Key Design Decisions

- **Heterogeneous test strategies**: 5 prompt templates targeting different test-writing
  approaches, increasing coverage diversity.
- **10-verdict classifier**: Handles pytest, Django unittest (`./tests/runtests.py`),
  and sympy (`bin/test`) output formats natively.
- **Frequency-rank selector**: Picks the test with highest kill rate across all fixes,
  then selects the surviving fix.

## Models

- **Test generation**: `openai/gpt-5.4-2026-03-05` (via `/v1/responses` API)
- **Fix generation**: `openai/gpt-5.4-2026-03-05` (via `/v1/responses` API)
- **Verdict classification**: Pure regex (no LLM)
- **Selection**: Deterministic frequency-rank algorithm (no LLM)

## Baseline

175/276 instances resolved using Jina Test + GPT-5.4 (single-pass inference).

## CXV-Guided Selection Results

- **Selector subset**: 13/24 resolved (54%)
- **Total new resolves**: **18**
- **Combined: 193/276 = 69.9%**

## Locally Determined Performance

- **Baseline**: 175/276 = 63.4%
- **Baseline + CXV selection**: 193/276 = 69.9% (+6.5pp lift, +18 new resolves)
- **Evaluation method**: standard SWE-bench Lite harness with `princeton-nlp/SWE-bench_Lite`

## Resolved Instance IDs (18 new, beyond baseline)

django__django-12470, django__django-12856, django__django-13447,
django__django-13933, django__django-14580, django__django-15781,
django__django-16408, matplotlib__matplotlib-23964,
matplotlib__matplotlib-23987, matplotlib__matplotlib-25079,
matplotlib__matplotlib-25311, sphinx-doc__sphinx-8435,
sympy__sympy-12236, sympy__sympy-15346, sympy__sympy-17022,
sympy__sympy-17139, sympy__sympy-18057, sympy__sympy-21612

## Infrastructure

- **VM**: GCP e2-highmem-16 (16 vCPU, 128GB RAM), us-central1-a
- **Wall-clock time**: ~32h total
- **Estimated cost**: ~$500 (API calls) + ~$100 (compute)

## Reproduction

1. Set up Jina Test harness with CXV pipeline
2. Set up GCP VM with Docker dependencies
3. Configure `OPENAI_API_KEY` for GPT-5.4 access
4. Execute the CXV pipeline on the target instance set
5. Merge CXV-selected predictions with the baseline prediction set
6. Evaluate with the SWT-bench environment
