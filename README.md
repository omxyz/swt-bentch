# Cross-Validation Proxy Oracle for Automated Bug Fix Selection

**69.9% (193/276)** on SWT-Bench Lite unit-test mode | +6.5pp over baseline

## Results

| Method | J | Patched | Resolved | Rate | P→R |
|---|---|---|---|---|---|
| e-Otter++† | 40 | -- | 145 | 52.5% | -- |
| Baseline (OpenHands + GPT-5.4) | 1 | 199 | 175 | 63.4% | 87.9% |
| **Jina Test + GPT-5.4** | **3** | **234** | **193** | **69.9%** | **82.5%** |

J = fix candidates. Patched = candidate patches. P→R = resolved/patched. †Prior published result.

**Per-repository breakdown:**

| Repository | Baseline | Jina Test |
|---|---|---|
| django/django | 89 | 96 |
| sympy/sympy | 49 | 55 |
| matplotlib/matplotlib | 6 | 10 |
| sphinx-doc/sphinx | 1 | 2 |
| Others (6 repos) | 30 | 30 |
| **Total** | **175** | **193** |

## Method

Jina Test generates K test candidates and J fix candidates, evaluates all K×J pairs in Docker sandboxes, and selects the fix that survives the most discriminating test.

**Pipeline:**
1. **Test Generation** — K=10 candidates via 5 prompt strategies (direct, snippet, assertflip, contract, exception) × 2 temperatures
2. **Fix Generation** — J=3 candidates at temperatures (0.0, 0.5, 1.0) via dedicated fix-agent prompt
3. **CXV Matrix** — 30-cell cross-validation matrix with 10-verdict regex classifier
4. **Frequency-Rank Selection** — test with highest kill rate becomes proxy oracle; fix that survives it becomes the output patch

Implementation built on OpenHands 1.16.1. Model: `openai/gpt-5.4-2026-03-05`. Verdict classification and selection are deterministic (no LLM).

## Key Findings

- **Multi-candidate generation is a major driver.** A naive "always pick fix 0" strategy resolves 45.8% of attempted instances. CXV selection raises this to 54.2%, giving +8.4pp realized selector lift.
- **Test generation quality is high leverage.** 18.2% of CXV cells produce actionable signal, leaving clear room for stronger generated tests.
- **All winners used `direct` at temperature 0.0.** The four alternative prompt strategies never won, contrasting with e-Otter++'s reported gains from heterogeneous prompting.

## Repository Structure

```
├── code/                          # Reproduction code
│   ├── runner.py                  # Pipeline orchestrator
│   ├── bon_runner.py              # OpenHands Agent wrapper
│   ├── cxv.py                     # K×J matrix builder
│   ├── verdict.py                 # 10-verdict regex classifier
│   ├── selector.py                # Frequency-rank selection
│   ├── instance_sets.py           # Instance set loaders
│   ├── submit.py                  # JSONL packaging
│   └── prompts/                   # 5 test strategies + fix-agent
├── submission/                    # Jina Test + GPT-5.4 (193/276 = 69.9%)
├── arxiv/                         # Paper (PRIMEarxiv template)
│   ├── main.tex
│   ├── main.pdf
│   └── references.bib
├── arxiv/                         # Paper (main.tex, main.pdf, references.bib)
└── SETUP.md                       # Reproduction and evaluation guide
```

Data files (`.gitignore`d, available on request) are stored locally under `data/`.

## Reproduction

See **[SETUP.md](SETUP.md)** for detailed instructions covering three workflows:

1. **Reproduce predictions** — run the full pipeline (OpenHands + CXV) from scratch
2. **Evaluate our predictions** — verify the leaderboard score using the official [logic-star-ai/swt-bench](https://github.com/logic-star-ai/swt-bench) harness
3. **Fast local eval** — directional check with the SWE-bench harness

**Prerequisites:** OpenAI API key (GPT-5.4), Docker, VM with 16+ vCPU and 64+ GB RAM (e2-highmem-16 recommended).

## Citation

```bibtex
@article{kim2026cxv,
  title={Cross-Validation Proxy Oracle for Automated Bug Fix Selection},
  author={Kim, Keon and Chelikavada, Krish},
  year={2026}
}
```
