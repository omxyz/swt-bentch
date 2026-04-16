Subject: SWT-Bench Lite Submission: Jina Test + GPT-5.4 (69.9%)

To: submit@swtbench.com

---

Hello SWT-Bench team,

We would like to submit our method for inclusion on the SWT-Bench Lite unit-test mode leaderboard.

**Method name:** Jina Test + GPT-5.4

**Locally determined performance:** 193/276 = 69.9% resolved

**Approach:**
Jina Test is a cross-validation proxy oracle pipeline built on OpenHands 1.16.1.
For each instance it generates K=10 test candidates (5 prompt strategies × 2 temperatures)
and J=3 fix candidates, evaluates all K×J pairs in Docker sandboxes using a 10-verdict
regex classifier, and selects the fix that survives the most discriminating test
(frequency-rank selection). Verdict classification and selection are deterministic.
Base model: openai/gpt-5.4-2026-03-05.

**Project homepage and traces:**
https://github.com/omxyz/swt-bentch

The repository contains:
- `submission/predictions.jsonl`: the 276-instance JSONL we are submitting
- `trajectories/`: per-instance agent event traces for all 18 newly resolved instances
  (includes 10 test candidates, 3 fix candidates, 30 CXV cell verdicts, final patch,
  and agent step-by-step events)
- `code/`: full pipeline implementation (runner, bon_runner, cxv, verdict, selector, prompts)
- `arxiv/main.pdf`: technical report with honest selector decomposition analysis
- `README.md`, `SETUP.md`: overview and three reproduction workflows

**Reproduction:**
See SETUP.md in the repo. Summary:
1. Set OPENAI_API_KEY to a GPT-5.4-enabled key
2. Graft `code/` into an OpenHands benchmarks checkout
3. Run: `python -m evaluation.benchmarks.swt_bench_v1.runner --phase 2 --track lite --instance-set failing100 --num-workers 8`
4. Merge with baseline predictions
5. Evaluate with `python -m src.main --dataset_name princeton-nlp/SWE-bench_Lite --predictions_path <...> --filter_swt --max_workers 8 --run_id jina_test`

Infrastructure used: GCP e2-highmem-16 (16 vCPU, 128 GB RAM). Mean execution time
~788 s per instance.

**Notes on honest framing:**
Our technical report decomposes the 69.9% score. The 175 already-resolved baseline
instances use vanilla OpenHands + GPT-5.4 single-pass predictions. The 18 new
resolutions come from the CXV pipeline applied to the 100 failing instances.
On the 24-instance selector subset, a naive "pick fix 0" strategy resolves 11,
while CXV selection resolves 13 — the realized selector lift is +8.4pp. The
primary driver of the gain is multi-candidate fix generation, with CXV providing
incremental but real selection benefit.

Happy to answer any questions or provide additional artifacts.

Thanks,
Keon Kim, Krish Chelikavada
Om Labs
{keon,krish}@omlabs.xyz
