# Setup Guide

The Jina Test code is developed inside the OpenHands benchmarks repo and expects
that package layout. This guide shows two setups: **reproduction** (full pipeline)
and **evaluation-only** (using our predictions with the official SWT-Bench harness).

## Repositories

| Repo | Purpose | Required for |
|------|---------|--------------|
| [omxyz/swt-bentch](https://github.com/omxyz/swt-bentch) (this repo) | Code, predictions, paper | All |
| [All-Hands-AI/benchmarks](https://github.com/All-Hands-AI/benchmarks) | OpenHands agent runner | Reproduction |
| [logic-star-ai/swt-bench](https://github.com/logic-star-ai/swt-bench) | Official SWT-Bench eval harness | Leaderboard evaluation |

## Setup 1 — Reproduction (Generate Predictions)

Requires GPT-5.4 API access, Docker, and a beefy VM (16+ vCPU, 64+ GB RAM).

### Install

```bash
# 1. Clone the OpenHands benchmarks repo
git clone https://github.com/All-Hands-AI/benchmarks.git ~/benchmarks
cd ~/benchmarks
uv sync

# 2. Clone this repo
git clone https://github.com/omxyz/swt-bentch.git ~/swt-bentch

# 3. Graft our code into the benchmarks layout
mkdir -p ~/benchmarks/evaluation/benchmarks
cp -r ~/swt-bentch/code ~/benchmarks/evaluation/benchmarks/swt_bench_v1

# 4. Provide baseline report for instance set loaders
mkdir -p ~/swt-bench-root/submission
cp ~/swt-bentch/data/baseline/report.json ~/swt-bench-root/submission/report.json
export SWT_BENCH_REPO_ROOT=~/swt-bench-root

# 5. API key
export OPENAI_API_KEY=<your-key>

# 6. Run
cd ~/benchmarks
export PYTHONPATH=~/benchmarks:~/swt-bench-root
python -m evaluation.benchmarks.swt_bench_v1.runner \
  --phase 2 --track lite \
  --instance-set failing100 \
  --num-workers 8
```

Predictions land in `~/benchmarks/results/swt_bench_v1_*/predictions.jsonl`.

### Merge with Baseline

```python
import json
baseline = {json.loads(l)['instance_id']: json.loads(l)
            for l in open('data/baseline/predictions.jsonl')}
phase2 = {json.loads(l)['instance_id']: json.loads(l)
          for l in open('results/swt_bench_v1_option_e_phase2_lite_failing100/predictions.jsonl')}
merged = {**baseline, **phase2}
with open('submission/predictions.jsonl', 'w') as f:
    for p in merged.values():
        p['model_name_or_path'] = 'Jina Test + GPT-5.4'
        f.write(json.dumps(p) + '\n')
```

## Setup 2 — Evaluation Only (Use Our Predictions)

To verify our leaderboard score without regenerating predictions, use the
official SWT-Bench harness with our `submission/predictions.jsonl`.

### Install the Official SWT-Bench Harness

```bash
git clone https://github.com/logic-star-ai/swt-bench.git ~/swt-bench
cd ~/swt-bench

# Create a venv for the harness
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Run the Harness on Our Predictions

```bash
# Copy our predictions into the swt-bench dir
cp ~/swt-bentch/submission/predictions.jsonl ~/swt-bench/jina_test_predictions.jsonl

cd ~/swt-bench
python -m src.main \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path jina_test_predictions.jsonl \
  --max_workers 8 \
  --run_id jina_test
```

The harness writes `run_instance_swt_logs/jina_test.swtbench/<model>/<instance>/`
logs and produces a report at `GPT-5.4 + Jina Test.jina_test.json`.

### Read the Score

```bash
python -m src.report \
  run_instance_swt_logs/jina_test.swtbench/<model-name> \
  --dataset lite
```

Expected output (matching our locally determined score):
```
| Method             | Jina Test + GPT-5.4 |
| Applicability (W)  | ~85                 |
| Success Rate (S)   | 69.9                |
| F->P               | 69.9                |
```

## Setup 3 — Fast Evaluation Path (SWE-bench Harness)

The official swt-bench harness is slow (Docker per cell). For faster local
validation, use the SWE-bench harness directly:

```bash
pip install swebench
python -m swebench.harness.run_evaluation \
  -p submission/predictions.jsonl \
  -d princeton-nlp/SWE-bench_Lite \
  -id jina-test \
  --max_workers 8 \
  --report_dir ./eval_out \
  -t 900
```

**Caveat**: The SWE-bench harness uses different Docker images than SWT-Bench.
Patches valid under SWT-Bench may fail to apply (or vice versa). Use only for
directional verification; the leaderboard requires the SWT-Bench harness.

## Troubleshooting

### `ModuleNotFoundError: No module named 'evaluation'`
You are running `runner.py` outside the benchmarks layout. Use Setup 1
(graft into `~/benchmarks/evaluation/benchmarks/swt_bench_v1/`).

### `FileNotFoundError: predictions file not found`
The instance set loaders need the baseline report. Either:
- Set `SWT_BENCH_REPO_ROOT=<path-to-repo-with-submission/report.json>`, or
- Place `data/baseline/report.json` at the repo root

### Docker image mismatch errors in evaluation
This happens when the evaluation harness builds images against a different
commit of the SWT-Bench code than the one used for your predictions. Pin the
swt-bench repo to a specific commit if reproducing: `git checkout <sha>`.
