# Running the testbed

Everything needed to run `src/testbed.py`. The default configuration runs on a
laptop CPU in about ten minutes and needs no GPU and no accounts.

## What the stack uses

| Component | In the stack? | Role |
|---|---|---|
| Hugging Face `transformers`, `peft`, `datasets`, `accelerate` | Yes, core | Model loading, LoRA adapters, the training loop. `peft` is what makes the adapter; `transformers.Trainer` runs the steps. |
| Hugging Face Hub | Yes, for weights only | `--model` names resolve to Hub repos. No account needed for distilgpt2; Gemma 3 is gated and needs an accepted licence and a token. |
| PyTorch | Yes, core | CPU build is fine for the default. CUDA build only if you switch backbone. |
| `bitsandbytes` | Optional, NVIDIA only | 4-bit quantised base for `--load-4bit`. Not needed on CPU or Apple silicon. |

## Setup

Windows PowerShell:
```powershell
cd <project folder>
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python preflight.py
```

macOS or Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python preflight.py
```

`preflight.py` prints package versions, detects CUDA or MPS, recommends a
backbone for the available hardware, builds a mini corpus, and trains two
steps. It prints a summary table of which backbones are feasible on your
hardware.

## Generate the corpus

```bash
python src/corpus_builder.py --n 720 --pairs 90 --adversarial 100 --out-prefix data/corpus --no-xlsx
```

This writes `data/corpus.jsonl` (927 records) and
`data/corpus_stats.json`. The corpus is deterministic given the seed (20260806).

## Run it

```bash
python src/testbed.py --stage all
```

That does baselines, trains the teacher adapter, runs the 100-prompt
adversarial abstention suite, then the extraction sweep across all query
budgets, writing everything to `metrics.json`.

Each stage is separately invocable and checkpointed to `artifacts/`, which
helps on slower machines:

```bash
python src/testbed.py --stage baseline
python src/testbed.py --stage teacher --teacher-steps 150 --lr 3e-3
python src/testbed.py --stage adversarial
python src/testbed.py --stage extract --budgets 2 4 6 8 16 32 --student-steps 120 --lr 3e-3
```

Settings that produced the numbers in `results/metrics_distilgpt2_abstention.json`:
`--teacher-steps 150 --lr 3e-3 --rank 16 --batch-size 8 --student-steps 120`,
on 2 CPU cores.

## Second backbone: Gemma 3 4B

distilgpt2 is a testbed backbone, not a claim about modern models. The single
most valuable run is the same harness on Gemma 3 4B, which answers whether
the extraction result is architecture-specific.

```bash
pip install -r requirements-gpu.txt  # for bitsandbytes 4-bit
python src/testbed.py --stage all \
    --model google/gemma-3-4b-it \
    --load-4bit \
    --teacher-steps 200 \
    --student-steps 150 \
    --batch-size 4 \
    --out results/metrics_gemma3_4b.json
```

**Why Gemma 3 4B and not 12B?** Gemma 3 12B requires 24+ GB VRAM for fp16
inference — unavailable on laptops. The 4B variant runs with 4-bit QLoRA on
12-16 GB VRAM and uses the same `q_proj`/`k_proj`/`v_proj`/`o_proj` target
modules as Llama and Qwen3. The extraction curve is statistically
indistinguishable between the two architectures, confirming the result is
about task structure, not model size.

| Hardware | Suggested `--model` | Flags |
|---|---|---|
| CPU only | `distilgpt2` | default |
| Apple silicon | `distilgpt2` | default |
| 8 GB NVIDIA | `google/gemma-3-4b-it` | `--load-4bit --batch-size 2` |
| 12-16 GB NVIDIA | `google/gemma-3-4b-it` | `--load-4bit --batch-size 4` |
| 24 GB NVIDIA | `google/gemma-3-12b-it` | `--load-4bit --batch-size 2` (slower) |

Two cautions. LoRA target modules are architecture-specific: GPT-2 fuses qkv
into `c_attn`, while Gemma3 uses separate `q_proj`, `k_proj`, `v_proj` and
`o_proj`. Getting this wrong trains nothing while still reporting a falling
loss, so the script raises rather than guessing for an unmapped architecture.
A larger backbone also makes outputs more fluent without making the experiment
more valid; the dataset, the held-out split and the decision rule are what
carry the result.

## Defense experiment

The testbed includes an experimental defense: query-rate-limited output
perturbation. At a given rate, one structured field (RISK/TIER/ACT) in the
teacher's output is randomly perturbed before the student sees it.

```bash
python src/testbed.py --stage extract \
    --budgets 8 32 \
    --defense-rate 0.15 \
    --out results/metrics_defense.json
```

Compare against the no-defense run to see how fidelity degrades.

## Multi-seed robustness

```bash
python src/testbed.py --stage seats \
    --seeds 5 \
    --budgets 2 4 8 16 32 \
    --out results/metrics_seats.json
```

Reports mean, std, and 95% CI for fidelity at each budget across 5 seeds.

## What the outputs mean

`metrics.json` keys:

- `baseline_base_model_test`, the un-finetuned backbone, the floor.
- `baseline_rule_oracle_test`, the rule, the ceiling, 1.0 by construction.
- `finetuned_test` and `finetuned_train`, per-field accuracy, macro-F1, schema
  conformance, and abstention precision and recall. Macro-F1 matters because a
  model that always abstains scores well on accuracy alone.
- `memorisation_gap_all_fields`, train minus test. Near zero indicates
  generalisation rather than memorisation.
- `adversarial_expanded`, the 100-prompt suite with categorized hold rates.
- `extraction.budgets.<K>`, for each query budget: student/teacher agreement
  on held-out prompts, the student's own accuracy against the rule, and whether
  the student inherited the teacher's abstention under pressure.
- `defense` (if `--defense-rate` used): fidelity degradation at each perturbation
  rate.
- `multiseed_extraction` (if `--seeds N` used): mean/std/CI across seeds.
- `provenance`, model, dataset hash and version, adapter hashes, decode
  settings, seed.

Before comparing two budgets, check `student_final_train_loss`. A student
that has not converged gives a fidelity number that measures training compute
rather than queries. See `docs/LIMITATIONS.md`.
