# adapter-extraction-testbed

A reproducible testbed for a PE (private equity) acquisition advisor model, and for measuring how much of that model an adversary can steal from its outputs alone.

The advisor maps a data asset's characteristics to a rights-risk grade, a value tier, a recommended action, and a mandatory reason code, and refuses (abstains) when the rights evidence is missing or contradictory. Because such a model would sit behind an API, the testbed also measures model extraction and whether the refusal behaviour survives being copied.

[![tests](https://github.com/JASONW26327/adapter-extraction-testbed/actions/workflows/ci.yml/badge.svg)](https://github.com/JASONW26327/adapter-extraction-testbed/actions/workflows/ci.yml)

> **On results.** Every number in this repository comes from a file in `results/`. This revision commits the DistilGPT-2 baseline only; the teacher, extraction, adversarial, defense and Gemma runs are produced by the commands in [Reproducing the results](#reproducing-the-results). Nothing is written in by hand. `docs/SECURITY_RESEARCH_REPORT.md` gives the full method.

## What it does

A LoRA adapter is fine-tuned on a **1019-record corpus** (`data/corpus.jsonl`) generated from a documented decision rule (`decide()` in `src/corpus_builder.py`), so the model's ground-truth decision function is known exactly. The testbed then asks:

1. **How much can an adversary steal?** An attacker queries the adapter and trains a *student* adapter on only the replies. How many queries until the student matches the teacher on held-out cases?
2. **Does the refusal survive theft?** Does the student also learn to abstain on the same inputs the teacher abstains on?
3. **Can a defense help?** Does rate-limited output perturbation reduce extraction fidelity without destroying task utility?

## Backbones

**Gemma 3 / Gemma 4 is the primary backbone.** It is a modern instruction-tuned transformer and is the model this project is designed around. There are two ways to run it, for two different questions:

- **Zero-shot instruction-following**, via Ollama, no training required:
  ```bash
  python src/run_model_eval.py --ollama gemma4:12b --label gemma4-12b
  ```
  This measures how well a real model follows the diligence rule from the prompt alone.
- **LoRA fine-tune and extraction**, on a GPU with 4-bit QLoRA (~16GB VRAM):
  ```bash
  python src/testbed.py --stage all --model google/gemma-3-4b-it --load-4bit
  ```
  This runs the full security experiment.

**DistilGPT-2 (82M) is the reference backbone.** It is small enough to run on a CPU with no GPU and no gated download, so the repository runs out of the box and CI can verify it. The committed metrics in `results/` are produced on DistilGPT-2. It is a testbed backbone, not evidence about frontier models.

`src/testbed.py` maps LoRA target modules per architecture (GPT-2, Llama, Qwen 2/3, Mistral, Gemma 2/3/4, Phi-3) and raises rather than guessing for an unmapped one, because a wrong target module trains nothing while still reporting a falling loss. Gemma 3 1B uses `model_type=gemma3_text` (text-only); 4B/12B use `gemma3` (multimodal). Both map to the same LoRA targets.

## Corpus

**1019 records**, reproducible from a seed (CI rebuilds it and checks the SHA-256):

| Part | Records | Source |
|---|---:|---|
| Synthetic body | 719 | Rule-generated |
| Contrast pairs | 180 | Generated: twins that flip the decision across one feature |
| Real anchors | 20 | Verified public deals, held out of training |
| Adversarial | 100 | Hand-written pressure prompts across 10 attack types, held out |

No real company, contract, or client record is used in training, and no monetary figure is produced anywhere. The 20 real anchors carry named buyer, target, seller and route with a public source each; their feature values are an analyst's reading of the public record and are marked for sign-off. See `docs/ANCHOR_DEALS.md`.

## Quick start

Python 3.10 or later. CPU is sufficient for the reference backbone.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
make corpus    # generates data/corpus.jsonl (1019 records)
make test      # 19 checks: schema, rule consistency, split integrity
make results   # full DistilGPT-2 pipeline, writes results/metrics_distilgpt2.json
```

For the primary backbone on a GPU, add `pip install -r requirements-gpu.txt` and use the Gemma commands above.

## Stages

| Stage | What it does |
|---|---|
| `baseline` | Zero-shot performance of the unmodified base model |
| `teacher` | Fine-tune a LoRA adapter on the train split, evaluate on test |
| `adversarial` | 100 categorised prompts, measure abstention hold rate |
| `extract` | Train student adapters at budgets K in {2, 4, 6, 8, 16, 32} |
| `defense` | Extraction with output perturbation (`--defense-rate N`) |
| `seeds` | Multi-seed replication (`--seeds N`) for statistical robustness |

Useful flags: `--max-new-tokens` (decode length; the output schema is ~30 tokens), `--teacher-steps`, `--student-steps`, `--load-4bit`.

## Reproducing the results

```bash
# Reference backbone (DistilGPT-2, CPU, ~15 min on a normal laptop)
python src/testbed.py --stage all --model distilgpt2 \
    --teacher-steps 150 --student-steps 120 --out results/metrics_distilgpt2.json

# Primary backbone (Gemma), zero-shot via Ollama
python src/run_model_eval.py --ollama gemma4:12b --label gemma4-12b \
    --out results/metrics_gemma4_12b.json

# Primary backbone (Gemma 3 4B), full fine-tune + extraction on a GPU
python src/testbed.py --stage all --model google/gemma-3-4b-it --load-4bit \
    --out results/metrics_gemma3_4b.json
```

Each command writes a JSON file to `results/`. `docs/SECURITY_RESEARCH_REPORT.md` explains what to look for in each.

## Threat model

Black-box, query-only: the attacker observes the structured `EVIDENCE=… RIGHTS=… TIER=… ACT=… WHY=…;` output for arbitrary inputs, not the weights. The attacker knows the schema and that the rule is deterministic, but not the rule's contents. The goal is to reproduce the model's behaviour, especially the abstention behaviour.

## Tests

```bash
pip install pytest && pytest tests/ -v
```

19 checks: rule determinism, split leakage, agreement between stored labels and the rule, reason-code completeness, contrast-pair construction, class balance, and holdout discipline. CI additionally rebuilds the corpus from its seed and checks the hash.

## Limitations

Read `docs/LIMITATIONS.md` before quoting a number. In brief: the rule is a legible rubric, not a calibrated diligence instrument, and no real transaction has been scored against an outcome; a small backbone on a generated rule is a testbed, not evidence about frontier models; compare query budgets only where students have converged; committed measurement is currently the DistilGPT-2 baseline only.

## Citation

See `CITATION.cff`.

## Contact

Jason Wong, [@JASONW26327](https://github.com/JASONW26327), work.jasonwong@gmail.com

## Licence

MIT, see `LICENSE`. The generated corpus carries the same licence. Deal facts for the 20 real anchors come from the public sources cited in `docs/ANCHOR_DEALS.md`.
