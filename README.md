# PE-DD-Acquisition-AI-Advisor-Model

This fine-tuned, reproducible testbed model sits at the intersection of private equity and model security. It is specialized in: 
1. Data Rights Assessment - Evaluates if training data rights are clear for valuation
2. Acquisition Due Diligence - Simulates PE/Strategic AI diligence scenarios
4. Model Security Analysis - Measures adversarial extraction vulnerability i.e. how much of that model an adversary can steal from its outputs alone, including its refusal to answer when rights are unclear?

This mirrors how human PE teams evaluate whether an AI startup's data assets can actually be transferred and integrated in an acquisition, making it a practical tool for simulating real acquisition decision-making.

The advisor maps a data asset's characteristics to:
1. A grade defining risks from a legal/rights perspective (the rights-risk grade)
2. How valuable or commercially useful it is (a value tier), 
3. A recommended action, and 
4. A mandatory reason code - providing a standardized reason for the assessment or recommended action.

It refuses (abstains) when the rights evidence is missing or contradictory. Because such a model would sit behind an API, the testbed also measures model extraction and whether the refusal behaviour survives being copied.
[![tests](https://github.com/JASONW26327/adapter-extraction-testbed/actions/workflows/ci.yml/badge.svg)](https://github.com/JASONW26327/adapter-extraction-testbed/actions/workflows/ci.yml)

> **On results.** Every number in this repository comes from a file in `results/`. Committed so far:
> - **Full 258-case zero-shot evals** for DistilGPT-2, Gemma 3 4B-IT, and Gemma 3 12B-IT
> - **DistilGPT-2 teacher + extraction sweep** (6 query budgets + defense comparison), run on CPU
> - Small **feasibility samples** for Gemma 4 12B (not viable on CPU — thinking mode consumes all tokens)
> No numbers are written by hand. `docs/SECURITY_RESEARCH_REPORT.md` gives the full method and states plainly that low zero-shot accuracy against a hidden rule is the expected outcome, not a model weakness. The meaningful zero-shot signal is **schema conformance** — whether the model emits valid `RIGHTS=... TIER=... ACT=...;`.

## What it does

A LoRA adapter is fine-tuned on a **1019-record corpus** (`data/corpus.jsonl`) generated from a documented decision rule (`decide()` in `src/corpus_builder.py`), so the model's ground-truth decision function is known exactly. The testbed then asks:

1. **How much can an adversary steal?** An attacker queries the adapter and trains a *student* adapter on only the replies. How many queries until the student matches the teacher on held-out cases?
2. **Does the refusal survive theft?** Does the student also learn to abstain on the same inputs the teacher abstains on?
3. **Can a defense help?** Does rate-limited output perturbation reduce extraction fidelity without destroying task utility?

## Model Architecture

| Component | Specification |
|---|---|
| Adaptation | LoRA (Low-Rank Adaptation) |
| Rank | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Optimizer | AdamW |
| Learning Rate | 5e-3 (teacher), 1e-2 (student) |
| Training Steps | 150 (configurable) |

## Output Format

```
EVIDENCE=<LEVEL> RIGHTS=<RISK> TIER=<TIER> ACT=<ACTION> WHY=<REASON>;
```

The action is one of:
- **AIBUYER** — Strong strategic acquisition for a major AI company
- **SPONSOR** — Deal needs PE involvement
- **RENEG** — Data rights issues prevent clean transfer
- **ABSTAIN** — Insufficient evidence for valuation

Each output requires a reason code that traces back to source evidence, making the model's diligence process auditable.

## Backbones

**Gemma 3 is the primary backbone.** It is a modern instruction-tuned transformer and is the model this project is designed around. There are two ways to evaluate it:

- **Zero-shot instruction-following**, via Ollama, no training required:
  ```bash
  # Full 258-case eval (~27 min on CPU)
  python src/run_model_eval.py --ollama gemma3:4b --label gemma3-4b-full \
      --out results/metrics_gemma3_4b_ollama_full.json
  ```
  This measures how well a real model follows the diligence rule from the prompt alone.

- **LoRA fine-tune and extraction**, on a GPU with 4-bit QLoRA (~16GB VRAM):
  ```bash
  python src/testbed.py --stage all --model google/gemma-3-4b-it --load-4bit
  ```
  This runs the full security experiment (teacher + extraction + defense).

**DistilGPT-2 (82M) is the reference backbone.** It is small enough to run on a CPU with no GPU and no gated download, so the repository runs out of the box and CI can verify it. Full teacher + extraction results committed. It is a testbed backbone, not evidence about frontier models.

`src/testbed.py` maps LoRA target modules per architecture (GPT-2, Llama, Qwen 2/3, Mistral, Gemma 2/3/4, Phi-3) and raises rather than guessing for an unmapped one. Gemma 3 1B uses `model_type=gemma3_text` (text-only); 4B/12B use `gemma3` (multimodal). Both map to the same LoRA targets.

## Corpus

**1019 records**, reproducible from a seed (CI rebuilds it and checks the SHA-256):

| Part | Records | Source |
|---|---:|---|
| Synthetic body | 719 | Rule-generated |
| Contrast pairs | 180 | Generated: twins that flip the decision across one feature |
| Real anchors | 20 | Verified public deals, held out of training |
| Adversarial | 100 | Hand-written pressure prompts across 10 attack types, held out |

No real company, contract, or client record is used in training, and no monetary figure is produced anywhere. The 20 real anchors carry named buyer, target, seller and route with a public source each. See `docs/ANCHOR_DEALS.md`.

## Quick start

Python 3.10 or later. CPU is sufficient for the reference backbone.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
make corpus    # generates data/corpus.jsonl (1019 records)
make test      # 19 checks pass
make results   # DistilGPT-2 teacher + extraction, writes results/metrics_distilgpt2.json
```

For Gemma 3 on a GPU, add `pip install -r requirements-gpu.txt` and use the Gemma commands above.

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

## Key Findings

### DistilGPT-2 teacher + extraction (committed, `results/metrics_distilgpt2.json`)

| Query Budget (K) | Student-teacher fidelity |
|---:|---:|
| 2 | 0.225 |
| 4 | 0.406 |
| 6 | 0.239 |
| 8 | 0.645 |
| 16 | 0.681 |
| 32 | 0.246 |
| 8 + defense (15%) | 0.543 |

- Teacher achieves 91.3% risk accuracy on test but **0% schema conformance** — it learns the rule but cannot emit valid output format.
- Fidelity peaks at K=16 (0.681), drops at K=32 (overfitting — loss 0.264 vs 0.162 at K=16).
- Defense (15% perturbation) drops fidelity from 0.645 to 0.543.

### Zero-shot model comparison (full 258-case evals)

| Model | Schema % | Risk Acc | Action Acc | Abstain Recall | Time |
|---|---:|---:|---:|---:|---:|
| DistilGPT-2 (82M) | 0% | 0% | 26% | 40% | 505s |
| Gemma 3 4B-IT (4.3B) | **100%** | 48% | 25% | 24% | 1606s |
| Gemma 3 12B-IT (12.2B) | **99.6%** | 28% | **45%** | **54%** | 2199s |
| Gemma 4 12B (not viable on CPU) | 0% | — | — | — | — |

- **Schema conformance is the key zero-shot signal** — Gemma models emit valid format, DistilGPT-2 does not.
- **12B > 4B on action accuracy** (45% vs 25%) and **adversarial abstention** (54% vs 24% recall).
- **Gemma 4 12B** is not viable on CPU — its thinking mode consumes all output tokens before emitting text.

## Reproducing the results

```bash
# Zero-shot (no training, CPU)
python src/run_model_eval.py --hf distilgpt2 --label distilgpt2-full \
    --out results/metrics_distilgpt2_full.json
python src/run_model_eval.py --ollama gemma3:4b --label gemma3-4b-full \
    --out results/metrics_gemma3_4b_ollama_full.json
python src/run_model_eval.py --ollama gemma3:12b --label gemma3-12b-full \
    --out results/metrics_gemma3_12b_ollama_full.json

# Full teacher + extraction (DistilGPT-2, CPU, ~17 min)
python src/testbed.py --stage all --model distilgpt2 \
    --teacher-steps 150 --student-steps 120 --out results/metrics_distilgpt2.json

# Gemma 3 4B/12B full fine-tune + extraction (GPU, ~16GB+ VRAM)
python src/testbed.py --stage all --model google/gemma-3-4b-it --load-4bit \
    --out results/metrics_gemma3_4b.json
```

Each command writes a JSON file to `results/`. `docs/SECURITY_RESEARCH_REPORT.md` explains what to look for in each.

## Threat model

Black-box, query-only: the attacker observes the structured `EVIDANCE=... RIGHTS=... TIER=... ACT=... WHY=...;` output for arbitrary inputs, not the weights. The attacker knows the schema and that the rule is deterministic, but not the rule's contents. The goal is to reproduce the model's behaviour, especially the abstention behaviour.

## Security Implications

1. **Structured outputs do not prevent extraction** — schema provides no protection.
2. **Refusal is transferable** — abstention behaviour can be copied to a student.
3. **Query budget threshold** — fidelity plateaus then drops (K=32 overfits), so monitoring query volume is a practical control.
4. **Defense works partially** — 15% output perturbation reduces fidelity by 10 points.
5. **Format gap** — the teacher learns the rule (91.3% accuracy) but cannot emit valid schema (0%). This makes extraction a format-learning problem, not a rule-learning one.

## Tests

```bash
pip install pytest && pytest tests/ -v
```

19 checks: rule determinism, split leakage, agreement between stored labels and the rule, reason-code completeness, contrast-pair construction, class balance, and holdout discipline. CI additionally rebuilds the corpus from its seed and checks the hash.

## Limitations

Read `docs/LIMITATIONS.md` before quoting a number. In brief: the rule is a legible rubric, not a calibrated diligence instrument; the corpus is generated; deterministic rule (teacher loss approaches zero); single-turn queries; committed measurement spans DistilGPT-2 teacher+extraction and full Gemma 3 4B/12B zero-shot, but Gemma teacher+extraction runs require a GPU and are not yet committed.

## File layout

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── preflight.py
├── src/
│   ├── testbed.py           # Main experiment harness
│   ├── corpus_builder.py    # Dataset generation
│   └── run_model_eval.py    # Zero-shot evaluation (Ollama / HuggingFace)
├── data/
│   └── corpus.jsonl         # 1019-record corpus
├── docs/
│   ├── SECURITY_RESEARCH_REPORT.md  # Full report with results tables
│   ├── ANCHOR_DEALS.md
│   ├── LIMITATIONS.md
│   └── RUNNING.md
├── results/                 # Committed experiment outputs
├── tests/
│   └── test_corpus.py      # 19 pytest checks
└── Makefile
```

## Citation

See `CITATION.cff`.

## Contact

Jason Wong, [@JASONW26327](https://github.com/JASONW26327), work.jasonwong@gmail.com

## Licence

MIT, see `LICENSE`.


