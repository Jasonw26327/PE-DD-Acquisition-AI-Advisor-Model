# adapter-extraction-testbed

A reproducible testbed for measuring model extraction and abstention-transfer
against a task-specific LoRA adapter, with an experimental defense mechanism.

## What it does

A PE (private equity) acquisition advisor model is fine-tuned with LoRA on a
**927-record corpus** (`data/corpus.jsonl`). The corpus is generated from a
documented decision rule (`decide()` in `src/corpus_builder.py`), so the model's
ground-truth decision function is known exactly rather than asserted. The
testbed then asks three security questions:

1. **How much can an adversary steal?** An attacker queries the adapter and
   trains a *student* adapter on only the replies. How many queries until the
   student matches the teacher on held-out cases?
2. **Does the refusal survive theft?** Does the student also learn to abstain on
   the same inputs the teacher abstains on?
3. **Can a defense help?** A rate-limited output-perturbation defense degrades
   structured outputs above a query budget — does this meaningfully reduce
   extraction fidelity?

## Key design decisions

- **Two backbones**: DistilGPT-2 (82M, CPU-friendly) and **Gemma 3 4B-IT**
  (modern transformer). Comparing across architectures tests whether extraction
  is a general phenomenon or specific to one model family.
- **927-record corpus**: 719 synthetic (rule-generated) + 180 contrast pairs
  (twin cases that flip the decision across one feature) + 20 real anchors
  (verified public deals) + 100 categorized adversarial prompts.
- **Adversarial suite expanded from 8 → 100**: Each prompt targets an abstaining
  feature config and tries to override the refusal, categorized across 10 attack
  types (direct extraction, rule reconstruction, prompt injection, instruction
  hierarchy, refusal bypass, context manipulation, label flipping, reason-code
  extraction, boundary probing, multi-turn extraction).
- **Provenance**: Every run records a corpus SHA-256 prefix, adapter hash,
  dataset version, and split sizes so results are reproducible.
- **Defense experiment**: `--defense-rate N` perturbs one output field at rate N
  to measure extraction fidelity degradation.

## Quick start

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install openpyxl  # optional, for the review spreadsheet

make corpus    # generates data/corpus.jsonl (927 records)
make test      # validates schema, rule consistency, split integrity
make run       # full pipeline: baseline → teacher → adversarial → extract
```

For GPU/QLoRA:
```bash
pip install -r requirements-gpu.txt
python src/testbed.py --stage teacher --model google/gemma-3-4b-it --load-4bit
```

## Stages

| Stage | What it does |
|-------|-------------|
| `baseline` | Zero-shot performance of the unmodified base model |
| `teacher` | Fine-tune LoRA adapter on the train split, evaluate on test |
| `adversarial` | 100 categorized prompts, measure abstention hold rate |
| `extract` | Steal student adapters at budgets K ∈ {2, 4, 6, 8, 16, 32} |
| `seats` | Multi-seed replication (5 seeds) for statistical robustness |
| `defense` | Extraction with output perturbation defense |

## Results

See `results/metrics_distilgpt2_*.json` for pre-computed metrics on the
DistilGPT-2 backbone. To reproduce and add Gemma 3 4B:

```bash
python src/testbed.py --stage all --out metrics_gemma3_4b.json \
    --model google/gemma-3-4b-it --load-4bit --teacher-steps 150 --student-steps 120
```

## Threat model

This testbed assumes an attacker who can query the model's output for arbitrary
inputs and observes the structured `RISK=… TIER=… ACT=…` response, not the
weights. The adversary's goal is to reproduce the **abstention behaviour**
(the hardest part to learn and the most critical for safety) as well as the
scoring tiers. This is a white-box *algorithm* threat model: the attacker knows
the corpus exists and the rule is deterministic, but not the rule's contents.

## Citation

```
@software{wong_adapter-extraction-testbed,
  author = {Wong, Jason},
  title = {adapter-extraction-testbed},
  year = {2026},
  url = {https://github.com/JASONW26327/adapter-extraction-testbed}
}
```

See `CITATION.cff` for the full record.
