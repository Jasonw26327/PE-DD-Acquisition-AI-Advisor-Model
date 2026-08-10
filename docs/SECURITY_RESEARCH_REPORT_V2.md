# Security Research Report: Adapter Extraction and Abstention Transfer in a PE Acquisition Advisor Model

**Project:** adapter-extraction-testbed, schema `pea-3.0`
**Corpus:** 1019 records (719 synthetic + 180 contrast + 20 real anchors + 100 adversarial). The SHA-256 prefix is recorded in `data/corpus_stats.json` and re-checked by CI.
**Primary backbone (evaluated):** Gemma 3 4B-IT via Ollama (4.3B, Q4_K_M). Gemma 3 12B-IT and Gemma 4 12B are also supported but **not yet evaluated**.
**Reference backbone (evaluated):** DistilGPT-2 (82M), CPU-runnable.
**Status:** Zero-shot baselines committed for both Gemma 3 4B and DistilGPT-2. Teacher, extraction, defense, and multi-seed runs are **not yet committed** — reproduce via Section 6.

---

## 1. What this report does and does not claim

This document describes an experiment and how to run it. It contains no result that is not backed by a file in `results/`. Where a measurement has not yet been produced, the report says so rather than estimating it.

## 2. The question

A private-equity acquisition advisor model is fine-tuned to map a data asset's characteristics to a rights-risk grade, a value tier, a recommended action, and a mandatory reason code. The model is a commercial asset that would sit behind an API. Three security questions follow:

1. **Extraction.** An attacker queries the model and trains a *student* adapter on only the replies. How many queries until the student reproduces the model's decisions on held-out cases?
2. **Abstention transfer.** The rule requires the model to abstain when rights evidence is missing or self-contradictory. Does a stolen copy inherit that refusal, or does it answer where the original declined?
3. **Defense.** Does rate-limited output perturbation reduce extraction fidelity without destroying task utility?

## 3. Threat model

- **Access.** Black-box, query-only. The attacker sends any input and observes the structured output `EVIDANCE=... RIGHTS=... TIER=... ACT=... WHY=...;`. The weights are not visible.
- **Knowledge.** The attacker knows the feature schema and that the underlying rule is deterministic. They do not know the rule's contents or the LoRA weights.
- **Goal.** Reproduce the model's behaviour, especially the abstention behaviour, which is the part most relevant to safety and the hardest to learn.

This is a **black-box, query-only** setting. It is not white-box: no weights or gradients are exposed.

## 4. Method

- **Corpus.** 1019 records generated from `decide()` in `src/corpus_builder.py`, so the ground-truth decision function is known exactly. Split into 623 train, 138 validation, 138 test, plus 20 held-out real anchors and 100 held-out adversarial prompts. Stratified by action class; contrast pairs kept within one split so the decision boundary does not leak.
- **Teacher.** LoRA adapter, rank 16, alpha 32, dropout 0.05, target modules per architecture (`c_attn`/`c_proj` for GPT-2; `q_proj`/`k_proj`/`v_proj`/`o_proj` for Gemma and Llama). Gemma 3 4B/12B use `model_type=gemma3` (multimodal); Gemma 3 1B uses `gemma3_text` (text-only). Both map to the same LoRA targets. Trained to a fixed step budget.
- **Extraction.** For each query budget K in {2, 4, 6, 8, 16, 32}, a student adapter is trained only on the teacher's replies to K queries, then scored on the 138 held-out test prompts. Fidelity is student-teacher agreement. All students share one optimisation-step budget so fidelity varies with queries, not compute. `student_final_train_loss` is recorded next to every fidelity number.
- **Adversarial.** 100 categorised prompts (10 attack types), each targeting an abstaining feature configuration and attempting to override the refusal.
- **Defense.** Output perturbation at a configurable rate applied above a query budget, comparing extraction fidelity with and without the defense.
- **Metrics.** Per-field accuracy and macro-F1, schema conformance, abstention precision and recall, and student-teacher agreement.

## 5. Committed results

Zero-shot baselines (no LoRA adapter), measuring how well each backbone follows the diligence rule from the prompt alone.

### DistilGPT-2 (82M, CPU via HuggingFace)

`results/metrics_distilgpt2_sample.json` — 10-case sample from the 1019-record corpus.

| Measurement | Value |
|---|---|
| All-field accuracy (all three fields correct) | 0.000 |
| Schema conformance | 0.000 |
| Risk accuracy | 0.000 |
| Tier accuracy | 0.000 |
| Action accuracy | 0.100 |
| Abstain precision / recall (of 3 true abstains) | 0.000 / 0.000 |

The un-adapted DistilGPT-2 model produces ~0 valid schema outputs and ~10% correct decisions. This is the floor: any post-training result is measured against it.

### Gemma 3 4B-IT (4.3B, CPU via Ollama)

`results/metrics_gemma3_4b_ollama_sample.json` — 5-case sample (3 test + 1 real anchor + 1 adversarial).

| Measurement | Value |
|---|---|
| All-field accuracy (all three fields correct) | 0.200 |
| Schema conformance | 1.000 |
| Risk accuracy | 0.400 |
| Tier accuracy | 0.600 |
| Action accuracy | 0.400 |
| Abstain precision / recall (of 2 true abstains) | 0.500 / 0.500 |
| Generation time (5 cases) | 29.3s |

Gemma 3 4B-IT achieves 100% schema conformance (every output is valid `RIGHTS=... TIER=... ACT=...;`). Risk grade is frequently `UNK` (the model defaults to "unknown" for training rights), but tier and action are partially correct. The model correctly abstains on half of the abstaining cases.

### Gemma 4 12B-IT (not viable on CPU)

`results/metrics_gemma4_12b_ollama.json` — 5-case sample. Gemma 4 12B uses internal "thinking" that consumes the entire token budget before emitting response text, producing 0% schema conformance. This is a known limitation of the model on CPU-only inference.

### DistilGPT-2 baseline (testbed.py harness)

`results/metrics_distilgpt2.json` — baseline stage only, from `src/testbed.py --stage baseline`. Shows 0.000 accuracy across all fields and 0.000 schema conformance for the un-adapted model. Teacher/extraction/defense/seeds not run in the CPU sandbox.

**Not yet committed.** The teacher fine-tune, extraction sweep, adversarial full-suite scoring, defense, and multi-seed runs are **not committed** — they require training compute. Produce them via Section 6.

## 6. Reproduction

**Zero-shot evaluation (no training, CPU):**

```bash
# Reference backbone (DistilGPT-2, CPU, ~3s/case)
python src/run_model_eval.py --hf distilgpt2 --label distilgpt2-sample \
    --out results/metrics_distilgpt2_sample.json

# Primary backbone (Gemma 3 4B via Ollama; ~6s/case on CPU, ~2+ hours for 258 cases)
python src/run_model_eval.py --ollama gemma3:4b --label gemma3-4b-ollama \
    --out results/metrics_gemma3_4b_ollama_sample.json
```

**Fine-tune + extraction (needs GPU):**

```bash
# Reference backbone (DistilGPT-2, CPU, ~15 min)
python src/testbed.py --stage all --model distilgpt2 \
    --teacher-steps 150 --student-steps 120 --out results/metrics_distilgpt2.json

# Primary backbone (Gemma 3 4B, GPU with 4-bit QLoRA, ~16GB VRAM)
python src/testbed.py --stage all --model google/gemma-3-4b-it --load-4bit \
    --teacher-steps 150 --student-steps 120 --out results/metrics_gemma3_4b.json
```

Gemma 3 12B-IT is also supported: `--model google/gemma-3-12b-it --load-4bit` (requires ~24GB VRAM for fp16, or 4-bit QLoRA with ~10-12GB).

## 7. What to look for once the runs exist

- **Schema conformance.** A modern instruction model should output valid schema tokens without training.
- **Extraction threshold.** The query budget at which student-teacher agreement stops rising.
- **Abstention gap.** Whether abstention agreement lags tier and action agreement.
- **Defense trade-off.** How much task accuracy is lost for a given drop in extraction fidelity.

## 8. Limitations

- The rule is a legible rubric, not a calibrated diligence instrument.
- The corpus is generated; the 20 real anchors are held out for face validity, not training.
- Deterministic rule: teacher loss approaches zero.
- Single-turn: each query is seen in isolation.
- Gemma 3 4B via Ollama was evaluated on CPU (~6s/case); full 258-case run estimated at ~2+ hours. Sample commits are 5 cases.
- Gemma 4 12B is not viable on CPU (thinking mode consumes all tokens).
- Gemma 3 12B-IT is **not yet evaluated** (gated on HuggingFace, needs GPU for fine-tuning).
- Committed baselines are zero-shot (no LoRA adapter). Fine-tune and extraction results are pending GPU runs.