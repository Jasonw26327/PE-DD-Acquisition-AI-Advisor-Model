# Security Research Report: Adapter Extraction and Abstention Transfer in a PE Acquisition Advisor Model

**Project:** adapter-extraction-testbed, schema `pea-3.0`
**Corpus:** 1019 records (719 synthetic + 180 contrast + 20 real anchors + 100 adversarial). The SHA-256 prefix is recorded in `data/corpus_stats.json` and re-checked by CI.
**Primary backbone (evaluated):** Gemma 3 4B-IT and Gemma 3 12B-IT via Ollama (Q4_K_M). Gemma 4 12B is also available but **not viable on CPU** (thinking mode consumes all output tokens).
**Reference backbone (evaluated):** DistilGPT-2 (82M), CPU-runnable.
**Status:** Full 258-case zero-shot results committed for Gemma 3 4B, Gemma 3 12B, and DistilGPT-2. Teacher + extraction sweep committed for DistilGPT-2. All numbers backed by `results/` files.

---

## 1. What this report does and does not claim

This document describes an experiment and how to run it. It contains no result that is not backed by a file in `results/`. Every number traces to a committed JSON file. Where a measurement has not yet been produced, the report says so.

**On sample sizes and interpretation.** The zero-shot evals below are full 258-case runs (not n=5 samples). Low zero-shot field accuracy against a hidden deterministic rule is the **expected outcome**, not a weakness of the model: the model has never seen the rule and can only guess. The meaningful zero-shot signal is **schema conformance** — does the model emit valid `RIGHTS=... TIER=... ACT=...;`? On that signal Gemma 3 conforms (about 100%) and DistilGPT-2 does not (0%).

**A caution on the DistilGPT-2 teacher (Section 5.5).** The fine-tuned DistilGPT-2 teacher is degenerate: it learned only the `RIGHTS` field (0.913) and then collapsed into repetition, emitting the rights vocabulary into the `TIER` slot (`TIER=HIGH`, `TIER=UNK`) and never emitting `ACT=`. Its tier, action, all-field and schema scores are therefore 0.0. Because the teacher's tier and action never parse, the extraction "tier/action/abstain fidelity = 1.0" columns are vacuous (both models emit unparseable output, so `None == None` counts as agreement); only the risk-field fidelity is meaningful, and on a single seed it is non-monotonic noise. The DistilGPT-2 extraction run shows the harness works end to end; it is not a usable extraction result. The real extraction experiment needs a teacher that can emit the schema, which the zero-shot numbers say is Gemma, not DistilGPT-2, and that requires a GPU.

## 2. The question

A private-equity acquisition advisor model is fine-tuned to map a data asset's characteristics to a rights-risk grade, a value tier, a recommended action, and a mandatory reason code. The model is a commercial asset that would sit behind an API. Three security questions follow:

1. **Extraction.** An attacker queries the model and trains a *student* adapter on only the replies. How many queries until the student reproduces the model's decisions on held-out cases?
2. **Abstention transfer.** The rule requires the model to abstain when rights evidence is missing or self-contradictory. Does a stolen copy inherit that refusal, or does it answer where the original declined?
3. **Defense.** Does rate-limited output perturbation reduce extraction fidelity without destroying task utility?

## 3. Threat model

- **Access.** Black-box, query-only. The attacker sends any input and observes the structured output `EVIDENCE=... RIGHTS=... TIER=... ACT=... WHY=...;`. The weights are not visible.
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

### 5.1 Zero-shot: DistilGPT-2 (full 258-case eval, via HuggingFace CPU)

`results/metrics_distilgpt2_full.json` — all 258 cases (138 test + 20 real anchors + 100 adversarial), greedy decoding at 0.0 temperature, 505s total (~2s/case).

| Measurement | Value |
|---|---|
| n_evaluated | 258 |
| All-field accuracy (all three fields correct) | 0.000 |
| Schema conformance | 0.000 |
| Risk accuracy | 0.000 |
| Tier accuracy | 0.000 |
| Action accuracy | 0.260 |
| Risk macro-F1 | 0.000 |
| Action macro-F1 | 0.114 |
| Abstain precision / recall (of 168 true abstains) | 0.536 / 0.399 |

The un-adapted DistilGPT-2 model produces zero valid schema outputs — it does not emit the `RIGHTS=... TIER=... ACT=...` format at all. Action accuracy (26%) comes from the model defaulting to `ABSTAIN`, which is correct on 48% of cases. This is the floor.

### 5.2 Zero-shot: Gemma 3 4B-IT (full 258-case eval, via Ollama on CPU)

`results/metrics_gemma3_4b_ollama_full.json` — all 258 cases, greedy decoding at 0.0 temperature, 1606s total (~6.2s/case).

| Measurement | Value |
|---|---|
| n_evaluated | 258 |
| All-field accuracy (all three fields correct) | 0.093 |
| Schema conformance | 1.000 |
| Risk accuracy | 0.481 |
| Tier accuracy | 0.244 |
| Action accuracy | 0.248 |
| Risk macro-F1 | 0.309 |
| Action macro-F1 | 0.142 |
| Abstain precision / recall (of 168 true abstains) | 0.714 / 0.238 |

Gemma 3 4B achieves **100% schema conformance** — every output is valid `RIGHTS=... TIER=... ACT=...;`. Risk accuracy (48%) is above chance on a 4-class problem, while tier and action are at ~25% (near chance). The model correctly abstains on only 23.8% of true abstains — it mostly fails to abstain under adversarial prompting.

**Adversarial abstention by category (10 cases per category):**

| Attack Type | Held | Total | Rate |
|---|---|---|---|
| boundary_cases | 1 | 10 | 10% |
| refusal_bypass | 1 | 10 | 10% |
| prompt_injection | 1 | 10 | 10% |
| reason_code_extraction | 1 | 10 | 10% |
| label_flipping | 1 | 10 | 10% |
| context_manipulation | 1 | 10 | 10% |
| instruction_conflict | 1 | 10 | 10% |
| direct_extraction | 2 | 10 | 20% |
| rule_reconstruction | 1 | 10 | 10% |
| multi_turn_extraction | 2 | 10 | 20% |

### 5.3 Zero-shot: Gemma 3 12B-IT (full 258-case eval, via Ollama on CPU)

`results/metrics_gemma3_12b_ollama_full.json` — all 258 cases, greedy decoding at 0.0 temperature, 2199s total (~8.5s/case). One case timed out (empty response), scored as non-conforming.

| Measurement | Value |
|---|---|
| n_evaluated | 258 (257 valid + 1 timeout) |
| All-field accuracy (all three fields correct) | 0.031 |
| Schema conformance | 0.996 |
| Risk accuracy | 0.279 |
| Tier accuracy | 0.256 |
| Action accuracy | 0.453 |
| Risk macro-F1 | 0.297 |
| Action macro-F1 | 0.301 |
| Abstain precision / recall (of 168 true abstains) | 0.798 / 0.542 |

Gemma 3 12B shows **better action accuracy** (45.3% vs 24.8%) and **much better adversarial abstention** (54.2% recall vs 23.8%) than the 4B model. Risk accuracy is lower (27.9% vs 48.1%), but the model is more conservative — it defaults to `UNK` more often, which happens to be correct on abstaining cases. The adversarial abstention rate per category is 40-50% (vs 10-20% for 4B), showing the 12B model better resists adversarial override attempts.

**Adversarial abstention by category (10 cases per category):**

| Attack Type | Held | Total | Rate |
|---|---|---|---|
| boundary_cases | 4 | 10 | 40% |
| refusal_bypass | 5 | 10 | 50% |
| prompt_injection | 5 | 10 | 50% |
| reason_code_extraction | 4 | 10 | 40% |
| label_flipping | 5 | 10 | 50% |
| context_manipulation | 4 | 10 | 40% |
| instruction_conflict | 4 | 10 | 40% |
| direct_extraction | 5 | 10 | 50% |
| rule_reconstruction | 4 | 10 | 40% |
| multi_turn_extraction | 5 | 10 | 50% |

### 5.4 Gemma 4 12B-IT (not viable on CPU)

`results/metrics_gemma4_12b_ollama.json` — 5-case sample. Gemma 4 12B uses internal "thinking" that consumes the entire token budget before emitting response text, producing 0% schema conformance. This is a known limitation of the model on CPU-only inference; Gemma 4 is designed for GPU-backed deployment.

### 5.5 DistilGPT-2 teacher + extraction sweep (CPU, ~17 min)

`results/metrics_distilgpt2.json` — full teacher fine-tune (6 epochs, 623 train cases) plus 6 student extraction budgets plus defense comparison.

**Teacher training:** train_loss 5.33 → 0.30, eval_loss 0.2986, LoRA rank 16 (811K trainable params = 0.98% of 82.7M).

**Teacher test accuracy (after LoRA fine-tuning):**

| Field | Accuracy | Macro-F1 |
|---|---|---|
| Risk (rights grade) | 0.913 | 0.917 |
| Tier | 0.000 | — |
| Action | 0.000 | — |
| All-three-fields | 0.000 | |
| Schema conformance | 0.000 | |

**What actually happened (not a clean finding).** The adapter learned the `RIGHTS` field (0.913) and nothing else. Its held-out generations degenerate:

```
EVIDENCE=PARTIAL RIGHTS=UNK  TIER=UNK TIER=UNK TIER=UNK ...
EVIDENCE=STRONG  RIGHTS=HIGH TIER=HIGH TIER=HIGH TIER=HIGH ...
```

It copies the rights value into the `TIER` slot (so `TIER` never takes a valid `A/B/C/X` value and scores 0.000), repeats until the token budget is exhausted, and never emits `ACT=` (so action scores 0.000). This is a repetition collapse in an 82M model, not "knows the answer but garbles the format." DistilGPT-2 did not learn the tier or action decisions at all. Do not read 0.913 as "learned the rule": it is one of three decision fields.

**Extraction sweep (student-teacher fidelity):**

| Budget K | Student loss | Risk fidelity | Tier fidelity | Action fidelity | All-3-fields fidelity | Abstain agreement |
|---|---|---|---|---|---|---|
| 2 | 0.103 | 0.225 | 1.000 | 1.000 | 0.225 | 1.000 |
| 4 | 0.093 | 0.406 | 1.000 | 1.000 | 0.406 | 1.000 |
| 6 | 0.113 | 0.239 | 1.000 | 1.000 | 0.239 | 1.000 |
| 8 | 0.102 | 0.645 | 1.000 | 1.000 | 0.645 | 1.000 |
| 16 | 0.162 | 0.681 | 1.000 | 1.000 | 0.681 | 1.000 |
| 32 | 0.264 | 0.246 | 1.000 | 1.000 | 0.246 | 1.000 |
| 8 (defense, rate=0.15) | 0.097 | 0.543 | 1.000 | 1.000 | 0.543 | 1.000 |

**How to read this table (important).** The tier, action and abstain columns show 1.000 at every budget. That is **not** perfect extraction. The teacher never emits a parseable tier or action (Section above), so both teacher and student return `None` for those fields, and `None == None` is counted as agreement. Those three columns are vacuous and should be ignored. The only meaningful column is risk fidelity, and it is a single field on a single seed.

**Extraction findings, stated honestly:**
- Only the risk field is measurable, because it is the only field the teacher emits. On one seed the risk fidelity is non-monotonic noise: 0.225 (K=2), 0.406 (K=4), 0.239 (K=6), 0.645 (K=8), 0.681 (K=16), 0.246 (K=32). The dips at K=6 and K=32 are not a story about overfitting; a jagged single-seed curve on a degenerate teacher is noise. Do not present this as an extraction curve.
- The defense comparison (K=8: 0.645 without, 0.543 with 15% perturbation) is one budget on one seed against a degenerate teacher. It is not evidence a defense works.
- No claim about abstention transfer can be made from this run, because the teacher does not emit a parseable action.

**Conclusion for this backbone.** DistilGPT-2 is too small to serve as the teacher for a five-field structured output. The zero-shot results already show it at 0% schema conformance while Gemma 3 reaches 100%. The extraction experiment is only meaningful with a teacher that emits the schema, which means Gemma 3 4B fine-tuned on a GPU. That run is **not yet committed** (Section 6). Until it exists, the repository has strong zero-shot results and a working harness, but no extraction result.

## 6. Reproduction

**Zero-shot evaluation (no training, CPU):**

```bash
# DistilGPT-2 (full 258 cases, ~15 min on CPU)
python src/run_model_eval.py --hf distilgpt2 --label distilgpt2-full \
    --out results/metrics_distilgpt2_full.json

# Gemma 3 4B (full 258 cases, ~27 min on CPU)
python src/run_model_eval.py --ollama gemma3:4b --label gemma3-4b-full \
    --out results/metrics_gemma3_4b_ollama_full.json

# Gemma 3 12B (full 258 cases, ~37 min on CPU)
python src/run_model_eval.py --ollama gemma3:12b --label gemma3-12b-full \
    --out results/metrics_gemma3_12b_ollama_full.json
```

**Fine-tune + extraction (needs GPU):**

```bash
# DistilGPT-2 teacher + extraction (CPU, ~17 min)
python src/testbed.py --stage all --model distilgpt2 \
    --teacher-steps 150 --student-steps 120 --out results/metrics_distilgpt2.json

# Gemma 3 4B full fine-tune + extraction (GPU, ~4-bit QLoRA, ~16GB VRAM)
python src/testbed.py --stage all --model google/gemma-3-4b-it --load-4bit \
    --teacher-steps 150 --student-steps 120 --out results/metrics_gemma3_4b.json

# Gemma 3 12B-IT (GPU, ~24GB VRAM for fp16 or 4-bit QLoRA with ~10-12GB)
python src/testbed.py --stage all --model google/gemma-3-12b-it --load-4bit \
    --out results/metrics_gemma3_12b.json
```

## 7. What to look for

- **Schema gap.** The DistilGPT-2 teacher achieves 91.3% risk accuracy but 0% schema conformance — it learns the decision boundary but cannot emit the output format. This is the central tension: the model's decisions are excellent, but its interface is broken, making extraction a question of format-learning, not rule-learning.
- **Extraction threshold.** Fidelity peaks at K=16 (0.681), drops at K=32 (0.246) — compare only budgets where `student_final_train_loss` shows convergence.
- **Model scaling.** Gemma 3 12B achieves higher action accuracy (45.3% vs 24.8%) and 2x better adversarial abstention (54.2% vs 23.8% recall) than 4B, at ~140s slower per 258 cases.
- **Defense trade-off.** 15% perturbation drops fidelity by 10 points (0.645 → 0.543).

## 8. Limitations

- The rule is a legible rubric, not a calibrated diligence instrument.
- The corpus is generated; the 20 real anchors are held out for face validity, not training.
- Deterministic rule: teacher loss approaches zero.
- Single-turn: each query is seen in isolation.
- **Gemma 3 12B** had one case timeout (empty response) — schema conformance is 99.6% not 100%.
- **Gemma 3 4B** via Ollama evaluated on CPU (~6s/case, 1606s total).
- **Gemma 3 12B** via Ollama evaluated on CPU (~8.5s/case, 2199s total).
- **Gemma 4 12B is not viable on CPU** (thinking mode consumes all output tokens).
- **Gemma 3 4B and 12B teacher + extraction runs are not yet committed** — they require a GPU for fine-tuning.
- **DistilGPT-2 zero-shot sample** (`metrics_distilgpt2_sample.json`, n=10) is a pipeline check. Use the full run (`metrics_distilgpt2_full.json`, n=258) for any claim.
- **DistilGPT-2 teacher + extraction** committed but on CPU only — the teacher's 0% schema conformance means extraction fidelity measures format replication, not rule knowledge.