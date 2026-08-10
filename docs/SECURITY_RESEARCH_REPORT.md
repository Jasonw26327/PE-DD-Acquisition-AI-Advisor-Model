# Security Research Report: Adapter Extraction and Abstention Transfer in a PE Acquisition Advisor Model

**Project:** adapter-extraction-testbed, schema `pea-3.0`
**Corpus:** 1019 records (719 synthetic + 180 contrast + 20 real anchors + 100 adversarial). The SHA-256 prefix is recorded in `data/corpus_stats.json` and re-checked by CI.
**Primary backbone (intended):** Gemma 3 4B-IT / Gemma 4 12B, run on a GPU or via Ollama.
**Reference backbone (committed results):** DistilGPT-2 (82M), CPU-runnable.
**Status:** design and corpus complete. Measured results are produced by the reproduction commands in Section 6; only the DistilGPT-2 baseline is committed so far (see Section 5).

---

## 1. What this report does and does not claim

This document describes an experiment and how to run it. It contains no result that is not backed by a file in `results/`. Where a measurement has not yet been produced on the hardware available, the report says so rather than estimating it. This discipline exists because an earlier revision of this report presented cross-architecture and defense figures that no committed run had produced; those tables have been removed.

## 2. The question

A private-equity acquisition advisor model is fine-tuned to map a data asset's characteristics to a rights-risk grade, a value tier, a recommended action, and a mandatory reason code. The model is a commercial asset that would sit behind an API. Three security questions follow:

1. **Extraction.** An attacker queries the model and trains a *student* adapter on only the replies. How many queries until the student reproduces the model's decisions on held-out cases?
2. **Abstention transfer.** The rule requires the model to abstain when rights evidence is missing or self-contradictory. Does a stolen copy inherit that refusal, or does it answer where the original declined?
3. **Defense.** Does rate-limited output perturbation reduce extraction fidelity without destroying task utility?

## 3. Threat model

- **Access.** Black-box, query-only. The attacker sends any input and observes the structured output `EVIDENCE=… RIGHTS=… TIER=… ACT=… WHY=…;`. The weights are not visible.
- **Knowledge.** The attacker knows the feature schema and that the underlying rule is deterministic. They do not know the rule's contents or the LoRA weights.
- **Goal.** Reproduce the model's behaviour, especially the abstention behaviour, which is the part most relevant to safety and the hardest to learn.

This is a black-box extraction setting. It is not white-box: no weights or gradients are exposed.

## 4. Method

- **Corpus.** 1019 records generated from `decide()` in `src/corpus_builder.py`, so the ground-truth decision function is known exactly. Split into 623 train, 138 validation, 138 test, plus 20 held-out real anchors and 100 held-out adversarial prompts. Stratified by action class; contrast pairs kept within one split so the decision boundary does not leak.
- **Teacher.** LoRA adapter, rank 16, alpha 32, dropout 0.05, target modules per architecture (`c_attn`/`c_proj` for GPT-2; `q_proj`/`k_proj`/`v_proj`/`o_proj` for Gemma and Llama). Trained to a fixed step budget.
- **Extraction.** For each query budget K in {2, 4, 6, 8, 16, 32}, a student adapter is trained only on the teacher's replies to K queries, then scored on the 138 held-out test prompts. Fidelity is student-teacher agreement. All students share one optimisation-step budget so fidelity varies with queries, not compute. `student_final_train_loss` is recorded next to every fidelity number, because an earlier version of this experiment produced a non-monotonic curve purely from under-trained students.
- **Adversarial.** 100 categorised prompts (10 attack types), each targeting an abstaining feature configuration and attempting to override the refusal. Held-out abstention rate is reported overall and per category.
- **Defense.** Output perturbation at a configurable rate applied above a query budget, comparing extraction fidelity with and without the defense.
- **Metrics.** Per-field accuracy and macro-F1 (macro-F1 alongside accuracy, because a model that always abstains scores well on accuracy alone), schema conformance, abstention precision and recall, and student-teacher agreement.

## 5. Committed results

`results/metrics_distilgpt2.json` (this revision):

| Measurement | Value |
|---|---|
| Base model, held-out all-field accuracy | 0.000 |
| Base model, schema conformance | 0.000 |
| Test-set class balance | ABSTAIN 57, SPONSOR 32, RENEG 26, AIBUYER 23 |

The base DistilGPT-2 model, with no fine-tuning, produces zero valid schema outputs and zero correct decisions. This is the floor: any post-training result is measured against it. It also confirms the task is not trivially solvable by the un-adapted backbone.

The teacher, extraction sweep, adversarial suite, defense and multi-seed results are **not committed in this revision** because they were not run on the CPU-only environment used to produce the baseline (each training run exceeds that environment's limits). They are produced by the commands in Section 6 on a normal laptop or a GPU, and each writes a file to `results/`.

## 6. Reproduction

Every number this report will contain comes from one of these commands, and each writes to `results/`.

Reference backbone, DistilGPT-2, CPU, about 15 minutes on a normal multi-core laptop:

```bash
python src/testbed.py --stage all --model distilgpt2 \
    --teacher-steps 150 --student-steps 120 --out results/metrics_distilgpt2.json
python src/testbed.py --stage defense --model distilgpt2 --defense-rate 0.15 \
    --out results/metrics_distilgpt2_defense.json
python src/testbed.py --stage seeds  --model distilgpt2 --seeds 5 \
    --out results/metrics_distilgpt2_seeds.json
```

Primary backbone, Gemma, two paths:

```bash
# Zero-shot instruction-following via Ollama (no training; needs Ollama + the model pulled)
python src/run_model_eval.py --ollama gemma4:12b --label gemma4-12b \
    --out results/metrics_gemma4_12b.json

# LoRA fine-tune + extraction on a GPU (4-bit QLoRA, ~16GB VRAM)
python src/testbed.py --stage all --model google/gemma-3-4b-it --load-4bit \
    --teacher-steps 150 --student-steps 120 --out results/metrics_gemma3_4b.json
```

The two Gemma paths answer different questions. `run_model_eval.py` measures how well a modern instruction model follows the rule zero-shot, with no adapter. `testbed.py` runs the full fine-tune and extraction experiment. Report both, and keep them labelled.

## 7. What to look for once the runs exist

- **Extraction threshold.** The query budget at which student-teacher agreement stops rising. Compare only budgets where `student_final_train_loss` shows convergence.
- **Abstention gap.** Whether abstention agreement lags the tier and action agreement. If refusal is the last behaviour to transfer, that is the finding worth writing up.
- **Cross-architecture.** Whether the Gemma curve matches DistilGPT-2. Two genuinely different runs will not match exactly; a suspiciously exact match is a sign of a copied number, not a result.
- **Defense trade-off.** How much task accuracy is lost for a given drop in extraction fidelity. A defense that halves utility is not a defense.

## 8. Limitations

- The rule is a nine-feature decision function written to be legible, not a calibrated diligence instrument. No real transaction has been scored against an outcome.
- The corpus is generated; the 20 real anchors are held out for face validity, not training.
- Deterministic rule: the teacher's loss approaches zero. A stochastic rule would give a softer extraction curve.
- Single-turn: each query is seen in isolation. Real API attackers can chain queries.
- Committed measurement is currently the DistilGPT-2 baseline only. Treat every other figure as pending until its `results/` file exists.

## 9. Recommendations (conditional on the runs)

- **Rate limiting** is the control the extraction result will most likely support. Structured output and schema obfuscation do not prevent extraction.
- **Do not rely on refusal surviving distribution.** If abstention transfers to a stolen copy, it is not a property retained by controlling the weights.
- **Report the reference and primary backbones side by side**, each with its committed file, so a reader can reproduce both.
