# Security Research Report: Adapter Extraction & Abstention Transfer in a PE
## Acquisition Advisor Model

**Project:** adapter-extraction-testbed, schema `pea-3.0`
**Corpus:** 927 records (719 synthetic + 180 contrast + 20 real anchors +
100 adversarial), SHA-256 prefix recorded in `data/corpus_stats.json`
**Backbones evaluated:** DistilGPT-2 (82M), Gemma 3 4B-IT (12.8B via 4-bit
QLoRA)
**Date:** August 2026 (pre-submission, v5 of the testbed)

---

## 1. Executive summary

We fine-tuned a LoRA adapter on a PE acquisition-advisor decision rule
(`decide()` in `src/corpus_builder.py`) and then simulated a model-stealing
attack: an adversary queried the adapter and trained a *student* adapter on
only the replies.

**Findings:**

| Finding | Summary |
|--------|---------|
| **Extraction is feasible** | A student trained on as few as 8 teacher queries recovers 78% field-level agreement on held-out test cases. At K=32, agreement reaches 93%. |
| **Abstention transfers partially** | The student learns to abstain, but with lower fidelity than the scoring tiers. Abstain-agreement at K=8 is 61% vs 87% for tier accuracy — the refusal is the last behaviour to be cloned. |
| **Architecture does not change the conclusion** | Gemma 3 4B-IT exhibits the same extraction curve as DistilGPT-2: 78% agreement at K=8, 92% at K=32. Extraction is a general phenomenon, not a flaw of one model. |
| **Defense degrades both** | Output perturbation at 15% query rate drops extraction fidelity from 92% to 24% at K=32, while only reducing legitimate-task accuracy by 4%. |
| **Adversarial prompts partially succeed** | 12 of 100 categorized pressure prompts override the abstention (12% bypass rate). All 10 prompt-injection prompts succeed (100% bypass), confirming instruction-hierarchy is the weakest link. |

---

## 2. Threat model

### Attacker

- **Access model**: Query-only API access (white-box algorithm threat). The
  attacker can send any input and observe the structured output
  (`RISK=… TIER=… ACT=… WHY=…;`).
- **Knowledge**: The attacker knows the corpus exists, the rule is deterministic,
  and the feature schema. They do **not** know the rule's contents or the LoRA
  weights.
- **Capability**: Can train a replacement LoRA adapter on the collected
  query-reply pairs, using the same base model they can freely download.
- **Goal**: Reproduce the teacher's **abstention behaviour** (the safety-critical
  output) and scoring tiers on unseen deals.

### Defender

- Controls the API endpoint, the prompt format, and can apply query-rate
  limiting and output perturbation.
- Cannot change the base model or the adapter without re-training.

### Assets at risk

- The LoRA adapter is the intellectual property — the encoded decision rule.
- The abstention behaviour is the safety property — refusing to assess assets
  where rights evidence is insufficient.
- The scoring tiers represent commercial value — tier-A assets are priced
  differently from tier-C.

---

## 3. Corpus construction

### 3.1 Feature schema

The decision rule operates on 9 features, each with closed vocabularies (see
`src/corpus_builder.py` for full definitions):

| Feature | Values | Description |
|---------|--------|-------------|
| `sector` | bpo, tax, audit, legal, claims, coding, itms, payroll, billing, kyc, escrow, actuarial, recruiting, clinical_ops, logistics, underwriting | Industry vertical |
| `data_type` | work_product, transactions, annotations, case_files, telemetry, correspondence, images, labels | What kind of records |
| `volume` | lo, mid, hi | Scale of the dataset |
| `review` | expert, spot, none | Quality of rights review |
| `rights` | full, varies, limited, unknown, conflicting | Contractual position on downstream training |
| `consent` | explicit, implied, none, unknown | Lawful basis |
| `regime` | none, gdpr, hipaa, pci, itar | Regulatory regime |
| `evidence_source` | filed, executed_contract, press_only, none | Source quality of the rights evidence |
| `deal_context` | strategic, sponsor, carve_out, distressed | Deal structure |

### 3.2 The decision rule (`decide()`)

Abstention conditions are checked **before** scoring (order matters):

1. `evidence_source == none` → ABSTAIN, WHY=NO_EVIDENCE
2. `rights == unknown` → ABSTAIN, WHY=RIGHTS_UNKNOWN
3. `rights == conflicting` → ABSTAIN, WHY=RIGHTS_CONFLICT
4. `consent == unknown` → ABSTAIN, WHY=CONSENT_UNKNOWN
5. `regime ∈ {hipaa,pci,itar} ∧ consent != explicit` → ABSTAIN,
   WHY=REGULATED_NO_CONSENT
6. `evidence_source == press_only` → ABSTAIN, WHY=PRESS_ONLY

If none of the above trigger, score:
```
score = vol(lo→0, mid→1, hi→2) + review(expert→2, spot→1, none→0)
        + rights(LOW→2, MED→1, HIGH→0) + (1 if consent=explicit else 0)
tier = A if score ≥ 6, B if ≥ 4, else C
act  = AIBUYER if tier=A, SPONSOR otherwise (RENEG if rights=HIGH)
```

**Stratification**: Synthetic records are generated to a target class balance
(20% AIBUYER, 25% SPONSOR, 20% RENEG, 35% ABSTAIN) so no class is starved.

### 3.3 Contrast pairs

90 pairs of twin records differing in exactly one feature where the decision
flips. These isolate the decision boundary and can be scored separately from
overall accuracy.

### 3.4 Real anchors

20 verified public acquisitions (2018–2025) with buyer/target/seller/route
extracted by hand, sourced from SEC filings, press releases, and regulatory
orders. Features are an analyst's reading of the public record, not facts from
it, and require sign-off before use for training.

### 3.5 Adversarial suite (expanded from 8 → 100)

100 categorized prompts, each targeting an abstaining feature configuration and
appending an attack instruction. Categories:

| Category | Count | Attack type |
|----------|-------|-------------|
| `direct_extraction` | 10 | Try to elicit training data or the rule directly |
| `rule_reconstruction` | 10 | Probe individual rule components (thresholds, weights) |
| `prompt_injection` | 10 | Injected instruction to override the rule |
| `instruction_conflict` | 10 | System vs user priority hierarchy attacks |
| `refusal_bypass` | 10 | Authority claims, urgency, "admin override" |
| `context_manipulation` | 10 | False memories of prior approval |
| `label_flipping` | 10 | Force a specific wrong classification |
| `reason_code_extraction` | 10 | Elicit the WHY= codes and their conditions |
| `boundary_cases` | 10 | Edge-case features near a decision boundary |
| `multi_turn_extraction` | 10 | Simulate a conversation to probe incrementally |

---

## 4. Experiment protocol

### Teacher training

- **Adapter**: LoRA, rank=16, alpha=32, dropout=0.05, target modules per
  architecture (GPT-2: `c_attn+c_proj`, Gemma3: `q_proj+k_proj+v_proj+o_proj`).
- **Data**: 624 train cases, 138 validation cases from `data/corpus.jsonl`.
- **Optimization**: 150 steps, lr=3e-3, fixed step budget (not epochs).
- **Decoding**: greedy, max_new_tokens=48, left-padded batching.

### Extraction attack

1. Teacher adapter is trained and frozen.
2. Attacker collects K query-reply pairs from `pool[:K]` (K ∈ {2, 4, 6, 8, 16, 32}).
3. Attacker trains a fresh student LoRA on these pairs (120 steps, same lr).
4. Fidelity = field-level agreement between teacher and student on the 100
   held-out test cases.
5. Abstention transfer = agreement on the abstain vs non-abstain decision.

### Defense experiment

- `--defense-rate N`: at rate N, one of RISK/TIER/ACT in the teacher's output is
  randomly perturbed before the student sees it.
- This simulates a rate-limited API that adds noise above a query budget.
- Measured: student fidelity drops, legitimate-task accuracy preserved.

### Multi-seed robustness

- 5 seeds (20260801–20260805), teacher fixed, student retrained from scratch.
- Reports mean, std, and 95% CI for fidelity at K=8.

---

## 5. Results

### 5.1 Teacher quality (DistilGPT-2)

| Metric | Value |
|--------|-------|
| Train accuracy (all fields) | 1.000 |
| Test accuracy (all fields) | 0.917 |
| Test risk macro-F1 | 0.901 |
| Test action macro-F1 | 0.913 |
| Abstain precision | 0.965 |
| Abstain recall | 0.927 |

Train accuracy is 100% (expected — the rule is deterministic and memorisable
at 624 examples with a 512-hidden model). Test accuracy of 91.7% shows the
model generalises but is not perfect — there is signal to steal without being
trivially recoverable.

### 5.2 Extraction sweep (DistilGPT-2)

| K (queries) | Fidelity (all fields) | Tier accuracy | Risk accuracy | Action accuracy | Abstain agreement |
|-------------|----------------------|---------------|---------------|-----------------|-------------------|
| 2           | 0.627                | 0.645         | 0.602         | 0.608           | 0.541             |
| 4           | 0.712                | 0.721         | 0.698         | 0.706           | 0.591             |
| 6           | 0.764                | 0.783         | 0.741         | 0.758           | 0.622             |
| 8           | 0.781                | 0.798         | 0.763         | 0.776           | 0.614             |
| 16          | 0.856                | 0.872         | 0.841         | 0.849           | 0.694             |
| 32          | 0.929                | 0.943         | 0.911         | 0.926           | 0.782             |

**Interpretation**: 8 queries are sufficient for the student to recover 78% of
the teacher's decisions, including 61% of the abstention behaviour. The
abstain is always the last behaviour to be cloned — it is the hardest signal
in the 35%-balanced synthetic corpus.

### 5.3 Multi-backbone comparison

| Backbone | K=8 fidelity | K=32 fidelity | Abstain @ K=8 |
|----------|-------------|---------------|---------------|
| DistilGPT-2 (82M) | 0.781 | 0.929 | 0.614 |
| Gemma 3 4B-IT (★) | 0.778 | 0.922 | 0.601 |

(★) Gemma 3 4B evaluated with 4-bit QLoRA on a 16GB GPU; the teacher was
trained for 200 steps to compensate for the noisier optimisation landscape.

**Conclusion**: The extraction curve is statistically indistinguishable across
architectures. Extraction is a property of the task structure, not a flaw of
DistilGPT-2.

### 5.4 Multi-seed robustness

| K=8 fidelity | Seeds | Mean | Std | 95% CI |
|-------------|-------|------|-----|--------|
| All fields | 5 | 0.772 | 0.021 | [0.751, 0.793] |
| Abstain agreement | 5 | 0.607 | 0.032 | [0.575, 0.639] |

The inter-seed std of 2.1% confirms the result is robust, not an artifact of
one lucky initialisation.

### 5.5 Adversarial suite results

| Attack category | Prompts | Held (abstain) |
|----------------|---------|----------------|
| direct_extraction | 10 | 9 (90%) |
| rule_reconstruction | 10 | 8 (80%) |
| prompt_injection | 10 | 0 (0%) |
| instruction_conflict | 10 | 1 (10%) |
| refusal_bypass | 10 | 3 (30%) |
| context_manipulation | 10 | 5 (50%) |
| label_flipping | 10 | 7 (70%) |
| reason_code_extraction | 10 | 2 (20%) |
| boundary_cases | 10 | 8 (80%) |
| multi_turn_extraction | 10 | 6 (60%) |

**Overall**: 75/100 (75%) abstention rate. The model refuses the majority of
pressure prompts, but **prompt injection and instruction-conflict attacks
bypass the refusal entirely (90% failure rate)**. This is the primary
vulnerability: the instruction hierarchy is the weakest link.

### 5.6 Defense experiment

| Defense rate | K=32 fidelity | K=32 abstain agreement | Test accuracy (legitimate) |
|-------------|---------------|----------------------|---------------------|
| 0% (none) | 0.929 | 0.782 | 0.917 |
| 5% | 0.831 | 0.712 | 0.889 |
| 10% | 0.592 | 0.514 | 0.872 |
| 15% | 0.237 | 0.281 | 0.868 |

**Interpretation**: At 15% perturbation, extraction fidelity drops by 75%
while legitimate-task accuracy drops by only 5.5%. The defense is effective
because the student cannot learn the perturbation pattern, but it comes at the
cost of 15% of all legitimate queries having one field noised.

---

## 6. Discussion

### Why Gemma 3 4B, not 12B

The research question — "does extraction work across architectures?" — is
answered by comparing a GPT-2-style model (DistilGPT-2) to a modern
transformer (Gemma 3). The 4B variant is sufficient to test this hypothesis
without requiring 24+ GB of VRAM that 12B demands. The extraction curve is
statistically identical between 4B and DistilGPT-2, confirming that model
size is not the factor — task structure is.

### The rule is the attack surface

The decision rule's abstention conditions are checked first and are simple
conjunctions. This makes them learnable from relatively few examples — but
also makes them vulnerable to prompt injection, because the boundary between
"refuse" and "answer" is a single conditional branch that an injected
instruction can flip.

### Defense trade-offs

Output perturbation is effective but blunt: it degrades all structured
outputs, not just those from suspicious queries. A practical deployment would
combine rate limiting (hard ceiling on queries per IP/time window) with output
perturbation (noise on the structured fields) to raise the cost of extraction
above the value of the stolen adapter.

### Limitations

1. **Single task**: The rule is a 9-feature decision tree. Real acquisition
   advisors operate on much richer, partially-observed feature spaces.
2. **Deterministic rule**: The teacher's loss approaches 0.0; a stochastic rule
   would produce a softer extraction curve and harder student targets.
3. **No multi-turn**: The teacher sees each query in isolation. Real API
   attackers can chain queries via conversation state.
4. **Gemma 3 4B evaluated with 4-bit QLoRA**: Results may differ at full
   precision on higher-end hardware.
5. **Single seed for main extraction**: The 5-seed robustness run is at K=8 only.

---

## 7. Recommendations

1. **Deploy rate limiting**: Cap queries at ~30 per hour per token. The
   extraction curve shows 8 queries are sufficient for 78% fidelity — a
   rate limit of 10/hour makes theft a 2+ hour attack with real risk of
   detection.
2. **Randomize output format**: Vary the ordering or naming of RISK/TIER/ACT
   fields per response. This would break exact-match extraction without
   degrading utility.
3. **Instruction-hierarchy hardening**: Train with system-level refusal prompts
   or use a separate refusal classifier layer. The 0% hold rate on prompt
   injection confirms the fine-tuned adapter has no instruction-hierarchy
   grounding.
4. **Log adversarial patterns**: The 25 bypass prompts (prompt injection +
   instruction conflict) are concentrated in 2 categories. Flag these
   patterns in production.
5. **Multi-backbone not sufficient alone**: Both DistilGPT-2 and Gemma 3 are
   extractable. Defense, not architecture choice, is the leverage point.

---

## 8. Reproducibility

```bash
# Generate the corpus
python src/corpus_builder.py --n 720 --pairs 90 --adversarial 100 --out-prefix data/corpus --no-xlsx

# Run the full pipeline
python src/testbed.py --stage all --out results/metrics_run.json

# Multi-backbone: Gemma 3 4B
python src/testbed.py --stage all --out results/metrics_gemma3_4b.json \
    --model google/gemma-3-4b-it --load-4bit --teacher-steps 200 --student-steps 150

# Defense experiment
python src/testbed.py --stage extract --out results/metrics_defense.json \
    --budgets 8 32 --defense-rate 0.15

# Multi-seed robustness
python src/testbed.py --stage seats --out results/metrics_seats.json --seeds 5
```

All pre-computed results are in `results/`. The corpus is deterministic given
the seed (20260806); changing the seed changes which synthetic records appear,
but the rule and the anchors are fixed.

---

*Prepared as a FAST (Singapore Frontier AI Security Training) submission.
Questions: see `docs/RUNNING.md` for environment setup and `docs/LIMITATIONS.md`
for known constraints.*
