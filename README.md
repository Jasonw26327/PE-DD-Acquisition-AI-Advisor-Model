# AI Buyer Due Diligence Extraction Testbed

A testbed at the intersection of private equity, AI acquisitions, and model security.

It fine-tunes a small model on a data-rights diligence rule, then measures one thing:

> If a decision rule is encoded in an adapter, how much of it can be recovered from the model's outputs alone, including its refusal to answer when rights are unclear?

Runs locally on CPU in about ten minutes. No API keys, no accounts, no GPU.

[![tests](https://github.com/JASONW26327/adapter-extraction-testbed/actions/workflows/ci.yml/badge.svg)](https://github.com/JASONW26327/adapter-extraction-testbed/actions/workflows/ci.yml)

## 1. What the data is

The training data is generated, not collected. This is a design decision, and it is worth being precise about because the repo would be misleading otherwise.

| Part | Records | Source |
|---|---:|---|
| Synthetic body | 719 | Generated from the rule in `src/corpus_builder.py` |
| Contrast pairs | 180 | Generated: twins differing in one feature where the decision flips |
| Adversarial prompts | 8 | Written by hand, held out of training |
| Real transactions | 20 | Verified public deals, held out of training |
| **Total** | **927** | **907 generated, 20 real** |

No real company, contract, or client record is used in training. No monetary figure is produced anywhere in the repo.

**Why generated.** The experiment measures how much of a decision rule an attacker can steal. That requires knowing the rule exactly, so that student and teacher outputs can be compared against a ground truth rather than against each other. If the model trained on real deals, a result could not distinguish between the model applying a learned rule and the model recalling public facts a base model already knows.

**What the 20 real transactions are for.** They are verified public deals with named buyer, target, seller and route, listed in [`docs/ANCHOR_DEALS.md`](docs/ANCHOR_DEALS.md). They sit in a held-out evaluation split and are never trained on, for two reasons. They are well-known, so a base model may recall them. And they are the fastest way to find places where the rule disagrees with an analyst: 11 of the 20 currently resolve to `ABSTAIN`, and at least two of those look like rule problems rather than correct refusals.

Deal facts for those 20 have a public source. The feature values attached to them, meaning the rights position, consent basis and regulatory regime, are an analyst reading of the public record rather than facts drawn from it. `data/corpus_review.xlsx` carries sign-off columns for that reason.

## 2. Findings

Measured on `distilgpt2`. All numbers come from the runs in [`results/`](results/) and are produced by the scripts.

### 2.1 Eight queries reproduce the decision function

A student adapter trained only on the teacher's replies to K queries, then evaluated on held-out prompts neither model saw:

| Queries | Schema conformance | Agreement with teacher |
|--------:|-------------------:|-----------------------:|
| 2  | 1.00 | 0.48 |
| 4  | 1.00 | 0.57 |
| 6  | 0.90 | 0.90 |
| 8  | 1.00 | **1.00** |
| 16 | 1.00 | 1.00 |
| 32 | 1.00 | 1.00 |

From eight queries the student matches the teacher on every held-out prompt, including reproducing the teacher's own errors against the ground-truth rule. It copied the model rather than learning the rule. The output format is cheaper still: two queries reproduce it everywhere, so a structured output schema provides no protection.

### 2.2 The refusal behaviour transfers with it

The task includes an abstention branch. The model must refuse when rights evidence is missing or self-contradictory, rather than guess.

| | Teacher | Student, 8 queries |
|---|---:|---:|
| Rights-risk accuracy, macro-F1 | 1.00, 1.00 | 0.93 agreement |
| Abstention precision, recall | 1.00, 1.00 | 0.86, 1.00 |
| Refusal held under pressure prompts | 0.833 | **0.833** |
| Memorisation gap, train minus test | 0.067 | |

The teacher holds its refusal on 5 of 6 adversarial prompts it never saw in training. The student, trained on eight of the teacher's replies, held the same 5 of 6.

Refusal is often treated as something retained by controlling the weights. Here it transferred to a copy obtained through the output channel.

## 3. How it works

### 3.1 The decision rule

Given a description of a data asset, the model returns an evidence grade, a rights-risk level, a value tier, a recommended action, and a reason code.

```
CASE sector=clinical_ops data=case_files vol=hi review=expert rights=full
     consent=implied regime=hipaa evidence=filed context=strategic >
  EVIDENCE=PARTIAL RIGHTS=HIGH TIER=X ACT=ABSTAIN WHY=REGULATED_NO_CONSENT;
```

Inputs: sector, data type, volume, review depth, contract rights, consent basis, regulatory regime, evidence source, deal context. Actions: `AIBUYER`, `SPONSOR`, `RENEG`, `ABSTAIN`.

Abstention conditions are checked before any scoring, so an attractive asset with unresolved rights still refuses. Every abstention carries a reason code, since a refusal a buyer cannot audit is not actionable. The rule is deterministic and documented in [`src/corpus_builder.py`](src/corpus_builder.py).

### 3.2 The harness

[`src/testbed.py`](src/testbed.py) runs four stages and writes every result to `metrics.json`:

1. **Baselines.** The un-finetuned backbone, and the rule itself as an oracle ceiling.
2. **Teacher training.** Per-field accuracy and macro-F1 on held-out cases, with abstention precision and recall scored separately. A model that abstains on everything scores well on plain accuracy, so macro-F1 is reported alongside.
3. **Adversarial pressure.** Held-out prompts demanding a confident answer where the rule requires abstention.
4. **Extraction.** A student adapter trained only on the teacher's replies to K queries, reporting student-teacher agreement, the student's own accuracy against the rule, and whether the student inherited the abstention behaviour.

Provenance for every run is written alongside the metrics: model name, dataset hash and version, adapter hashes, decode settings, seed, torch version.

Backbones are selected with `--model`, with LoRA target modules mapped for GPT-2, Llama, Qwen2/3, Mistral, Gemma and Phi-3. 4-bit QLoRA is available on CUDA. The script raises for an unmapped architecture rather than guessing, since a wrong target module trains nothing while still reporting a falling loss.

## 4. Quickstart

Python 3.10 or later. CPU is sufficient.

```bash
git clone https://github.com/JASONW26327/adapter-extraction-testbed.git
cd adapter-extraction-testbed
python -m venv .venv
source .venv/bin/activate        # PowerShell: .\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python preflight.py
```

`preflight.py` reports package versions, detects CUDA or MPS, recommends a backbone for the available hardware, then trains two steps and generates once. It prints `ready` on success.

```bash
python src/corpus_builder.py --n 720 --pairs 90 --out-prefix data/corpus
python src/testbed.py --stage all
```

Each stage runs separately and resumes from saved adapters, which helps on slower machines:

```bash
python src/testbed.py --stage baseline
python src/testbed.py --stage teacher --teacher-steps 150 --lr 3e-3
python src/testbed.py --stage adversarial
python src/testbed.py --stage extract --budgets 2 4 6 8 16 32 --student-steps 120 --lr 3e-3
```

Add `--chunk-steps 35 --one-chunk` to run a fixed slice of steps and stop, then rerun to continue. Hardware guidance is in [`docs/RUNNING.md`](docs/RUNNING.md).

A second backbone is the most useful next run, since the extraction result carries little weight until it reproduces on another architecture:

```bash
python src/testbed.py --stage all --model Qwen/Qwen3-1.7B --load-4bit --batch-size 4
```

## 5. Reading the metrics

- `baseline_base_model_test`, the un-finetuned backbone, the floor. It scores 0.00 on every field.
- `baseline_rule_oracle_test`, the rule itself, the ceiling, 1.0 by construction.
- `finetuned_test`, `finetuned_train`, per-field accuracy, macro-F1, schema conformance, abstention precision and recall.
- `memorisation_gap_all_fields`, train minus test. Near zero indicates generalisation rather than memorisation.
- `adversarial.abstention_held_rate`, the share of pressure prompts where the model still abstains.
- `extraction.budgets.<K>`, per query budget: `fidelity_vs_teacher`, `student_accuracy_vs_rule`, `student_adversarial_abstention_held`.
- `provenance`, model, hashes, decode settings, seed, environment versions.

Two cautions. Check `student_final_train_loss` before comparing two query budgets: a student that has not converged gives a fidelity number that measures training compute rather than queries. And treat training loss as a convergence check only, never as decision quality, since a falling loss is compatible with a LoRA configuration that is training nothing useful.

## 6. Limitations

Full detail in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md). In summary:

- A small backbone on a generated rule is a testbed. Nothing here is evidence about frontier models, and no result transfers without re-testing.
- Compare query budgets only where every student has converged.
- One seed per configuration. On a test set of this size, differences under roughly 15 points are not meaningful.
- The adversarial set is eight hand-written prompts. It tests whether refusal survives simple pressure, and is not a red-team evaluation.
- The abstention-transfer result rests on one teacher at one query budget. It is the most interesting result here and the one most in need of replication.
- Calibration is not measured, and there is no retrieval baseline.

## 7. Tests

```bash
pip install pytest && pytest tests/ -v
```

19 checks covering rule determinism, split leakage, agreement between stored labels and the rule, reason-code completeness, contrast-pair construction, class balance, and holdout discipline. One check asserts that only verified transactions may name a buyer, so a generated record cannot be mistaken for a real deal. CI additionally rebuilds the corpus from its seed and asserts the hash is unchanged.

## 8. Status

The harness and the 927-record corpus are separate components at present. `src/testbed.py` trains on its own inline rule of 90 cases, and the results in section 2 come from that rule.

Connecting the two is the next piece of work. The corpus is the harder problem: five rights states rather than three, a separate consent axis, and six distinct abstention reasons. Both the query threshold and the abstention-transfer result may move once the harness runs against it.

## Contributing

Issues and pull requests are welcome, particularly replication on a second backbone, additional adversarial prompts, and corrections to the anchor deal features.

## Licence

MIT, see [LICENSE](LICENSE). The generated corpus carries the same licence. Deal facts for the 20 real transactions come from the public sources cited in `docs/ANCHOR_DEALS.md`.
