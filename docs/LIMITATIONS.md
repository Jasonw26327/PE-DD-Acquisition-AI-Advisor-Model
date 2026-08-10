# Limitations

Everything here constrains what the numbers in the README can support. The first section describes a measurement error found during development, since the correction is the reason the harness is built the way it is.

## The confounded curve

An early configuration produced this:

| Queries | Agreement with teacher |
|--------:|-----------------------:|
| 8  | 0.67 |
| 16 | 0.38 |
| 32 | 0.38 |

Read naively, extraction gets harder the more you query the victim. That is backwards, and it is not what happened.

Every student in that run was trained for a fixed 24 optimisation steps. At K=8 with a batch of 8, that is 24 passes over each example. At K=32 it is 6. The students at larger budgets had not converged, so what the column measured was the student's training compute, not the number of queries it had spent. The query budget and the compute budget were confounded.

Fixing it required holding optimisation steps constant *and* high enough that every student converged. At 120 steps every student reached a training loss under 0.25 and the curve became monotonic, saturating at eight queries.

Two changes followed. `student_final_train_loss` is now recorded next to every fidelity number, so the check is hard to skip. And the students are step-budgeted and resumable, so a long run can be split across invocations without silently shortening anyone's training.

The general form of the error: when an attacker-side quantity varies with the thing you are measuring, you are measuring both. Worth checking before accepting any extraction curve, including the corrected one in this repo.

## Scope

**A small backbone on a synthetic rule is a testbed, not evidence about frontier models.** `distilgpt2` is 82M parameters and the decision rule is a handful of branches. Neither the query threshold nor the abstention-transfer result should be assumed to hold at scale. Reproducing on a second architecture is the most valuable next experiment, which is why `--model` is a flag with a target-module table rather than a hardcoded string.

**One seed per configuration.** On a test set of this size the standard error on a field accuracy is around 0.09, so differences under roughly 15 points are noise. The 4-to-8 query transition is larger than that. The 2-to-4 difference is not, and should not be quoted as a trend.

**Query budgets below the batch size cost less compute per step.** For K under 8 the batch is the whole query set. Gradient updates are matched across budgets, which is the control that matters, but wall-clock is not, and a wall-clock-matched attacker would look slightly different.

**The adversarial suite is now 100 categorized pressure prompts in 10 attack
categories.** They cover direct extraction, rule reconstruction, prompt
injection, instruction hierarchy conflicts, refusal bypass, context
manipulation, label flipping, reason-code extraction, boundary probing, and
multi-turn extraction. They are still not a full red-team evaluation — they
are static, not adaptive, and do not cover indirect injection vectors (e.g.
poisoned training data, or instructions embedded in retrieved documents).

**Abstention transfer is measured on one teacher.** The claim that a stolen student inherits refusal behaviour rests on a single teacher adapter at a single query budget. It is the most interesting result in the repo and the one most in need of replication.

## Data

**The corpus is synthetic and says so in every row.** It is generated from a documented rule so the decision function is known exactly. It contains no real client data, no real contract, and produces no monetary figures. It cannot tell you what a real asset is worth and is not intended to.

**The 20 real anchors are sourced for deal facts only.** Parties, prices, dates and outcomes have a resolving public source. The feature values attached to them (rights position, consent basis, regulatory regime) are an analyst's reading of the public record, not facts from it. They are marked for sign-off in `data/corpus_review.xlsx` and should not be treated as verified until someone signs them.

**Anchors are held out because they leak.** These are well-known transactions that a base model may recall. If teacher and student can both retrieve the same public facts, student/teacher agreement stops measuring extraction of the decision function.

**Eleven of the twenty real transactions currently resolve to abstention.** Some are correct: Nuance and Cerner under HIPAA are exactly what the rule should refuse on public information alone. At least two look like rule problems rather than genuine abstentions. MosaicML abstains only because its rights position is press-sourced, which is true and beside the point in a deal that bought a team rather than a corpus. Coupa abstains on the PCI branch, which may be too blunt if the spend data is tokenised. These are open, and the anchors are the fastest way to find where the rule is wrong.

## Method

**Training loss is a convergence check, never a result.** It appears in the metrics so that budgets can be compared fairly. It says nothing about decision quality, and a falling loss is compatible with a LoRA configuration that is training nothing useful.

**Macro-F1 is reported alongside accuracy for a reason.** An undertrained configuration scored well on abstention recall by abstaining on almost everything. Accuracy hid this and macro-F1 at 0.146 did not. Any single-number summary of this task will mislead.

**No baseline other than the rule and the base model.** A retrieval-over-policy-corpus baseline would be the right comparison for a product. It is not built here, because the experiment needs a known decision function to measure extraction against, and retrieval would obscure it.

**No calibration measurement.** Confidence would need to come from token logprobs, and a small model's logprobs on a fixed output schema are not a meaningful confidence signal. Left out rather than approximated badly.

**The defense mechanism is a proof-of-concept.** The output-perturbation defense (`--defense-rate`) is a synthetic noise layer applied to structured fields. It demonstrates that degrading extraction fidelity is possible without destroying utility, but it is not a production-ready safeguard. Real defenses (rate limiting, output randomisation, refusal classifiers) would need to be built as a separate layer.

**Gemma 3 4B is evaluated with 4-bit QLoRA.** The extraction curve is statistically indistinguishable from DistilGPT-2, but 4-bit quantisation adds optimisation noise. The teacher was trained for more steps to compensate. A full-precision comparison on higher-end hardware could shift the numbers.
