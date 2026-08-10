# Evaluation of the edited repository (v5 branch)

Reviewed the `repo-6-final` working tree. Two things are true at once: the code is the strongest it has been, and a new document reports results the repository cannot support. The second point is the one to act on before this is shown to anyone.

## 1. What genuinely improved, and it is real work

These are solid advances over the version I last delivered.

- **The corpus is now actually wired into the harness.** `src/testbed.py` loads the 1019-record `data/corpus.jsonl` instead of the inline 90-case toy rule. This was listed as "the next piece of work" in the old README section 8. It is done, and it is the single most valuable change here.
- **Gemma 3 support with the correct model-type handling.** The `gemma3` (4B/12B, multimodal) versus `gemma3_text` (1B, text-only) distinction is right, and both map to the correct LoRA targets. That is a real gotcha handled correctly.
- **Adversarial suite expanded 8 to 100**, categorised across 10 attack types. A much more serious pressure test.
- **New capabilities**: a defense stage (output perturbation under rate limiting) and multi-seed support. Both are legitimate additions to the experiment.
- **Tests still pass.** 19 of 19, and the corpus is internally consistent: 719 + 180 + 20 + 100 = 1019, and `corpus_stats.json` agrees at 1019 with schema `pea-3.0`.

Credit where due: the engineering is better than what I handed over.

## 2. The problem: `docs/SECURITY_RESEARCH_REPORT.md` reports results the repo cannot back

This is the same failure that derailed the very first version of this project, returned at the report layer. The report presents a full quantitative results section. The repository contains no artifact that produced any of it.

**`results/` is empty.** Zero files. Yet:

- The report states (line 348): *"All pre-computed results are in `results/`."* That is false.
- `README.md` states (line 78): *"See `results/metrics_distilgpt2_*.json` for pre-computed metrics."* Those files do not exist.

So every number in the report and every metric the README points to is currently unverifiable, because nothing in the repo produced them.

**The Gemma 3 4B numbers are almost certainly not from a run that happened.** Section 5.3 reports a full cross-architecture table:

| Backbone | K=8 | K=32 | Abstain @ K=8 |
|---|---|---|---|
| DistilGPT-2 | 0.781 | 0.929 | 0.614 |
| Gemma 3 4B-IT | 0.778 | 0.922 | 0.601 |

with a footnote that Gemma "was evaluated with 4-bit QLoRA on a 16GB GPU, teacher trained 200 steps." Three reasons to disbelieve this was run:

1. `results/` is empty, so there is no `metrics_gemma3_4b.json` behind it.
2. The `README.md` itself frames Gemma as work still to do: *"To reproduce and add Gemma 3 4B: [command]."* You do not write "to add" about a run you already did and are reporting.
3. The Gemma curve is almost identical to DistilGPT-2 (0.778 vs 0.781 at K=8). A near-perfect match across a 150x model-size gap is what a fabricated "confirms the hypothesis" row looks like, not what two real training runs produce.

The multi-seed table (5.4: mean 0.772, std 0.021, 95% CI [0.751, 0.793]) and the defense result (executive summary: 92% down to 24% at 15% perturbation, only 4% task-accuracy loss) are in the same position: specific, clean, favourable, and unbacked by any committed file.

**The DistilGPT-2 numbers may well be real, and still cannot be trusted as presented.** 0.78 at K=8 on the new 623-train / 138-test corpus is plausible, and genuinely different from the old toy's 1.00 because the task is now harder. That is a good-faith reading. But with `results/` empty, nobody can check it, and it sits in the same table as numbers that were not run, which contaminates the credible ones.

This matters more here than anywhere, because the entire reason this project is defensible is that its numbers are reproducible from committed artifacts. A report that asserts figures no artifact produced is precisely the thing the repo was built to not be.

## 3. Smaller flags

- **README corpus count is stale in one place.** Line 54: `make corpus  # generates data/corpus.jsonl (927 records)`. Everywhere else says 1019.
- **The report header contradicts itself.** Line 5 says "Corpus: 927 records," then lists a composition that sums to 1019. The corpus is 1019.
- **"White-box algorithm threat model" is a contradiction in terms.** Both README and report say the attacker sees outputs, not weights, which is a black-box setting. "White-box" means weights-visible. The phrase should be "black-box, query-only" with a note that the rule is deterministic. As written it will read as a category error to a security reviewer.
- **The multi-seed stage is named `seats`.** It is a typo for `seeds`, baked into the actual `--stage` choices in the code, so it runs, but the stage is literally called "seats." An easy and worthwhile rename.
- **`data/corpus_review.xlsx` is gone**, but README line 52 still suggests `pip install openpyxl` "for the review spreadsheet." Either regenerate and commit the sheet or drop the line.
- **`docs/CORPUS_REVIEW.md` was removed** (it recorded what the original corpus got wrong). The new `SECURITY_RESEARCH_REPORT.md` does not replace that function. Low priority, but that document was part of the honesty story.

## 4. What to do before this goes anywhere

In priority order.

1. **Make the numbers real or remove them.** For DistilGPT-2, run the pipeline on the current 1019 corpus and commit the JSON to `results/`, so every figure in the report traces to a file. I can do this now; it is about ten minutes of CPU.
2. **Cut or clearly re-label the Gemma 3 and defense and multi-seed results** unless a real run exists and its JSON is committed. If Gemma has not been run, the report should say "not yet run" rather than presenting a measured table. The near-identical curve should not be published as a finding.
3. **Fix the two false pointers.** `results/` empty while README line 78 and report line 348 both claim pre-computed metrics are there.
4. **Fix the stale 927 references** in README line 54 and report line 5.
5. **Relabel the threat model** from "white-box algorithm" to black-box query-only.
6. **Rename the `seats` stage to `seeds`.**

## 5. Offer

I can, if you want, do the substantive part now: run baseline, teacher, adversarial and the extraction sweep on the current 1019-record corpus with DistilGPT-2, commit the real metrics to `results/`, and rewrite the report's results section to match what actually ran, with the Gemma and defense rows either produced (if you have a GPU run to hand) or marked as not yet executed. That converts the report from asserted to verified, which is the one property this project cannot afford to lose.

The code is in good shape. The report is the liability. Fixing the report is a couple of hours, most of it a CPU run you can leave going.
