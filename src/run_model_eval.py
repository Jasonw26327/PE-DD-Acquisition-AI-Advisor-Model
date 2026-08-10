#!/usr/bin/env python3
"""
Run a model (via Ollama API or HuggingFace transformers) against the PE
acquisition advisor test set, and score the structured outputs.

Usage (Ollama / Gemma 4):
    python src/run_model_eval.py --ollama gemma4:12b --label "gemma4-12b-ollama"

Usage (HuggingFace / DistilGPT-2, CPU):
    python src/run_model_eval.py --hf distilgpt2 --label "distilgpt2"

Usage (HuggingFace / Gemma 3, GPU with 4-bit QLoRA):
    python src/run_model_eval.py --hf google/gemma-3-4b-it --load-4bit \
        --label "gemma3-4b"

The script loads the corpus, formats prompts in the schema format, decodes
the structured RIGHTS=/TIER=/ACT= fields, and writes metrics to args.out.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from corpus_builder import decide, SCHEMA_VERSION


def load_cases():
    """Load test + anchor + adversarial cases from the corpus."""
    corpus_path = _find_corpus()
    if not corpus_path:
        sys.exit("Run `python src/corpus_builder.py` first to generate data/corpus.jsonl")

    train, val, test, anchor_eval, adversarial_eval, stats = load_corpus()

    # Use test + anchor_eval + adversarial for zero-shot evaluation
    eval_cases = list(test) + list(anchor_eval) + list(adversarial_eval)
    return eval_cases, stats


def _find_corpus():
    candidates = [
        "data/corpus.jsonl",
        "../data/corpus.jsonl",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return os.path.abspath(p)
    return None


from testbed import load_corpus, parse, score, held_rate, macro_f1
import re


def format_instruction_prompt(features, abstain_hint=True):
    """Format a case as an instruction-following prompt.

    The model must output in the strict schema:
      RIGHTS=<LOW|MED|HIGH|UNK> TIER=<A|B|C|X> ACT=<AIBUYER|SPONSOR|RENEG|ABSTAIN>;
    """
    f = features
    prompt = (
        "You are an AI buyer due-diligence advisor.\n\n"
        f"Sector: {f['sector']}\n"
        f"Data type: {f['data_type']}\n"
        f"Volume: {f['volume']}\n"
        f"Review: {f['review']}\n"
        f"Rights: {f['rights']}\n"
        f"Consent: {f['consent']}\n"
        f"Regime: {f['regime']}\n"
        f"Evidence source: {f['evidence_source']}\n"
        f"Deal context: {f['deal_context']}\n\n"
        "Classify the acquisition according to this rule:\n"
        "  RIGHTS = the strength of training/inference rights (LOW/MED/HIGH/UNK)\n"
        "  TIER = investment tier (A/B/C/X)\n"
        "  ACT = action to take (AIBUYER, SPONSOR, RENEG, ABSTAIN)\n\n"
        "Respond with EXACTLY: RIGHTS=<...> TIER=<...> ACT=<...>;\n"
        "If you cannot determine the answer, abstain (ACT=ABSTAIN).\n\n"
        f"Case: {f['sector']} / {f['data_type']} / {f['volume']} volume\n"
        "Response:"
    )
    return prompt


def query_ollama(model_name, prompt, base_url="http://localhost:11434"):
    """Send a prompt to an Ollama model and return the generated text."""
    import urllib.request

    # Gemma 4 models use internal "thinking" that can consume tokens.
    # Gemma 3 models are faster but still benefit from a generous token budget.
    model_name_lower = model_name.lower()
    if "gemma4" in model_name_lower or "gemma-4" in model_name_lower:
        num_predict = 500
    else:
        num_predict = 200

    url = f"{base_url}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 1.0,
            "num_ctx": 1024,
            "num_predict": num_predict,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers)

    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read())
    return result.get("response", "")


def query_hf(distilgpt2_model, tokenizer, prompt, model, max_new_tokens=48):
    """Generate using a HuggingFace model."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(gen[0], skip_special_tokens=True)
    # Return just the generated part (after the prompt)
    return text[len(prompt):].strip()


def evaluate(cases_raw, fn_generate, label, stats):
    """Run a model against cases and score the results."""
    from testbed import corpus_to_cases

    # Convert raw corpus records to the case format that score() expects
    cases = corpus_to_cases(cases_raw)
    prompts = [format_instruction_prompt(c["features"], abstain_hint=True)
               for c in cases_raw]

    print(f"[{label}] Generating {len(prompts)} responses...")
    t0 = time.time()
    gens = []
    for i, prompt in enumerate(prompts):
        try:
            out = fn_generate(prompt)
            gens.append(out.strip())
        except Exception as e:
            print(f"  [{label}] error on case {i}: {e}")
            gens.append("")
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{label}] {i+1}/{len(prompts)} done ({elapsed:.1f}s, "
                  f"{(i+1)/elapsed:.1f} cases/s)")

    # Score
    results = score(cases, gens)
    results["n_evaluated"] = len(cases)
    results["held_out_abstention_rate"] = held_rate(gens)
    results["generation_time_seconds"] = round(time.time() - t0, 1)

    # Category breakdown for adversarial cases
    cat_data = {}
    for c, g in zip(cases, gens):
        cat = c.get("category", "standard")
        if cat not in cat_data:
            cat_data[cat] = {"total": 0, "held": 0}
        cat_data[cat]["total"] += 1
        if parse(g)["action"] == "ABSTAIN":
            cat_data[cat]["held"] += 1

    results["adversarial_by_category"] = {
        k: {"held": v["held"], "total": v["total"],
            "rate": round(v["held"] / v["total"], 3) if v["total"] else 0.0}
        for k, v in cat_data.items()
    }

    # Provenance
    results["provenance"] = {
        "model_label": label,
        "evaluation_type": "zero-shot (no LoRA adapter applied)",
        "n_evaluated": len(cases),
        "splits_combined": "test + anchor_eval + adversarial_eval",
        "corpus_hash": stats.get("corpus_sha256_16"),
        "dataset_version": stats.get("version"),
        "corpus_counts": stats.get("counts"),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_method": "zero-shot via API" if fn_generate.__name__ == "query_ollama" else "zero-shot via transformers",
    }

    return results


def main():
    p = argparse.ArgumentParser(description="Evaluate a model against the PE advisor corpus")
    p.add_argument("--ollama", default=None, help="Ollama model name (e.g. gemma4:12b)")
    p.add_argument("--hf", default=None, help="HuggingFace model name (e.g. distilgpt2)")
    p.add_argument("--load-4bit", action="store_true", help="Use 4-bit QLoRA (HF models)")
    p.add_argument("--label", required=True, help="Label for this run (used in output filename)")
    p.add_argument("--out", default=None, help="Output file path (default: results/metrics_<label>.json)")
    args = p.parse_args()

    cases, stats = load_cases()
    print(f"Loaded {len(cases)} evaluation cases from corpus")
    print(f"  Corpus stats: {json.dumps(stats.get('counts', {}), indent=2)}")

    if args.ollama:
        fn = lambda prompt, m=args.ollama: query_ollama(m, prompt)
        model_label = f"ollama:{args.ollama}"
    elif args.hf:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(args.hf)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        kwargs = {"torch_dtype": "auto"}
        if args.load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(args.hf, **kwargs)
        fn = lambda prompt, m=model, t=tok: query_hf(args.hf, t, prompt, m)
        model_label = f"hf:{args.hf}"
    else:
        sys.exit("Must specify --ollama or --hf")

    results = evaluate(cases, fn, args.label, stats)

    out_path = args.out or f"results/metrics_{args.label.replace('-', '_')}.json"
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== {args.label} results ===")
    print(json.dumps({k: v for k, v in results.items()
                      if k not in ("adversarial_by_category", "provenance")}, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()