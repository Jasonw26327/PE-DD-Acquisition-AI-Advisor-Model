#!/usr/bin/env python3
"""Check the environment before running the testbed, and recommend a backbone.

    python preflight.py

Reports Python and package versions, whether CUDA or MPS is available, how much
VRAM you have, and which --model setting fits. Then runs a 20-second smoke test:
loads distilgpt2, attaches a LoRA adapter, takes two optimisation steps and
generates once. If that passes, the full script will run.
"""

import platform
import shutil
import sys

MIN_PY = (3, 9)

RECOMMENDATIONS = [
    # (min VRAM GB, model, flags, note)
    (24, "Qwen/Qwen3-8B", "--load-4bit --batch-size 2",
     "headroom for longer structured outputs; do not use a larger model to "
     "cover weak data or evaluation"),
    (12, "Qwen/Qwen3-4B-Instruct-2507", "--load-4bit --batch-size 4",
     "best balance of a credible modern backbone and feasible adapter training"),
    (8, "Qwen/Qwen3-1.7B", "--load-4bit --batch-size 4",
     "comfortable on an 8GB card, still far more current than distilgpt2"),
    (0, "Qwen/Qwen3-0.6B", "--batch-size 4",
     "CPU or small GPU. Slow on CPU; use distilgpt2 for the first pass"),
]


def main():
    print(f"python   {platform.python_version()}  ({platform.machine()})")
    if sys.version_info < MIN_PY:
        print(f"  needs >= {'.'.join(map(str, MIN_PY))}")
        return 1

    try:
        import torch
    except ImportError:
        print("torch    MISSING")
        print("\n  pip install torch --index-url "
              "https://download.pytorch.org/whl/cpu")
        print("  pip install -r requirements.txt")
        return 1
    print(f"torch    {torch.__version__}")

    missing = []
    for mod in ("transformers", "peft", "datasets", "accelerate"):
        try:
            m = __import__(mod)
            print(f"{mod:<9}{getattr(m, '__version__', 'unknown')}")
        except ImportError:
            print(f"{mod:<9}MISSING")
            missing.append(mod)
    if missing:
        print("\n  pip install -r requirements.txt")
        return 1

    vram = 0
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\naccel    CUDA, {torch.cuda.get_device_name(0)}, "
              f"{vram:.0f} GB VRAM")
        try:
            import bitsandbytes  # noqa: F401
            print("4bit     bitsandbytes present, --load-4bit available")
        except ImportError:
            print("4bit     bitsandbytes missing; install "
                  "requirements-gpu.txt to use --load-4bit")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("\naccel    Apple MPS. Works without --load-4bit; bitsandbytes "
              "is NVIDIA-only, so skip 4-bit here.")
    else:
        print(f"\naccel    CPU only, {shutil.os.cpu_count()} cores")

    print("\nrecommended backbone")
    for min_vram, model, flags, note in RECOMMENDATIONS:
        if vram >= min_vram:
            print(f"  --model {model} {flags}")
            print(f"  {note}")
            break
    print("  start with the default distilgpt2 either way: it is the fastest "
          "way to confirm the harness runs before you spend GPU time.")

    print("\nsmoke test ...")
    try:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("distilgpt2")
        tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        model = get_peft_model(model, LoraConfig(
            r=8, lora_alpha=16, target_modules=["c_attn", "c_proj"],
            task_type="CAUSAL_LM"))
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3)
        batch = tok(["case bpo|hi|expert|full > RISK=LOW TIER=A ACT=AIBUYER;"],
                    return_tensors="pt")
        batch["labels"] = batch["input_ids"].clone()
        for _ in range(2):
            loss = model(**batch).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
        out = model.generate(**tok("case bpo|hi|expert|full >",
                                   return_tensors="pt"),
                             max_new_tokens=8, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        print(f"  trained 2 steps, final loss {loss.item():.3f}")
        print(f"  generated: {tok.decode(out[0], skip_special_tokens=True)!r}")
        print("\nready. next:  python ai_buyer_due_diligence_finetune_v4.py "
              "--stage all")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        print("  fix this before running the full script")
        return 1


if __name__ == "__main__":
    sys.exit(main())
