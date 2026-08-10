#!/usr/bin/env python3
"""Preflight checker for adapter-extraction-testbed.

Validates that the environment can run the corpus builder and the testbed,
and recommends model backbones based on available hardware.

Usage:
    python preflight.py
"""
import json
import os
import subprocess
import sys


def check(label, fn):
    try:
        result = fn()
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {label}")
        return result
    except Exception as e:
        print(f"  [WARN] {label}: {e}")
        return False


def main():
    print("=== Preflight: adapter-extraction-testbed ===\n")

    print("1. Python packages")
    def has_torch():
        import torch; print(f"       torch {torch.__version__}, cuda={torch.cuda.is_available()}")
        return True
    def has_transformers():
        import transformers; print(f"       transformers {transformers.__version__}")
        return True
    def has_peft():
        import peft; print(f"       peft {peft.__version__}")
        return True
    def has_datasets():
        import datasets; print(f"       datasets {datasets.__version__}")
        return True

    check("torch", has_torch)
    check("transformers", has_transformers)
    check("peft", has_peft)
    check("datasets", has_datasets)

    print("\n2. Hardware")
    def hw_info():
        try:
            import torch
            if torch.cuda.is_available():
                print(f"       GPU: {torch.cuda.get_device_name(0)}")
                print(f"       GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                print("       GPU: None (CPU only)")
            import os
            ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            print(f"       RAM: {ram / 1e9:.1f} GB")
        except Exception:
            import os
            mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            print(f"       RAM: {mem / 1e9:.1f} GB")
        return True
    check("hardware detection", hw_info)

    print("\n3. Model backbone recommendations")
    print("  For this experiment you need two backbones:")
    print("    - Teacher: distilgpt2 (82M params, CPU-friendly, fast iteration)")
    print("    - Second backbone: Gemma 3 4B-IT (12-layer, 512-hidden, 4-bit QLoRA)")
    print("")
    print("  Why Gemma 3 4B and not Gemma 3 12B:")
    print("    - 12B requires 24+ GB VRAM for fp16 inference (unavailable on laptops)")
    print("    - 4B runs with 4-bit quantization on 12-16 GB VRAM, or on CPU with patience")
    print("    - The research question ('extraction across architectures') is answered")
    print("      by comparing GPT-2-style (distilgpt2) vs modern transformer (Gemma 3)")
    print("      — 4B is sufficient; 12B adds cost with no additional evidence value.")
    print("")
    print("  For comparison, the target modules mapping supports:")
    from testbed import TARGET_MODULES
    for mt in TARGET_MODULES:
        print(f"    - {mt}: {TARGET_MODULES[mt]}")

    print("\n4. Corpus builder")
    def build():
        result = subprocess.run(
            [sys.executable, "src/corpus_builder.py", "--n", "30", "--pairs", "6",
             "--adversarial", "100", "--out-prefix", "/tmp/preflight_corpus", "--no-xlsx"],
            capture_output=True, timeout=30)
        if result.returncode != 0:
            print(f"       stderr: {result.stderr.decode()[:200]}")
        return result.returncode == 0
    check("corpus builder (mini run)", build)

    print("\n5. Test suite (schema/rule tests only, no models)")
    def tests():
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_corpus.py", "-q"],
            capture_output=True, timeout=120)
        if result.returncode != 0:
            print(f"       stderr: {result.stderr.decode()[:500]}")
        return result.returncode == 0
    check("pytest tests/test_corpus.py", tests)

    print("\n=== Preflight complete ===")


if __name__ == "__main__":
    sys.path.insert(0, "src")
    main()
