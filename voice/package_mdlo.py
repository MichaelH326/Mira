"""
package_mdlo.py — Convert a fine-tuned HF model directory into a single
self-contained `mira.mdlo` v2 file.

Pipeline:  HF dir --(llama.cpp converter)--> GGUF --(optional quantize)-->
           wrapped in the MDLO v2 container (JSON header + GGUF payload).

The v2 header carries the persona (system prompt), generation settings, and
provenance, so run_mira.py needs nothing but the .mdlo file itself.

Usage:
    python package_mdlo.py --model mira-hf --out mira.mdlo \
        --llama_cpp ./llama.cpp --quant Q4_K_M --quantize_bin ./llama-quantize
    # --quant q8_0 needs no external binary (the converter emits it directly)
"""

import os
import json
import struct
import hashlib
import argparse
import subprocess
import sys
import tempfile

MDLO_MAGIC = b"MDLO"
MDLO_VERSION = 2

DEFAULT_SYSTEM_PROMPT = (
    "You are Mira, a friendly voice assistant on a casual phone call. "
    "Speak in short, natural spoken sentences with contractions, like a relaxed "
    "conversation between friends. Never use lists, markdown, emojis, or symbols; "
    "your words go straight to text to speech. Say numbers the way people say them "
    "out loud. Keep most replies to one or two short sentences. If you don't have "
    "enough context to answer well, don't guess: ask one short, specific follow-up "
    "question instead."
)


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mira-hf", help="merged HF model directory")
    p.add_argument("--out", default="mira.mdlo")
    p.add_argument("--llama_cpp", default="llama.cpp", help="path to a llama.cpp checkout")
    p.add_argument("--quant", default="q8_0",
                   help="q8_0/f16 (converter-native) or Q4_K_M/Q5_K_M/... (needs --quantize_bin)")
    p.add_argument("--quantize_bin", default=None, help="path to llama-quantize binary for K-quants")
    p.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--base_model", default="unknown", help="recorded in header for provenance")
    p.add_argument("--finetune_steps", type=int, default=-1, help="recorded in header")
    args = p.parse_args()

    convert = os.path.join(args.llama_cpp, "convert_hf_to_gguf.py")
    if not os.path.exists(convert):
        raise SystemExit(f"converter not found at {convert} — pass --llama_cpp path/to/llama.cpp")

    with tempfile.TemporaryDirectory() as td:
        native = args.quant.lower() in ("f16", "f32", "bf16", "q8_0")
        first_type = args.quant.lower() if native else "f16"
        gguf_a = os.path.join(td, f"model-{first_type}.gguf")
        env = dict(os.environ, PYTHONPATH=os.path.join(args.llama_cpp, "gguf-py"))
        run([sys.executable, convert, args.model, "--outfile", gguf_a, "--outtype", first_type], env=env)

        gguf_final = gguf_a
        if not native:
            if not args.quantize_bin or not os.path.exists(args.quantize_bin):
                raise SystemExit(f"--quant {args.quant} needs --quantize_bin (llama-quantize). "
                                 f"Use --quant q8_0 to avoid the binary.")
            gguf_final = os.path.join(td, f"model-{args.quant}.gguf")
            run([args.quantize_bin, gguf_a, gguf_final, args.quant])

        with open(gguf_final, "rb") as f:
            payload = f.read()

    header = {
        "format": "mdlo",
        "format_version": MDLO_VERSION,
        "model_name": "mira",
        "engine": "gguf",           # payload is a complete GGUF file
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_bytes": len(payload),
        "chat": {
            "system_prompt": args.system_prompt,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_reply_tokens": 120,
            "style": "casual-voice",
        },
        "provenance": {
            "base_model": args.base_model,
            "finetune_steps": args.finetune_steps,
            "quantization": args.quant,
        },
    }
    hb = json.dumps(header).encode("utf-8")

    with open(args.out, "wb") as f:
        f.write(MDLO_MAGIC)
        f.write(struct.pack("<I", MDLO_VERSION))
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(payload)
    print(f"Wrote {args.out} ({(16 + len(hb) + len(payload))/1e6:.1f} MB, quant={args.quant})")


if __name__ == "__main__":
    main()
