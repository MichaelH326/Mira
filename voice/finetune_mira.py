"""
finetune_mira.py — Fine-tune a pretrained small LM (360M to ~2B) on CPU with
LoRA so Mira speaks in the casual phone-call voice, then merge and save a
plain HF model directory ready for package_mdlo.py.

Why fine-tune instead of training from scratch: pretraining a 500M+ model
needs billions of tokens — years of CPU time. Fine-tuning an open base
(SmolLM2, Qwen2.5, Llama 3.2) needs only thousands of examples and gives a
model that genuinely speaks. CPU-friendly because LoRA trains <1% of weights.

Usage:
    python make_voice_dataset.py --out voice_data.jsonl
    python finetune_mira.py --base HuggingFaceTB/SmolLM2-360M-Instruct \
        --data voice_data.jsonl --out mira-hf --steps 300

    # packaging-only mode (no training, persona comes from the system prompt):
    python finetune_mira.py --base Qwen/Qwen2.5-0.5B-Instruct --out mira-hf --steps 0
"""

import os
import json
import time
import math
import random
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"
)


def load_conversations(path):
    convs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                convs.append(json.loads(line)["messages"])
    return convs


def encode_conversation(tok, conv, seq_len, loss_on="assistant"):
    """Tokenize one conversation. Returns (input_ids, labels) or None.

    With loss_on="assistant", labels are -100 everywhere except Mira's own
    replies (and the end-of-turn token after each, so she learns to stop).
    Training on every token would spend most of the gradient memorizing the
    system prompt, identical in every example, and imitating the caller's
    lines; masking puts all of it on how Mira responds to what was said.
    """
    text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
    enc = tok(text, truncation=True, max_length=seq_len,
              return_offsets_mapping=loss_on == "assistant")
    ids = enc["input_ids"]
    if loss_on != "assistant":
        return ids, list(ids)

    # Character span of each assistant turn: from where its content starts
    # (the prompt rendered up to that turn, plus the generation header) to
    # the end of the turn rendered in full.
    spans = []
    for i, msg in enumerate(conv):
        if msg["role"] != "assistant":
            continue
        start = tok.apply_chat_template(conv[:i], tokenize=False, add_generation_prompt=True)
        end = tok.apply_chat_template(conv[: i + 1], tokenize=False, add_generation_prompt=False)
        if not (text.startswith(end) and end.startswith(start)):
            raise ValueError("chat template is not prefix-stable; use --loss_on all")
        spans.append((len(start), len(end)))

    labels = [-100] * len(ids)
    for k, (s, _) in enumerate(enc["offset_mapping"]):
        if any(a <= s < b for a, b in spans):
            labels[k] = ids[k]
    if all(l == -100 for l in labels):
        return None        # every reply was truncated away: nothing to learn
    return ids, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="HuggingFaceTB/SmolLM2-360M-Instruct",
                   help="HF model id or local path (SmolLM2-360M/1.7B, Qwen2.5-0.5B/1.5B, Llama-3.2-1B...)")
    p.add_argument("--data", default="voice_data.jsonl")
    p.add_argument("--out", default="mira-hf")
    p.add_argument("--steps", type=int, default=150,
                   help="optimizer steps; 0 = skip training and just re-save the base (persona-only mode). "
                        "The dataset is template-based, so too many steps memorizes the templates and "
                        "erodes the base model's general ability — 100-150 is usually the sweet spot.")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--loss_on", choices=["assistant", "all"], default="assistant",
                   help="assistant = learn only from Mira's replies (recommended); "
                        "all = also from the system prompt and the caller's lines")
    p.add_argument("--save_every", type=int, default=25,
                   help="save a resumable LoRA checkpoint every N steps (0 = off)")
    p.add_argument("--resume", action="store_true",
                   help="resume from the last checkpoint in <out>/ckpt if present")
    p.add_argument("--grad_checkpoint", action="store_true",
                   help="trade speed for memory (recommended for 1B+ bases)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"CPU fine-tune: {os.cpu_count()} threads, base={args.base}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.chat_template is None:
        tok.chat_template = CHATML_TEMPLATE  # embedded into the GGUF later
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32)
    n_params = sum(pr.numel() for pr in model.parameters())
    print(f"Base model: {n_params/1e6:.0f}M parameters", flush=True)

    if args.steps <= 0:
        print("steps=0: packaging-only mode, saving base model unchanged")
        os.makedirs(args.out, exist_ok=True)
        model.save_pretrained(args.out)
        tok.save_pretrained(args.out)
        return

    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.train()

    convs = load_conversations(args.data)
    print(f"Dataset: {len(convs)} conversations", flush=True)

    print("Tokenizing (this takes a couple of minutes)...", flush=True)
    encoded = [e for c in convs
               if (e := encode_conversation(tok, c, args.seq_len, args.loss_on))]
    n_tok = sum(len(ids) for ids, _ in encoded)
    n_lab = sum(sum(l != -100 for l in labels) for _, labels in encoded)
    print(f"Tokenized {len(encoded)} conversations; loss on {n_lab:,} of {n_tok:,} tokens "
          f"({100 * n_lab / max(n_tok, 1):.0f}%, --loss_on {args.loss_on}); starting training",
          flush=True)

    def get_batch(rng):
        picks = rng.sample(encoded, args.batch_size)
        maxlen = max(len(ids) for ids, _ in picks)
        input_ids, labels, attn = [], [], []
        for ids, lab in picks:
            pad = maxlen - len(ids)
            input_ids.append(ids + [tok.pad_token_id] * pad)
            labels.append(lab + [-100] * pad)          # no loss on padding
            attn.append([1] * len(ids) + [0] * pad)
        return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn))

    optim = torch.optim.AdamW([pr for pr in model.parameters() if pr.requires_grad], lr=args.lr)
    rng = random.Random(args.seed)

    ckpt_dir = os.path.join(args.out, "ckpt")
    ckpt_file = os.path.join(ckpt_dir, "state.pt")
    start_step = 1
    if args.resume and os.path.exists(ckpt_file):
        state = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        optim.load_state_dict(state["optim"])
        rng.setstate(state["rng"])
        start_step = state["step"] + 1
        print(f"Resumed from step {state['step']} ({ckpt_file})", flush=True)

    def save_ckpt(step):
        os.makedirs(ckpt_dir, exist_ok=True)
        tmp = ckpt_file + ".tmp"
        torch.save({
            # only the trainable LoRA params: small and fast to write
            "model": {k: v for k, v in model.state_dict().items() if "lora" in k.lower()},
            "optim": optim.state_dict(),
            "rng": rng.getstate(),
            "step": step,
        }, tmp)
        os.replace(tmp, ckpt_file)   # atomic: never leaves a half-written file
        print(f"  [checkpoint saved at step {step}]", flush=True)

    t0, running = time.time(), None
    for step in range(start_step, args.steps + 1):
        optim.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(args.grad_accum):
            input_ids, labels, attn = get_batch(rng)
            loss = model(input_ids=input_ids, labels=labels, attention_mask=attn).loss / args.grad_accum
            loss.backward()
            total += loss.item()
        optim.step()
        running = total if running is None else 0.9 * running + 0.1 * total
        if step % 10 == 0 or step == 1 or step == args.steps:
            done = step - start_step + 1
            print(f"step {step:4d}/{args.steps}  loss {total:.4f}  ema {running:.4f}  "
                  f"({time.time()-t0:.0f}s, {(time.time()-t0)/done:.1f}s/step)", flush=True)
        if args.save_every and step % args.save_every == 0 and step != args.steps:
            save_ckpt(step)

    print("Merging LoRA into base weights...", flush=True)
    model = model.merge_and_unload()
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"Saved merged model to {args.out}/")


if __name__ == "__main__":
    main()
