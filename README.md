# Mira

Mira is a small language model (SLM) you can **train and run entirely on a CPU laptop** — no GPU, no cloud, no API keys. It is a compact GPT-style decoder-only transformer with a character-level tokenizer, written in plain PyTorch in a few hundred lines.

## Why it works on a CPU

- **Character-level tokenizer** (~65 symbols): the embedding table stays tiny, so the whole model does too.
- **Small presets** (0.8M–6.5M parameters): sized so a full training run finishes in minutes to hours on a laptop.
- **Fused attention** via `torch.scaled_dot_product_attention`, which has an efficient CPU kernel.
- **No exotic dependencies**: just PyTorch.

## Quick start

```bash
pip install -r requirements.txt

# Train the ~0.8M-parameter model on the included Shakespeare corpus
python -m mira.train --data data/tinyshakespeare.txt --preset tiny --max-iters 3000

# Generate text
python -m mira.generate --checkpoint checkpoints/mira-tiny --prompt "ROMEO:"

# Or chat with it interactively
python -m mira.generate --checkpoint checkpoints/mira-tiny --interactive
```

Tip: install the CPU-only PyTorch wheel to skip the multi-GB CUDA download:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Model presets

| Preset  | Params | Layers | Heads | Embed | Context | Laptop training time* |
|---------|--------|--------|-------|-------|---------|-----------------------|
| `tiny`  | ~0.8M  | 4      | 4     | 128   | 256     | ~30–60 min            |
| `small` | ~2.7M  | 6      | 6     | 192   | 256     | ~1–3 h                |
| `base`  | ~6.5M  | 8      | 8     | 256   | 512     | an evening            |

\* for the default 3000 iterations; varies with core count. Training resumes from the last checkpoint with `--resume`, so you can stop and continue any time.

On a 4-core machine, `tiny` trains at roughly 0.9 s/iteration and drops from ~4.2 loss (random) to ~2.45 within the first 200 iterations; a full run reaches ~1.5 and produces recognizably Shakespeare-shaped dialogue.

## Training on your own text

Any UTF-8 text file works — books, code, song lyrics, your notes:

```bash
python -m mira.train --data path/to/your.txt --preset small --out checkpoints/my-mira
python -m mira.generate --checkpoint checkpoints/my-mira --interactive
```

Useful flags (see `--help` for all):

- `--max-iters`, `--batch-size`, `--lr` — the usual knobs
- `--block-size` — override the preset's context length
- `--threads N` — cap torch CPU threads (defaults to all cores)
- `--resume` — continue from the checkpoint in `--out`

Training logs per-eval losses to `train_log.jsonl` in the checkpoint directory.

## Project layout

```
mira/
├── config.py     # MiraConfig + tiny/small/base presets
├── tokenizer.py  # character-level tokenizer (JSON-serialized)
├── model.py      # GPT-style transformer: pre-norm blocks, weight tying
├── data.py       # corpus loading and random-crop batching
├── train.py      # training loop: AdamW, cosine LR schedule, checkpointing
└── generate.py   # sampling CLI (temperature + top-k), interactive mode
data/
└── tinyshakespeare.txt  # 1.1MB starter corpus
```

## What to expect

Mira is an educational-scale model: at 0.8M–6.5M parameters trained on ~1MB of text, it learns spelling, punctuation, dialogue structure, and local style — not facts or reasoning. It's a great way to watch a language model learn from scratch on hardware you already own, and a clean, hackable base for experiments (swap the tokenizer, scale the presets, try new corpora).
