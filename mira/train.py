"""Train Mira on a text corpus, entirely on CPU.

Example:
    python -m mira.train --data data/tinyshakespeare.txt --preset tiny \
        --max-iters 3000 --out checkpoints/mira-tiny
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from mira.config import PRESETS, MiraConfig
from mira.data import get_batch, load_corpus, prepare_data
from mira.model import MiraModel
from mira.tokenizer import SPECIAL_TOKENS, USER_TOK, CharTokenizer


def get_lr(it: int, warmup: int, max_iters: int, max_lr: float, min_lr: float) -> float:
    if it < warmup:
        return max_lr * (it + 1) / warmup
    if it >= max_iters:
        return min_lr
    ratio = (it - warmup) / (max_iters - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(
    model: MiraModel,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
) -> dict[str, float]:
    model.eval()
    out = {}
    for name, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(
    out_dir: Path,
    model: MiraModel,
    tokenizer: CharTokenizer,
    iter_num: int,
    val_loss: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": model.config.to_dict(),
            "iter": iter_num,
            "val_loss": val_loss,
        },
        out_dir / "model.pt",
    )
    tokenizer.save(out_dir / "tokenizer.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Mira on a CPU")
    parser.add_argument("--data", default="data/tinyshakespeare.txt", help="path to a UTF-8 text file")
    parser.add_argument("--preset", default="tiny", choices=sorted(PRESETS), help="model size preset")
    parser.add_argument("--out", default=None, help="checkpoint directory (default: checkpoints/mira-<preset>)")
    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=None, help="override the preset's context length")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = all cores)")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", action="store_true", help="resume from an existing checkpoint in --out")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    out_dir = Path(args.out or f"checkpoints/mira-{args.preset}")

    text = load_corpus(args.data)
    # Dialogue-formatted data gets the special turn markers as single tokens.
    specials = SPECIAL_TOKENS if USER_TOK in text else []
    tokenizer = CharTokenizer.from_text(text, specials)
    train_data, val_data = prepare_data(text, tokenizer)

    start_iter = 0
    if args.resume and (out_dir / "model.pt").exists():
        ckpt = torch.load(out_dir / "model.pt", map_location="cpu")
        config = MiraConfig.from_dict(ckpt["config"])
        tokenizer = CharTokenizer.load(out_dir / "tokenizer.json")
        model = MiraModel(config)
        model.load_state_dict(ckpt["model"])
        start_iter = ckpt["iter"] + 1
        print(f"resumed from {out_dir} at iter {start_iter}")
    else:
        config = PRESETS[args.preset]
        config.vocab_size = tokenizer.vocab_size
        if args.block_size:
            config.block_size = args.block_size
        model = MiraModel(config)

    n_params = model.num_params()
    print(f"mira-{args.preset}: {n_params / 1e6:.2f}M params, vocab={config.vocab_size}, "
          f"block={config.block_size}, corpus={len(text):,} chars "
          f"({len(train_data):,} train / {len(val_data):,} val tokens)")

    # Fused AdamW is not available on CPU; the default implementation is fine.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )

    model.train()
    best_val = float("inf")
    log_path = out_dir / "train_log.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for it in range(start_iter, args.max_iters):
        lr = get_lr(it, args.warmup, args.max_iters, args.lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, config.block_size, args.batch_size)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if it % 50 == 0:
            elapsed = time.time() - t0
            print(f"iter {it:5d} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.0f}s")

        last_iter = it == args.max_iters - 1
        if (it > 0 and it % args.eval_interval == 0) or last_iter:
            losses = estimate_loss(
                model, train_data, val_data, config.block_size, args.batch_size, args.eval_iters
            )
            print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"iter": it, **losses, "time": time.time() - t0}) + "\n")
            if losses["val"] < best_val or last_iter:
                best_val = min(best_val, losses["val"])
                save_checkpoint(out_dir, model, tokenizer, it, losses["val"])
                print(f"saved checkpoint to {out_dir} (val {losses['val']:.4f})")

    print(f"done in {time.time() - t0:.0f}s | best val loss {best_val:.4f}")


if __name__ == "__main__":
    main()
