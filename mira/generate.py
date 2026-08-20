"""Generate text from a trained Mira checkpoint on CPU.

Examples:
    python -m mira.generate --checkpoint checkpoints/mira-tiny --prompt "ROMEO:"
    python -m mira.generate --checkpoint checkpoints/mira-tiny --interactive
"""

import argparse
from pathlib import Path

import torch

from mira.config import MiraConfig
from mira.model import MiraModel
from mira.tokenizer import CharTokenizer


def load_model(checkpoint_dir: str | Path) -> tuple[MiraModel, CharTokenizer]:
    checkpoint_dir = Path(checkpoint_dir)
    ckpt = torch.load(checkpoint_dir / "model.pt", map_location="cpu")
    config = MiraConfig.from_dict(ckpt["config"])
    model = MiraModel(config)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tokenizer = CharTokenizer.load(checkpoint_dir / "tokenizer.json")
    return model, tokenizer


def sample(
    model: MiraModel,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    ids = tokenizer.encode(prompt) or [0]
    idx = torch.tensor([ids], dtype=torch.long)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from a trained Mira model")
    parser.add_argument("--checkpoint", default="checkpoints/mira-tiny", help="checkpoint directory")
    parser.add_argument("--prompt", default="\n", help="text to continue")
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--interactive", action="store_true", help="read prompts in a loop")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, tokenizer = load_model(args.checkpoint)
    print(f"loaded {args.checkpoint} ({model.num_params() / 1e6:.2f}M params)")

    if args.interactive:
        print("type a prompt and press enter (ctrl-d or ctrl-c to quit)")
        while True:
            try:
                prompt = input("mira> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            print(sample(model, tokenizer, prompt or "\n", args.max_new_tokens,
                         args.temperature, args.top_k))
            print("-" * 60)
    else:
        print(sample(model, tokenizer, args.prompt, args.max_new_tokens,
                     args.temperature, args.top_k))


if __name__ == "__main__":
    main()
