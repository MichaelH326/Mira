"""Mira assistant: an interactive, fully offline, CPU-only chat runtime.

Reliability layers on top of the trained model:

1. The model itself is instruction-tuned to answer known facts directly, ask a
   follow-up question when required context is missing, and say "I don't know"
   instead of inventing facts (see mira/instruct_data.py).
2. Greedy decoding by default — reproducible and most faithful to training.
3. A confidence gate: the mean per-token log-probability of the generated
   reply is measured, and low-confidence replies are replaced with an honest
   fallback instead of being shown as if they were reliable.

Usage:
    python -m mira.assistant --checkpoint checkpoints/mira-assistant
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from mira.model import MiraModel
from mira.tokenizer import END_TOK, MIRA_TOK, USER_TOK, CharTokenizer

FALLBACK = ("I'm not confident I can answer that reliably. "
            "Could you rephrase it or give me a bit more context?")


def load_model(checkpoint_dir: str | Path) -> tuple[MiraModel, CharTokenizer]:
    from mira.config import MiraConfig

    checkpoint_dir = Path(checkpoint_dir)
    ckpt = torch.load(checkpoint_dir / "model.pt", map_location="cpu")
    config = MiraConfig.from_dict(ckpt["config"])
    model = MiraModel(config)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tokenizer = CharTokenizer.load(checkpoint_dir / "tokenizer.json")
    return model, tokenizer


@torch.no_grad()
def generate_reply(
    model: MiraModel,
    tokenizer: CharTokenizer,
    user_text: str,
    history: list[tuple[str, str]] | None = None,
    max_new_tokens: int = 200,
    temperature: float = 0.0,
) -> tuple[str, float]:
    """Generate Mira's reply. Returns (reply, mean token log-probability).

    temperature 0 means greedy decoding. The mean log-probability is the
    model's own confidence in what it said; low values indicate the model is
    off its training distribution and should not be trusted.
    """
    prompt = ""
    for u, m in history or []:
        prompt += f"{USER_TOK}{u}{MIRA_TOK}{m}{END_TOK}"
    prompt += f"{USER_TOK}{user_text}{MIRA_TOK}"

    ids = tokenizer.encode(prompt)
    block = model.config.block_size
    ids = ids[-(block - 1):]  # leave room to generate at least one token
    idx = torch.tensor([ids], dtype=torch.long)
    end_id = tokenizer.stoi[END_TOK]

    reply_ids: list[int] = []
    logps: list[float] = []
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -block:])
        logits = logits[0, -1]
        logp = F.log_softmax(logits, dim=-1)
        if temperature <= 0:
            next_id = int(torch.argmax(logp))
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, 1))
        logps.append(logp[next_id].item())
        if next_id == end_id:
            break
        reply_ids.append(next_id)
        idx = torch.cat([idx, torch.tensor([[next_id]])], dim=1)

    reply = tokenizer.decode(reply_ids).strip()
    confidence = sum(logps) / max(len(logps), 1)
    return reply, confidence


def respond(
    model: MiraModel,
    tokenizer: CharTokenizer,
    user_text: str,
    history: list[tuple[str, str]] | None = None,
    min_confidence: float = -0.60,
    temperature: float = 0.0,
) -> tuple[str, float, bool]:
    """Reply with the confidence gate applied. Returns (reply, conf, gated)."""
    reply, conf = generate_reply(model, tokenizer, user_text, history,
                                 temperature=temperature)
    if conf < min_confidence or not reply:
        return FALLBACK, conf, True
    return reply, conf, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with Mira (offline, CPU-only)")
    parser.add_argument("--checkpoint", default="checkpoints/mira-assistant")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = greedy (most reliable)")
    parser.add_argument("--min-confidence", type=float, default=-0.60,
                        help="mean log-prob below this triggers the honest fallback")
    parser.add_argument("--history-turns", type=int, default=2,
                        help="previous exchanges kept in the prompt")
    parser.add_argument("--show-confidence", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_model(args.checkpoint)
    print(f"Mira is ready ({model.num_params() / 1e6:.2f}M params, CPU-only, offline).")
    print("Type a message; ctrl-d or ctrl-c to quit.\n")

    history: list[tuple[str, str]] = []
    while True:
        try:
            user_text = input("you>  ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        reply, conf, gated = respond(
            model, tokenizer, user_text, history,
            min_confidence=args.min_confidence, temperature=args.temperature,
        )
        suffix = f"   [conf {conf:.2f}{' gated' if gated else ''}]" if args.show_confidence else ""
        print(f"mira> {reply}{suffix}")
        if not gated:
            history.append((user_text, reply))
            history = history[-args.history_turns:]


if __name__ == "__main__":
    main()
