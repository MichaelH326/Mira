"""Dependency-free smoke tests: python -m tests.test_smoke (or pytest)."""

import random

import torch

from mira.config import MiraConfig
from mira.instruct_data import build_examples, render_stream
from mira.model import MiraModel
from mira.tokenizer import END_TOK, MIRA_TOK, SPECIAL_TOKENS, USER_TOK, CharTokenizer


def test_tokenizer_round_trip():
    text = f"{USER_TOK}Hi!{MIRA_TOK}Hello! What can I help you with?{END_TOK}"
    tok = CharTokenizer.from_text(text, SPECIAL_TOKENS)
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert ids[0] == tok.stoi[USER_TOK]
    assert ids[-1] == tok.stoi[END_TOK]


def test_model_forward_and_generate():
    config = MiraConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = MiraModel(config)
    x = torch.randint(0, 32, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, 32)
    assert loss.item() > 0
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=5)
    assert out.shape == (1, 6)


def test_dataset_deterministic_and_disjoint():
    train1, evals1 = build_examples(random.Random(7))
    train2, evals2 = build_examples(random.Random(7))
    assert train1 == train2 and evals1 == evals2
    stream = render_stream(train1[:5])
    assert stream.startswith(USER_TOK) and stream.endswith(END_TOK)
    # Held-out eval prompts must not appear verbatim in training data.
    train_users = {u for u, _ in train1}
    leaked = [e for e in evals1 if e["prompt"] in train_users]
    assert not leaked, f"eval prompts leaked into training: {leaked[:3]}"


if __name__ == "__main__":
    test_tokenizer_round_trip()
    test_model_forward_and_generate()
    test_dataset_deterministic_and_disjoint()
    print("all smoke tests passed")
