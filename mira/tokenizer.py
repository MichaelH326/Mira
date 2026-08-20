"""Character-level tokenizer with optional special tokens.

Character-level keeps the vocabulary tiny (~65-100 symbols), which keeps the
embedding table — and therefore the whole model — small enough to train
comfortably on a CPU. Special tokens (dialogue markers like ``<|u|>``) are
encoded as single ids so the model sees turn structure atomically.
No external dependencies, fully reversible.
"""

import json
from pathlib import Path

# Dialogue markers used by the instruction-tuned assistant.
USER_TOK = "<|u|>"
MIRA_TOK = "<|m|>"
END_TOK = "<|e|>"
SPECIAL_TOKENS = [USER_TOK, MIRA_TOK, END_TOK]


class CharTokenizer:
    def __init__(self, chars: list[str], specials: list[str] | None = None):
        self.specials = specials or []
        self.chars = chars
        vocab = self.specials + self.chars
        self.stoi = {tok: i for i, tok in enumerate(vocab)}
        self.itos = {i: tok for i, tok in enumerate(vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.specials) + len(self.chars)

    @classmethod
    def from_text(cls, text: str, specials: list[str] | None = None) -> "CharTokenizer":
        specials = specials or []
        for tok in specials:
            text = text.replace(tok, "")
        return cls(sorted(set(text)), specials)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        i = 0
        while i < len(text):
            matched = False
            for tok in self.specials:
                if text.startswith(tok, i):
                    ids.append(self.stoi[tok])
                    i += len(tok)
                    matched = True
                    break
            if not matched:
                # Unknown characters are skipped rather than crashing on prompts.
                ch = text[i]
                if ch in self.stoi:
                    ids.append(self.stoi[ch])
                i += 1
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids if i in self.itos)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"chars": self.chars, "specials": self.specials}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["chars"], data.get("specials", []))
