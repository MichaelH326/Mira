"""Model configuration and CPU-friendly presets for Mira."""

from dataclasses import dataclass, asdict


@dataclass
class MiraConfig:
    vocab_size: int = 65      # set from the tokenizer at training time
    block_size: int = 256     # max context length
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = False        # no bias in Linears/LayerNorms: slightly faster, often better

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MiraConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# Presets sized so a full training run finishes in minutes-to-hours on a laptop CPU.
# Parameter counts assume a ~65-symbol character vocabulary.
PRESETS = {
    # ~0.8M params: trains to coherent text in ~10-20 min on a modern laptop CPU
    "tiny": MiraConfig(block_size=256, n_layer=4, n_head=4, n_embd=128),
    # ~2.7M params: noticeably better quality, ~1-2 hours
    "small": MiraConfig(block_size=256, n_layer=6, n_head=6, n_embd=192),
    # ~6.5M params: best quality of the presets, an evening of training
    "base": MiraConfig(block_size=512, n_layer=8, n_head=8, n_embd=256),
}
