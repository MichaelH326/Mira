"""Mira: a small language model you can train and run on a CPU laptop."""

__version__ = "0.1.0"

from mira.config import MiraConfig, PRESETS
from mira.model import MiraModel
from mira.tokenizer import CharTokenizer

__all__ = ["MiraConfig", "PRESETS", "MiraModel", "CharTokenizer"]
