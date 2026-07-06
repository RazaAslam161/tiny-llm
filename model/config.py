"""Model configuration for tiny-llm."""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 256  # max context length
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
