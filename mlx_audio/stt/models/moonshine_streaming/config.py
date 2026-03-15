import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EncoderConfig:
    hidden_size: int = 320
    intermediate_size: int = 1280
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    head_dim: int = 40
    hidden_act: str = "gelu"
    frame_ms: float = 5.0
    sample_rate: int = 16000
    max_position_embeddings: int = 4096
    attention_bias: bool = False
    sliding_windows: Optional[List[List[int]]] = None

    @property
    def frame_len(self) -> int:
        return int(self.frame_ms * self.sample_rate / 1000)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EncoderConfig":
        return cls(**{k: v for k, v in d.items() if k in inspect.signature(cls).parameters})


@dataclass
class ModelConfig:
    model_type: str = "moonshine_streaming"

    # Decoder dimensions (top-level fields in config.json)
    hidden_size: int = 320
    intermediate_size: int = 1280
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    head_dim: int = 40
    hidden_act: str = "silu"
    vocab_size: int = 32768
    max_position_embeddings: int = 4096
    attention_bias: bool = False
    tie_word_embeddings: bool = False

    # Encoder (may differ from decoder)
    encoder_hidden_size: Optional[int] = None
    encoder_config: Optional[Dict[str, Any]] = None

    # Tokens
    bos_token_id: int = 1
    eos_token_id: int = 2
    decoder_start_token_id: int = 1
    pad_token_id: int = 0

    # RoPE
    rope_parameters: Optional[Dict[str, Any]] = None
    partial_rotary_factor: float = 0.8
    rope_theta: float = 10000.0

    # Generation limits
    max_tokens_per_second: float = 6.5

    # Resolved encoder config (set in __post_init__)
    _encoder: Optional[EncoderConfig] = field(default=None, repr=False)

    def __post_init__(self):
        # Build encoder config from nested dict or defaults
        if self.encoder_config is not None and isinstance(self.encoder_config, dict):
            self._encoder = EncoderConfig.from_dict(self.encoder_config)
        else:
            self._encoder = EncoderConfig(
                hidden_size=self.encoder_hidden_size or self.hidden_size,
                intermediate_size=(self.encoder_hidden_size or self.hidden_size) * 4,
                num_hidden_layers=self.num_hidden_layers,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
            )

        if self.encoder_hidden_size is None:
            self.encoder_hidden_size = self._encoder.hidden_size

        # Extract rope parameters if provided as nested dict
        if self.rope_parameters is not None and isinstance(self.rope_parameters, dict):
            self.partial_rotary_factor = self.rope_parameters.get(
                "partial_rotary_factor", self.partial_rotary_factor
            )
            self.rope_theta = self.rope_parameters.get("rope_theta", self.rope_theta)

    @property
    def enc(self) -> EncoderConfig:
        return self._encoder

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ModelConfig":
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
