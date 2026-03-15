"""
Moonshine V2 Streaming — native MLX implementation.

Target models: UsefulSensors/moonshine-streaming-{tiny,small,medium}
Reference impl: moonshine/core/moonshine-streaming-model.{h,cpp}
"""

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from ..base import STTOutput
from .config import ModelConfig


# ---------------------------------------------------------------------------
# Rotary Embeddings (decoder self-attention only)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self._inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))

    def __call__(self, position_ids: mx.array) -> Tuple[mx.array, mx.array]:
        freqs = position_ids[:, :, None].astype(mx.float32) * self._inv_freq[None, None, :]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb), mx.sin(emb)


def _rotate_half(x: mx.array) -> mx.array:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return mx.stack([-x2, x1], axis=-1).reshape(x.shape)


def _apply_rotary_pos_emb(q, k, cos, sin):
    cos = mx.expand_dims(cos, axis=1)  # [B, 1, T, D]
    sin = mx.expand_dims(sin, axis=1)
    half = cos.shape[-1] // 2
    cos = mx.repeat(cos[..., :half], 2, axis=-1)
    sin = mx.repeat(sin[..., :half], 2, axis=-1)
    rot_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]
    q_out = q_rot * cos + _rotate_half(q_rot) * sin
    k_out = k_rot * cos + _rotate_half(k_rot) * sin
    return mx.concatenate([q_out, q_pass], axis=-1), mx.concatenate([k_out, k_pass], axis=-1)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Unified attention for encoder (no RoPE) and decoder (optional RoPE)."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
        is_causal: bool = False,
        use_rope: bool = False,
        partial_rotary_factor: float = 0.8,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.is_causal = is_causal
        self.scale = head_dim ** -0.5

        attn_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim

        self.q_proj = nn.Linear(input_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, kv_dim, bias=False)
        self.o_proj = nn.Linear(attn_dim, input_dim, bias=False)

        self.use_rope = use_rope
        if use_rope:
            rot_ndims = int(head_dim * partial_rotary_factor)
            self.rotary_ndims = rot_ndims - (rot_ndims % 2)
            self.rotary_emb = RotaryEmbedding(self.rotary_ndims, base=rope_theta)

    def __call__(
        self,
        x: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        position_ids: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        return_weights: bool = False,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array], Optional[mx.array]]:
        B, T, _ = x.shape
        is_cross = encoder_hidden_states is not None

        q = self.q_proj(x)
        kv_input = encoder_hidden_states if is_cross else x
        k = self.k_proj(kv_input)
        v = self.v_proj(kv_input)

        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        S = k.shape[1]
        k = k.reshape(B, S, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.use_rope and not is_cross:
            if position_ids is None:
                offset = cache[0].shape[2] if cache is not None else 0
                position_ids = mx.arange(offset, offset + T)[None, :]
            cos, sin = self.rotary_emb(position_ids)
            q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        if cache is not None:
            if is_cross:
                k, v = cache
            else:
                k = mx.concatenate([cache[0], k], axis=2)
                v = mx.concatenate([cache[1], v], axis=2)

        if self.num_kv_groups > 1:
            k = mx.repeat(k, self.num_kv_groups, axis=1)
            v = mx.repeat(v, self.num_kv_groups, axis=1)

        attn_mask = mask
        if self.is_causal and T > 1:
            causal = nn.MultiHeadAttention.create_additive_causal_mask(T)
            if k.shape[2] > T:
                prefix = mx.zeros((T, k.shape[2] - T))
                causal = mx.concatenate([prefix, causal], axis=1)
            attn_mask = causal if attn_mask is None else attn_mask + causal

        qk = None
        if return_weights and is_cross:
            # Compute attention weights explicitly so we can return them
            qk = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # [B, H, T, S]
            w = mx.softmax(qk, axis=-1)
            o = w @ v
        else:
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=attn_mask)

        o = o.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.o_proj(o), (k, v), qk


# ---------------------------------------------------------------------------
# MLPs
# ---------------------------------------------------------------------------

class EncoderMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class DecoderMLP(nn.Module):
    """SwiGLU: fc1 -> split(x, gate) -> silu(gate) * x -> fc2"""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 2 * intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x, gate = mx.split(x, 2, axis=-1)
        return self.fc2(nn.silu(gate) * x)


# ---------------------------------------------------------------------------
# Transformer layers
# ---------------------------------------------------------------------------

class UnitOffsetLayerNorm(nn.Module):
    """LayerNorm(elementwise_affine=False) followed by (gamma + 1) scaling.

    The HuggingFace MoonshineStreamingLayerNorm stores gamma with a unit offset:
    gamma is initialized near 0 and 1.0 is added at runtime.  This matches the
    weight key ``*.gamma`` in the safetensors checkpoint.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.weight = mx.zeros((dim,))  # stored as gamma (near 0); renamed from .gamma by sanitize
        self._dim = dim

    def __call__(self, x: mx.array) -> mx.array:
        mean = x.mean(axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        normed = (x - mean) / mx.sqrt(var + 1e-5)
        return normed * (self.weight + 1.0)


class EncoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        ec = cfg.enc
        self.self_attn = Attention(
            input_dim=ec.hidden_size, num_heads=ec.num_attention_heads,
            head_dim=ec.head_dim, num_kv_heads=ec.num_key_value_heads,
            is_causal=False, use_rope=False,
        )
        self.mlp = EncoderMLP(ec.hidden_size, ec.intermediate_size)
        self.input_layernorm = UnitOffsetLayerNorm(ec.hidden_size)
        self.post_attention_layernorm = UnitOffsetLayerNorm(ec.hidden_size)

        # Per-layer sliding window [left, right]
        if ec.sliding_windows and layer_idx < len(ec.sliding_windows):
            self.window_left, self.window_right = ec.sliding_windows[layer_idx]
        else:
            self.window_left, self.window_right = None, None

    def _sliding_mask(self, seq_len: int) -> Optional[mx.array]:
        if self.window_left is None:
            return None
        pos = mx.arange(seq_len)
        diff = pos[:, None] - pos[None, :]  # q_pos - k_pos
        # HF uses strict < (not <=): left positions 0..left-1, right positions -1..-right+1
        valid = ((diff >= 0) & (diff < self.window_left)) | ((diff < 0) & (-diff < self.window_right))
        return mx.where(valid, mx.array(0.0), mx.array(-1e9))

    def __call__(self, x: mx.array) -> mx.array:
        mask = self._sliding_mask(x.shape[1])
        r = x
        x = self.input_layernorm(x)
        x, _, _ = self.self_attn(x, mask=mask)
        x = r + x
        r = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return r + x


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.hidden_size
        self.self_attn = Attention(
            input_dim=d, num_heads=cfg.num_attention_heads,
            head_dim=cfg.head_dim, num_kv_heads=cfg.num_key_value_heads,
            is_causal=True, use_rope=True,
            partial_rotary_factor=cfg.partial_rotary_factor,
            rope_theta=cfg.rope_theta,
        )
        self.encoder_attn = Attention(
            input_dim=d, num_heads=cfg.num_attention_heads,
            head_dim=cfg.head_dim, num_kv_heads=cfg.num_key_value_heads,
            is_causal=False, use_rope=False,
        )
        self.mlp = DecoderMLP(d, cfg.intermediate_size)
        self.input_layernorm = nn.LayerNorm(d, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(d, bias=False)
        self.final_layernorm = nn.LayerNorm(d, bias=False)

    def __call__(self, x, memory, self_cache=None, cross_cache=None, return_cross_weights=False):
        r = x
        x = self.input_layernorm(x)
        x, new_self, _ = self.self_attn(x, cache=self_cache)
        x = r + x

        r = x
        x = self.post_attention_layernorm(x)
        x, new_cross, cross_qk = self.encoder_attn(
            x, encoder_hidden_states=memory, cache=cross_cache,
            return_weights=return_cross_weights,
        )
        x = r + x

        r = x
        x = self.final_layernorm(x)
        x = self.mlp(x)
        return r + x, new_self, new_cross, cross_qk


# ---------------------------------------------------------------------------
# Embedder (audio frontend)
# ---------------------------------------------------------------------------

class FrameCMVN(nn.Module):
    """Per-frame cepstral mean and variance normalization (no learnable params)."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, T, frame_len]
        mean = x.mean(axis=-1, keepdims=True)
        centered = x - mean
        rms = mx.sqrt(mx.mean(centered * centered, axis=-1, keepdims=True) + self.eps)
        return centered / rms


class AsinhCompression(nn.Module):
    """Asinh compression with learnable scale: asinh(exp(log_k) * x)."""
    def __init__(self):
        super().__init__()
        self.log_k = mx.array(0.0)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.arcsinh(mx.exp(self.log_k) * x)


class Embedder(nn.Module):
    """
    Converts raw 16 kHz audio into feature frames.

    Architecture (from HuggingFace transformers MoonshineStreamingEncoderEmbedder):
      1. Chunk audio into frame_len-sample frames
      2. cmvn  : per-frame mean/variance normalization
      3. comp  : asinh(exp(log_k) * x)
      4. silu(linear(x))  : Linear(frame_len, enc_dim, bias=False) + SiLU
      5. silu(conv1(x))   : Conv1d(enc_dim, 2*enc_dim, k=5, s=2) + SiLU
      6. conv2(x)         : Conv1d(2*enc_dim, enc_dim, k=5, s=2) — no activation

    Output rate: 50 Hz (one feature every 20 ms).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        ec = cfg.enc
        dim = ec.hidden_size
        self.frame_len = ec.frame_len
        self.dim = dim

        self.cmvn = FrameCMVN()
        self.comp = AsinhCompression()
        self.linear = nn.Linear(self.frame_len, dim, bias=False)
        self.conv1 = nn.Conv1d(dim, 2 * dim, kernel_size=5, stride=2, bias=True)
        self.conv2 = nn.Conv1d(2 * dim, dim, kernel_size=5, stride=2, bias=True)

    def __call__(self, audio: mx.array) -> mx.array:
        """Batch forward. audio: [B, N] or [N]."""
        if audio.ndim == 1:
            audio = audio[None, :]
        B, N = audio.shape
        n_frames = N // self.frame_len
        trimmed = audio[:, : n_frames * self.frame_len]
        frames = trimmed.reshape(B, n_frames, self.frame_len)   # [B, T, 80]
        x = self.cmvn(frames)                                   # [B, T, 80]
        x = self.comp(x)                                        # [B, T, 80]
        x = nn.silu(self.linear(x))                             # [B, T, dim]
        # MLX Conv1d expects [B, T, C] (channels-last)
        x = mx.pad(x, [(0, 0), (4, 0), (0, 0)])                # causal pad
        x = nn.silu(self.conv1(x))                              # [B, ~T/2, 2*dim]
        x = mx.pad(x, [(0, 0), (4, 0), (0, 0)])
        x = self.conv2(x)                                       # [B, ~T/4, dim]
        return x

    def process_chunk(self, audio_chunk: mx.array, state: "StreamingState") -> Tuple[mx.array, "StreamingState"]:
        """Streaming forward for a single chunk of audio."""
        # Prepend leftover samples
        if state.sample_len > 0:
            buf = state.sample_buffer[: state.sample_len]
            audio_chunk = mx.concatenate([buf, audio_chunk])

        n_samples = audio_chunk.shape[0]
        n_frames = n_samples // self.frame_len
        remainder = n_samples - n_frames * self.frame_len

        if remainder > 0:
            state.sample_buffer = audio_chunk[-remainder:]
            state.sample_len = remainder
        else:
            state.sample_len = 0

        if n_frames == 0:
            return mx.zeros((1, 0, self.dim)), state

        trimmed = audio_chunk[: n_frames * self.frame_len]
        frames = trimmed.reshape(1, n_frames, self.frame_len)
        x = self.cmvn(frames)
        x = self.comp(x)
        x = nn.silu(self.linear(x))

        # Conv1 with causal buffer (4 frames history, stride=2)
        if state.conv1_buffer is not None:
            x_in = mx.concatenate([state.conv1_buffer, x], axis=1)
        else:
            x_in = mx.pad(x, [(0, 0), (4, 0), (0, 0)])
        state.conv1_buffer = x_in[:, -4:, :]
        x = nn.silu(self.conv1(x_in))

        # Conv2 with causal buffer — no activation
        if state.conv2_buffer is not None:
            x_in = mx.concatenate([state.conv2_buffer, x], axis=1)
        else:
            x_in = mx.pad(x, [(0, 0), (4, 0), (0, 0)])
        state.conv2_buffer = x_in[:, -4:, :]
        x = self.conv2(x_in)

        return x, state


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        ec = cfg.enc
        self.embedder = Embedder(cfg)
        self.layers = [EncoderLayer(cfg, i) for i in range(ec.num_hidden_layers)]
        self.final_norm = UnitOffsetLayerNorm(ec.hidden_size)

    def __call__(self, features: mx.array) -> mx.array:
        """Encode pre-extracted features (no positional embedding)."""
        x = features
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        enc_dim = cfg.encoder_hidden_size
        dec_dim = cfg.hidden_size

        self.embed_tokens = nn.Embedding(cfg.vocab_size, dec_dim)
        self.pos_emb = nn.Embedding(cfg.max_position_embeddings, enc_dim)

        if enc_dim != dec_dim:
            self.proj = nn.Linear(enc_dim, dec_dim, bias=False)
        else:
            self.proj = None

        self.layers = [DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        self.norm = nn.LayerNorm(dec_dim, bias=False)

    def prepare_memory(self, encoder_out: mx.array, pos_offset: int = 0) -> mx.array:
        """Add learned positional embedding and project to decoder dim."""
        T = encoder_out.shape[1]
        positions = mx.arange(pos_offset, pos_offset + T)
        x = encoder_out + self.pos_emb(positions)
        if self.proj is not None:
            x = self.proj(x)
        return x

    def __call__(self, tokens, memory, cache=None, return_cross_qk=False):
        x = self.embed_tokens(tokens)
        if cache is None:
            cache = [{"self": None, "cross": None} for _ in self.layers]
        new_cache = []
        cross_qk_all = [] if return_cross_qk else None
        for i, layer in enumerate(self.layers):
            x, sc, cc, layer_qk = layer(
                x, memory,
                self_cache=cache[i]["self"], cross_cache=cache[i]["cross"],
                return_cross_weights=return_cross_qk,
            )
            new_cache.append({"self": sc, "cross": cc})
            if return_cross_qk:
                cross_qk_all.append(layer_qk)
        return self.norm(x), new_cache, cross_qk_all


# ---------------------------------------------------------------------------
# Streaming state
# ---------------------------------------------------------------------------

@dataclass
class StreamingState:
    # Frontend
    sample_buffer: Optional[mx.array] = None
    sample_len: int = 0
    conv1_buffer: Optional[mx.array] = None
    conv2_buffer: Optional[mx.array] = None

    # Accumulated encoder-input features [1, T, enc_dim]
    accumulated_features: Optional[mx.array] = None
    accumulated_feature_count: int = 0

    # Encoder progress
    encoder_frames_emitted: int = 0

    # Positional offset for pos_emb
    pos_offset: int = 0

    # Memory [1, M, dec_dim]
    memory: Optional[mx.array] = None
    memory_len: int = 0

    # Decoder cache (reset each transcribe)
    decoder_cache: Optional[List[dict]] = None

    active: bool = False


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        self.config = config

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.proj_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self._tokenizer = None

    @staticmethod
    def model_quant_predicate(path: str, module) -> bool:
        """Only quantize nn.Linear layers — skip conv, embedding, norms, comp."""
        return isinstance(module, nn.Linear)

    @property
    def sample_rate(self) -> int:
        return self.config.enc.sample_rate

    def _get_logits(self, hidden: mx.array) -> mx.array:
        if self.config.tie_word_embeddings:
            return self.decoder.embed_tokens.as_linear(hidden)
        return self.proj_out(hidden)

    # ===================================================================
    #  Offline (batch) generation
    # ===================================================================

    def generate(self, audio, *, max_tokens: int = 500, temperature: float = 0.0,
                 verbose: bool = False, stream: bool = False,
                 dtype: mx.Dtype = mx.float32, **kwargs) -> STTOutput:
        for k in ("generation_stream", "language", "source_lang", "target_lang"):
            kwargs.pop(k, None)
        start = time.time()

        if isinstance(audio, (str, Path)):
            from mlx_audio.stt.utils import load_audio
            audio = load_audio(str(audio), sr=self.sample_rate, dtype=dtype)
        elif not isinstance(audio, mx.array):
            audio = mx.array(audio)
        if audio.dtype != dtype:
            audio = audio.astype(dtype)

        # Encode (fused: single eval for entire encode pipeline)
        features = self.encoder.embedder(audio)
        encoded = self.encoder(features)
        memory = self.decoder.prepare_memory(encoded)
        mx.eval(memory)

        # Limit tokens by audio duration
        dur = audio.shape[-1] / self.sample_rate
        max_tokens = min(max_tokens, int(math.ceil(dur * self.config.max_tokens_per_second)))

        # Decode (async eval: pipeline GPU compute with Python)
        tokens = [self.config.decoder_start_token_id]
        cache = None

        # First token (sync to prime the cache)
        tok = mx.array([[tokens[-1]]], dtype=mx.int32)
        hidden, cache, _ = self.decoder(tok, memory, cache=cache)
        logits = self._get_logits(hidden[:, -1, :])
        if temperature > 0:
            next_tok_arr = mx.random.categorical(logits / temperature)
        else:
            next_tok_arr = logits.argmax()
        mx.eval(next_tok_arr)

        for _ in range(max_tokens - 1):
            nt = int(next_tok_arr)
            if nt == self.config.eos_token_id:
                break
            tokens.append(nt)
            tok = mx.array([[nt]], dtype=mx.int32)
            hidden, cache, _ = self.decoder(tok, memory, cache=cache)
            logits = self._get_logits(hidden[:, -1, :])
            if temperature > 0:
                next_tok_arr = mx.random.categorical(logits / temperature)
            else:
                next_tok_arr = logits.argmax()
            mx.async_eval(next_tok_arr)

        # Collect final token
        nt = int(next_tok_arr)
        if nt != self.config.eos_token_id:
            tokens.append(nt)

        gen = tokens[1:]
        text = self._decode_tokens(gen)
        elapsed = time.time() - start
        if verbose:
            print(f"Generated {len(gen)} tokens in {elapsed:.2f}s")
            print(f"Text: {text}")

        return STTOutput(
            text=text.strip(),
            segments=[{"text": text.strip(), "start": 0.0, "end": dur}],
            prompt_tokens=1, generation_tokens=len(gen),
            total_tokens=1 + len(gen), total_time=elapsed,
            prompt_tps=1 / elapsed if elapsed else 0,
            generation_tps=len(gen) / elapsed if elapsed else 0,
        )

    # ===================================================================
    #  Word-level timestamps
    # ===================================================================

    def generate_with_word_timestamps(
        self, audio, *, max_tokens: int = 500,
        dtype: mx.Dtype = mx.float32, time_offset: float = 0.0,
        **kwargs,
    ) -> Tuple[STTOutput, list]:
        """
        Transcribe audio and extract word-level timestamps via cross-attention DTW.

        Returns (STTOutput, list[WordTiming]).
        """
        from .timing import find_alignment, WordTiming

        for k in ("generation_stream", "language", "source_lang", "target_lang"):
            kwargs.pop(k, None)
        start = time.time()

        if isinstance(audio, (str, Path)):
            from mlx_audio.stt.utils import load_audio
            audio = load_audio(str(audio), sr=self.sample_rate, dtype=dtype)
        elif not isinstance(audio, mx.array):
            audio = mx.array(audio)
        if audio.dtype != dtype:
            audio = audio.astype(dtype)

        # Encode (fused)
        features = self.encoder.embedder(audio)
        encoded = self.encoder(features)
        memory = self.decoder.prepare_memory(encoded)
        mx.eval(memory)

        num_frames = memory.shape[1]
        dur = audio.shape[-1] / self.sample_rate
        max_tokens = min(max_tokens, int(math.ceil(dur * self.config.max_tokens_per_second)))
        cfg = self.config

        # Decode with per-step cross-attention weight extraction.
        # Uses manual attention (not fused sdpa) for cross-attention only,
        # since mx.fast.scaled_dot_product_attention doesn't return weights.
        tokens = [cfg.decoder_start_token_id]
        cache = None
        cross_qk_per_step = []
        token_probs = []

        for _ in range(max_tokens):
            tok = mx.array([[tokens[-1]]], dtype=mx.int32)
            hidden, cache, cross_qk = self.decoder(
                tok, memory, cache=cache, return_cross_qk=True,
            )
            mx.eval(hidden)
            if cross_qk is not None:
                cross_qk_per_step.append(cross_qk)

            logits = self._get_logits(hidden[:, -1, :])
            probs = mx.softmax(logits, axis=-1)
            nt = int(logits.argmax())
            mx.eval(probs)
            token_probs.append(float(probs.reshape(-1)[nt]))
            if nt == cfg.eos_token_id:
                break
            tokens.append(nt)

        gen = tokens[1:]
        text = self._decode_tokens(gen)
        elapsed = time.time() - start

        # Extract word timestamps
        word_timings = find_alignment(
            cross_qk_per_step, gen, self._tokenizer, num_frames,
            time_offset=time_offset,
            token_probs=token_probs,
        )

        # Build segments with word-level detail
        words = [
            {"word": wt.word, "start": wt.start, "end": wt.end, "probability": wt.probability}
            for wt in word_timings
        ]

        output = STTOutput(
            text=text.strip(),
            segments=[{"text": text.strip(), "start": 0.0, "end": dur, "words": words}],
            prompt_tokens=1, generation_tokens=len(gen),
            total_tokens=1 + len(gen), total_time=elapsed,
            prompt_tps=1 / elapsed if elapsed else 0,
            generation_tps=len(gen) / elapsed if elapsed else 0,
        )
        return output, word_timings

    # ===================================================================
    #  Streaming API
    # ===================================================================

    def create_stream(self) -> StreamingState:
        return StreamingState(
            sample_buffer=mx.zeros((self.config.enc.frame_len - 1,)),
        )

    def start_stream(self, state: StreamingState) -> StreamingState:
        state.active = True
        state.accumulated_features = None
        state.accumulated_feature_count = 0
        state.encoder_frames_emitted = 0
        state.pos_offset = 0
        state.memory = None
        state.memory_len = 0
        state.decoder_cache = None
        return state

    def stop_stream(self, state: StreamingState) -> StreamingState:
        state.active = False
        return state

    def add_audio(self, state: StreamingState, chunk: mx.array) -> StreamingState:
        if chunk.ndim != 1:
            chunk = chunk.squeeze()
        features, state = self.encoder.embedder.process_chunk(chunk, state)
        mx.eval(features)
        if features.shape[1] > 0:
            if state.accumulated_features is None:
                state.accumulated_features = features
            else:
                state.accumulated_features = mx.concatenate(
                    [state.accumulated_features, features], axis=1)
            state.accumulated_feature_count = state.accumulated_features.shape[1]
        return state

    def transcribe(self, state: StreamingState, *, is_final: bool = False,
                   max_tokens: int = 256, temperature: float = 0.0) -> Tuple[str, StreamingState]:
        cfg = self.config
        ec = cfg.enc

        if state.accumulated_features is None or state.accumulated_feature_count == 0:
            return "", state

        total = state.accumulated_feature_count

        # Determine stable frames (hold back lookahead unless final)
        lookahead_frames = 0
        if ec.sliding_windows:
            lookahead_frames = max(r for _, r in ec.sliding_windows)
        stable = total if is_final else max(0, total - lookahead_frames)
        new_frames = stable - state.encoder_frames_emitted

        if new_frames <= 0:
            if state.memory_len > 0:
                return self._decode_memory(state, max_tokens, temperature), state
            return "", state

        # Encoder: sliding window with left context (fused eval)
        left_ctx = 16 * ec.num_hidden_layers
        win_start = max(0, state.encoder_frames_emitted - left_ctx)
        window = state.accumulated_features[:, win_start:total, :]

        encoded = self.encoder(window)
        offset = state.encoder_frames_emitted - win_start
        new_encoded = encoded[:, offset: offset + new_frames, :]
        new_memory = self.decoder.prepare_memory(new_encoded, pos_offset=state.pos_offset)
        mx.eval(new_memory)  # single eval for encode + memory prep

        state.pos_offset += new_frames
        state.encoder_frames_emitted = stable

        if state.memory is None:
            state.memory = new_memory
        else:
            state.memory = mx.concatenate([state.memory, new_memory], axis=1)
        state.memory_len = state.memory.shape[1]

        return self._decode_memory(state, max_tokens, temperature), state

    def _decode_memory(self, state: StreamingState, max_tokens: int, temperature: float) -> str:
        """Auto-regressive decode with async eval pipelining."""
        cfg = self.config
        if state.memory is None or state.memory_len == 0:
            return ""
        dur = state.memory_len * 0.020
        limit = int(math.ceil(dur * cfg.max_tokens_per_second))
        max_tokens = min(max_tokens, limit)

        tokens = [cfg.decoder_start_token_id]
        cache = None  # fresh each decode pass

        # First token (sync)
        tok = mx.array([[tokens[-1]]], dtype=mx.int32)
        hidden, cache, _ = self.decoder(tok, state.memory, cache=cache)
        logits = self._get_logits(hidden[:, -1, :])
        if temperature > 0:
            next_tok_arr = mx.random.categorical(logits / temperature)
        else:
            next_tok_arr = logits.argmax()
        mx.eval(next_tok_arr)

        for _ in range(max_tokens - 1):
            nt = int(next_tok_arr)
            if nt == cfg.eos_token_id:
                break
            tokens.append(nt)
            tok = mx.array([[nt]], dtype=mx.int32)
            hidden, cache, _ = self.decoder(tok, state.memory, cache=cache)
            logits = self._get_logits(hidden[:, -1, :])
            if temperature > 0:
                next_tok_arr = mx.random.categorical(logits / temperature)
            else:
                next_tok_arr = logits.argmax()
            mx.async_eval(next_tok_arr)

        nt = int(next_tok_arr)
        if nt != cfg.eos_token_id:
            tokens.append(nt)
        return self._decode_tokens(tokens[1:])

    def _decode_tokens(self, tokens: List[int]) -> str:
        if self._tokenizer is not None:
            return self._tokenizer.decode(tokens, skip_special_tokens=True)
        return "".join(chr(t) if t < 128 else f"<{t}>" for t in tokens)

    # ===================================================================
    #  Weight loading
    # ===================================================================

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        out: Dict[str, mx.array] = {}

        # Detect if weights are in HuggingFace format (keys start with "model.")
        # vs already-converted MLX format (no "model." prefix).
        needs_hf_mapping = any(k.startswith("model.") for k in weights)

        for key, val in weights.items():
            k = key

            if needs_hf_mapping:
                # Strip "model." prefix
                if k.startswith("model."):
                    k = k[len("model."):]

                # Encoder norms: .gamma -> .weight (UnitOffsetLayerNorm)
                k = k.replace(".gamma", ".weight")

                # Conv1d weights: PyTorch [out, in, k] -> MLX [out, k, in]
                if "conv" in k and k.endswith(".weight") and val.ndim == 3:
                    val = val.transpose(0, 2, 1)

            out[k] = val
        return out

    def apply_quantization(self, quant_config: dict):
        """Set up QuantizedLinear layers from a quantization config dict."""
        group_size = quant_config.get("group_size", 64)
        bits = quant_config.get("bits", 8)

        def class_predicate(path: str, module):
            if not isinstance(module, nn.Linear):
                return False
            if not hasattr(module, "weight"):
                return False
            if module.weight.shape[-1] % group_size != 0:
                return False
            return {"group_size": group_size, "bits": bits}

        nn.quantize(self, class_predicate=class_predicate)

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        model_path = Path(model_path)
        try:
            from transformers import AutoTokenizer
            model._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        except Exception:
            pass
        return model
