# Moonshine V2 MLX Rewrite Plan

## Context

We're implementing `moonshine_v2/moonshine_v2.py` — a native MLX reimplementation of the
Moonshine V2 streaming speech-to-text model.

**Target models:** `UsefulSensors/moonshine-streaming-{tiny,small,medium}` on HuggingFace
**Reference implementation:** `/Users/kylehowells/Developer/Example-Projects/Moonshine/moonshine/core/`
**Pattern to follow:** existing `mlx_audio/stt/models/` conventions (like voxtral_realtime)

## What We Learned from the Actual Weights

The safetensors from HuggingFace revealed the actual architecture differs from our initial
assumptions. Here's the ground truth:

### Actual Weight Structure (from `moonshine-streaming-tiny`)

```
# ENCODER FRONTEND ("embedder")
model.encoder.embedder.linear.weight:     [320, 80]        # Linear(80, enc_dim), no bias
model.encoder.embedder.conv1.weight:      [640, 320, 5]    # Conv1d(enc_dim, 2*enc_dim, k=5), bias
model.encoder.embedder.conv1.bias:        [640]
model.encoder.embedder.conv2.weight:      [320, 640, 5]    # Conv1d(2*enc_dim, enc_dim, k=5), bias
model.encoder.embedder.conv2.bias:        [320]
model.encoder.embedder.comp.log_k:        []               # scalar compression parameter

# ENCODER TRANSFORMER (uses RMSNorm with .gamma)
model.encoder.layers.N.input_layernorm.gamma:           [320]
model.encoder.layers.N.self_attn.{q,k,v,o}_proj.weight: [320, 320]
model.encoder.layers.N.post_attention_layernorm.gamma:  [320]
model.encoder.layers.N.mlp.fc1.{weight,bias}:           [1280, 320] / [1280]
model.encoder.layers.N.mlp.fc2.{weight,bias}:           [320, 1280] / [320]
model.encoder.final_norm.gamma:                         [320]

# DECODER (uses LayerNorm with .weight)
model.decoder.embed_tokens.weight:                      [32768, 320]
model.decoder.pos_emb.weight:                           [4096, 320]   # LEARNED pos emb
model.decoder.layers.N.input_layernorm.weight:          [320]
model.decoder.layers.N.self_attn.{q,k,v,o}_proj.weight: [320, 320]
model.decoder.layers.N.post_attention_layernorm.weight: [320]
model.decoder.layers.N.encoder_attn.{q,k,v,o}_proj.weight: [320, 320]
model.decoder.layers.N.final_layernorm.weight:          [320]
model.decoder.layers.N.mlp.fc1.{weight,bias}:           [2560, 320] / [2560]  # SwiGLU
model.decoder.layers.N.mlp.fc2.{weight,bias}:           [320, 1280] / [320]
model.decoder.norm.weight:                              [320]

# LM HEAD (separate, NOT tied)
proj_out.weight:                                        [32768, 320]
```

### For small/medium: encoder_dim != decoder_dim

Small has `encoder_hidden_size=620`, `decoder hidden_size=512`, plus:
```
model.decoder.proj.weight:    [512, 620]   # enc_dim -> dec_dim projection
model.decoder.pos_emb.weight: [4096, 620]  # pos_emb is in enc_dim space!
```
The cross-attention k/v projections are `[512, 512]` — they take the already-projected
memory (in decoder dim), not the raw encoder output.

### Config Structure (from config.json)

```json
{
  "model_type": "moonshine_streaming",
  "encoder_config": {           // NESTED encoder config
    "hidden_size": 320,
    "intermediate_size": 1280,
    "num_hidden_layers": 6,
    "num_attention_heads": 8,
    "head_dim": 40,
    "hidden_act": "gelu",
    "sliding_windows": [[16,4],[16,4],[16,0],[16,0],[16,4],[16,4]]
  },
  "encoder_hidden_size": 320,   // top-level alias
  "hidden_size": 320,           // decoder hidden_size
  "intermediate_size": 1280,    // decoder intermediate
  "num_hidden_layers": 6,       // decoder layers
  "num_attention_heads": 8,
  "head_dim": 40,
  "partial_rotary_factor": 0.8, // in rope_parameters
  "tie_word_embeddings": false,
  "vocab_size": 32768
}
```

## What Needs to Change

### config.py — Full rewrite

- Handle nested `encoder_config` dict with its own dims/layers/heads
- Track `encoder_hidden_size` vs `hidden_size` (decoder) separately
- Store `sliding_windows` per-layer config
- Extract `rope_parameters.partial_rotary_factor` and `rope_parameters.rope_theta`
- `tie_word_embeddings` is always `false` for these models
- `model_type` should match `"moonshine_streaming"` (what's in the config.json)

### moonshine_v2.py — Module structure changes

#### Frontend → Embedder

**Old (wrong):** Conv1d(1, dim, k=80, s=80) + GroupNorm + Conv1d + Conv1d
**New (correct):**
- `linear`: Linear(80, enc_dim) — no bias, projects 80-sample frames
- `conv1`: Conv1d(enc_dim, 2*enc_dim, k=5, s=1) + GELU — with bias
- `conv2`: Conv1d(2*enc_dim, enc_dim, k=5, s=1) + GELU — with bias
- `comp.log_k`: learnable scalar — log-compression: `x = sign(x) * log1p(k * |x|) / log1p(k)`

Frame extraction: chunk audio into 80-sample frames, project each with `linear`.

#### Encoder norms

**Old:** nn.LayerNorm (stores `.weight`)
**New:** nn.RMSNorm (stores `.weight`, sanitize maps `.gamma` → `.weight`)

#### Adapter → Removed

**Old:** Separate Adapter module with sinusoidal PE
**New:** No adapter. Instead:
- `decoder.pos_emb`: nn.Embedding(4096, enc_dim) — learned positional embedding
- `decoder.proj`: nn.Linear(enc_dim, dec_dim) — only when dims differ
- Applied to encoder output before it becomes decoder memory

#### Decoder changes

- Add `pos_emb` (Embedding) and optional `proj` (Linear)
- Cross-attention operates on projected memory (decoder dim), not raw encoder output
- Separate `proj_out` Linear for LM head (never tied)

#### Sanitize

Key mappings:
- Strip `model.` prefix
- `encoder.embedder.*` stays as-is
- `encoder.*.gamma` → `encoder.*.weight` (RMSNorm)
- `encoder.final_norm.gamma` → `encoder.final_norm.weight`
- Conv weights: transpose (0, 2, 1) for PyTorch → MLX
- `proj_out.*` stays as-is (separate module)

### MODEL_REMAPPING

Add `"moonshine_streaming": "moonshine_v2"` so the `model_type` from config.json
resolves to our module.

## Encoder Attention: head_dim from config

For small model: encoder has hidden_size=620, head_dim=64, num_heads=8.
So attention internal dim = 8*64=512 ≠ 620. The q/k/v project from 620→512,
o_proj projects 512→620. This is NOT the standard `head_dim = hidden_size // num_heads`.
The `head_dim` must come from config explicitly.

## Sliding Window Attention

The encoder config has per-layer `sliding_windows` entries like `[16, 4]` meaning
left=16, right=4 context. For streaming, the right context (lookahead) is what gets
held back. Layers with `[16, 0]` have no lookahead.

For the initial implementation, we can implement this as a standard attention mask
rather than a true sliding-window kernel.

## Implementation Order

1. Rewrite `config.py` — handle the actual config.json format
2. Rewrite `moonshine_v2.py` — all modules to match actual weights
3. Rewrite `sanitize()` — match actual weight key names
4. Update MODEL_REMAPPING
5. Test: load `moonshine-streaming-tiny` weights and run on test audio
