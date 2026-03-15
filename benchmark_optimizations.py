#!/usr/bin/env python3
"""
Benchmark individual and cumulative optimizations to the Moonshine Streaming
decode pipeline.  Measures latency, peak GPU memory, and WER for each variant.
"""

import json
import math
import os
import gc
import glob
import re
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

AUDIO_DIR = "benchmark_audio"
RESULTS_DIR = "benchmark_results"

# Use 3 test files for fast iteration
TEST_FILES = [
    "How To Make Sugar Rockets [12fR9neVnS8]",                    # 354s, short
    "Attention in transformers, step-by-step ｜ DL6 [eMlx5fFNoYc]",  # 1569s, medium
    "How AI Datacenters Eat the World [dhqoTku-HAA]",             # 1815s, long
]

# Test on these model variants
MODEL_CONFIGS = [
    ("tiny-fp16",   "models/moonshine-streaming-tiny-fp16"),
    ("small-fp16",  "models/moonshine-streaming-small-fp16"),
    ("small-8bit",  "models/moonshine-streaming-small-8bit"),
]

CHUNK_SEC = 30
CHUNK_SIZE = CHUNK_SEC * 16000
WARMUP_RUNS = 1  # warmup before timed run


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def normalize_text(text):
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', text.lower())).strip()

def compute_wer(reference, hypothesis):
    r = normalize_text(reference).split()
    h = normalize_text(hypothesis).split()
    if not r:
        return 0.0 if not h else 1.0
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1): d[i][0] = i
    for j in range(len(h) + 1): d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = d[i-1][j-1] if r[i-1] == h[j-1] else 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
    return d[len(r)][len(h)] / len(r)


# ---------------------------------------------------------------------------
# Generate variants — each adds one optimization
# ---------------------------------------------------------------------------

def generate_baseline(model, audio, max_tokens):
    """Current implementation: separate evals, no cross-KV cache, sync decode."""
    features = model.encoder.embedder(audio)
    mx.eval(features)
    encoded = model.encoder(features)
    mx.eval(encoded)
    memory = model.decoder.prepare_memory(encoded)
    mx.eval(memory)

    tokens = [model.config.decoder_start_token_id]
    cache = None
    for _ in range(max_tokens):
        tok = mx.array([[tokens[-1]]], dtype=mx.int32)
        hidden, cache = model.decoder(tok, memory, cache=cache)
        mx.eval(hidden)
        logits = model._get_logits(hidden[:, -1, :])
        nt = int(logits.argmax())
        if nt == model.config.eos_token_id:
            break
        tokens.append(nt)
    return model._decode_tokens(tokens[1:])


def generate_fused_encode(model, audio, max_tokens):
    """Opt 1: Fuse encoder evals — one mx.eval for the full encode pipeline."""
    features = model.encoder.embedder(audio)
    encoded = model.encoder(features)
    memory = model.decoder.prepare_memory(encoded)
    mx.eval(memory)  # single eval for entire encode pipeline

    tokens = [model.config.decoder_start_token_id]
    cache = None
    for _ in range(max_tokens):
        tok = mx.array([[tokens[-1]]], dtype=mx.int32)
        hidden, cache = model.decoder(tok, memory, cache=cache)
        mx.eval(hidden)
        logits = model._get_logits(hidden[:, -1, :])
        nt = int(logits.argmax())
        if nt == model.config.eos_token_id:
            break
        tokens.append(nt)
    return model._decode_tokens(tokens[1:])


def generate_cross_kv_cache(model, audio, max_tokens):
    """Opt 2: + Cache cross-attention KV across decode steps."""
    features = model.encoder.embedder(audio)
    encoded = model.encoder(features)
    memory = model.decoder.prepare_memory(encoded)
    mx.eval(memory)

    # Pre-compute cross-attention K/V for all decoder layers
    cross_caches = []
    for layer in model.decoder.layers:
        # Run k_proj and v_proj on memory once
        B, S, _ = memory.shape
        k = layer.encoder_attn.k_proj(memory)
        v = layer.encoder_attn.v_proj(memory)
        k = k.reshape(B, S, layer.encoder_attn.num_kv_heads, layer.encoder_attn.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, layer.encoder_attn.num_kv_heads, layer.encoder_attn.head_dim).transpose(0, 2, 1, 3)
        if layer.encoder_attn.num_kv_groups > 1:
            k = mx.repeat(k, layer.encoder_attn.num_kv_groups, axis=1)
            v = mx.repeat(v, layer.encoder_attn.num_kv_groups, axis=1)
        cross_caches.append((k, v))
    mx.eval(*[c for pair in cross_caches for c in pair])

    tokens = [model.config.decoder_start_token_id]
    cache = [{"self": None, "cross": cc} for cc in cross_caches]
    for _ in range(max_tokens):
        tok = mx.array([[tokens[-1]]], dtype=mx.int32)
        hidden, cache = model.decoder(tok, memory, cache=cache)
        mx.eval(hidden)
        logits = model._get_logits(hidden[:, -1, :])
        nt = int(logits.argmax())
        if nt == model.config.eos_token_id:
            break
        tokens.append(nt)
        # Restore cross caches (decoder overwrites them)
        for i, cc in enumerate(cross_caches):
            cache[i]["cross"] = cc
    return model._decode_tokens(tokens[1:])


def generate_async_eval(model, audio, max_tokens):
    """Opt 3: + async_eval double-buffering in decode loop."""
    features = model.encoder.embedder(audio)
    encoded = model.encoder(features)
    memory = model.decoder.prepare_memory(encoded)
    mx.eval(memory)

    # Pre-compute cross KV
    cross_caches = []
    for layer in model.decoder.layers:
        B, S, _ = memory.shape
        k = layer.encoder_attn.k_proj(memory)
        v = layer.encoder_attn.v_proj(memory)
        k = k.reshape(B, S, layer.encoder_attn.num_kv_heads, layer.encoder_attn.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, layer.encoder_attn.num_kv_heads, layer.encoder_attn.head_dim).transpose(0, 2, 1, 3)
        if layer.encoder_attn.num_kv_groups > 1:
            k = mx.repeat(k, layer.encoder_attn.num_kv_groups, axis=1)
            v = mx.repeat(v, layer.encoder_attn.num_kv_groups, axis=1)
        cross_caches.append((k, v))
    mx.eval(*[c for pair in cross_caches for c in pair])

    tokens = [model.config.decoder_start_token_id]
    cache = [{"self": None, "cross": cc} for cc in cross_caches]

    # First token
    tok = mx.array([[tokens[-1]]], dtype=mx.int32)
    hidden, cache = model.decoder(tok, memory, cache=cache)
    logits = model._get_logits(hidden[:, -1, :])
    next_tok_arr = logits.argmax()
    mx.eval(next_tok_arr)

    for _ in range(max_tokens - 1):
        nt = int(next_tok_arr)
        if nt == model.config.eos_token_id:
            break
        tokens.append(nt)
        for i, cc in enumerate(cross_caches):
            cache[i]["cross"] = cc

        # Submit next computation
        tok = mx.array([[nt]], dtype=mx.int32)
        hidden, cache = model.decoder(tok, memory, cache=cache)
        logits = model._get_logits(hidden[:, -1, :])
        next_tok_arr = logits.argmax()
        # Async eval: submit to GPU, continue python work
        mx.async_eval(next_tok_arr)
        # (in a real pipeline we'd do useful work here; the eval completes by next iteration)

    # Final token
    nt = int(next_tok_arr)
    if nt != model.config.eos_token_id:
        tokens.append(nt)

    return model._decode_tokens(tokens[1:])


def generate_compiled(model, audio, max_tokens):
    """Opt 4: + mx.compile on the decoder step."""
    features = model.encoder.embedder(audio)
    encoded = model.encoder(features)
    memory = model.decoder.prepare_memory(encoded)
    mx.eval(memory)

    # Pre-compute cross KV
    cross_caches = []
    for layer in model.decoder.layers:
        B, S, _ = memory.shape
        k = layer.encoder_attn.k_proj(memory)
        v = layer.encoder_attn.v_proj(memory)
        k = k.reshape(B, S, layer.encoder_attn.num_kv_heads, layer.encoder_attn.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, layer.encoder_attn.num_kv_heads, layer.encoder_attn.head_dim).transpose(0, 2, 1, 3)
        if layer.encoder_attn.num_kv_groups > 1:
            k = mx.repeat(k, layer.encoder_attn.num_kv_groups, axis=1)
            v = mx.repeat(v, layer.encoder_attn.num_kv_groups, axis=1)
        cross_caches.append((k, v))
    mx.eval(*[c for pair in cross_caches for c in pair])

    # Compiled decode step
    @mx.compile
    def decode_step(tok, memory, *flat_cache):
        # Reconstruct cache from flat args
        n_layers = len(model.decoder.layers)
        cache = []
        idx = 0
        for i in range(n_layers):
            if idx + 1 < len(flat_cache) and flat_cache[idx] is not None:
                self_cache = (flat_cache[idx], flat_cache[idx + 1])
            else:
                self_cache = None
            idx += 2
            cross_cache = cross_caches[i]
            cache.append({"self": self_cache, "cross": cross_cache})
        hidden, new_cache = model.decoder(tok, memory, cache=cache)
        logits = model._get_logits(hidden[:, -1, :])
        next_token = logits.argmax()
        # Flatten cache for output
        flat_out = []
        for c in new_cache:
            if c["self"] is not None:
                flat_out.extend(c["self"])
            else:
                flat_out.extend([mx.array(0.0), mx.array(0.0)])
        return next_token, *flat_out

    # Can't easily compile with dynamic cache shapes, fall back to non-compiled
    # with async_eval (compilation of dynamic shapes is tricky in MLX)
    # Instead, just use async_eval which is the main win
    tokens = [model.config.decoder_start_token_id]
    cache = [{"self": None, "cross": cc} for cc in cross_caches]

    tok = mx.array([[tokens[-1]]], dtype=mx.int32)
    hidden, cache = model.decoder(tok, memory, cache=cache)
    logits = model._get_logits(hidden[:, -1, :])
    next_tok_arr = logits.argmax()
    mx.eval(next_tok_arr)

    for _ in range(max_tokens - 1):
        nt = int(next_tok_arr)
        if nt == model.config.eos_token_id:
            break
        tokens.append(nt)
        for i, cc in enumerate(cross_caches):
            cache[i]["cross"] = cc
        tok = mx.array([[nt]], dtype=mx.int32)
        hidden, cache = model.decoder(tok, memory, cache=cache)
        logits = model._get_logits(hidden[:, -1, :])
        next_tok_arr = logits.argmax()
        mx.async_eval(next_tok_arr)

    nt = int(next_tok_arr)
    if nt != model.config.eos_token_id:
        tokens.append(nt)
    return model._decode_tokens(tokens[1:])


# ---------------------------------------------------------------------------
# Optimization variants
# ---------------------------------------------------------------------------

VARIANTS = [
    ("baseline",        generate_baseline),
    ("fused_encode",    generate_fused_encode),
    ("cross_kv_cache",  generate_cross_kv_cache),
    ("async_eval",      generate_async_eval),
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path):
    from mlx_audio.stt.models.moonshine_streaming import Model, ModelConfig
    from transformers import AutoTokenizer

    with open(os.path.join(model_path, 'config.json')) as f:
        raw = json.load(f)
    cfg = ModelConfig.from_dict(raw)
    model = Model(cfg)
    if "quantization" in raw:
        model.apply_quantization(raw["quantization"])
    weights = {}
    for sf in sorted(glob.glob(os.path.join(model_path, '*.safetensors'))):
        weights.update(mx.load(sf))
    model.load_weights(list(model.sanitize(weights).items()), strict=False)
    base = os.path.basename(model_path).rsplit('-', 1)[0]
    model._tokenizer = AutoTokenizer.from_pretrained(f'UsefulSensors/{base}')
    return model


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_optimization_benchmark():
    from mlx_audio.stt.utils import load_audio

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load test data
    test_data = []
    for tf in TEST_FILES:
        wav = os.path.join(AUDIO_DIR, tf + ".wav")
        txt = os.path.join(AUDIO_DIR, tf + ".txt")
        if not os.path.exists(wav) or not os.path.exists(txt):
            print(f"SKIP (missing): {tf}")
            continue
        with open(txt) as f:
            ref = f.read().strip()
        dur = float(os.popen(f'ffprobe -v quiet -show_entries format=duration -of csv=p=0 "{wav}"').read().strip())
        test_data.append({"name": tf.split(" [")[0][:50], "wav": wav, "ref": ref, "dur": dur})
        print(f"Test: {test_data[-1]['name']} ({dur:.0f}s)")

    print(f"\nTest files: {len(test_data)}")
    print(f"Variants: {len(VARIANTS)}")
    print(f"Models: {len(MODEL_CONFIGS)}")
    print()

    all_results = []

    for model_name, model_path in MODEL_CONFIGS:
        print(f"\n{'='*70}")
        print(f" Model: {model_name}")
        print(f"{'='*70}")

        model = load_model(model_path)

        for variant_name, generate_fn in VARIANTS:
            print(f"\n  --- {variant_name} ---")

            for td in test_data:
                audio_full = load_audio(td["wav"], sr=16000)
                total_samples = audio_full.shape[0]

                # Warmup run (first chunk only)
                warmup_chunk = audio_full[:min(CHUNK_SIZE, total_samples)]
                dur_w = warmup_chunk.shape[0] / 16000
                max_tok_w = int(math.ceil(dur_w * model.config.max_tokens_per_second))
                generate_fn(model, warmup_chunk[None, :] if warmup_chunk.ndim == 1 else warmup_chunk, max_tok_w)

                # Clear memory stats
                gc.collect()
                mx.metal.reset_peak_memory()

                # Timed run — process in chunks
                all_text = []
                t0 = time.time()
                for start in range(0, total_samples, CHUNK_SIZE):
                    end = min(start + CHUNK_SIZE, total_samples)
                    chunk = audio_full[start:end]
                    if chunk.ndim == 1:
                        chunk = chunk[None, :]
                    dur_c = chunk.shape[-1] / 16000
                    max_tok = int(math.ceil(dur_c * model.config.max_tokens_per_second))
                    text = generate_fn(model, chunk, max_tok)
                    all_text.append(text)
                elapsed = time.time() - t0

                peak_mem = mx.metal.get_peak_memory() / (1024 * 1024)  # MB
                hypothesis = " ".join(all_text)
                wer = compute_wer(td["ref"], hypothesis)
                rtf = elapsed / td["dur"]

                entry = {
                    "model": model_name,
                    "variant": variant_name,
                    "test_file": td["name"],
                    "audio_duration_s": round(td["dur"], 1),
                    "inference_time_s": round(elapsed, 2),
                    "rtf": round(rtf, 5),
                    "speed_x": round(td["dur"] / elapsed, 1),
                    "peak_memory_mb": round(peak_mem, 1),
                    "wer": round(wer, 4),
                }
                all_results.append(entry)
                print(f"    {td['name'][:35]:35s} RTF={rtf:.5f} ({elapsed:6.1f}s) "
                      f"Mem={peak_mem:6.0f}MB WER={wer:.1%}")

        del model
        gc.collect()

    # Save results
    out_path = os.path.join(RESULTS_DIR, "optimization_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return all_results


def print_summary(results):
    import numpy as np

    models = sorted(set(r["model"] for r in results))
    variants = []
    seen = set()
    for r in results:
        if r["variant"] not in seen:
            variants.append(r["variant"])
            seen.add(r["variant"])

    for model_name in models:
        print(f"\n{'='*75}")
        print(f" {model_name}")
        print(f"{'='*75}")
        print(f"  {'Variant':<20} {'Speed':>7} {'RTF':>10} {'Peak Mem':>10} {'WER':>7} {'vs base':>8}")
        print(f"  {'-'*65}")

        model_results = [r for r in results if r["model"] == model_name]
        baseline_rtf = None

        for vn in variants:
            entries = [r for r in model_results if r["variant"] == vn]
            if not entries:
                continue
            total_audio = sum(e["audio_duration_s"] for e in entries)
            total_infer = sum(e["inference_time_s"] for e in entries)
            avg_rtf = total_infer / total_audio
            speed_x = total_audio / total_infer
            avg_mem = np.mean([e["peak_memory_mb"] for e in entries])
            avg_wer = np.mean([e["wer"] for e in entries])

            if baseline_rtf is None:
                baseline_rtf = avg_rtf
                delta = ""
            else:
                pct = (avg_rtf - baseline_rtf) / baseline_rtf * 100
                delta = f"{pct:+.1f}%"

            print(f"  {vn:<20} {speed_x:>5.0f}x {avg_rtf:>10.5f} {avg_mem:>8.0f}MB {avg_wer*100:>6.1f}% {delta:>8}")


if __name__ == "__main__":
    results = run_optimization_benchmark()
    print_summary(results)
