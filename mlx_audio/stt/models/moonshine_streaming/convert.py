#!/usr/bin/env python3
"""
Convert Moonshine Streaming models to MLX format with optional quantization.

Usage:
    # Float16
    python -m mlx_audio.stt.models.moonshine_streaming.convert \
        --model UsefulSensors/moonshine-streaming-tiny \
        --dtype float16 \
        -o moonshine-streaming-tiny-fp16

    # 8-bit quantized
    python -m mlx_audio.stt.models.moonshine_streaming.convert \
        --model UsefulSensors/moonshine-streaming-tiny \
        --quantize --q-bits 8 \
        -o moonshine-streaming-tiny-8bit

    # 4-bit quantized
    python -m mlx_audio.stt.models.moonshine_streaming.convert \
        --model UsefulSensors/moonshine-streaming-tiny \
        --quantize --q-bits 4 \
        -o moonshine-streaming-tiny-4bit
"""

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten

from .config import ModelConfig
from .moonshine_streaming import Model


def get_model_path(repo_or_path: str) -> Path:
    p = Path(repo_or_path)
    if p.exists():
        return p
    return Path(snapshot_download(
        repo_or_path,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
    ))


def load_model(model_path: Path) -> tuple:
    with open(model_path / "config.json") as f:
        raw_config = json.load(f)

    config = ModelConfig.from_dict(raw_config)
    model = Model(config)

    weights = {}
    for sf in sorted(model_path.glob("*.safetensors")):
        weights.update(mx.load(str(sf)))

    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()))

    return model, config, raw_config


def quantize_model(model: nn.Module, group_size: int = 64, bits: int = 8):
    """Quantize only nn.Linear layers, skipping conv, embedding, norms, comp."""

    def class_predicate(path: str, module: nn.Module):
        if not isinstance(module, nn.Linear):
            return False
        # Skip if weight dim not divisible by group_size
        if not hasattr(module, "weight"):
            return False
        if module.weight.shape[-1] % group_size != 0:
            return False
        return {"group_size": group_size, "bits": bits}

    nn.quantize(model, class_predicate=class_predicate)


def convert(
    model_repo: str,
    output_path: str,
    dtype: str = None,
    quantize: bool = False,
    q_bits: int = 8,
    q_group_size: int = 64,
):
    print(f"Loading {model_repo}...")
    model_path = get_model_path(model_repo)
    model, config, raw_config = load_model(model_path)

    # When quantizing, always convert remaining weights to fp16 first so that
    # layers that can't be quantized (odd dimensions, conv, etc.) don't stay
    # at fp32 and bloat the file.
    effective_dtype = dtype or ("float16" if quantize else None)

    if effective_dtype:
        print(f"Converting to {effective_dtype}...")
        mx_dtype = getattr(mx, effective_dtype)
        weights = dict(tree_flatten(model.parameters()))
        weights = {
            k: v.astype(mx_dtype) if mx.issubdtype(v.dtype, mx.floating) else v
            for k, v in weights.items()
        }
        model.load_weights(list(weights.items()), strict=False)

    # Quantization
    if quantize:
        print(f"Quantizing to {q_bits}-bit (group_size={q_group_size})...")
        quantize_model(model, group_size=q_group_size, bits=q_bits)
        raw_config["quantization"] = {
            "group_size": q_group_size,
            "bits": q_bits,
        }

    # Save
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    # Save weights
    weights = dict(tree_flatten(model.parameters()))
    total_params = sum(v.size for v in weights.values())

    # Count bits
    total_bits = 0
    for v in weights.values():
        if v.dtype == mx.uint32:
            # quantized weight — bits stored in config
            total_bits += v.size * q_bits if quantize else v.size * 32
        else:
            total_bits += v.size * v.dtype.size * 8
    avg_bits = total_bits / total_params if total_params > 0 else 0

    mx.save_safetensors(str(out / "model.safetensors"), weights)

    # Save config
    raw_config["model_type"] = "moonshine_streaming"
    with open(out / "config.json", "w") as f:
        json.dump(raw_config, f, indent=2)

    # Copy tokenizer and other supporting files
    for pattern in ["tokenizer*", "special_tokens*", "preprocessor*", "processor*", "generation_config*"]:
        for src in model_path.glob(pattern):
            shutil.copy2(src, out / src.name)

    # Summary
    sf_size = (out / "model.safetensors").stat().st_size / (1024 * 1024)
    print(f"Saved to {out}/")
    print(f"  Weights: {sf_size:.1f} MB ({avg_bits:.1f} bits/param avg)")
    print(f"  Parameters: {total_params:,}")

    return out


def main():
    parser = argparse.ArgumentParser(description="Convert Moonshine Streaming models to MLX format")
    parser.add_argument("--model", required=True, help="HuggingFace repo ID or local path")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default=None, help="Convert weights to dtype")
    parser.add_argument("--quantize", action="store_true", help="Quantize the model")
    parser.add_argument("--q-bits", type=int, default=8, choices=[4, 8], help="Quantization bits (default: 8)")
    parser.add_argument("--q-group-size", type=int, default=64, help="Quantization group size (default: 64)")
    args = parser.parse_args()

    convert(
        model_repo=args.model,
        output_path=args.output,
        dtype=args.dtype,
        quantize=args.quantize,
        q_bits=args.q_bits,
        q_group_size=args.q_group_size,
    )


if __name__ == "__main__":
    main()
