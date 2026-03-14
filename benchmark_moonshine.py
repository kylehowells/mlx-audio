#!/usr/bin/env python3
"""
Benchmark Moonshine Streaming models across sizes and quantizations.
Computes WER, ChrF++ against reference transcripts.

Results are saved per-(model, file) as individual JSON files so that a
crash or interruption loses at most one result.  Re-running skips any
combination that already has a result on disk.
"""

import json
import os
import glob
import time
import re
import sys
from pathlib import Path

import mlx.core as mx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AUDIO_DIR = "benchmark_audio"  # pre-extracted 16kHz mono WAVs + .txt transcripts
RESULTS_DIR = "benchmark_results"
INDIVIDUAL_DIR = os.path.join(RESULTS_DIR, "individual")  # one JSON per (model, file)

MODEL_VARIANTS = [
    # (display_name, path, is_hf_repo)
    ("tiny-fp32",   "UsefulSensors/moonshine-streaming-tiny",   True),
    ("tiny-fp16",   "models/moonshine-streaming-tiny-fp16",     False),
    ("tiny-8bit",   "models/moonshine-streaming-tiny-8bit",     False),
    ("tiny-4bit",   "models/moonshine-streaming-tiny-4bit",     False),
    ("small-fp16",  "models/moonshine-streaming-small-fp16",    False),
    ("small-8bit",  "models/moonshine-streaming-small-8bit",    False),
    ("small-4bit",  "models/moonshine-streaming-small-4bit",    False),
    ("medium-fp16", "models/moonshine-streaming-medium-fp16",   False),
    ("medium-8bit", "models/moonshine-streaming-medium-8bit",   False),
    ("medium-4bit", "models/moonshine-streaming-medium-4bit",   False),
]

CHUNK_SEC = 30  # process audio in 30-second chunks to avoid OOM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def result_key(model_name: str, test_name: str) -> str:
    """Filesystem-safe key for a (model, test_file) pair."""
    safe = re.sub(r'[^\w\-]', '_', f"{model_name}___{test_name}")
    return safe[:200]  # cap length


def result_path(model_name: str, test_name: str) -> str:
    return os.path.join(INDIVIDUAL_DIR, result_key(model_name, test_name) + ".json")


def result_exists(model_name: str, test_name: str) -> bool:
    return os.path.exists(result_path(model_name, test_name))


def save_result(entry: dict):
    os.makedirs(INDIVIDUAL_DIR, exist_ok=True)
    p = result_path(entry["model"], entry["test_file"])
    with open(p, "w") as f:
        json.dump(entry, f, indent=2)


def load_all_results() -> list:
    """Read every individual result JSON back into a list."""
    results = []
    if not os.path.isdir(INDIVIDUAL_DIR):
        return results
    for p in sorted(glob.glob(os.path.join(INDIVIDUAL_DIR, "*.json"))):
        with open(p) as f:
            results.append(json.load(f))
    return results


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_wer(reference: str, hypothesis: str) -> float:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def compute_chrf(reference: str, hypothesis: str, n: int = 6, beta: float = 2.0) -> float:
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    def char_ngrams(text, order):
        return [text[i:i + order] for i in range(len(text) - order + 1)]

    def word_ngrams(text, order):
        words = text.split()
        return [tuple(words[i:i + order]) for i in range(len(words) - order + 1)]

    total_prec = 0.0
    total_rec = 0.0
    count = 0
    for order in range(1, n + 1):
        ref_ng = char_ngrams(ref_norm, order)
        hyp_ng = char_ngrams(hyp_norm, order)
        if not ref_ng or not hyp_ng:
            continue
        ref_counts = {}
        for ng in ref_ng:
            ref_counts[ng] = ref_counts.get(ng, 0) + 1
        hyp_counts = {}
        for ng in hyp_ng:
            hyp_counts[ng] = hyp_counts.get(ng, 0) + 1
        matches = sum(min(hyp_counts.get(ng, 0), c) for ng, c in ref_counts.items())
        total_prec += matches / len(hyp_ng)
        total_rec += matches / len(ref_ng)
        count += 1
    for order in range(1, 3):
        ref_ng = word_ngrams(ref_norm, order)
        hyp_ng = word_ngrams(hyp_norm, order)
        if not ref_ng or not hyp_ng:
            continue
        ref_counts = {}
        for ng in ref_ng:
            ref_counts[ng] = ref_counts.get(ng, 0) + 1
        hyp_counts = {}
        for ng in hyp_ng:
            hyp_counts[ng] = hyp_counts.get(ng, 0) + 1
        matches = sum(min(hyp_counts.get(ng, 0), c) for ng, c in ref_counts.items())
        total_prec += matches / len(hyp_ng)
        total_rec += matches / len(ref_ng)
        count += 1
    if count == 0:
        return 0.0
    avg_prec = total_prec / count
    avg_rec = total_rec / count
    if avg_prec + avg_rec == 0:
        return 0.0
    return (1 + beta ** 2) * avg_prec * avg_rec / (beta ** 2 * avg_prec + avg_rec)


# ---------------------------------------------------------------------------
# Discover test files
# ---------------------------------------------------------------------------

def discover_test_files() -> list:
    """Find all .wav files in AUDIO_DIR that have a matching .txt transcript."""
    test_data = []
    for wav in sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav"))):
        txt = wav.replace(".wav", ".txt")
        if not os.path.exists(txt) or os.path.getsize(txt) == 0:
            continue
        # Skip denoised / intermediate files
        base = os.path.basename(wav)
        if "denoised" in base or "48k" in base:
            continue
        with open(txt) as f:
            ref = f.read().strip()
        dur_s = float(os.popen(
            f'ffprobe -v quiet -show_entries format=duration -of csv=p=0 "{wav}"'
        ).read().strip())
        short_name = os.path.basename(wav).replace(".wav", "")
        # Trim for display
        display = short_name.split(" [")[0][:60]
        test_data.append({
            "name": short_name,
            "display": display,
            "media": wav,
            "reference": ref,
            "duration_s": dur_s,
        })
    return test_data


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(name, path, is_hf):
    from mlx_audio.stt.models.moonshine_v2 import Model, ModelConfig
    from transformers import AutoTokenizer

    if is_hf:
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download(path, 'config.json')
        sf_path = hf_hub_download(path, 'model.safetensors')
        with open(cfg_path) as f:
            raw = json.load(f)
        cfg = ModelConfig.from_dict(raw)
        model = Model(cfg)
        weights = mx.load(sf_path)
        model.load_weights(list(model.sanitize(weights).items()))
        model._tokenizer = AutoTokenizer.from_pretrained(path)
        sf_size = os.path.getsize(sf_path) / (1024 * 1024)
    else:
        with open(os.path.join(path, 'config.json')) as f:
            raw = json.load(f)
        cfg = ModelConfig.from_dict(raw)
        model = Model(cfg)
        if "quantization" in raw:
            model.apply_quantization(raw["quantization"])
        weights = {}
        for sf in sorted(glob.glob(os.path.join(path, '*.safetensors'))):
            weights.update(mx.load(sf))
        model.load_weights(list(model.sanitize(weights).items()), strict=False)
        base = os.path.basename(path).rsplit('-', 1)[0]
        model._tokenizer = AutoTokenizer.from_pretrained(f'UsefulSensors/{base}')
        sf_size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(path, '*.safetensors'))) / (1024 * 1024)

    return model, sf_size


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark():
    from mlx_audio.stt.utils import load_audio

    os.makedirs(INDIVIDUAL_DIR, exist_ok=True)

    test_data = discover_test_files()
    total_audio = sum(t["duration_s"] for t in test_data)

    print(f"Test files: {len(test_data)}")
    for td in test_data:
        print(f"  {td['display']} ({td['duration_s']:.0f}s, {len(td['reference'].split())} words)")
    print(f"\nTotal test audio: {total_audio/60:.1f} min ({total_audio/3600:.1f} hrs)")
    print(f"Model variants: {len(MODEL_VARIANTS)}")
    print(f"Total runs: {len(test_data) * len(MODEL_VARIANTS)}")

    # Count how many are already done
    already_done = sum(
        1 for mn, _, _ in MODEL_VARIANTS
        for td in test_data
        if result_exists(mn, td["name"])
    )
    remaining = len(test_data) * len(MODEL_VARIANTS) - already_done
    print(f"Already completed: {already_done}, remaining: {remaining}")
    print()

    if remaining == 0:
        print("All results already exist. Skipping to chart generation.")
        return load_all_results()

    chunk_size = CHUNK_SEC * 16000

    for model_name, model_path, is_hf in MODEL_VARIANTS:
        # Check if all files for this model are done
        needed = [td for td in test_data if not result_exists(model_name, td["name"])]
        if not needed:
            print(f"=== {model_name} === (all done, skipping)")
            continue

        print(f"=== {model_name} === ({len(needed)} files remaining)")
        model, sf_mb = load_model(model_name, model_path, is_hf)

        for td in needed:
            sys.stdout.write(f"  {td['display'][:45]}... ")
            sys.stdout.flush()

            audio = load_audio(td["media"], sr=16000)
            total_samples = audio.shape[0]
            all_text = []
            t0 = time.time()
            total_gen_tokens = 0

            for start in range(0, total_samples, chunk_size):
                end = min(start + chunk_size, total_samples)
                r = model.generate(audio[start:end])
                all_text.append(r.text)
                total_gen_tokens += r.generation_tokens

            elapsed = time.time() - t0
            hypothesis = " ".join(all_text)

            wer = compute_wer(td["reference"], hypothesis)
            chrf = compute_chrf(td["reference"], hypothesis)
            rtf = elapsed / td["duration_s"]

            entry = {
                "model": model_name,
                "model_size_mb": round(sf_mb, 1),
                "test_file": td["name"],
                "test_display": td["display"],
                "audio_duration_s": round(td["duration_s"], 1),
                "inference_time_s": round(elapsed, 2),
                "rtf": round(rtf, 4),
                "wer": round(wer, 4),
                "chrf": round(chrf, 4),
                "num_tokens": total_gen_tokens,
                "tokens_per_sec": round(total_gen_tokens / elapsed, 1) if elapsed > 0 else 0,
                "hypothesis": hypothesis,
            }
            save_result(entry)
            print(f"WER={wer:.1%} ChrF={chrf:.1%} RTF={rtf:.4f} ({elapsed:.1f}s)")

        del model
        print()

    return load_all_results()


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def generate_charts(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save combined results
    combined_path = os.path.join(RESULTS_DIR, "benchmark_results.json")
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)

    models = []
    seen = set()
    for r in results:
        if r["model"] not in seen:
            models.append(r["model"])
            seen.add(r["model"])

    # Aggregate metrics per model
    agg = {}
    for m in models:
        entries = [r for r in results if r["model"] == m]
        total_audio = sum(e["audio_duration_s"] for e in entries)
        total_infer = sum(e["inference_time_s"] for e in entries)
        agg[m] = {
            "size_mb": entries[0]["model_size_mb"],
            "avg_wer": np.mean([e["wer"] for e in entries]),
            "avg_chrf": np.mean([e["chrf"] for e in entries]),
            "avg_rtf": total_infer / total_audio if total_audio else 0,
            "avg_tps": np.mean([e["tokens_per_sec"] for e in entries]),
            "speed_x": total_audio / total_infer if total_infer else 0,
            "num_files": len(entries),
            "total_audio_min": total_audio / 60,
        }

    # Color scheme
    colors = {}
    for m in models:
        if "tiny" in m:
            base = "#3498db"
        elif "small" in m:
            base = "#2ecc71"
        else:
            base = "#e74c3c"
        if "4bit" in m:
            colors[m] = base + "99"
        elif "8bit" in m:
            colors[m] = base + "CC"
        else:
            colors[m] = base

    # --- Chart 1: Model Size vs WER ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in models:
        ax.scatter(agg[m]["size_mb"], agg[m]["avg_wer"] * 100,
                   s=120, c=colors[m], edgecolors='black', linewidth=0.5, zorder=3)
        ax.annotate(m, (agg[m]["size_mb"], agg[m]["avg_wer"] * 100),
                    fontsize=8, ha='left', va='bottom', xytext=(5, 3),
                    textcoords='offset points')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Model Size (MB)")
    ax.set_ylabel("Word Error Rate (%)")
    ax.set_title("Model Size vs Word Error Rate")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "size_vs_wer.png"), dpi=150)
    plt.close()

    # --- Chart 2: Speed comparison ---
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    bars = ax.bar(x, [agg[m]["speed_x"] for m in models],
                  color=[colors[m] for m in models], edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Speed (x realtime)")
    ax.set_title("Transcription Speed (higher = faster)")
    for bar, m in zip(bars, models):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{agg[m]["speed_x"]:.0f}x', ha='center', va='bottom', fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "speed_comparison.png"), dpi=150)
    plt.close()

    # --- Chart 3: Quality metrics ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(models))
    width = 0.6
    ax1.bar(x, [agg[m]["avg_wer"] * 100 for m in models], width,
            color=[colors[m] for m in models], edgecolor='black', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax1.set_ylim(bottom=0)
    ax1.set_ylabel("WER (%)")
    ax1.set_title("Word Error Rate (lower = better)")
    ax1.grid(True, axis='y', alpha=0.3)
    ax2.bar(x, [agg[m]["avg_chrf"] * 100 for m in models], width,
            color=[colors[m] for m in models], edgecolor='black', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel("ChrF++ (%)")
    ax2.set_title("ChrF++ Score (higher = better)")
    ax2.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "quality_metrics.png"), dpi=150)
    plt.close()

    # --- Chart 4: Size vs Speed vs Quality ---
    fig, ax = plt.subplots(figsize=(10, 7))
    for m in models:
        bubble_size = (1 - agg[m]["avg_wer"]) * 500
        ax.scatter(agg[m]["size_mb"], agg[m]["speed_x"],
                   s=bubble_size, c=colors[m], alpha=0.7,
                   edgecolors='black', linewidth=0.5, zorder=3)
        ax.annotate(f'{m}\nWER={agg[m]["avg_wer"]*100:.1f}%',
                    (agg[m]["size_mb"], agg[m]["speed_x"]),
                    fontsize=7, ha='center', va='bottom', xytext=(0, 8),
                    textcoords='offset points')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Model Size (MB)")
    ax.set_ylabel("Speed (x realtime)")
    ax.set_title("Size vs Speed vs Quality (bubble size = quality)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "size_speed_quality.png"), dpi=150)
    plt.close()

    # --- Chart 5: Per-file WER heatmap ---
    test_names = sorted(set(r["test_file"] for r in results))
    test_displays = {}
    for r in results:
        test_displays[r["test_file"]] = r.get("test_display", r["test_file"][:35])

    fig, ax = plt.subplots(figsize=(max(14, len(test_names) * 1.2), max(6, len(models) * 0.5)))
    data = np.full((len(models), len(test_names)), np.nan)
    for i, m in enumerate(models):
        for j, t in enumerate(test_names):
            entries = [r for r in results if r["model"] == m and r["test_file"] == t]
            if entries:
                data[i, j] = entries[0]["wer"] * 100
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn_r')
    ax.set_xticks(range(len(test_names)))
    ax.set_xticklabels([test_displays.get(t, t)[:30] for t in test_names],
                       rotation=40, ha='right', fontsize=7)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    for i in range(len(models)):
        for j in range(len(test_names)):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f'{data[i,j]:.1f}', ha='center', va='center', fontsize=6,
                        color='white' if data[i,j] > 30 else 'black')
    ax.set_title("WER (%) per Model per Test File")
    fig.colorbar(im, label="WER %")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "wer_heatmap.png"), dpi=150)
    plt.close()

    print(f"Charts saved to {RESULTS_DIR}/")

    # --- Summary table ---
    print(f"\n{'Model':<16} {'Size':>7} {'Speed':>7} {'WER':>7} {'ChrF++':>7} {'Files':>6}")
    print("=" * 55)
    for m in models:
        a = agg[m]
        print(f"{m:<16} {a['size_mb']:>6.0f}M {a['speed_x']:>5.0f}x "
              f"{a['avg_wer']*100:>6.1f}% {a['avg_chrf']*100:>6.1f}% {a['num_files']:>5}")

    return agg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_benchmark()
    agg = generate_charts(results)
