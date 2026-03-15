#!/usr/bin/env python3
"""
Cross-model STT benchmark on synthetic audio with known-perfect transcripts.

Compares Moonshine Streaming (all quants) vs Whisper (turbo, large-v2) vs Parakeet.
Uses TTS-generated audio so the reference transcript is ground truth.
"""

import json
import math
import os
import glob
import re
import subprocess
import sys
import time
from pathlib import Path

SYNTH_DIR = "benchmark_synthetic"
RESULTS_DIR = "benchmark_results"
INDIVIDUAL_DIR = os.path.join(RESULTS_DIR, "cross_model")
CHUNK_SEC = 30
CHUNK_SIZE = CHUNK_SEC * 16000


# ---------------------------------------------------------------------------
# Metrics (same as benchmark_moonshine.py)
# ---------------------------------------------------------------------------

def normalize_text(text):
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', text.lower())).strip()

def compute_wer(ref, hyp):
    r = normalize_text(ref).split()
    h = normalize_text(hyp).split()
    if not r: return 0.0 if not h else 1.0
    d = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): d[i][0] = i
    for j in range(len(h)+1): d[0][j] = j
    for i in range(1,len(r)+1):
        for j in range(1,len(h)+1):
            d[i][j] = d[i-1][j-1] if r[i-1]==h[j-1] else 1+min(d[i-1][j],d[i][j-1],d[i-1][j-1])
    return d[len(r)][len(h)] / len(r)

def compute_bleu(ref, hyp, max_n=4):
    r = normalize_text(ref).split()
    h = normalize_text(hyp).split()
    if not r or not h: return 0.0
    bp = min(1.0, len(h) / len(r))
    log_avg = 0.0
    for n in range(1, max_n+1):
        rc = {}
        for i in range(len(r)-n+1):
            ng = tuple(r[i:i+n]); rc[ng] = rc.get(ng,0)+1
        hc = {}
        for i in range(len(h)-n+1):
            ng = tuple(h[i:i+n]); hc[ng] = hc.get(ng,0)+1
        clipped = sum(min(hc.get(ng,0),c) for ng,c in rc.items())
        total = max(len(h)-n+1, 1)
        prec = clipped/total if total>0 else 0.0
        if prec == 0: return 0.0
        log_avg += math.log(prec) / max_n
    return bp * math.exp(log_avg)

def compute_chrf(ref, hyp, n=6, beta=2.0):
    rn = normalize_text(ref); hn = normalize_text(hyp)
    def cng(t,o): return [t[i:i+o] for i in range(len(t)-o+1)]
    def wng(t,o):
        w=t.split(); return [tuple(w[i:i+o]) for i in range(len(w)-o+1)]
    tp=0.0; tr=0.0; cnt=0
    for o in range(1,n+1):
        rg=cng(rn,o); hg=cng(hn,o)
        if not rg or not hg: continue
        rc={}
        for g in rg: rc[g]=rc.get(g,0)+1
        hc={}
        for g in hg: hc[g]=hc.get(g,0)+1
        m=sum(min(hc.get(g,0),c) for g,c in rc.items())
        tp+=m/len(hg); tr+=m/len(rg); cnt+=1
    for o in range(1,3):
        rg=wng(rn,o); hg=wng(hn,o)
        if not rg or not hg: continue
        rc={}
        for g in rg: rc[g]=rc.get(g,0)+1
        hc={}
        for g in hg: hc[g]=hc.get(g,0)+1
        m=sum(min(hc.get(g,0),c) for g,c in rc.items())
        tp+=m/len(hg); tr+=m/len(rg); cnt+=1
    if cnt==0: return 0.0
    ap=tp/cnt; ar=tr/cnt
    if ap+ar==0: return 0.0
    return (1+beta**2)*ap*ar/(beta**2*ap+ar)


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def result_key(model, test):
    return re.sub(r'[^\w\-]', '_', f"{model}___{test}")[:200]

def result_path(model, test):
    return os.path.join(INDIVIDUAL_DIR, result_key(model, test) + ".json")

def result_exists(model, test):
    return os.path.exists(result_path(model, test))

def save_result(entry):
    os.makedirs(INDIVIDUAL_DIR, exist_ok=True)
    with open(result_path(entry["model"], entry["test_file"]), "w") as f:
        json.dump(entry, f, indent=2)

def load_all_results():
    results = []
    if not os.path.isdir(INDIVIDUAL_DIR): return results
    for p in sorted(glob.glob(os.path.join(INDIVIDUAL_DIR, "*.json"))):
        with open(p) as f: results.append(json.load(f))
    return results


# ---------------------------------------------------------------------------
# Discover test files
# ---------------------------------------------------------------------------

def discover_tests():
    tests = []
    for wav in sorted(glob.glob(os.path.join(SYNTH_DIR, "*.wav"))):
        txt = wav.replace(".wav", "_excerpt.txt")
        if not os.path.exists(txt): continue
        with open(txt) as f: ref = f.read().strip()
        dur = float(os.popen(f'ffprobe -v quiet -show_entries format=duration -of csv=p=0 "{wav}"').read().strip())
        name = os.path.basename(wav).replace(".wav", "")
        tests.append({"name": name, "display": name[:50], "wav": wav, "ref": ref, "dur": dur})
    return tests


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------

def run_moonshine(model_name, model_path, wav_path, is_hf=False):
    """Run Moonshine Streaming MLX model."""
    import mlx.core as mx
    from mlx_audio.stt.models.moonshine_streaming import Model, ModelConfig
    from mlx_audio.stt.utils import load_audio
    from transformers import AutoTokenizer

    if is_hf:
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download(model_path, 'config.json')
        sf_path = hf_hub_download(model_path, 'model.safetensors')
        with open(cfg_path) as f: raw = json.load(f)
        cfg = ModelConfig.from_dict(raw)
        model = Model(cfg)
        weights = mx.load(sf_path)
        model.load_weights(list(model.sanitize(weights).items()))
        model._tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        with open(os.path.join(model_path, 'config.json')) as f: raw = json.load(f)
        cfg = ModelConfig.from_dict(raw)
        model = Model(cfg)
        if "quantization" in raw: model.apply_quantization(raw["quantization"])
        weights = {}
        for sf in sorted(glob.glob(os.path.join(model_path, '*.safetensors'))):
            weights.update(mx.load(sf))
        model.load_weights(list(model.sanitize(weights).items()), strict=False)
        base = os.path.basename(model_path).rsplit('-', 1)[0]
        model._tokenizer = AutoTokenizer.from_pretrained(f'UsefulSensors/{base}')

    audio = load_audio(wav_path, sr=16000)
    all_text = []
    t0 = time.time()
    for start in range(0, audio.shape[0], CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, audio.shape[0])
        r = model.generate(audio[start:end])
        all_text.append(r.text)
    elapsed = time.time() - t0
    del model
    return " ".join(all_text), elapsed


def run_whisper_turbo(wav_path):
    """Run MLX Whisper Turbo."""
    venv = "/Users/kylehowells/Developer/Playgrounds/STT/venv/bin/python"
    cmd = [venv, "-c", f"""
import time, json, mlx_whisper
t0 = time.time()
r = mlx_whisper.transcribe("{wav_path}", path_or_hf_repo="mlx-community/whisper-turbo")
elapsed = time.time() - t0
print(json.dumps({{"text": r["text"].strip(), "time": elapsed}}))
"""]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    data = json.loads(result.stdout.strip().split('\n')[-1])
    return data["text"], data["time"]


def run_whisper_large_v2(wav_path):
    """Run Lightning Whisper MLX large-v2."""
    venv = "/Users/kylehowells/Developer/Playgrounds/STT/venv/bin/python"
    cmd = [venv, "-c", f"""
import time, json
from lightning_whisper_mlx import LightningWhisperMLX
whisper = LightningWhisperMLX(model="large-v2", batch_size=12, quant=None)
t0 = time.time()
r = whisper.transcribe(audio_path="{wav_path}")
elapsed = time.time() - t0
print(json.dumps({{"text": r["text"].strip(), "time": elapsed}}))
"""]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    data = json.loads(result.stdout.strip().split('\n')[-1])
    return data["text"], data["time"]


def run_parakeet(wav_path):
    """Run parakeet-mlx CLI."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = ["/Users/kylehowells/.local/bin/parakeet-mlx", wav_path,
               "--output-dir", tmpdir, "--output-format", "txt"]
        t0 = time.time()
        subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        elapsed = time.time() - t0
        # Read output
        txt_files = glob.glob(os.path.join(tmpdir, "*.txt"))
        text = ""
        if txt_files:
            with open(txt_files[0]) as f: text = f.read().strip()
    return text, elapsed


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

MODELS = [
    # (name, runner_fn, args)
    ("moonshine-tiny-fp16",   "moonshine", ("models/moonshine-streaming-tiny-fp16", False)),
    ("moonshine-tiny-8bit",   "moonshine", ("models/moonshine-streaming-tiny-8bit", False)),
    ("moonshine-tiny-4bit",   "moonshine", ("models/moonshine-streaming-tiny-4bit", False)),
    ("moonshine-small-fp16",  "moonshine", ("models/moonshine-streaming-small-fp16", False)),
    ("moonshine-small-8bit",  "moonshine", ("models/moonshine-streaming-small-8bit", False)),
    ("moonshine-medium-fp16", "moonshine", ("models/moonshine-streaming-medium-fp16", False)),
    ("moonshine-medium-8bit", "moonshine", ("models/moonshine-streaming-medium-8bit", False)),
    ("whisper-turbo",         "whisper_turbo", ()),
    ("whisper-large-v2",      "whisper_large_v2", ()),
    ("parakeet",              "parakeet", ()),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark():
    os.makedirs(INDIVIDUAL_DIR, exist_ok=True)
    tests = discover_tests()

    print(f"Test files: {len(tests)}")
    for t in tests:
        print(f"  {t['display']} ({t['dur']:.0f}s, {len(t['ref'].split())} words)")
    total_audio = sum(t['dur'] for t in tests)
    print(f"\nTotal audio: {total_audio/60:.1f} min")
    print(f"Models: {len(MODELS)}")

    already = sum(1 for m,_,_ in MODELS for t in tests if result_exists(m, t['name']))
    remaining = len(MODELS) * len(tests) - already
    print(f"Already done: {already}, remaining: {remaining}\n")

    for model_name, runner_type, runner_args in MODELS:
        needed = [t for t in tests if not result_exists(model_name, t['name'])]
        if not needed:
            print(f"=== {model_name} === (all done)")
            continue

        print(f"=== {model_name} === ({len(needed)} remaining)")

        for t in needed:
            sys.stdout.write(f"  {t['display'][:40]}... ")
            sys.stdout.flush()

            try:
                if runner_type == "moonshine":
                    text, elapsed = run_moonshine(model_name, runner_args[0], t['wav'], runner_args[1])
                elif runner_type == "whisper_turbo":
                    text, elapsed = run_whisper_turbo(t['wav'])
                elif runner_type == "whisper_large_v2":
                    text, elapsed = run_whisper_large_v2(t['wav'])
                elif runner_type == "parakeet":
                    text, elapsed = run_parakeet(t['wav'])
                else:
                    raise ValueError(f"Unknown runner: {runner_type}")
            except Exception as e:
                print(f"ERROR: {e}")
                continue

            wer = compute_wer(t['ref'], text)
            bleu = compute_bleu(t['ref'], text)
            chrf = compute_chrf(t['ref'], text)
            rtf = elapsed / t['dur']

            entry = {
                "model": model_name,
                "test_file": t['name'],
                "test_display": t['display'],
                "audio_duration_s": round(t['dur'], 1),
                "inference_time_s": round(elapsed, 2),
                "rtf": round(rtf, 5),
                "speed_x": round(t['dur'] / elapsed, 1) if elapsed > 0 else 0,
                "wer": round(wer, 4),
                "bleu": round(bleu, 4),
                "chrf": round(chrf, 4),
                "hypothesis": text,
            }
            save_result(entry)
            print(f"WER={wer:.1%} BLEU={bleu:.1%} ({elapsed:.1f}s)")

        print()

    return load_all_results()


def print_summary(results):
    import numpy as np

    models = []
    seen = set()
    for r in results:
        if r["model"] not in seen:
            models.append(r["model"])
            seen.add(r["model"])

    print(f"\n{'Model':<25} {'Speed':>7} {'WER':>7} {'BLEU':>7} {'ChrF++':>7} {'Files':>6}")
    print("=" * 65)
    for m in models:
        entries = [r for r in results if r["model"] == m]
        total_audio = sum(e["audio_duration_s"] for e in entries)
        total_infer = sum(e["inference_time_s"] for e in entries)
        speed = total_audio / total_infer if total_infer > 0 else 0
        avg_wer = np.mean([e["wer"] for e in entries])
        avg_bleu = np.mean([e["bleu"] for e in entries])
        avg_chrf = np.mean([e["chrf"] for e in entries])
        print(f"{m:<25} {speed:>5.0f}x {avg_wer*100:>6.1f}% {avg_bleu*100:>6.1f}% "
              f"{avg_chrf*100:>6.1f}% {len(entries):>5}")


if __name__ == "__main__":
    results = run_benchmark()
    print_summary(results)
