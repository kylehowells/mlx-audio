# Optimization Benchmark Plan

## Goal
Measure the latency, memory, and accuracy impact of each optimization individually
and cumulatively (stacked).

## Optimizations to test (in order)
1. **Baseline** — current implementation as-is
2. **Fuse encoder evals** — remove intermediate `mx.eval()` between embedder/encoder/memory
3. **Cross-attention KV cache** — cache cross-KV across decode steps instead of recomputing
4. **async_eval double-buffer** — use `mx.async_eval()` in decode loop for pipelining
5. **mx.compile decoder step** — compile the hot decoder forward pass

Each optimization is tested both standalone AND stacked on top of previous ones.

## Metrics
- **Latency**: wall-clock time for full transcription (RTF = inference_time / audio_duration)
- **Memory**: peak GPU memory via `mx.metal.get_peak_memory()`
- **Accuracy**: WER against reference transcript (must not regress)

## Test files
Use 3 files of varying length for fast iteration (~5 min total audio):
- Short: How To Make Sugar Rockets (354s)
- Medium: How AI Datacenters Eat the World (1815s) -- use first 5 minutes only
- Use small-fp16 model as primary test target (bandwidth-bound, most to gain)

## Implementation
- Write optimizations as flags/switches in the model code
- Run each variant 2x, take the faster run (warm cache)
- Record all results to JSON, generate comparison table
