"""
Word-level timestamp extraction via cross-attention DTW alignment.

Extracts the cross-attention weights computed during autoregressive decoding,
then applies Dynamic Time Warping to find a monotonic token-to-audio-frame
alignment.  Encoder frames are at 50 Hz (20 ms each).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx
import numpy as np


FRAME_DURATION_S = 0.02  # 50 Hz encoder output = 20 ms per frame


@dataclass
class WordTiming:
    word: str
    tokens: List[int]
    start: float
    end: float
    probability: float


# ---------------------------------------------------------------------------
# DTW
# ---------------------------------------------------------------------------

def _dtw(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Dynamic Time Warping on a cost matrix. Returns (text_indices, time_indices)."""
    N, M = x.shape
    cost = np.full((N + 1, M + 1), np.inf, dtype=np.float32)
    trace = np.full((N + 1, M + 1), -1, dtype=np.int32)
    cost[0, 0] = 0.0

    for j in range(1, M + 1):
        for i in range(1, N + 1):
            c0 = cost[i - 1, j - 1]  # diagonal
            c1 = cost[i - 1, j]      # vertical (skip text)
            c2 = cost[i, j - 1]      # horizontal (skip time)
            if c0 <= c1 and c0 <= c2:
                c, t = c0, 0
            elif c1 <= c2:
                c, t = c1, 1
            else:
                c, t = c2, 2
            cost[i, j] = x[i - 1, j - 1] + c
            trace[i, j] = t

    # Backtrace
    i, j = N, M
    trace[0, :] = 2
    trace[:, 0] = 1
    path = []
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        t = trace[i, j]
        if t == 0:
            i -= 1; j -= 1
        elif t == 1:
            i -= 1
        else:
            j -= 1

    path = np.array(path[::-1]).T  # [2, path_len]: row 0 = text, row 1 = time
    return path[0], path[1]


def _median_filter(x: np.ndarray, width: int) -> np.ndarray:
    """Apply median filter along the last dimension."""
    if width <= 1:
        return x
    pad = width // 2
    if x.shape[-1] <= pad:
        return x
    # Pad and apply 1D median filter per row
    from scipy.ndimage import median_filter as _mf
    return _mf(x.astype(np.float32), size=(1, width))


# ---------------------------------------------------------------------------
# Alignment extraction
# ---------------------------------------------------------------------------

def find_alignment(
    cross_qk_per_step: List[List[mx.array]],
    tokens: List[int],
    tokenizer,
    num_frames: int,
    *,
    time_offset: float = 0.0,
    medfilt_width: int = 7,
    token_probs: Optional[List[float]] = None,
) -> List[WordTiming]:
    """
    Build word-level timestamps from per-step cross-attention weights.

    Parameters
    ----------
    cross_qk_per_step : list of list of mx.array
        Outer list: one entry per decode step.
        Inner list: one entry per decoder layer.
        Each array: [B, num_heads, 1, cross_len] — the raw QK scores.
    tokens : list of int
        The decoded token IDs (excluding BOS).
    tokenizer : tokenizer with decode() and convert_ids_to_tokens()
    num_frames : int
        Number of encoder memory frames (= cross_len).
    time_offset : float
        Time offset to add to all timestamps.
    medfilt_width : int
        Width of median filter applied to attention weights.
    token_probs : list of float, optional
        Per-token generation probabilities from softmax(logits).

    Returns
    -------
    list of WordTiming
    """
    if not tokens or not cross_qk_per_step:
        return []

    n_steps = len(cross_qk_per_step)
    n_layers = len(cross_qk_per_step[0])

    # Build attention matrix: average across all layers and heads
    attn_rows = []
    for step_qks in cross_qk_per_step:
        step_weights = []
        for layer_qk in step_qks:
            w = mx.softmax(layer_qk[0, :, 0, :], axis=-1)  # [H, cross_len]
            step_weights.append(w.mean(axis=0))              # [cross_len]
        attn_rows.append(mx.stack(step_weights).mean(axis=0))

    attn_matrix = mx.stack(attn_rows)  # [n_steps, cross_len]
    attn_np = np.array(attn_matrix.astype(mx.float32))

    # Normalize (z-score per row)
    mean = attn_np.mean(axis=-1, keepdims=True)
    std = attn_np.std(axis=-1, keepdims=True) + 1e-6
    attn_np = (attn_np - mean) / std

    # Median filter
    if medfilt_width > 1:
        attn_np = _median_filter(attn_np, medfilt_width)

    # DTW on negative attention (DTW finds minimum-cost path)
    text_indices, time_indices = _dtw(-attn_np)

    # Default token probs to 1.0 if not provided
    if token_probs is None:
        token_probs = [1.0] * len(tokens)

    # Group tokens into words using sentencepiece markers (▁ = word boundary)
    pieces = tokenizer.convert_ids_to_tokens(tokens)
    words = []
    current_word_tokens = []
    current_text_tokens = []

    for i, (tok_id, piece) in enumerate(zip(tokens, pieces)):
        if piece.startswith("\u2581") and current_word_tokens:
            word_text = tokenizer.decode(current_text_tokens)
            words.append((word_text, list(current_word_tokens)))
            current_word_tokens = []
            current_text_tokens = []
        current_word_tokens.append(i)
        current_text_tokens.append(tok_id)

    if current_word_tokens:
        word_text = tokenizer.decode(current_text_tokens)
        words.append((word_text, list(current_word_tokens)))

    # Map token indices to frame indices via DTW path
    jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
    jump_times = time_indices[jumps]

    # Build word timings
    result = []
    for word_text, word_tok_indices in words:
        if not word_tok_indices:
            continue
        first_tok = word_tok_indices[0]
        last_tok = word_tok_indices[-1]

        if first_tok < len(jump_times):
            start_frame = jump_times[first_tok]
        else:
            start_frame = jump_times[-1] if len(jump_times) > 0 else 0
        if last_tok + 1 < len(jump_times):
            end_frame = jump_times[last_tok + 1]
        elif len(jump_times) > 0:
            end_frame = jump_times[-1]
        else:
            end_frame = num_frames

        start_time = time_offset + float(start_frame) * FRAME_DURATION_S
        end_time = time_offset + float(end_frame) * FRAME_DURATION_S

        # Average token probability for this word
        word_probs = [token_probs[i] for i in word_tok_indices if i < len(token_probs)]
        avg_prob = float(np.mean(word_probs)) if word_probs else 1.0

        result.append(WordTiming(
            word=word_text.strip(),
            tokens=[tokens[i] for i in word_tok_indices],
            start=round(start_time, 3),
            end=round(end_time, 3),
            probability=round(avg_prob, 4),
        ))

    # Fix overlapping word boundaries by splitting at midpoints
    for i in range(1, len(result)):
        prev = result[i - 1]
        curr = result[i]
        if prev.end > curr.start:
            mid = round((prev.end + curr.start) / 2, 3)
            result[i - 1] = WordTiming(
                word=prev.word, tokens=prev.tokens,
                start=prev.start, end=mid, probability=prev.probability,
            )
            result[i] = WordTiming(
                word=curr.word, tokens=curr.tokens,
                start=mid, end=curr.end, probability=curr.probability,
            )

    return result
