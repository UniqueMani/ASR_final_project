"""
Pure-numpy MFCC implementation (no librosa).
Designed for classic audio classification baselines (Language ID).

MFCC pipeline:
wav -> pre-emphasis -> framing -> window -> FFT power -> Mel filterbank -> log -> DCT-II -> MFCC
Then compute delta and delta-delta, and take global mean/std stats.

Output feature dim (default): n_mfcc * 6  (mfcc mean+std, delta mean+std, delta2 mean+std)
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import hashlib
import json
import os
import wave
from typing import Tuple, Optional, Dict, Any

import numpy as np


def _read_wav_mono(path: str | Path) -> Tuple[np.ndarray, int]:
    """Read PCM WAV to float32 mono in [-1,1]. Uses stdlib wave."""
    path = str(path)
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        dtype = np.int16
        scale = 32768.0
    elif sampwidth == 1:
        dtype = np.uint8  # unsigned
        scale = 128.0
    elif sampwidth == 4:
        dtype = np.int32
        scale = 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    x = np.frombuffer(raw, dtype=dtype).astype(np.float32)

    if sampwidth == 1:
        x = (x - 128.0) / scale
    else:
        x = x / scale

    if n_channels > 1:
        x = x.reshape(-1, n_channels).mean(axis=1)

    return x, sr


def _resample_linear(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Very simple linear resample (good enough for baseline)."""
    if sr == target_sr:
        return x
    if len(x) == 0:
        return x
    ratio = target_sr / float(sr)
    n = int(round(len(x) * ratio))
    if n <= 1:
        return x[:1]
    xp = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    xq = np.linspace(0.0, 1.0, num=n, endpoint=False)
    return np.interp(xq, xp, x).astype(np.float32)


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Create Mel filterbank matrix of shape (n_mels, n_fft//2+1)."""
    # FFT bin frequencies
    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_freqs)

    mmin = _hz_to_mel(np.array([fmin], dtype=np.float32))[0]
    mmax = _hz_to_mel(np.array([fmax], dtype=np.float32))[0]
    mels = np.linspace(mmin, mmax, n_mels + 2)
    hz = _mel_to_hz(mels)

    # Convert hz to bin indices
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center == left:
            center = min(center + 1, n_freqs - 1)
        if right == center:
            right = min(right + 1, n_freqs - 1)
        # rising slope
        if center > left:
            fb[i, left:center] = (np.arange(left, center) - left) / float(center - left)
        # falling slope
        if right > center:
            fb[i, center:right] = (right - np.arange(center, right)) / float(right - center)

    # Normalize filters to unit area (optional but helps)
    enorm = 2.0 / (hz[2:n_mels + 2] - hz[:n_mels])
    fb *= enorm[:, None].astype(np.float32)
    return fb


def _dct_type2(x: np.ndarray, n_out: int) -> np.ndarray:
    """
    DCT-II along last axis, keep first n_out coefficients.
    Implemented via cosine matrix multiply (pure numpy).
    """
    n = x.shape[-1]
    k = np.arange(n_out)[:, None]
    n_idx = np.arange(n)[None, :]
    # DCT-II basis: cos(pi/n * (n_idx + 0.5) * k)
    basis = np.cos(np.pi / n * (n_idx + 0.5) * k).astype(np.float32)  # (n_out, n)
    return x @ basis.T  # (..., n_out)


def _frame_signal(x: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    if len(x) < frame_len:
        # pad at least one frame
        pad = frame_len - len(x)
        x = np.pad(x, (0, pad), mode="constant")
    n_frames = 1 + (len(x) - frame_len) // hop_len
    shape = (n_frames, frame_len)
    strides = (x.strides[0] * hop_len, x.strides[0])
    frames = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return frames.copy()


def _delta(feat: np.ndarray, N: int = 2) -> np.ndarray:
    """Simple delta with window N (like HTK)."""
    T = feat.shape[0]
    denom = 2.0 * sum(i * i for i in range(1, N + 1))
    padded = np.pad(feat, ((N, N), (0, 0)), mode="edge")
    out = np.zeros_like(feat, dtype=np.float32)
    for t in range(T):
        acc = 0.0
        for n in range(1, N + 1):
            acc += n * (padded[t + N + n] - padded[t + N - n])
        out[t] = acc / denom
    return out


@dataclass
class MfccConfig:
    target_sr: int = 16000
    pre_emphasis: float = 0.97
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    n_fft: int = 512
    n_mels: int = 40
    n_mfcc: int = 13
    fmin: float = 20.0
    fmax: float = 7600.0
    eps: float = 1e-10


def mfcc(x: np.ndarray, sr: int, cfg: MfccConfig) -> np.ndarray:
    """Return MFCC frames: (T, n_mfcc)."""
    # resample
    if sr != cfg.target_sr:
        x = _resample_linear(x, sr, cfg.target_sr)
        sr = cfg.target_sr

    # pre-emphasis
    if len(x) >= 2 and cfg.pre_emphasis > 0:
        x = np.append(x[0], x[1:] - cfg.pre_emphasis * x[:-1])

    frame_len = int(round(cfg.frame_ms / 1000.0 * sr))
    hop_len = int(round(cfg.hop_ms / 1000.0 * sr))
    frames = _frame_signal(x, frame_len, hop_len)

    # window
    win = np.hamming(frame_len).astype(np.float32)
    frames = frames * win[None, :]

    # FFT power spectrum
    spec = np.fft.rfft(frames, n=cfg.n_fft, axis=1)
    power = (np.abs(spec) ** 2).astype(np.float32) / float(cfg.n_fft)

    # Mel filterbank
    fb = _mel_filterbank(sr, cfg.n_fft, cfg.n_mels, cfg.fmin, min(cfg.fmax, sr / 2.0 - 1.0))
    mel = power @ fb.T  # (T, n_mels)
    mel = np.maximum(mel, cfg.eps)
    log_mel = np.log(mel).astype(np.float32)

    # DCT-II -> MFCC
    mf = _dct_type2(log_mel, cfg.n_mfcc).astype(np.float32)
    return mf


def mfcc_stats_from_wav(
    wav_path: str | Path,
    sr_target: int = 16000,
    n_mfcc: int = 13,
    cfg: Optional[MfccConfig] = None,
    cache_dir: Optional[str | Path] = None,
    force: bool = False,
    return_meta: bool = False,
):
    """Compute utterance-level MFCC statistics.

    This is a thin compatibility wrapper for training/eval scripts that expect:
        mfcc_stats_from_wav(wav_path, sr_target=16000) -> np.ndarray

    Internally we call `_mfcc_stats_from_wav_impl` which returns (feat, meta).
    Set `return_meta=True` to get (feat, meta).
    """
    if cfg is None:
        cfg = MfccConfig(target_sr=sr_target, n_mfcc=n_mfcc)
    feat, meta = _mfcc_stats_from_wav_impl(
        wav_path=wav_path,
        cfg=cfg,
        cache_dir=cache_dir,
        force=force,
    )
    return (feat, meta) if return_meta else feat

def _mfcc_stats_from_wav_impl(
    wav_path: str | Path,
    cfg: Optional[MfccConfig] = None,
    cache_dir: Optional[str | Path] = None,
    force: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute MFCC stats feature vector for a wav file.
    Returns (feature, meta), where feature is (n_mfcc*6,).
    """
    cfg = cfg or MfccConfig()
    wav_path = Path(wav_path)

    meta: Dict[str, Any] = {"wav": str(wav_path), "sr": None, "frames": None, "cache_hit": False}

    # Cache key based on absolute path + size + mtime
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        st = wav_path.stat()
        key = f"{wav_path.resolve()}|{st.st_size}|{int(st.st_mtime)}|{cfg.__dict__}"
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()
        cache_path = cache_dir / f"{h}.npy"
        if cache_path.exists() and not force:
            feat = np.load(cache_path)
            meta["cache_hit"] = True
            return feat.astype(np.float32), meta

    x, sr = _read_wav_mono(wav_path)
    meta["sr"] = sr
    m = mfcc(x, sr, cfg)  # (T, n_mfcc)
    meta["frames"] = int(m.shape[0])

    d1 = _delta(m, N=2)
    d2 = _delta(d1, N=2)

    def _mean_std(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu = a.mean(axis=0)
        sd = a.std(axis=0)
        return mu, sd

    mu0, sd0 = _mean_std(m)
    mu1, sd1 = _mean_std(d1)
    mu2, sd2 = _mean_std(d2)

    feat = np.concatenate([mu0, sd0, mu1, sd1, mu2, sd2], axis=0).astype(np.float32)

    if cache_path is not None:
        np.save(cache_path, feat)

    return feat, meta


# ---------------------------
# Frame-level MFCC helper for language ID
# ---------------------------

def mfcc_frames_from_wav(
    wav_path: str,
    sr_target: int = 16000,
    n_mfcc: int = 13,
    cmvn: bool = True,
    vad: bool = True,
    vad_percentile: float = 20.0,
    max_frames: int | None = None,
    *,
    norm: str | None = None,
    use_deltas: bool = False,
):
    """
    Return MFCC frames (T, D) for one wav.

    - norm: {"none","cmn","cmvn"}; if None, falls back to legacy `cmvn` bool.
    - use_deltas: if True, concatenate [static, delta, delta2] -> D = n_mfcc*3
    - vad: simple energy-based VAD computed on waveform frames (more stable than MFCC-norm VAD)
    """
    x, sr = _read_wav_mono(wav_path)

    if sr != sr_target:
        x = _resample_linear(x, sr, sr_target)
        sr = sr_target

    cfg = MfccConfig(target_sr=sr_target, n_mfcc=n_mfcc)

    # --- Energy VAD on waveform frames (aligned with MFCC framing) ---
    keep = None
    if vad:
        # mimic pre-emphasis to align with MFCC
        x_pe = x
        if len(x_pe) >= 2 and cfg.pre_emphasis > 0:
            x_pe = np.append(x_pe[0], x_pe[1:] - cfg.pre_emphasis * x_pe[:-1])

        frame_len = int(round(cfg.frame_ms / 1000.0 * sr))
        hop_len = int(round(cfg.hop_ms / 1000.0 * sr))
        wf_frames = _frame_signal(x_pe.astype(np.float32), frame_len, hop_len)
        wf_frames = wf_frames * np.hamming(frame_len).astype(np.float32)[None, :]
        energy = (wf_frames * wf_frames).mean(axis=1)
        if energy.size >= 5:
            thr = np.percentile(energy, vad_percentile)
            keep = energy > thr
            if keep.sum() < 3:
                keep = None

    # --- MFCC frames ---
    M = mfcc(x, sr=sr, cfg=cfg)  # (T, n_mfcc)
    if M.size == 0:
        return M

    if keep is not None:
        # keep mask should align with MFCC frame count; guard just in case
        T = M.shape[0]
        keep = keep[:T]
        if keep.sum() >= 3:
            M = M[keep]

    # --- Optional deltas ---
    if use_deltas and M.shape[0] >= 3:
        d1 = _delta(M, N=2)
        d2 = _delta(d1, N=2)
        M = np.concatenate([M, d1, d2], axis=1).astype(np.float32)

    # --- Normalization ---
    mode = norm if norm is not None else ("cmvn" if cmvn else "none")
    if mode not in ("none", "cmn", "cmvn"):
        raise ValueError(f"norm must be one of none/cmn/cmvn, got: {mode}")

    if mode != "none" and M.size > 0:
        mu = M.mean(axis=0, keepdims=True)
        M = M - mu
        if mode == "cmvn":
            sd = M.std(axis=0, keepdims=True) + 1e-6
            M = M / sd

    # --- frame cap (deterministic) ---
    if max_frames is not None and M.shape[0] > max_frames:
        rng = np.random.default_rng(0)
        idx = rng.choice(M.shape[0], size=max_frames, replace=False)
        M = M[idx]

    return M
    # Simple VAD using frame energy proxy
    if vad and M.shape[0] >= 5:
        e = np.sqrt((M**2).sum(axis=1))
        thr = np.percentile(e, vad_percentile)
        keep = e > thr
        if keep.sum() >= 3:
            M = M[keep]

    if cmvn and M.size > 0:
        mu = M.mean(axis=0, keepdims=True)
        sd = M.std(axis=0, keepdims=True) + 1e-6
        M = (M - mu) / sd

    if max_frames is not None and M.shape[0] > max_frames:
        rng = np.random.default_rng(0)
        idx = rng.choice(M.shape[0], size=max_frames, replace=False)
        M = M[idx]

    return M