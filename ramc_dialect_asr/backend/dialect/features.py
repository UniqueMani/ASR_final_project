# backend/dialect/features.py
from __future__ import annotations
import numpy as np
from pathlib import Path
import wave

EPS = 1e-10

# -----------------------------
# WAV loading (no librosa)
# -----------------------------
def load_wav_mono(wav_path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Load PCM WAV using Python stdlib wave.
    Supports 16-bit PCM, mono/stereo. Returns float32 in [-1, 1].
    If sample rate != target_sr, do a simple linear resample.
    """
    p = Path(wav_path)
    with wave.open(str(p), "rb") as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError(f"Only 16-bit PCM wav supported, got sampwidth={sampwidth} bytes")

    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_ch > 1:
        pcm = pcm.reshape(-1, n_ch).mean(axis=1)  # mixdown to mono

    if sr != target_sr:
        pcm = resample_linear(pcm, sr, target_sr)
        sr = target_sr

    return pcm.astype(np.float32), sr


def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Simple linear resampling (fast enough for coursework)."""
    if sr_in == sr_out:
        return x
    n_in = x.shape[0]
    n_out = int(round(n_in * (sr_out / sr_in)))
    if n_out <= 1:
        return np.zeros((0,), dtype=np.float32)
    t_in = np.linspace(0.0, 1.0, n_in, endpoint=False)
    t_out = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float32)


# -----------------------------
# MFCC building blocks
# -----------------------------
def pre_emphasis(x: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    if x.size == 0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - coeff * x[:-1]
    return y


def frame_signal(x: np.ndarray, sr: int, frame_len_s: float = 0.025, frame_step_s: float = 0.010) -> np.ndarray:
    frame_len = int(round(frame_len_s * sr))
    frame_step = int(round(frame_step_s * sr))
    if frame_len <= 0 or frame_step <= 0:
        raise ValueError("Bad frame_len/frame_step")
    if x.size < frame_len:
        # pad to at least one frame
        pad = frame_len - x.size
        x = np.pad(x, (0, pad), mode="constant")
    num_frames = 1 + int(np.floor((x.size - frame_len) / frame_step))
    idx = (np.arange(frame_len)[None, :] + frame_step * np.arange(num_frames)[:, None]).astype(np.int64)
    frames = x[idx]
    # Hamming window
    win = np.hamming(frame_len).astype(np.float32)
    return frames * win[None, :]


def power_spectrum(frames: np.ndarray, nfft: int = 512) -> np.ndarray:
    # rfft -> (num_frames, nfft//2+1)
    mag = np.abs(np.fft.rfft(frames, n=nfft, axis=1))
    pow_spec = (1.0 / nfft) * (mag ** 2)
    return pow_spec.astype(np.float32)


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(sr: int, nfft: int, n_filt: int = 26, fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    if fmax is None:
        fmax = sr / 2.0
    # mel points
    mmin = hz_to_mel(np.array([fmin], dtype=np.float32))[0]
    mmax = hz_to_mel(np.array([fmax], dtype=np.float32))[0]
    m_points = np.linspace(mmin, mmax, n_filt + 2, dtype=np.float32)
    hz_points = mel_to_hz(m_points)
    bins = np.floor((nfft + 1) * hz_points / sr).astype(np.int32)

    fb = np.zeros((n_filt, nfft // 2 + 1), dtype=np.float32)
    for m in range(1, n_filt + 1):
        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]
        if right <= left:
            continue
        # rising
        for k in range(left, center):
            if 0 <= k < fb.shape[1] and center != left:
                fb[m - 1, k] = (k - left) / (center - left + EPS)
        # falling
        for k in range(center, right):
            if 0 <= k < fb.shape[1] and right != center:
                fb[m - 1, k] = (right - k) / (right - center + EPS)

    return fb


def dct_type2(log_mel: np.ndarray, n_mfcc: int) -> np.ndarray:
    """
    Orthonormal DCT-II on last dimension.
    log_mel: (num_frames, n_filt)
    returns: (num_frames, n_mfcc)
    """
    n_filt = log_mel.shape[1]
    n = np.arange(n_filt, dtype=np.float32)
    k = np.arange(n_mfcc, dtype=np.float32)[:, None]
    # DCT-II basis
    basis = np.cos(np.pi * (n + 0.5)[None, :] * k / n_filt).astype(np.float32)
    # orthonormal scaling
    basis[0, :] *= np.sqrt(1.0 / n_filt)
    if n_mfcc > 1:
        basis[1:, :] *= np.sqrt(2.0 / n_filt)
    return (log_mel @ basis.T).astype(np.float32)


def mfcc_from_pcm(
    pcm: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 13,
    n_filt: int = 26,
    nfft: int = 512,
    preemph: float = 0.97,
) -> np.ndarray:
    """
    Returns MFCC matrix shape (n_mfcc, T)
    """
    x = pcm.astype(np.float32).reshape(-1)
    x = pre_emphasis(x, preemph)
    frames = frame_signal(x, sr=sr)
    pow_spec = power_spectrum(frames, nfft=nfft)
    fb = mel_filterbank(sr=sr, nfft=nfft, n_filt=n_filt)
    mel_energy = pow_spec @ fb.T  # (num_frames, n_filt)
    mel_energy = np.maximum(mel_energy, EPS)
    log_mel = np.log(mel_energy).astype(np.float32)
    mfcc = dct_type2(log_mel, n_mfcc=n_mfcc)  # (num_frames, n_mfcc)
    return mfcc.T  # (n_mfcc, num_frames)


def delta(feat: np.ndarray, N: int = 2) -> np.ndarray:
    """
    Compute delta along time axis.
    feat: (D, T)
    """
    D, T = feat.shape
    if T == 0:
        return feat.copy()
    denom = 2.0 * sum(n * n for n in range(1, N + 1))
    # pad with edge values
    padded = np.pad(feat, ((0, 0), (N, N)), mode="edge")
    out = np.zeros_like(feat, dtype=np.float32)
    for t in range(T):
        acc = 0.0
        for n in range(1, N + 1):
            acc += n * (padded[:, t + N + n] - padded[:, t + N - n])
        out[:, t] = acc / (denom + EPS)
    return out.astype(np.float32)


# -----------------------------
# Public APIs (same as before)
# -----------------------------
def wav_to_mfcc_stats(wav_path: str, sr: int = 16000, n_mfcc: int = 13) -> np.ndarray:
    """
    MFCC + delta + delta2 -> mean+std embedding
    Output dim = (n_mfcc*3)*2
    """
    y, _sr = load_wav_mono(wav_path, target_sr=sr)
    if y.size < sr * 0.3:
        raise ValueError("too short")
    mfcc = mfcc_from_pcm(y, sr=sr, n_mfcc=n_mfcc)
    d1 = delta(mfcc, N=2)
    d2 = delta(d1, N=2)
    feat = np.vstack([mfcc, d1, d2]).T  # (T, 3*n_mfcc)
    mean = feat.mean(axis=0)
    std = feat.std(axis=0) + 1e-6
    return np.concatenate([mean, std], axis=0).astype(np.float32)


def pcm_to_mfcc_stats(pcm: np.ndarray, sr: int = 16000, n_mfcc: int = 13) -> np.ndarray:
    y = pcm.astype(np.float32).reshape(-1)
    dim = (n_mfcc * 3) * 2
    if y.size < sr * 0.3:
        return np.zeros((dim,), dtype=np.float32)
    mfcc = mfcc_from_pcm(y, sr=sr, n_mfcc=n_mfcc)
    d1 = delta(mfcc, N=2)
    d2 = delta(d1, N=2)
    feat = np.vstack([mfcc, d1, d2]).T
    mean = feat.mean(axis=0)
    std = feat.std(axis=0) + 1e-6
    return np.concatenate([mean, std], axis=0).astype(np.float32)
