#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Language-ID models on merged FLEURS manifests.

Supported methods:
  1) stats_gauss : utterance-level MFCC stats -> per-language diagonal Gaussian (fast baseline)
  2) frame_gmm   : frame-level MFCC (optionally +deltas) -> per-language diagonal GMM (better baseline)
  3) utt_lr      : utterance-level MFCC stats -> Logistic Regression (often strongest among "classic" baselines)

Outputs under --out:
  - stats_gauss: gauss_diag.pkl (+ supervised_gauss.pkl for compatibility) + labels.json
  - frame_gmm : frame_gmms.pkl + labels.json
  - utt_lr    : utt_lr.pkl + labels.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from backend.langid.mfcc import mfcc_stats_from_wav, mfcc_frames_from_wav
from backend.langid.gauss_model import GaussDiagModel


def read_jsonl(p: Path) -> List[Dict[str, Any]]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_labels(out: Path, labels: List[str]) -> None:
    with (out / "labels.json").open("w", encoding="utf-8") as f:
        json.dump({"labels": labels}, f, ensure_ascii=False, indent=2)


def _frames_cache_path(cache_dir: Path, wav_path: str, key: str) -> Path:
    import hashlib
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return cache_dir / f"{h}.npy"


def load_or_compute_frames(
    wav_path: str,
    cache_dir: Path | None,
    *,
    sr: int,
    n_mfcc: int,
    vad: bool,
    vad_percentile: float,
    max_frames: int | None,
    norm: str,
    use_deltas: bool,
) -> np.ndarray:
    """
    Cache is optional and keyed on ALL important params.
    """
    key = f"{wav_path}|sr={sr}|n_mfcc={n_mfcc}|vad={int(vad)}|vadp={vad_percentile}|max={max_frames}|norm={norm}|dd={int(use_deltas)}"
    if cache_dir is None:
        return mfcc_frames_from_wav(
            wav_path,
            sr_target=sr,
            n_mfcc=n_mfcc,
            vad=vad,
            vad_percentile=vad_percentile,
            max_frames=max_frames,
            norm=norm,
            use_deltas=use_deltas,
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = _frames_cache_path(cache_dir, wav_path, key)
    if fp.exists():
        try:
            return np.load(fp, allow_pickle=False).astype(np.float32)
        except Exception:
            pass

    M = mfcc_frames_from_wav(
        wav_path,
        sr_target=sr,
        n_mfcc=n_mfcc,
        vad=vad,
        vad_percentile=vad_percentile,
        max_frames=max_frames,
        norm=norm,
        use_deltas=use_deltas,
    )
    np.save(fp, M.astype(np.float16))
    return M


def train_stats_gauss(rows: List[Dict[str, Any]], out: Path, cache_dir: Path | None, sr: int, log_every: int) -> None:
    # collect utterance stats
    feats: Dict[str, List[np.ndarray]] = {}
    for i, r in enumerate(rows, 1):
        lab = str(r["language"])
        wav = r["wav"]
        feat = mfcc_stats_from_wav(wav, sr_target=sr, cache_dir=cache_dir, return_meta=False)
        if isinstance(feat, tuple):
            feat = feat[0]
        feats.setdefault(lab, []).append(np.asarray(feat, dtype=np.float32))
        if log_every and i % log_every == 0:
            print(f"[PROG] {i}/{len(rows)} utts")

    labels = sorted(feats.keys())
    X = [np.stack(feats[l], axis=0) for l in labels]
    model = GaussDiagModel.fit(X, labels=labels, reg=1e-3)

    # Save to BOTH names to avoid "train saved A, eval loaded B"
    model.save(out)  # -> gauss_diag.pkl
    import pickle
    with (out / "supervised_gauss.pkl").open("wb") as f:
        pickle.dump(model, f)
    save_labels(out, labels)
    print(f"[OK] Saved stats_gauss: {out/'gauss_diag.pkl'}")


def train_frame_gmm(
    rows: List[Dict[str, Any]],
    out: Path,
    cache_dir: Path | None,
    *,
    sr: int,
    n_mfcc: int,
    n_components: int,
    reg_covar: float,
    max_frames_per_utt: int,
    vad: bool,
    vad_percentile: float,
    norm: str,
    use_deltas: bool,
    global_cmvn: bool,
    balance_frames_per_lang: int | None,
    seed: int,
    log_every: int,
) -> None:
    try:
        from sklearn.mixture import GaussianMixture
    except Exception as e:
        raise SystemExit("frame_gmm requires scikit-learn. Install: python -m pip install -U scikit-learn") from e

    rng = np.random.default_rng(seed)

    # If using global CMVN, we MUST extract frames without per-utt norm (otherwise language cues get washed out)
    extract_norm = "none" if global_cmvn else norm

    labels = sorted({str(r["language"]) for r in rows})
    per_lab_frames: Dict[str, List[np.ndarray]] = {lab: [] for lab in labels}
    kept_utts: Dict[str, int] = {lab: 0 for lab in labels}

    for i, r in enumerate(rows, 1):
        lab = str(r["language"])
        wav = r["wav"]

        M = load_or_compute_frames(
            wav,
            cache_dir,
            sr=sr,
            n_mfcc=n_mfcc,
            vad=vad,
            vad_percentile=vad_percentile,
            max_frames=max_frames_per_utt,
            norm=extract_norm,
            use_deltas=use_deltas,
        )
        if M.size == 0:
            continue
        per_lab_frames[lab].append(M.astype(np.float32))
        kept_utts[lab] += 1

        if log_every and i % log_every == 0:
            print(f"[PROG] {i}/{len(rows)} utts processed")

    # Diagnostics: how much data per label?
    print("[INFO] Training data kept:")
    for lab in labels:
        n_utts = kept_utts[lab]
        n_frames = int(sum(m.shape[0] for m in per_lab_frames[lab]))
        print(f"  - {lab}: utts={n_utts}, frames={n_frames}")

    # Compute GLOBAL CMVN on training frames (streaming)
    global_mu = None
    global_sd = None
    priors = {}

    if global_cmvn:
        # infer feature dim
        D = None
        for lab in labels:
            if per_lab_frames[lab]:
                D = per_lab_frames[lab][0].shape[1]
                break
        if D is None:
            raise RuntimeError("No frames collected. Check paths / VAD settings.")

        s = np.zeros((D,), dtype=np.float64)
        ss = np.zeros((D,), dtype=np.float64)
        n = 0

        for lab in labels:
            for M in per_lab_frames[lab]:
                s += M.sum(axis=0, dtype=np.float64)
                ss += (M.astype(np.float64) ** 2).sum(axis=0)
                n += M.shape[0]

        mu = s / max(n, 1)
        var = ss / max(n, 1) - mu * mu
        var = np.maximum(var, 1e-6)
        sd = np.sqrt(var)

        global_mu = mu.astype(np.float32)
        global_sd = sd.astype(np.float32)

        # apply in-place
        for lab in labels:
            new_list = []
            for M in per_lab_frames[lab]:
                Z = (M - global_mu[None, :]) / global_sd[None, :]
                new_list.append(Z.astype(np.float32))
            per_lab_frames[lab] = new_list

    # Fit GMM per language
    gmms = {}
    frame_counts = {}
    for lab in labels:
        mats = per_lab_frames[lab]
        if not mats:
            print(f"[WARN] label={lab} has 0 utts, skip")
            continue
        Xlab = np.concatenate(mats, axis=0).astype(np.float32)
        if balance_frames_per_lang is not None and Xlab.shape[0] > balance_frames_per_lang:
            idx = rng.choice(Xlab.shape[0], size=balance_frames_per_lang, replace=False)
            Xlab = Xlab[idx]
        frame_counts[lab] = int(Xlab.shape[0])

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            reg_covar=reg_covar,
            max_iter=300,
            random_state=seed,
            init_params="kmeans",
        )
        gmm.fit(Xlab)
        gmms[lab] = gmm

    # Priors by frames (after balancing)
    total_frames = float(sum(frame_counts.values())) if frame_counts else 1.0
    for lab, c in frame_counts.items():
        priors[lab] = float(c) / total_frames

    config = {
        "method": "frame_gmm",
        "sr": int(sr),
        "n_mfcc": int(n_mfcc),
        "n_components": int(n_components),
        "reg_covar": float(reg_covar),
        "max_frames_per_utt": int(max_frames_per_utt),
        "vad": bool(vad),
        "vad_percentile": float(vad_percentile),
        "norm": str(extract_norm),  # IMPORTANT: norm used at extraction time
        "use_deltas": bool(use_deltas),
        "global_cmvn": bool(global_cmvn),
        "balance_frames_per_lang": (int(balance_frames_per_lang) if balance_frames_per_lang is not None else None),
        "seed": int(seed),
    }

    import pickle
    with (out / "frame_gmms.pkl").open("wb") as f:
        pickle.dump(
            {
                "method": "frame_gmm",
                "gmms": gmms,
                "config": config,
                "global_mu": global_mu,
                "global_sd": global_sd,
                "priors": priors,
            },
            f,
        )
    save_labels(out, sorted(list(gmms.keys())))
    print(f"[OK] Saved frame_gmm: {out/'frame_gmms.pkl'}")


def train_utt_lr(rows: List[Dict[str, Any]], out: Path, cache_dir: Path | None, sr: int, log_every: int, seed: int) -> None:
    """
    Utterance-level MFCC stats -> (StandardScaler + LogisticRegression).
    Usually very strong and much less fragile than per-language GMM.
    """
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
    except Exception as e:
        raise SystemExit("utt_lr requires scikit-learn. Install: python -m pip install -U scikit-learn") from e

    X = []
    y = []
    for i, r in enumerate(rows, 1):
        lab = str(r["language"])
        wav = r["wav"]
        feat = mfcc_stats_from_wav(wav, sr_target=sr, cache_dir=cache_dir, return_meta=False)
        if isinstance(feat, tuple):
            feat = feat[0]
        X.append(np.asarray(feat, dtype=np.float32))
        y.append(lab)
        if log_every and i % log_every == 0:
            print(f"[PROG] {i}/{len(rows)} utts")

    labels = sorted(set(y))
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("lr", LogisticRegression(max_iter=2000, multi_class="auto", n_jobs=None, random_state=seed)),
        ]
    )
    clf.fit(np.stack(X, axis=0), np.array(y))

    import pickle
    with (out / "utt_lr.pkl").open("wb") as f:
        pickle.dump({"method": "utt_lr", "clf": clf, "labels": labels, "config": {"sr": int(sr)}}, f)
    save_labels(out, labels)
    print(f"[OK] Saved utt_lr: {out/'utt_lr.pkl'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="train split manifest.jsonl")
    ap.add_argument("--out", required=True, help="output dir for model files")
    ap.add_argument("--method", choices=["stats_gauss", "frame_gmm", "utt_lr"], default="utt_lr")
    ap.add_argument("--cache_dir", default=None, help="optional mfcc cache dir")
    ap.add_argument("--max_items", type=int, default=0, help="debug: cap utterances (0=all)")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=200, help="print progress every N utterances (0=disable)")

    # frame_gmm options
    ap.add_argument("--n_mfcc", type=int, default=20)
    ap.add_argument("--n_components", type=int, default=32)
    ap.add_argument("--reg_covar", type=float, default=1e-3)
    ap.add_argument("--max_frames_per_utt", type=int, default=600)
    ap.add_argument("--no_vad", action="store_true", help="disable simple energy VAD")
    ap.add_argument("--vad_percentile", type=float, default=10.0)
    ap.add_argument("--norm", choices=["none", "cmn", "cmvn"], default="none", help="per-utterance frame norm (usually off if --global_cmvn)")
    ap.add_argument("--use_deltas", action="store_true", help="append delta+delta2 for frame_gmm")
    ap.add_argument("--global_cmvn", action="store_true", help="apply ONE global CMVN (computed on training frames) [recommended]")
    ap.add_argument("--balance_frames_per_lang", type=int, default=250000, help="cap frames per language (0=disable)")

    args = ap.parse_args()
    rng = random.Random(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    rows = read_jsonl(Path(args.manifest))
    if args.max_items and args.max_items > 0:
        rng.shuffle(rows)
        rows = rows[: args.max_items]

    if args.method == "stats_gauss":
        train_stats_gauss(rows, out, cache_dir, args.sr, args.log_every)
        return

    if args.method == "utt_lr":
        train_utt_lr(rows, out, cache_dir, args.sr, args.log_every, args.seed)
        return

    # frame_gmm
    bal = None if (args.balance_frames_per_lang is None or args.balance_frames_per_lang <= 0) else int(args.balance_frames_per_lang)
    train_frame_gmm(
        rows,
        out,
        cache_dir,
        sr=args.sr,
        n_mfcc=args.n_mfcc,
        n_components=args.n_components,
        reg_covar=args.reg_covar,
        max_frames_per_utt=args.max_frames_per_utt,
        vad=(not args.no_vad),
        vad_percentile=args.vad_percentile,
        norm=args.norm,
        use_deltas=args.use_deltas,
        global_cmvn=bool(args.global_cmvn),
        balance_frames_per_lang=bal,
        seed=args.seed,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
