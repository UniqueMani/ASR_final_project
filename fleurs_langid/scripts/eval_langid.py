#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Language-ID models on a manifest.

Supports:
  - stats_gauss (gauss_diag.pkl)
  - frame_gmm  (frame_gmms.pkl)
  - utt_lr     (utt_lr.pkl)

Key additions vs earlier versions:
  - frame_gmm uses the EXACT config saved at training time (sr/n_mfcc/vad/deltas/norm)
  - supports global CMVN saved by training (global_mu/global_sd)
  - prints prediction distribution to detect collapse
"""
from __future__ import annotations

import argparse
import json
import time
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if hasattr(x, "item"):
        try:
            x = x.item()
        except Exception:
            pass
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--method", choices=["stats_gauss", "frame_gmm", "utt_lr"], default="utt_lr")
    ap.add_argument("--report", choices=["none", "per_label"], default="per_label")
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--score_mode", choices=["sum", "avg"], default="sum")
    ap.add_argument("--use_priors", action="store_true", help="only for frame_gmm: add log prior")
    args = ap.parse_args()

    t0 = time.time()
    rows = read_jsonl(Path(args.manifest))
    model_dir = Path(args.model_dir)

    labels: List[str] = []
    pred_counts: Dict[str, int] = {}

    # load model
    model_stats = None
    model_gmms = None
    gmm_cfg = None
    global_mu = None
    global_sd = None
    priors = None
    lr_clf = None
    lr_cfg = None

    if args.method == "stats_gauss":
        model_stats = GaussDiagModel.load(model_dir)
        labels = list(model_stats.labels)
        print(f"[INFO] Loaded stats_gauss with {len(labels)} labels")
    elif args.method == "frame_gmm":
        p_path = model_dir / "frame_gmms.pkl"
        if not p_path.exists():
            raise FileNotFoundError(f"Missing: {p_path}")
        with p_path.open("rb") as f:
            data = pickle.load(f)
        model_gmms = data["gmms"]
        gmm_cfg = data.get("config", {})
        global_mu = data.get("global_mu", None)
        global_sd = data.get("global_sd", None)
        priors = data.get("priors", None)
        labels = sorted(list(model_gmms.keys()))
        print(f"[INFO] Loaded frame_gmm with {len(labels)} labels")
        if gmm_cfg:
            print(f"[INFO] frame_gmm cfg: sr={gmm_cfg.get('sr')} n_mfcc={gmm_cfg.get('n_mfcc')} deltas={gmm_cfg.get('use_deltas')} global_cmvn={gmm_cfg.get('global_cmvn')}")
    else:
        p_path = model_dir / "utt_lr.pkl"
        if not p_path.exists():
            raise FileNotFoundError(f"Missing: {p_path}")
        with p_path.open("rb") as f:
            data = pickle.load(f)
        lr_clf = data["clf"]
        labels = list(data.get("labels", []))
        lr_cfg = data.get("config", {"sr": 16000})
        print(f"[INFO] Loaded utt_lr with {len(labels)} labels")

    # eval loop
    ok = 0
    n = 0
    per_lab_ok: Dict[str, int] = {}
    per_lab_n: Dict[str, int] = {}
    per_lab_conf: Dict[Tuple[str, str], int] = {}

    for i, r in enumerate(rows, 1):
        gold = _as_str(r.get("language", ""))
        wav = r.get("wav", "")
        if not gold or not wav:
            continue

        if args.method == "stats_gauss":
            feat = mfcc_stats_from_wav(wav, sr_target=16000, cache_dir=None, return_meta=False)
            if isinstance(feat, tuple):
                feat = feat[0]
            pred = model_stats.predict_one(np.asarray(feat, dtype=np.float32))

        elif args.method == "utt_lr":
            sr = int(lr_cfg.get("sr", 16000))
            feat = mfcc_stats_from_wav(wav, sr_target=sr, cache_dir=None, return_meta=False)
            if isinstance(feat, tuple):
                feat = feat[0]
            pred = lr_clf.predict(np.asarray(feat, dtype=np.float32).reshape(1, -1))[0]
            pred = _as_str(pred)

        else:
            # frame_gmm
            sr = int(gmm_cfg.get("sr", 16000))
            n_mfcc = int(gmm_cfg.get("n_mfcc", 13))
            vad = bool(gmm_cfg.get("vad", True))
            vad_p = float(gmm_cfg.get("vad_percentile", 20.0))
            max_frames = gmm_cfg.get("max_frames_per_utt", None)
            max_frames = int(max_frames) if max_frames is not None else None
            norm = str(gmm_cfg.get("norm", "none"))
            use_deltas = bool(gmm_cfg.get("use_deltas", False))

            frames = mfcc_frames_from_wav(
                wav,
                sr_target=sr,
                n_mfcc=n_mfcc,
                vad=vad,
                vad_percentile=vad_p,
                max_frames=max_frames,
                norm=norm,
                use_deltas=use_deltas,
            )
            if frames.size == 0:
                pred = labels[0]
            else:
                if global_mu is not None and global_sd is not None:
                    frames = (frames - global_mu[None, :]) / global_sd[None, :]

                best_score = -float("inf")
                best_lbl = labels[0]
                for lbl in labels:
                    gmm = model_gmms[lbl]
                    ll = gmm.score_samples(frames)
                    score = float(ll.sum() if args.score_mode == "sum" else ll.mean())
                    if args.use_priors and priors is not None and lbl in priors:
                        score += float(np.log(max(priors[lbl], 1e-12)))
                    if score > best_score:
                        best_score = score
                        best_lbl = lbl
                pred = best_lbl

        pred_counts[pred] = pred_counts.get(pred, 0) + 1

        n += 1
        per_lab_n[gold] = per_lab_n.get(gold, 0) + 1
        if pred == gold:
            ok += 1
            per_lab_ok[gold] = per_lab_ok.get(gold, 0) + 1
        else:
            per_lab_conf[(gold, pred)] = per_lab_conf.get((gold, pred), 0) + 1

        if args.log_every and i % args.log_every == 0:
            print(f"[PROG] {i}/{len(rows)} ok={ok} n={n}")

    acc = ok / max(n, 1)
    print(f"[RESULT] Method={args.method} Accuracy={acc*100:.4f}%")
    # top-5 preds
    top5 = sorted(pred_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
    print("[INFO] Pred top-5:", ", ".join([f"{k}:{v}" for k, v in top5]))

    if args.report == "per_label":
        print("\nper_label_accuracy:")
        for lab in sorted(per_lab_n.keys()):
            sup = per_lab_n[lab]
            a = per_lab_ok.get(lab, 0) / max(sup, 1)
            # top confuse
            confs = [(p, c) for (g, p), c in per_lab_conf.items() if g == lab]
            confs.sort(key=lambda x: x[1], reverse=True)
            if confs:
                top_p, top_c = confs[0]
                print(f"  - {lab:12s} acc={a*100:6.2f}% support={sup:5d} top_confuse={top_p}({top_c})")
            else:
                print(f"  - {lab:12s} acc={a*100:6.2f}% support={sup:5d}")

    print(f"[DONE] Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
