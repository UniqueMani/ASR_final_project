#!/usr/bin/env python3
from __future__ import annotations
import argparse
from backend.langid.gauss_model import GaussDiagModel
from backend.langid.mfcc import mfcc_stats_from_wav, MfccConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--cache_dir", default="data/.mfcc_cache")
    args = ap.parse_args()

    model = GaussDiagModel.load(args.model_dir)
    feat, meta = mfcc_stats_from_wav(args.wav, cfg=MfccConfig(), cache_dir=args.cache_dir)
    _, lab = model.predict(feat[None, :])
    print({"wav": args.wav, "pred_language": str(lab[0]), "cache_hit": bool(meta.get("cache_hit", False)), "frames": meta.get("frames")})

if __name__ == "__main__":
    main()
