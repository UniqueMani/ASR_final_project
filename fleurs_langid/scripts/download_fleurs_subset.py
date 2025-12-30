#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download a subset of google/fleurs for selected language configs.

Features:
- --list: list all available language configs
- download selected langs & splits
- save as:
  - arrow: datasets.save_to_disk() (fast, recommended)
  - wav: export audio to .wav + write manifest.jsonl (easy to use in classic pipelines)
- optional --max_per_split to limit size
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

def pick_text_field(columns) -> Optional[str]:
    """Pick a likely transcription field name."""
    candidates = ["transcription", "raw_transcription", "text", "sentence", "normalized_text"]
    for c in candidates:
        if c in columns:
            return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/fleurs_subset", help="Output root directory")
    ap.add_argument("--cache_dir", type=str, default=None, help="HF datasets cache dir (optional)")
    ap.add_argument("--langs", nargs="*", default=None,
                    help="Language config codes, e.g. en_us ja_jp cmn_hans_cn cmn_hant_tw yue_hant_hk")
    ap.add_argument("--splits", nargs="*", default=["train", "validation", "test"],
                    help="Splits to download: train/validation/test")
    ap.add_argument("--format", choices=["arrow", "wav"], default="arrow",
                    help="Save format: arrow (save_to_disk) or wav (export wav + manifest.jsonl)")
    ap.add_argument("--sr", type=int, default=16000, help="Target sampling rate for exported wav")
    ap.add_argument("--max_per_split", type=int, default=0,
                    help="Limit items per split per language (0 means no limit)")
    ap.add_argument("--list", action="store_true", help="List available language configs and exit")
    args = ap.parse_args()

    # Lazy imports so --list works quickly
    from datasets import get_dataset_config_names, load_dataset, Audio
    from tqdm import tqdm

    ds_name = "google/fleurs"

    if args.list:
        cfgs = get_dataset_config_names(ds_name)
        print(f"[INFO] {ds_name} configs = {len(cfgs)}")
        for c in cfgs:
            print(c)
        print("\nTip: use --langs <codes...> to download selected ones.")
        return

    if not args.langs:
        raise SystemExit("Please provide --langs (or run with --list to see available codes).")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # Optional: faster download if you have token
    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        print("[WARN] No HF token found. You may see rate-limit warnings. "
              "Optional: set HF_TOKEN for faster downloads.")

    for lang in args.langs:
        print(f"\n[LANG] {lang}")
        lang_out = out_root / lang
        lang_out.mkdir(parents=True, exist_ok=True)

        for split in args.splits:
            print(f"[SPLIT] {split}")

            ds = load_dataset(ds_name, name=lang, split=split, cache_dir=args.cache_dir)
            if args.max_per_split and args.max_per_split > 0 and len(ds) > args.max_per_split:
                ds = ds.select(range(args.max_per_split))

            # Make sure audio is decoded at target SR
            if "audio" in ds.column_names:
                ds = ds.cast_column("audio", Audio(sampling_rate=args.sr))

            if args.format == "arrow":
                save_dir = lang_out / split
                ds.save_to_disk(str(save_dir))
                print(f"[OK] saved_to_disk -> {save_dir}")
                continue

            # format == wav
            import soundfile as sf

            wav_dir = lang_out / split / "wav"
            wav_dir.mkdir(parents=True, exist_ok=True)

            text_field = pick_text_field(ds.column_names)
            manifest_path = lang_out / split / "manifest.jsonl"

            n = len(ds)
            with manifest_path.open("w", encoding="utf-8") as f:
                for i in tqdm(range(n), desc=f"{lang}:{split}", unit="utt"):
                    ex = ds[i]
                    audio = ex.get("audio", None)
                    if audio is None:
                        continue

                    arr = audio["array"]
                    sr = audio["sampling_rate"]

                    # file id
                    uid = ex.get("id", None) or ex.get("utterance_id", None) or f"{lang}_{split}_{i}"
                    uid = str(uid).replace("/", "_")

                    wav_path = wav_dir / f"{uid}.wav"
                    sf.write(str(wav_path), arr, sr)

                    txt = ex.get(text_field, "") if text_field else ""
                    dur = float(len(arr) / sr) if sr else None

                    row = {
                        "id": uid,
                        "lang": lang,
                        "split": split,
                        "wav": str(wav_path.as_posix()),
                        "sr": sr,
                        "duration": dur,
                        "text": txt,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            print(f"[OK] exported wav -> {wav_dir}")
            print(f"[OK] manifest -> {manifest_path}")

    print("\n[DONE] All requested languages downloaded/exported.")

if __name__ == "__main__":
    main()
