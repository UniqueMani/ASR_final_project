#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
merge_fleurs.py

Merge multiple per-language FLEURS subsets into a single merged dataset with unified train/test/validation splits.

Expected input layout (your current extracted subset format):
data/FLEURS_5/
  en_us/
    train/ (or train/metadata/)
      wav/...
      manifest.jsonl  (or train/metadata/manifest.jsonl)
    test/
      wav/...
      manifest.jsonl  (or test/metadata/manifest.jsonl)
    validation/
      wav/...
      manifest.jsonl  (or validation/metadata/manifest.jsonl)
  ja_jp/...
  cmn_hans_cn/...
  ko_kr/...
  yue_hant_hk/...

This script is tolerant: for each language+split it tries these manifest locations in order:
  1) <lang_dir>/<split>/metadata/manifest.jsonl
  2) <lang_dir>/<split>/manifest.jsonl
  3) <lang_dir>/metadata/<split>.manifest.jsonl
  4) <lang_dir>/manifest.jsonl   (fallback; if it contains a "split" field, we will filter)

Output layout:
out_root/
  train/
    wav/
    metadata/manifest.jsonl
  test/
    wav/
    metadata/manifest.jsonl
  validation/
    wav/
    metadata/manifest.jsonl
  manifest.jsonl          (all splits)
  stats.json

Each output manifest row adds "language" field (e.g., en_us, ja_jp, ...).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def read_jsonl(p: Path) -> List[dict]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(p: Path, rows: Iterable[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def safe_copy(src: Path, dst: Path, mode: str = "copy") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        # On Windows, symlink may require Developer Mode or Admin.
        if dst.exists():
            dst.unlink()
        os.symlink(src, dst)
    elif mode == "hardlink":
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def find_manifest_for_split(lang_dir: Path, split: str) -> Path:
    candidates = [
        lang_dir / split / "metadata" / "manifest.jsonl",
        lang_dir / split / "manifest.jsonl",
        lang_dir / "metadata" / f"{split}.manifest.jsonl",
        lang_dir / "manifest.jsonl",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"manifest not found for {lang_dir.name}/{split}. Tried: " + ", ".join(str(c) for c in candidates))


def resolve_wav_path(wav_field: str, manifest_path: Path, root: Path, lang: str, split: str) -> Path:
    """
    Resolve wav path from a manifest row. We try multiple bases to be robust.
    """
    p = Path(wav_field)

    # absolute
    if p.is_absolute() and p.exists():
        return p

    # as-is relative to cwd
    if p.exists():
        return p.resolve()

    lang_dir = root / lang
    split_dir = lang_dir / split

    bases = [
        manifest_path.parent,
        manifest_path.parent.parent,  # e.g. .../<split>/metadata -> .../<split>
        split_dir,
        lang_dir,
        root,
    ]
    for b in bases:
        cand = (b / p)
        if cand.exists():
            return cand.resolve()

    # try "wav/<basename>" under likely folders
    for b in [split_dir, manifest_path.parent, manifest_path.parent.parent]:
        cand = b / "wav" / p.name
        if cand.exists():
            return cand.resolve()

    raise FileNotFoundError(f"WAV not found: wav='{wav_field}' (lang={lang}, split={split}, manifest={manifest_path})")


def pick_text_field(row: dict) -> str:
    # Different dumps may use "text" or "transcript"/"transcription"
    for k in ("text", "transcription", "transcript", "sentence"):
        if k in row and isinstance(row[k], str):
            return row[k]
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder containing per-language subsets (e.g., data/FLEURS_5)")
    ap.add_argument("--out_root", required=True, help="Output merged dataset root (e.g., data/FLEURS_merged)")
    ap.add_argument("--langs", nargs="+", required=True, help="Language configs to merge (e.g., en_us ja_jp ...)")
    ap.add_argument("--splits", nargs="+", default=["train", "test", "validation"])
    ap.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="copy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_per_lang_per_split", type=int, default=0, help="0 = no limit")
    ap.add_argument("--max_total_per_split", type=int, default=0, help="0 = no limit")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_root)
    rng = random.Random(args.seed)

    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)

    # Collect + merge per split
    stats: Dict[str, dict] = {}
    all_rows: List[dict] = []

    for split in args.splits:
        merged_items: List[dict] = []
        per_lang_counts: Dict[str, int] = {}

        for lang in args.langs:
            lang_dir = root / lang
            if not lang_dir.exists():
                raise SystemExit(f"Language folder not found: {lang_dir}")

            manifest = find_manifest_for_split(lang_dir, split)
            rows = read_jsonl(manifest)

            # If this is a global manifest, filter by split if possible
            if manifest.name == "manifest.jsonl":
                # some manifests have "split", some don't
                if rows and "split" in rows[0]:
                    rows = [r for r in rows if r.get("split") == split]

            rng.shuffle(rows)

            if args.max_per_lang_per_split and args.max_per_lang_per_split > 0:
                rows = rows[: args.max_per_lang_per_split]

            per_lang_counts[lang] = len(rows)

            for r in rows:
                wav_field = r.get("wav") or r.get("audio") or r.get("path")
                if not wav_field:
                    raise SystemExit(f"Row missing wav/audio/path field in {manifest}: keys={list(r.keys())}")

                wav_abs = resolve_wav_path(str(wav_field), manifest, root, lang, split)
                merged_items.append({
                    "_src_lang": lang,
                    "_src_split": split,
                    "_src_manifest": str(manifest),
                    "_src_wav_field": str(wav_field),
                    "_src_wav_abs": str(wav_abs),
                    "_src_row": r,
                })

        rng.shuffle(merged_items)
        if args.max_total_per_split and args.max_total_per_split > 0:
            merged_items = merged_items[: args.max_total_per_split]

        out_wav_dir = out_root / split / "wav"
        out_manifest = out_root / split / "metadata" / "manifest.jsonl"
        out_wav_dir.mkdir(parents=True, exist_ok=True)

        out_rows_split: List[dict] = []
        for idx, it in enumerate(merged_items):
            lang = it["_src_lang"]
            src_wav = Path(it["_src_wav_abs"])
            new_id = f"{lang}_{split}_{idx:08d}"
            dst_wav = out_wav_dir / f"{new_id}.wav"

            safe_copy(src_wav, dst_wav, mode=args.mode)

            src_row = it["_src_row"]
            text = pick_text_field(src_row)

            out_row = {
                "id": new_id,
                "wav": dst_wav.as_posix(),
                "text": text,
                "split": split,
                "language": lang,
            }
            # Keep optional metadata if present (safe keys only)
            for k in ("gender", "age", "speaker_id", "speaker", "lang_id"):
                if k in src_row:
                    out_row[k] = src_row[k]

            out_rows_split.append(out_row)

        write_jsonl(out_manifest, out_rows_split)
        all_rows.extend(out_rows_split)

        stats[split] = {
            "total": len(out_rows_split),
            "per_lang": per_lang_counts,
            "manifest_written": str(out_manifest),
        }
        print(f"[OK] {split}: wrote {len(out_rows_split)} items -> {out_manifest}")

    write_jsonl(out_root / "manifest.jsonl", all_rows)
    (out_root / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] wrote merged manifest: {out_root/'manifest.jsonl'}")
    print(f"[OK] wrote stats: {out_root/'stats.json'}")


if __name__ == "__main__":
    main()
