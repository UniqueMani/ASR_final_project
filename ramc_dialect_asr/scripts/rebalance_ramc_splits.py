#!/usr/bin/env python
"""
Rebalance / re-split a small MagicData RAMC subset by dialect (province).

Why:
- Your current RAMC_small train/dev/test may have label mismatch:
  some dialect labels appear in test but not in train -> model can't learn them.

What this script does (speaker-level):
1) Read ALL speakers+utterances from existing splits under <root>:
   <root>/train, <root>/dev, <root>/test
   and split-local metadata:
   <root>/<split>/metadata/TRANS.subsampled.txt
   <root>/<split>/metadata/SPKINFO.subsampled.txt
   <root>/<split>/metadata/<split>.subsampled.scp
2) Pool them together.
3) Stratify speakers by dialect and re-assign into new train/dev/test with
   better coverage (every dialect appears in train/test when possible).
4) Copy/hardlink wavs into <out_root>/<split>/<spk_id>/*.wav
5) Write new split-local metadata into <out_root>/<split>/metadata/:
   - TRANS.subsampled.txt
   - SPKINFO.subsampled.txt
   - <split>.subsampled.scp
6) (Optional) Write pooled metadata into <out_root>/metadata/:
   - TRANS.all.txt, SPKINFO.all.txt, all.scp

After that, you can reuse your original manifest generator:
  python scripts/prepare_manifest_ramc.py --root <out_root> --split train --out <out_root>/train/metadata/manifest.jsonl
  python scripts/prepare_manifest_ramc.py --root <out_root> --split test  --out <out_root>/test/metadata/manifest.jsonl

Notes:
- Speaker-level split avoids speaker leakage across splits.
- If a dialect has only 1 speaker in the whole pool, it can only go to ONE split.
  We default to put it in train.

Author: course-project helper script
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ------------------------- IO helpers -------------------------

def read_tsv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))

def write_tsv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

def read_scp(path: Path) -> Dict[str, str]:
    """
    scp format used in this project:
      <utt_id_without_ext> <relative_wav_path>
    """
    m: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            utt_id = parts[0]
            rel = parts[1]
            m[utt_id] = rel
    return m

def safe_norm_label(s: str) -> str:
    return "_".join(str(s).strip().split())

def hardlink_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "link":
        try:
            os.link(src, dst)  # hardlink (works on NTFS)
            return
        except Exception:
            # fallback
            shutil.copy2(src, dst)
            return
    raise ValueError(f"unknown mode={mode}")

def find_existing(p_list: List[Path]) -> Optional[Path]:
    for p in p_list:
        if p.exists():
            return p
    return None


# ------------------------- data model -------------------------

@dataclass
class UttItem:
    utt_id: str          # stem
    utt_file: str        # with .wav (from TRANS UtteranceID)
    speaker_id: str
    dialect: str
    text: str
    rel_path: str        # split/spk/utt.wav (relative to root)
    abs_path: Path       # root/rel_path


# ------------------------- load pool -------------------------

def load_split_metadata(root: Path, split: str) -> Tuple[List[dict], Dict[str, dict], Dict[str, str]]:
    """Return (trans_rows, spkinfo_map, scp_map) for a split."""
    meta = root / split / "metadata"
    trans_p = find_existing([
        meta / "TRANS.subsampled.txt",
        root / split / "TRANS.txt",
    ])
    spk_p = find_existing([
        meta / "SPKINFO.subsampled.txt",
        meta / "SPKINFO.txt",
        root / split / "SPKINFO.txt",
    ])
    scp_p = find_existing([
        meta / f"{split}.subsampled.scp",
        meta / f"{split}.scp",
    ])
    if trans_p is None:
        raise SystemExit(f"TRANS not found for split={split}. Expected {meta/'TRANS.subsampled.txt'}")
    if spk_p is None:
        raise SystemExit(f"SPKINFO not found for split={split}. Expected {meta/'SPKINFO.subsampled.txt'}")
    if scp_p is None:
        raise SystemExit(f"SCP not found for split={split}. Expected {meta/(split+'.subsampled.scp')}")
    trans_rows = read_tsv(trans_p)

    # speaker meta
    spk_rows = read_tsv(spk_p)
    spk_map: Dict[str, dict] = {}
    for r in spk_rows:
        sid = (r.get("SPKID") or r.get("spkid") or r.get("SpeakerID") or r.get("speaker_id") or "").strip()
        if not sid:
            continue
        dialect = safe_norm_label((r.get("Dialect") or r.get("DIALECT") or r.get("dialect") or r.get("NativePlace") or r.get("nativeplace") or "").strip())
        spk_map[sid] = {
            "SPKID": sid,
            "Age": (r.get("Age") or r.get("AGE") or r.get("age") or "").strip(),
            "Gender": (r.get("Gender") or r.get("GENDER") or r.get("gender") or r.get("Sex") or "").strip(),
            "Dialect": dialect,
        }

    scp_map = read_scp(scp_p)
    return trans_rows, spk_map, scp_map


def build_pool(root: Path, splits: List[str]) -> Tuple[List[UttItem], Dict[str, dict]]:
    """Pool utterances across splits. Return (items, merged_spkinfo)."""
    merged_spk: Dict[str, dict] = {}
    scp_all: Dict[str, str] = {}
    trans_all: List[dict] = []

    # read each split and merge
    for sp in splits:
        trans_rows, spk_map, scp_map = load_split_metadata(root, sp)
        trans_all.extend([{**r, "_split": sp} for r in trans_rows])
        merged_spk.update(spk_map)  # later splits overwrite (should be same)
        # scp keys are utt stem -> relpath (includes split)
        scp_all.update(scp_map)

    items: List[UttItem] = []
    missing = 0
    for r in trans_all:
        utt_file = (r.get("UtteranceID") or "").strip()
        spk = (r.get("SpeakerID") or "").strip()
        txt = (r.get("Transcription") or "").strip()
        if not utt_file or not spk:
            continue
        utt_id = Path(utt_file).stem
        rel = scp_all.get(utt_id, "")
        if not rel:
            # fallback: assume it's under its original split
            sp = r.get("_split", "")
            rel_guess = (Path(sp) / spk / utt_file).as_posix()
            if (root / rel_guess).exists():
                rel = rel_guess
            else:
                missing += 1
                continue

        abs_path = (root / rel)
        if not abs_path.exists():
            missing += 1
            continue

        spk_meta = merged_spk.get(spk, {"Dialect": ""})
        dialect = safe_norm_label(spk_meta.get("Dialect", ""))
        items.append(UttItem(
            utt_id=utt_id,
            utt_file=utt_file,
            speaker_id=spk,
            dialect=dialect,
            text=txt,
            rel_path=rel,
            abs_path=abs_path,
        ))

    if missing:
        print(f"[WARN] {missing} TRANS rows could not be matched to wav files; they were skipped.")
    print(f"[INFO] Pooled utterances: {len(items)}, speakers(meta): {len(merged_spk)}")
    return items, merged_spk


# ------------------------- split logic -------------------------

def stratified_split_speakers(
    speakers: List[str],
    spk2dialect: Dict[str, str],
    seed: int,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    cap_spk_per_label: int,
    ensure_train: bool,
    ensure_test: bool,
) -> Dict[str, str]:
    """
    Return mapping speaker_id -> new_split ("train"/"dev"/"test")
    Speaker-level stratified split by dialect.
    """
    rng = random.Random(seed)

    # group speakers by dialect
    groups: Dict[str, List[str]] = {}
    for spk in speakers:
        lab = spk2dialect.get(spk, "") or ""
        lab = safe_norm_label(lab)
        groups.setdefault(lab, []).append(spk)

    # apply cap per label (optional)
    for lab, lst in groups.items():
        rng.shuffle(lst)
        if cap_spk_per_label and len(lst) > cap_spk_per_label:
            groups[lab] = lst[:cap_spk_per_label]

    assign: Dict[str, str] = {}
    for lab, lst in groups.items():
        rng.shuffle(lst)
        n = len(lst)
        if n == 0:
            continue

        # baseline counts from ratios
        n_train = int(round(n * train_ratio))
        n_dev   = int(round(n * dev_ratio))
        n_test  = n - n_train - n_dev

        # fix negatives / sum mismatch
        if n_test < 0:
            n_test = 0
        if n_train + n_dev + n_test != n:
            n_test = n - n_train - n_dev
            if n_test < 0:
                # reduce dev first, then train
                take = -n_test
                dec = min(take, n_dev); n_dev -= dec; take -= dec
                dec = min(take, n_train); n_train -= dec; take -= dec
                n_test = 0

        # ensure coverage
        if ensure_train and n >= 1 and n_train == 0:
            n_train = 1
            if n_test > 0:
                n_test -= 1
            elif n_dev > 0:
                n_dev -= 1

        if ensure_test and n >= 2 and n_test == 0:
            n_test = 1
            if n_dev > 0:
                n_dev -= 1
            elif n_train > 1:
                n_train -= 1

        # if label is super small (n==1) -> put into train
        if n == 1:
            n_train, n_dev, n_test = 1, 0, 0

        # final sanity
        while n_train + n_dev + n_test < n:
            n_train += 1
        while n_train + n_dev + n_test > n:
            if n_dev > 0:
                n_dev -= 1
            elif n_test > 0:
                n_test -= 1
            else:
                n_train -= 1

        # assign
        idx = 0
        for _ in range(n_train):
            assign[lst[idx]] = "train"; idx += 1
        for _ in range(n_dev):
            assign[lst[idx]] = "dev"; idx += 1
        for _ in range(n_test):
            assign[lst[idx]] = "test"; idx += 1

    return assign


def print_split_stats(assign: Dict[str, str], spk2dialect: Dict[str, str]) -> None:
    """Print per-split dialect counts (speakers)."""
    splits = ["train", "dev", "test"]
    counts: Dict[str, Dict[str, int]] = {s: {} for s in splits}
    for spk, sp in assign.items():
        lab = spk2dialect.get(spk, "") or ""
        lab = safe_norm_label(lab)
        counts[sp][lab] = counts[sp].get(lab, 0) + 1

    for sp in splits:
        total = sum(counts[sp].values())
        labs = sorted(counts[sp].items(), key=lambda x: (-x[1], x[0]))
        print(f"[STAT] {sp}: speakers={total}, dialects={len(labs)}")
        print("       top:", ", ".join([f"{k}:{v}" for k, v in labs[:10]]))


# ------------------------- write output -------------------------

def write_split_dataset(
    out_root: Path,
    split: str,
    items: List[UttItem],
    spk_rows: List[dict],
    mode: str,
) -> None:
    """Write wavs + metadata for a split."""
    # copy/link wavs
    for it in items:
        dst = out_root / split / it.speaker_id / it.utt_file
        hardlink_or_copy(it.abs_path, dst, mode)

    # metadata
    meta_dir = out_root / split / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # TRANS
    trans_rows = [
        {"UtteranceID": it.utt_file, "SpeakerID": it.speaker_id, "Transcription": it.text}
        for it in items
    ]
    write_tsv(meta_dir / "TRANS.subsampled.txt", trans_rows, ["UtteranceID", "SpeakerID", "Transcription"])

    # SPKINFO
    write_tsv(meta_dir / "SPKINFO.subsampled.txt", spk_rows, ["SPKID", "Age", "Gender", "Dialect"])

    # SCP
    scp_path = meta_dir / f"{split}.subsampled.scp"
    with scp_path.open("w", encoding="utf-8") as f:
        for it in items:
            rel = (Path(split) / it.speaker_id / it.utt_file).as_posix()
            f.write(f"{it.utt_id} {rel}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Existing dataset root (e.g., ./data/RAMC_small) containing train/dev/test")
    ap.add_argument("--out_root", required=True, help="Output dataset root (e.g., ./data/RAMC_balanced)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["copy", "link"], default="copy", help="copy wavs or hardlink them")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--dev_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--cap_spk_per_label", type=int, default=0, help="Optional: cap speakers per dialect to make distribution more uniform (0=disable)")
    ap.add_argument("--ensure_train", action="store_true", help="Ensure every dialect appears in train when possible")
    ap.add_argument("--ensure_test", action="store_true", help="Ensure every dialect appears in test when possible (needs >=2 speakers in that dialect)")
    ap.add_argument("--overwrite", action="store_true", help="If out_root exists, delete it first")
    ap.add_argument("--use_splits", type=str, default="train,dev,test", help="Which existing splits to pool from (default: train,dev,test)")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_root)

    if args.overwrite and out_root.exists():
        print(f"[INFO] Removing existing out_root: {out_root}")
        shutil.rmtree(out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.use_splits.split(",") if s.strip()]
    items, spkmeta = build_pool(root, splits)

    # speaker -> dialect
    spk2dialect: Dict[str, str] = {}
    for sid, meta in spkmeta.items():
        spk2dialect[sid] = safe_norm_label(meta.get("Dialect", ""))

    # speakers that actually have utterances
    spk_has_utts = sorted({it.speaker_id for it in items})
    print(f"[INFO] Speakers with utterances: {len(spk_has_utts)}")

    # assign speakers
    assign = stratified_split_speakers(
        speakers=spk_has_utts,
        spk2dialect=spk2dialect,
        seed=args.seed,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        cap_spk_per_label=args.cap_spk_per_label,
        ensure_train=args.ensure_train,
        ensure_test=args.ensure_test,
    )
    print_split_stats(assign, spk2dialect)

    # partition utterances by new split (speaker-level)
    split_items: Dict[str, List[UttItem]] = {"train": [], "dev": [], "test": []}
    for it in items:
        sp = assign.get(it.speaker_id, "train")
        split_items[sp].append(it)

    # partition spkinfo rows by new split
    split_spk_rows: Dict[str, List[dict]] = {"train": [], "dev": [], "test": []}
    for spk, sp in assign.items():
        meta = spkmeta.get(spk, {"SPKID": spk, "Age": "", "Gender": "", "Dialect": ""})
        split_spk_rows[sp].append({
            "SPKID": spk,
            "Age": meta.get("Age", ""),
            "Gender": meta.get("Gender", ""),
            "Dialect": safe_norm_label(meta.get("Dialect", "")),
        })

    # write each split
    for sp in ["train", "dev", "test"]:
        if not split_items[sp]:
            print(f"[WARN] split={sp} has 0 utterances, skip writing.")
            continue
        print(f"[INFO] Writing split={sp}: speakers={len(split_spk_rows[sp])}, utts={len(split_items[sp])}")
        write_split_dataset(out_root, sp, split_items[sp], split_spk_rows[sp], args.mode)

    # pooled metadata (optional, always write)
    meta_all = out_root / "metadata"
    meta_all.mkdir(parents=True, exist_ok=True)
    write_tsv(meta_all / "TRANS.all.txt",
              [{"UtteranceID": it.utt_file, "SpeakerID": it.speaker_id, "Transcription": it.text} for it in items],
              ["UtteranceID", "SpeakerID", "Transcription"])
    write_tsv(meta_all / "SPKINFO.all.txt",
              [{"SPKID": sid, "Age": spkmeta.get(sid, {}).get("Age",""), "Gender": spkmeta.get(sid, {}).get("Gender",""), "Dialect": spk2dialect.get(sid,"")}
               for sid in spk_has_utts],
              ["SPKID", "Age", "Gender", "Dialect"])
    with (meta_all / "all.scp").open("w", encoding="utf-8") as f:
        for it in items:
            # rel in new root is unknown (speaker assigned), so write original rel path under old root
            f.write(f"{it.utt_id} {it.rel_path}\n")

    print("[OK] Done.")
    print("Next steps (generate manifests):")
    print(f"  python scripts/prepare_manifest_ramc.py --root {out_root.as_posix()} --split train --out { (out_root/'train/metadata/manifest.jsonl').as_posix() }")
    print(f"  python scripts/prepare_manifest_ramc.py --root {out_root.as_posix()} --split test  --out { (out_root/'test/metadata/manifest.jsonl').as_posix() }")
    print(f"  python scripts/prepare_manifest_ramc.py --root {out_root.as_posix()} --split dev   --out { (out_root/'dev/metadata/manifest.jsonl').as_posix() }")

if __name__ == "__main__":
    main()
