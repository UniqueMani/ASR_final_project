"""Prepare a JSONL manifest for MagicData RAMC (dialect metadata available).

It reads TRANS + SPKINFO and creates a manifest JSONL with fields:
  utt_id, speaker_id, dialect, wav_path, text, split

This version supports *split-local metadata* written by scripts/subsample_ramc.py:

  <root>/<split>/metadata/TRANS.subsampled.txt
  <root>/<split>/metadata/SPKINFO.subsampled.txt

It will still fallback to the old global layout (<root>/metadata/*) if present.

Usage examples:

  # If you subsampled dev only (recommended for a small course project)
  python scripts/prepare_manifest_ramc.py --root ./data/RAMC_small --split dev --out ./data/RAMC_small/dev/metadata/manifest.jsonl

  # If you subsampled train+dev and want a combined manifest
  python scripts/prepare_manifest_ramc.py --root ./data/RAMC_small --split all --out ./data/manifest.jsonl
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List


def _norm_dialect(s: str) -> str:
    return "_".join(str(s).strip().split())


def read_tsv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [r for r in reader]


def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def load_trans_rows(root: Path, split: str, trans_arg: str) -> Tuple[List[dict], str]:
    """Load TRANS rows for a given split.

    Priority:
      1) --trans provided
      2) split-local: <root>/<split>/metadata/TRANS.subsampled.txt then <root>/<split>/TRANS.txt
      3) global: <root>/metadata/TRANS.subsampled.txt then <root>/metadata/TRANS.txt
      4) if split == 'all': merge split-local (or per-split TRANS.txt) from train/dev/test if present
    """
    if trans_arg:
        p = Path(trans_arg)
        if not p.exists():
            raise SystemExit(f"TRANS not found: {p}")
        return read_tsv(p), str(p)

    split = split.lower().strip()

    if split in ("train", "dev", "test"):
        cand = [
            root / split / "metadata" / "TRANS.subsampled.txt",
            root / split / "TRANS.txt",
            root / "metadata" / "TRANS.subsampled.txt",
            root / "metadata" / "TRANS.txt",
        ]
        p = _first_existing(cand)
        if p is not None:
            return read_tsv(p), str(p)

    if split == "all":
        # prefer a single merged TRANS if available
        p_global = _first_existing([
            root / "metadata" / "TRANS.subsampled.txt",
            root / "metadata" / "TRANS.txt",
        ])
        if p_global is not None:
            return read_tsv(p_global), str(p_global)

        merged: List[dict] = []
        used = []
        for sp in ("train", "dev", "test"):
            p = _first_existing([
                root / sp / "metadata" / "TRANS.subsampled.txt",
                root / sp / "TRANS.txt",
            ])
            if p is not None:
                merged.extend(read_tsv(p))
                used.append(str(p))
        if merged:
            return merged, " + ".join(used)

    raise SystemExit(
        "TRANS not found. Provide --trans, or place TRANS under "
        "<root>/<split>/metadata/TRANS.subsampled.txt, or under <root>/metadata/TRANS.txt."
    )


def _get_any(r: dict, keys: List[str]) -> str:
    for k in keys:
        v = (r.get(k) or "").strip()
        if v:
            return v
    return ""


def read_spkinfo(path: Path) -> Dict[str, dict]:
    spk = {}
    for r in read_tsv(path):
        sid = _get_any(r, ["SPKID", "spkid", "SpeakerID", "speaker_id", "spk_id", "spk"])
        if not sid:
            continue
        dialect = _norm_dialect(_get_any(r, ["Dialect", "DIALECT", "dialect", "NativePlace", "nativeplace", "Region", "region"]))
        spk[sid] = {
            "dialect": dialect,
            "age": _get_any(r, ["Age", "AGE", "age"]),
            "gender": _get_any(r, ["Gender", "GENDER", "gender", "Sex", "sex"]),
        }
    return spk


def load_spkmeta(root: Path, split: str, spkinfo_arg: str) -> Tuple[Dict[str, dict], str]:
    """Load SPKINFO.

    Priority:
      1) --spkinfo provided
      2) split-local: <root>/<split>/metadata/SPKINFO.subsampled.txt then SPKINFO.txt
      3) global: <root>/metadata/SPKINFO.subsampled.txt then SPKINFO.txt
      4) if split == 'all': merge split-local/global if found
    """
    if spkinfo_arg:
        p = Path(spkinfo_arg)
        if not p.exists():
            raise SystemExit(f"SPKINFO not found: {p}")
        return read_spkinfo(p), str(p)

    split = split.lower().strip()

    if split in ("train", "dev", "test"):
        cand = [
            root / split / "metadata" / "SPKINFO.subsampled.txt",
            root / split / "metadata" / "SPKINFO.txt",
            root / "metadata" / "SPKINFO.subsampled.txt",
            root / "metadata" / "SPKINFO.txt",
        ]
        p = _first_existing(cand)
        if p is not None:
            return read_spkinfo(p), str(p)

    if split == "all":
        p_global = _first_existing([
            root / "metadata" / "SPKINFO.subsampled.txt",
            root / "metadata" / "SPKINFO.txt",
        ])
        if p_global is not None:
            return read_spkinfo(p_global), str(p_global)

        merged: Dict[str, dict] = {}
        used = []
        for sp in ("train", "dev", "test"):
            p = _first_existing([
                root / sp / "metadata" / "SPKINFO.subsampled.txt",
                root / sp / "metadata" / "SPKINFO.txt",
            ])
            if p is not None:
                merged.update(read_spkinfo(p))
                used.append(str(p))
        if merged:
            return merged, " + ".join(used)

    raise SystemExit(
        "SPKINFO not found. Provide --spkinfo, or place SPKINFO under "
        "<root>/<split>/metadata/SPKINFO.subsampled.txt, or under <root>/metadata/SPKINFO.txt."
    )


def locate_wav(root: Path, split: str, spk: str, utt: str) -> Optional[Path]:
    p = root / split / spk / utt
    return p if p.exists() else None


def locate_wav_any(root: Path, spk: str, utt: str, order: list[str]) -> Optional[tuple[Path, str]]:
    for s in order:
        p = locate_wav(root, s, spk, utt)
        if p is not None:
            return p, s
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Dataset root containing train/dev/test + metadata")
    ap.add_argument("--split", type=str, default="train", help="train/dev/test/all")
    ap.add_argument("--trans", type=str, default="", help="Path to TRANS TSV (optional)")
    ap.add_argument("--spkinfo", type=str, default="", help="Path to SPKINFO TSV (optional)")
    ap.add_argument("--out", type=str, required=True, help="Output manifest.jsonl")
    args = ap.parse_args()

    root = Path(args.root)
    split = args.split.strip().lower()

    trans_rows, trans_src = load_trans_rows(root, split, args.trans)
    spkmeta, spk_src = load_spkmeta(root, split, args.spkinfo)

    print(f"[INFO] Using TRANS from: {trans_src}")
    print(f"[INFO] Using SPKINFO from: {spk_src}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    order = ["train", "dev", "test"]
    n_written = 0
    n_missing = 0
    with out.open("w", encoding="utf-8") as f:
        for r in trans_rows:
            utt = (r.get("UtteranceID") or "").strip()
            spk = (r.get("SpeakerID") or "").strip()
            txt = (r.get("Transcription") or "").strip()
            if not utt or not spk:
                continue

            if split == "all":
                found = locate_wav_any(root, spk, utt, order)
                if found is None:
                    n_missing += 1
                    continue
                wav_path, sp = found
            else:
                wav_path = locate_wav(root, split, spk, utt)
                if wav_path is None:
                    n_missing += 1
                    continue
                sp = split

            meta = spkmeta.get(spk, {})
            rec = {
                "utt_id": Path(utt).stem,
                "speaker_id": spk,
                "dialect": meta.get("dialect", ""),
                "wav_path": str(wav_path.resolve()),
                "text": txt,
                "split": sp,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[OK] Wrote {n_written} items -> {out.resolve()}")
    if n_missing:
        print(f"[WARN] {n_missing} rows in TRANS not found under split(s) {split}")


if __name__ == "__main__":
    main()
