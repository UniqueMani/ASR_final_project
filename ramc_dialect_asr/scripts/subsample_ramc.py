"""Subsample MagicData RAMC (or similar) dataset by speaker.

Goal: keep many speakers, but only N utterances per speaker (e.g., 20) to make
the dataset small enough for a course project.

Expected folder layout (example):

  <root>/train/<spk_id>/*.wav
  <root>/dev/<spk_id>/*.wav
  <root>/test/<spk_id>/*.wav
  <root>/metadata/TRANS.txt
  <root>/metadata/SPKINFO.txt

TRANS.txt header (TSV): UtteranceID\tSpeakerID\tTranscription
SPKINFO.txt: speaker meta with at least a speaker id column (e.g., SPKID / SpeakerID).

This script supports two modes:
  1) copy (recommended): create a NEW small dataset at --dst_root
  2) delete: delete files in-place under --root (dangerous)

NEW (split-local metadata):
  In copy mode, the script writes *split-local* metadata under:

    <dst_root>/<split>/metadata/
      TRANS.subsampled.txt
      SPKINFO.subsampled.txt
      <split>.subsampled.scp

  (So you can keep dev/train metadata next to the corresponding wav files.)
"""

import argparse
import csv
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def read_tsv(path: Path) -> Tuple[List[dict], List[str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = [row for row in reader]
        return rows, (reader.fieldnames or [])


def write_tsv(path: Path, fieldnames: List[str], rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def load_trans_rows(root: Path, splits: list[str], trans_arg: str) -> tuple[list[dict], str]:
    """Load TRANS rows.

    Priority:
      1) --trans provided
      2) <root>/metadata/TRANS.txt
      3) For each split in splits: <root>/<split>/TRANS.txt (then merged)

    Returns (rows, source_description)
    """
    if trans_arg:
        p = Path(trans_arg)
        if not p.exists():
            raise SystemExit(f"TRANS not found: {p}")
        rows, _ = read_tsv(p)
        return rows, str(p)

    p_meta = root / "metadata" / "TRANS.txt"
    if p_meta.exists():
        rows, _ = read_tsv(p_meta)
        return rows, str(p_meta)

    # try per-split TRANS.txt
    merged: list[dict] = []
    used = []
    for sp in splits:
        p_sp = root / sp / "TRANS.txt"
        if not p_sp.exists():
            raise SystemExit(
                f"TRANS not found: {p_meta}\n"
                f"Also tried per-split TRANS: {p_sp} (missing).\n"
                f"Fix: pass --trans <path/to/TRANS.txt> or place TRANS.txt under <root>/metadata/."
            )
        rows, _ = read_tsv(p_sp)
        merged.extend(rows)
        used.append(str(p_sp))
    return merged, " + ".join(used)


def _spk_id_from_row(r: dict) -> str:
    for k in ("SPKID", "spkid", "SpeakerID", "speaker_id", "spk_id", "spk"):
        v = (r.get(k) or "").strip()
        if v:
            return v
    return ""


def load_spkinfo_rows(spkinfo_path: Path) -> tuple[list[dict], list[str]]:
    rows, fns = read_tsv(spkinfo_path)
    if not fns:
        # fallback: keep common columns
        fns = list(rows[0].keys()) if rows else ["SPKID", "Age", "Gender", "Dialect"]
    return rows, fns


def index_rows_by_split(root: Path, split: str, trans_rows: List[dict]) -> Dict[str, List[dict]]:
    """Return spk -> list(rows) that actually exist in <root>/<split>/<spk>/<utt>."""
    groups: Dict[str, List[dict]] = {}
    for r in trans_rows:
        utt = r.get("UtteranceID")
        spk = r.get("SpeakerID")
        if not utt or not spk:
            continue
        wav = root / split / spk / utt
        if not wav.exists():
            continue
        rr = dict(r)
        rr["_split"] = split
        rr["_wav"] = str(wav)
        groups.setdefault(spk, []).append(rr)
    return groups


def select_per_speaker(groups: Dict[str, List[dict]], n_per_spk: int, rng: random.Random) -> Tuple[List[dict], Dict[str, set]]:
    selected_rows: List[dict] = []
    keep_map: Dict[str, set] = {}
    for spk, rows in groups.items():
        if not rows:
            continue
        k = min(n_per_spk, len(rows))
        pick = rng.sample(rows, k)
        selected_rows.extend(pick)
        keep_map[spk] = set([p["UtteranceID"] for p in pick])
    return selected_rows, keep_map


def copy_selected(selected_rows: List[dict], dst_root: Path):
    for r in selected_rows:
        split = r["_split"]
        spk = r["SpeakerID"]
        utt = r["UtteranceID"]
        src = Path(r["_wav"])
        dst = dst_root / split / spk / utt
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def delete_unselected(root: Path, split: str, keep_map: Dict[str, set]):
    """Delete wavs not in keep_map under <root>/<split>/<spk>"""
    split_dir = root / split
    if not split_dir.exists():
        return
    for spk_dir in split_dir.iterdir():
        if not spk_dir.is_dir():
            continue
        spk = spk_dir.name
        keep = keep_map.get(spk, set())
        for wav in spk_dir.glob("*.wav"):
            if wav.name not in keep:
                try:
                    wav.unlink()
                except Exception:
                    pass


def write_scp(path: Path, selected_rows: List[dict]):
    """Write a simple scp: <utt_id_without_ext> <relative_wav_path>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in selected_rows:
            utt = r["UtteranceID"]
            spk = r["SpeakerID"]
            split = r["_split"]
            rel = (Path(split) / spk / utt).as_posix()
            f.write(f"{Path(utt).stem} {rel}\n")


def filter_spkinfo(spkinfo_rows: list[dict], keep_speakers: set[str]) -> list[dict]:
    out = []
    for r in spkinfo_rows:
        sid = _spk_id_from_row(r)
        if sid and sid in keep_speakers:
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Dataset root containing train/dev/test + metadata")
    ap.add_argument("--splits", type=str, default="train", help="Comma-separated splits to process, e.g. train,dev")
    ap.add_argument("--n_per_spk", type=int, default=20, help="Keep N wavs per speaker per split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["copy", "delete"], default="copy")
    ap.add_argument("--dst_root", type=str, default="", help="Required when mode=copy. Output root for small dataset.")
    ap.add_argument("--trans", type=str, default="", help="Path to TRANS.txt (default: <root>/metadata/TRANS.txt)")
    ap.add_argument("--spkinfo", type=str, default="", help="Path to SPKINFO.txt (default: <root>/metadata/SPKINFO.txt)")
    args = ap.parse_args()

    root = Path(args.root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise SystemExit("No splits")

    trans_rows, trans_src = load_trans_rows(root, splits, args.trans)
    print(f"[INFO] Using TRANS from: {trans_src}")

    spkinfo_path = Path(args.spkinfo) if args.spkinfo else (root / "metadata" / "SPKINFO.txt")
    if not spkinfo_path.exists():
        raise SystemExit(f"SPKINFO not found: {spkinfo_path}")

    spkinfo_rows, spkinfo_fns = load_spkinfo_rows(spkinfo_path)
    rng = random.Random(args.seed)

    all_selected: List[dict] = []
    keep_maps: Dict[str, Dict[str, set]] = {}

    for split in splits:
        groups = index_rows_by_split(root, split, trans_rows)
        selected, keep_map = select_per_speaker(groups, args.n_per_spk, rng)
        all_selected.extend(selected)
        keep_maps[split] = keep_map
        print(f"[OK] split={split} speakers={len(groups)} selected_utts={len(selected)}")

    if args.mode == "copy":
        if not args.dst_root:
            raise SystemExit("mode=copy requires --dst_root")
        dst_root = Path(args.dst_root)
        dst_root.mkdir(parents=True, exist_ok=True)

        copy_selected(all_selected, dst_root)

        # write split-local metadata
        for split in splits:
            rows_sp = [r for r in all_selected if r["_split"] == split]
            keep_spk = set([r["SpeakerID"] for r in rows_sp])
            meta_out = dst_root / split / "metadata"
            meta_out.mkdir(parents=True, exist_ok=True)

            # TRANS for this split
            write_tsv(
                meta_out / "TRANS.subsampled.txt",
                ["UtteranceID", "SpeakerID", "Transcription"],
                rows_sp,
            )

            # SCP for this split
            write_scp(meta_out / f"{split}.subsampled.scp", rows_sp)

            # SPKINFO (filtered to speakers that appear in this split)
            spk_rows = filter_spkinfo(spkinfo_rows, keep_spk)
            write_tsv(meta_out / "SPKINFO.subsampled.txt", spkinfo_fns, spk_rows)

        print(f"[OK] Small dataset written -> {dst_root.resolve()}")

    else:
        # delete mode: remove unselected wavs in-place
        for split in splits:
            delete_unselected(root, split, keep_maps.get(split, {}))

        # write split-local metadata under <root>/<split>/metadata/
        for split in splits:
            rows_sp = [r for r in all_selected if r["_split"] == split]
            keep_spk = set([r["SpeakerID"] for r in rows_sp])
            meta_out = root / split / "metadata"
            meta_out.mkdir(parents=True, exist_ok=True)

            write_tsv(
                meta_out / "TRANS.subsampled.txt",
                ["UtteranceID", "SpeakerID", "Transcription"],
                rows_sp,
            )
            write_scp(meta_out / f"{split}.subsampled.scp", rows_sp)

            spk_rows = filter_spkinfo(spkinfo_rows, keep_spk)
            write_tsv(meta_out / "SPKINFO.subsampled.txt", spkinfo_fns, spk_rows)

        print("[OK] In-place deletion done. Split-local metadata written under <root>/<split>/metadata/.")


if __name__ == "__main__":
    main()
