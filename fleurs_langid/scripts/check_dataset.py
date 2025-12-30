#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset sanity checker for language-id manifests.

Checks:
- Missing wav files
- Basic audio info (sr, duration) per label
- SHA1 duplicates across labels (within split)
- SHA1 overlap between splits (train vs test) to detect leakage or path mixups
"""

import argparse, json, hashlib, wave
from pathlib import Path
from collections import defaultdict, Counter

def read_jsonl(p: Path):
    rows=[]
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: 
                continue
            rows.append(json.loads(line))
    return rows

def wav_info(path: Path):
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
    dur = n / float(sr) if sr else 0.0
    return sr, dur, n

def sha1_file(path: Path, chunk=1<<20):
    h=hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_manifest", type=str, default=None, help="train manifest jsonl")
    ap.add_argument("--test_manifest", type=str, default=None, help="test manifest jsonl")
    ap.add_argument("--manifest", type=str, default=None, help="single manifest jsonl")
    ap.add_argument("--max_per_label", type=int, default=200, help="max files per label for heavy checks")
    args=ap.parse_args()

    manifests=[]
    if args.manifest:
        manifests.append(("manifest", Path(args.manifest)))
    if args.train_manifest:
        manifests.append(("train", Path(args.train_manifest)))
    if args.test_manifest:
        manifests.append(("test", Path(args.test_manifest)))

    if not manifests:
        raise SystemExit("Provide --manifest or --train_manifest/--test_manifest")

    split_hashes = {}
    for name, mp in manifests:
        rows = read_jsonl(mp)
        missing=0
        by_label = defaultdict(list)
        for r in rows:
            lab = str(r.get("language") or r.get("label") or r.get("lang"))
            wav = Path(r.get("wav") or r.get("audio") or r.get("path"))
            if not wav.exists():
                missing += 1
                continue
            by_label[lab].append(wav)

        print(f"\n=== {name} ===")
        print(f"items={len(rows)} labels={len(by_label)} missing_wav={missing}")

        # audio stats (sample a subset)
        sr_ctr=Counter()
        dur_by=defaultdict(list)
        hashes_by_label=defaultdict(set)
        # sample up to max_per_label per label
        for lab, wavs in by_label.items():
            sample = wavs[:]
            if len(sample) > args.max_per_label:
                sample = sample[:args.max_per_label]
            for w in sample:
                try:
                    sr, dur, _ = wav_info(w)
                    sr_ctr[sr]+=1
                    dur_by[lab].append(dur)
                except Exception:
                    pass

        for lab, durs in sorted(dur_by.items(), key=lambda x: x[0]):
            if not durs:
                continue
            durs_sorted = sorted(durs)
            p50 = durs_sorted[len(durs_sorted)//2]
            p10 = durs_sorted[int(len(durs_sorted)*0.1)]
            p90 = durs_sorted[int(len(durs_sorted)*0.9)]
            print(f"  {lab:12s} n={len(durs):4d} dur_p10={p10:5.2f}s p50={p50:5.2f}s p90={p90:5.2f}s")

        print("  sample_rate_counts:", dict(sr_ctr.most_common(5)))

        # sha1 duplicates across labels (sampled)
        # compute hashes per label for sampled wavs
        for lab, wavs in by_label.items():
            sample = wavs[:]
            if len(sample) > args.max_per_label:
                sample = sample[:args.max_per_label]
            for w in sample:
                try:
                    hashes_by_label[lab].add(sha1_file(w))
                except Exception:
                    pass

        labs = sorted(hashes_by_label.keys())
        dup_pairs = []
        for i in range(len(labs)):
            for j in range(i+1, len(labs)):
                a,b=labs[i],labs[j]
                inter = hashes_by_label[a].intersection(hashes_by_label[b])
                if inter:
                    dup_pairs.append((a,b,len(inter)))
        if dup_pairs:
            dup_pairs.sort(key=lambda x: -x[2])
            print("  [WARN] sha1 overlaps across labels (sampled):")
            for a,b,k in dup_pairs[:10]:
                print(f"    {a} vs {b}: {k}")
        else:
            print("  sha1 overlaps across labels (sampled): 0")

        # store hashes for split overlap check (union across labels)
        all_hash=set()
        for s in hashes_by_label.values():
            all_hash |= s
        split_hashes[name]=all_hash

    if "train" in split_hashes and "test" in split_hashes:
        inter = split_hashes["train"].intersection(split_hashes["test"])
        print(f"\n=== train-test sha1 overlap (sampled) ===")
        print(f"overlap_count={len(inter)} (should be 0 for clean split)")

if __name__ == "__main__":
    main()
