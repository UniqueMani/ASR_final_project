# -*- coding: utf-8 -*-
"""
Train a dialect (accent) classifier from a RAMC-style manifest.jsonl.

Supervised mode:
  - Build one GMM per dialect label (labels taken from manifest or --label_file)
Unsupervised mode:
  - KMeans clustering on speaker embeddings (MFCC stats)

This file is intentionally self-contained and prints progress during feature extraction & training.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Make "backend" importable even when running from anywhere
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.dialect.features import wav_to_mfcc_stats  # noqa: E402


UNKNOWN_SET = {"unknown", "unk", "na", "n/a", "none", ""}


def load_manifest(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_labels_file(labels_path: Path):
    if not labels_path.exists():
        return None
    spk2lab = {}
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            spk, lab = line.split(maxsplit=1)
            spk2lab[spk.strip()] = lab.strip()
    return spk2lab


def labels_from_manifest(items):
    """majority-vote label per speaker from manifest fields"""
    spk2labs = {}
    for it in items:
        spk = it.get("speaker_id")
        lab = it.get("dialect") or it.get("accent") or it.get("label")
        if not spk or lab is None:
            continue
        spk2labs.setdefault(spk, []).append(str(lab))

    out = {}
    for spk, labs in spk2labs.items():
        if not labs:
            continue
        counts = {}
        for l in labs:
            counts[l] = counts.get(l, 0) + 1
        out[spk] = max(counts, key=counts.get)
    return out if out else None


def _fmt_secs(x: float) -> str:
    x = int(max(0, x))
    h = x // 3600
    m = (x % 3600) // 60
    s = x % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def is_unknown_label(lab: str) -> bool:
    return str(lab).strip().lower() in UNKNOWN_SET


def choose_n_components(n_spk: int) -> int:
    """Conservative choice to avoid singular covariance for small classes."""
    if n_spk < 15:
        return 1
    if n_spk < 40:
        return 2
    if n_spk < 80:
        return 4
    return 8


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--out", type=str, required=True, help="Output dir, e.g. ./data/dialect_model")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--max_utts_per_spk", type=int, default=30)
    p.add_argument("--supervised", action="store_true")
    p.add_argument("--label_file", type=str, default="", help="Optional speaker->label file (two columns). If empty, try labels in manifest.")
    p.add_argument(
        "--label_subset",
        type=str,
        default="all",
        choices=["all", "top10"],
        help="Supervised only: restrict dialect labels. top10 keeps the 10 most frequent dialect labels (by speaker count) in the given manifest.",
    )
    p.add_argument(
        "--topn",
        type=int,
        default=0,
        help="Supervised only: keep top-N dialect labels (by speaker count). Overrides --label_subset when > 0.",
    )
    # === NEW: label_exclude ===
    p.add_argument(
        "--label_exclude",
        type=str,
        default="",
        help="Comma-separated list of dialect labels to exclude (e.g. 'shang_hai,bei_jing'). Useful for removing noisy classes.",
    )
    # ==========================
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--log_every", type=int, default=10, help="Print progress every N speakers (default: 10). Use 1 for very verbose.")
    args = p.parse_args()

    t0 = time.time()

    manifest = Path(args.manifest)
    items = load_manifest(manifest)
    if not items:
        raise SystemExit("Empty manifest")

    # group by speaker
    spk2utts = {}
    for it in items:
        spk2utts.setdefault(it["speaker_id"], []).append(it)

    speakers = sorted(spk2utts.keys())
    total_speakers = len(speakers)
    total_utts = sum(len(v) for v in spk2utts.values())

    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Loaded manifest: {manifest}")
    print(f"[INFO] Utterances={total_utts}, Speakers={total_speakers}, max_utts_per_spk={args.max_utts_per_spk}, sr={args.sr}")
    print(f"[INFO] Mode={'supervised (GMM per dialect)' if args.supervised else 'unsupervised (KMeans)'}")

    spk2lab = None
    if args.supervised:
        if args.label_file:
            spk2lab = load_labels_file(Path(args.label_file))
        if spk2lab is None:
            spk2lab = labels_from_manifest(items)

        if spk2lab is None:
            raise SystemExit(
                "Supervised requested but no labels found. Provide one of:\n"
                "1) Put dialect labels into manifest (field: dialect/accent/label), or\n"
                "2) --label_file <speaker_id label>\n"
            )

        labeled = sum(1 for spk in speakers if spk in spk2lab and not is_unknown_label(spk2lab[spk]))
        print(f"[INFO] Supervised labels: speaker->dialect available for {labeled}/{total_speakers} speakers")

    X, y, kept = [], [], []

    kept_spk = 0
    skipped_too_few = 0
    utt_failures = 0
    utt_ok = 0

    for idx, spk in enumerate(speakers, start=1):
        utts = spk2utts[spk][: args.max_utts_per_spk]
        feats = []
        spk_fail = 0

        for u in utts:
            try:
                feats.append(wav_to_mfcc_stats(u["wav_path"], sr=args.sr))
                utt_ok += 1
            except Exception:
                spk_fail += 1
                utt_failures += 1
                continue

        if len(feats) < 3:
            skipped_too_few += 1
        else:
            # === MODIFIED: De-averaging (use all utterances) ===
            for f in feats:
                X.append(f)
                kept.append(spk)
                if spk2lab:
                    y.append(spk2lab.get(spk, "unknown"))
            
            kept_spk += 1

        if args.log_every > 0 and (idx == 1 or idx % args.log_every == 0 or idx == total_speakers):
            elapsed = time.time() - t0
            spk_per_s = idx / max(elapsed, 1e-9)
            eta = (total_speakers - idx) / max(spk_per_s, 1e-9)
            print(
                f"[PROG] {idx:>5}/{total_speakers} spk | kept_spk={kept_spk} | "
                f"utts_in_X={len(X)} | "
                f"last_spk={spk} (utts={len(utts)} ok={len(feats)}) | "
                f"elapsed={_fmt_secs(elapsed)} eta={_fmt_secs(eta)}"
            )

    X = np.asarray(X, dtype=np.float32)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Feature matrix X shape: {X.shape} (from {kept_spk} speakers)")

    if spk2lab:
        import pickle

        # === NEW FEATURE: Handle label_exclude BEFORE processing statistics ===
        if args.label_exclude:
            exclude_list = [s.strip() for s in args.label_exclude.split(",") if s.strip()]
            exclude_set = set(exclude_list)
            if exclude_set:
                print(f"[INFO] Filtering out labels: {exclude_set}")
                
                # Create mask to keep only labels NOT in exclude_set
                # y contains string labels corresponding to rows in X
                mask = np.array([yy not in exclude_set for yy in y], dtype=bool)
                
                n_before = len(y)
                X = X[mask]
                # Filter y and kept using list comprehension with zip
                # (kept needs to stay aligned with X for speaker info)
                y = [yy for yy, m in zip(y, mask) if m]
                kept = [k for k, m in zip(kept, mask) if m]
                
                n_after = len(y)
                print(f"[INFO] Excluded {n_before - n_after} samples. Remaining samples: {n_after}")
                
                # Update kept_spk count for logging (approximate)
                kept_spk = len(set(kept))
        # ====================================================================

        # Filter unknown labels (case-insensitive)
        labels_all = sorted(set(y))
        labels = [lab for lab in labels_all if not is_unknown_label(lab)]

        # Distribution (by unique speaker, to keep topN logic consistent)
        unique_kept_spks = set(kept)
        # Reconstruct speaker->label map for kept speakers only
        dist_spk_level = []
        for s in unique_kept_spks:
            if s in spk2lab and not is_unknown_label(spk2lab[s]):
                dist_spk_level.append(spk2lab[s])
        
        dist = Counter(dist_spk_level)

        if dist:
            print("[INFO] Kept speaker label distribution (top):")
            for k in sorted(dist, key=dist.get, reverse=True)[:20]:
                print(f"  - {k}: {dist[k]} speakers")

        # Optionally restrict to a subset of dialect labels (e.g., top10).
        topn = args.topn if (args.topn and args.topn > 0) else (10 if args.label_subset == "top10" else 0)
        if topn and dist:
            sorted_labels = sorted(dist, key=dist.get, reverse=True)
            keep_labels = sorted_labels[: min(topn, len(sorted_labels))]
            keep_set = set(keep_labels)

            mask = np.array([yy in keep_set for yy in y], dtype=bool)
            X = X[mask]
            # kept and y are parallel to X
            kept = [s for s, m in zip(kept, mask) if bool(m)]
            y = [yy for yy in y if yy in keep_set]
            labels = keep_labels

            # Re-check distribution after filtering
            unique_kept_spks = set(kept)
            dist_spk_level = []
            for s in unique_kept_spks:
                dist_spk_level.append(spk2lab[s])
            dist = Counter(dist_spk_level)

            print(f"[INFO] label_subset enabled: keeping top{len(labels)} labels by speaker count.")
            print(f"[INFO] kept_labels={labels}")
            print(f"[INFO] After label_subset: speakers={len(unique_kept_spks)} X shape={X.shape}")

        models = {}
        print(f"[INFO] Training GMMs for {len(labels)} dialect labels...")

        for lab in labels:
            mask = np.array([yy == lab for yy in y])
            Xi = X[mask]
            
            # Important: Calculate n_spk based on UNIQUE speakers, not total utterances
            # This prevents overfitting 10 speakers with 8 components just because they have 1000 lines.
            relevant_spks = np.array(kept)[mask]
            n_spk = len(set(relevant_spks))
            n_samples = Xi.shape[0]

            if n_spk < 5:
                print(f"[WARN] label={lab} has only {n_spk} speakers (<5), skip.")
                continue

            # Constraint: components depends on speaker diversity, but capped by n_samples safety
            n_comp = min(choose_n_components(n_spk), n_samples)
            print(f"[INFO] Fitting GMM: label={lab} speakers={n_spk} samples={n_samples} n_components={n_comp}")

            gmm = GaussianMixture(
                n_components=n_comp,
                covariance_type="diag",
                random_state=0,
                reg_covar=1e-5,
                init_params="random",
                max_iter=200,
            )

            try:
                gmm.fit(Xi)
            except ValueError as e:
                print(f"[WARN] GMM fit failed for label={lab} (n_comp={n_comp}): {e}")
                print("[WARN] Fallback -> n_components=1, reg_covar=1e-4")
                gmm = GaussianMixture(
                    n_components=1,
                    covariance_type="diag",
                    random_state=0,
                    reg_covar=1e-4,
                    init_params="random",
                    max_iter=200,
                )
                gmm.fit(Xi)

            models[lab] = gmm

        if not models:
            raise SystemExit("Not enough labeled speakers to train (after filtering small/unknown classes).")

        with open(out_dir / "supervised_gmms.pkl", "wb") as f:
            pickle.dump(models, f)

        with open(out_dir / "labels.json", "w", encoding="utf-8") as f:
            json.dump({"labels": list(models.keys())}, f, ensure_ascii=False, indent=2)

        print("[OK] Supervised dialect models:", list(models.keys()))
        print(f"[OK] Saved: {out_dir / 'supervised_gmms.pkl'}")
        print(f"[OK] Saved: {out_dir / 'labels.json'}")
    else:
        # Unsupervised mode on de-averaged data
        k = min(args.k, len(X))
        if k <= 0:
            raise SystemExit("No speakers kept. Check wav paths / manifest.")
        print(f"[INFO] Fitting KMeans: K={k}, samples={len(X)} (speakers={kept_spk})")
        
        km = KMeans(n_clusters=k, random_state=0, n_init="auto")
        clusters = km.fit_predict(X)
        
        # Aggregate cluster assignments by speaker (Majority Vote)
        spk_votes = {}
        for spk, c in zip(kept, clusters):
            spk_votes.setdefault(spk, []).append(int(c))
            
        spk2cluster = {}
        for spk, votes in spk_votes.items():
            # pick most common cluster
            best_c = Counter(votes).most_common(1)[0][0]
            spk2cluster[spk] = best_c

        np.save(out_dir / "cluster_centers.npy", km.cluster_centers_.astype(np.float32))
        (out_dir / "spk2cluster.json").write_text(
            json.dumps(spk2cluster, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print(f"[OK] Unsupervised clusters: K={k} -> {out_dir.resolve()}")
        print(f"[OK] Saved: {out_dir / 'cluster_centers.npy'}")
        print(f"[OK] Saved: {out_dir / 'spk2cluster.json'}")

    total_time = time.time() - t0
    print(f"[DONE] Total time: {_fmt_secs(total_time)}")


if __name__ == "__main__":
    main()