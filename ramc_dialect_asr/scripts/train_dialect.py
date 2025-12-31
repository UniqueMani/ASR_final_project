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
import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Make "backend" importable even when running from anywhere
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.dialect.features import wav_to_mfcc_stats  # noqa: E402


UNKNOWN_SET = {"unknown", "unk", "na", "n/a", "none", ""}

def parse_label_list(s: str):
    """Parse comma/space separated label list."""
    if not s:
        return []
    # allow comma or whitespace
    parts = []
    for chunk in str(s).replace(',', ' ').split():
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    return parts


def _cache_key(wav_path: str, sr: int) -> str:
    """Stable cache key for a wav file + sample rate."""
    p = Path(wav_path)
    try:
        abs_p = str(p.resolve())
    except Exception:
        abs_p = str(p.absolute())
    try:
        st = os.stat(abs_p)
        meta = f"{abs_p}|{st.st_size}|{int(st.st_mtime)}|{sr}"
    except Exception:
        meta = f"{abs_p}|{sr}"
    return hashlib.sha1(meta.encode('utf-8', errors='ignore')).hexdigest()


def wav_to_mfcc_stats_cached(wav_path: str, sr: int, cache_dir):
    """Compute MFCC stats, optionally caching to disk as .npy."""
    if cache_dir is None:
        return wav_to_mfcc_stats(wav_path, sr=sr), False
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(wav_path, sr)
    fp = cache_dir / f"{key}.npy"
    if fp.exists():
        try:
            return np.load(fp).astype(np.float32), True
        except Exception:
            # corrupted cache: ignore and recompute
            pass
    feat = wav_to_mfcc_stats(wav_path, sr=sr).astype(np.float32)
    # atomic write
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npy', dir=str(cache_dir)) as tmp:
            tmp_path = Path(tmp.name)
        np.save(tmp_path, feat)
        tmp_path.replace(fp)
    except Exception:
        # best-effort caching
        pass
    return feat, False


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
    p.add_argument(
        "--label_include",
        type=str,
        default="",
        help="Supervised only: comma/space-separated dialect labels to KEEP (whitelist). Applied after --label_subset/--topn.",
    )
    p.add_argument(
        "--label_exclude",
        type=str,
        default="",
        help="Supervised only: comma/space-separated dialect labels to DROP (blacklist). Applied after --label_subset/--topn.",
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default="./data/mfcc_cache_train",
        help="Cache utterance-level MFCC stats to disk (.npy) to speed up repeated runs. Use --no_cache to disable.",
    )
    p.add_argument("--no_cache", action="store_true", help="Disable MFCC caching even if --cache_dir is set.")
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

    # MFCC cache (utterance-level) to speed up repeated training/eval.
    cache_dir = None
    if (not args.no_cache) and args.cache_dir:
        cache_dir = Path(args.cache_dir)
        print(f"[INFO] MFCC cache: enabled dir={cache_dir}")
    else:
        print("[INFO] MFCC cache: disabled")

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
    cache_hit = 0
    cache_miss = 0

    for idx, spk in enumerate(speakers, start=1):
        utts = spk2utts[spk][: args.max_utts_per_spk]
        feats = []
        spk_fail = 0

        for u in utts:
            try:
                feat, hit = wav_to_mfcc_stats_cached(u["wav_path"], sr=args.sr, cache_dir=cache_dir)
                feats.append(feat)
                if hit:
                    cache_hit += 1
                else:
                    cache_miss += 1
                utt_ok += 1
            except Exception:
                spk_fail += 1
                utt_failures += 1
                continue

        if len(feats) < 3:
            skipped_too_few += 1
        else:
            spk_feat = np.mean(np.stack(feats, axis=0), axis=0)
            X.append(spk_feat)
            kept.append(spk)
            kept_spk += 1
            if spk2lab:
                y.append(spk2lab.get(spk, "unknown"))

        if args.log_every > 0 and (idx == 1 or idx % args.log_every == 0 or idx == total_speakers):
            elapsed = time.time() - t0
            spk_per_s = idx / max(elapsed, 1e-9)
            eta = (total_speakers - idx) / max(spk_per_s, 1e-9)
            print(
                f"[PROG] {idx:>5}/{total_speakers} spk | kept={kept_spk} skip={skipped_too_few} | "
                f"utts_ok={utt_ok} utts_fail={utt_failures} | cache_hit={cache_hit} miss={cache_miss} | "
                f"last_spk={spk} (utts={len(utts)} ok={len(feats)} fail={spk_fail}) | "
                f"elapsed={_fmt_secs(elapsed)} eta={_fmt_secs(eta)}"
            )

    X = np.asarray(X, dtype=np.float32)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Feature matrix X shape: {X.shape} (kept={len(kept)}/{total_speakers})")
    if cache_dir is not None:
        tot = cache_hit + cache_miss
        hr = (cache_hit / tot) if tot > 0 else 0.0
        print(f"[INFO] MFCC cache stats: hit={cache_hit} miss={cache_miss} hit_rate={hr:.2%}")

    if spk2lab:
        import pickle

        # Filter unknown labels (case-insensitive)
        labels_all = sorted(set(y))
        labels = [lab for lab in labels_all if not is_unknown_label(lab)]

        # Distribution (excluding unknown)
        dist = {}
        for yy in y:
            if is_unknown_label(yy):
                continue
            dist[yy] = dist.get(yy, 0) + 1

        if dist:
            print("[INFO] Kept speaker label distribution (top):")
            for k in sorted(dist, key=dist.get, reverse=True)[:20]:
                print(f"  - {k}: {dist[k]}")

        # Optionally restrict to a subset of dialect labels (e.g., top10), then apply include/exclude filters.
        topn = args.topn if (args.topn and args.topn > 0) else (10 if args.label_subset == "top10" else 0)

        keep_labels = list(labels)
        if topn and dist:
            sorted_labels = sorted(dist, key=dist.get, reverse=True)
            keep_labels = sorted_labels[: min(topn, len(sorted_labels))]

        # Additional manual label filtering (do NOT refill after excluding; useful for top10->top8 etc.)
        include_set = set(parse_label_list(args.label_include))
        exclude_set = set(parse_label_list(args.label_exclude))
        if include_set:
            keep_labels = [l for l in keep_labels if l in include_set]
        if exclude_set:
            keep_labels = [l for l in keep_labels if l not in exclude_set]

        if keep_labels != labels:
            keep_set = set(keep_labels)
            mask = np.array([yy in keep_set for yy in y], dtype=bool)
            X = X[mask]
            kept = [s for s, m in zip(kept, mask) if bool(m)]
            y = [yy for yy in y if yy in keep_set]
            labels = keep_labels

            if not labels:
                raise SystemExit("No labels left after --label_subset/--topn and --label_include/--label_exclude.")

            dist2 = {}
            for yy in y:
                dist2[yy] = dist2.get(yy, 0) + 1

            if topn:
                print(f"[INFO] label_subset enabled: keeping top{topn} labels by speaker count.")
            if include_set:
                print(f"[INFO] label_include enabled: {sorted(include_set)}")
            if exclude_set:
                print(f"[INFO] label_exclude enabled: {sorted(exclude_set)}")
            print(f"[INFO] kept_labels={labels}")
            print(f"[INFO] After label filtering: speakers={len(y)} X shape={X.shape}")

        models = {}
        print(f"[INFO] Training GMMs for {len(labels)} dialect labels...")

        for lab in labels:
            mask = np.array([yy == lab for yy in y])
            Xi = X[mask]
            n_spk = int(Xi.shape[0])  # ALWAYS define before using

            if n_spk < 5:
                print(f"[WARN] label={lab} has only {n_spk} speakers (<5), skip.")
                continue

            n_comp = min(choose_n_components(n_spk), n_spk)
            print(f"[INFO] Fitting GMM: label={lab} speakers={n_spk} n_components={n_comp}")

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
        k = min(args.k, len(X))
        if k <= 0:
            raise SystemExit("No speakers kept. Check wav paths / manifest.")
        print(f"[INFO] Fitting KMeans: K={k}, speakers={len(X)}")
        km = KMeans(n_clusters=k, random_state=0, n_init="auto")
        clusters = km.fit_predict(X)
        spk2cluster = {spk: int(c) for spk, c in zip(kept, clusters)}

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
