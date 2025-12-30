#!/usr/bin/env python
"""
Evaluate supervised dialect (accent) classifier on a manifest.jsonl.

Key point:
- train_dialect.py trains GMMs on SPEAKER embeddings (average of multiple utterances per speaker).
- So the recommended evaluation is:
    --level spk --spk_mode embed

This script also:
- Filters OOV labels (labels present in manifest but not in the trained model) so accuracy is meaningful
- Can output per-label accuracy and a confusion matrix for analysis

New additions:
- Progress logging (so you know \"where it is\")
- Optional MFCC-stat caching to disk to speed up repeated runs
  (works well when you increase max_utts_per_spk to 50/100)

Examples:
  python scripts/eval_dialect.py --manifest ./data/RAMC_small/test/metadata/manifest.jsonl --level spk --spk_mode embed --max_utts_per_spk 20
  python scripts/eval_dialect.py --manifest ./data/RAMC_small/test/metadata/manifest.jsonl --level spk --spk_mode embed --report per_label
  python scripts/eval_dialect.py --manifest ./data/RAMC_small/test/metadata/manifest.jsonl --level spk --spk_mode embed --report confusion
"""
import argparse
import hashlib
import json
import os
import sys
import time
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Ensure project root on sys.path so "backend" import works even without PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.dialect.features import wav_to_mfcc_stats  # noqa: E402


UNKNOWN_SET = {"unknown", "unk", "na", "n/a", "none", ""}


def is_unknown(x: str) -> bool:
    return str(x).strip().lower() in UNKNOWN_SET


def _speaker_key(it: dict) -> str:
    # Be tolerant: manifests in this project have used speaker_id, but some earlier versions used speaker.
    return (
        it.get("speaker")
        or it.get("speaker_id")
        or it.get("spk")
        or it.get("spk_id")
        or it.get("speakerId")
        or it.get("speakerID")
        or ""
    )



def parse_label_list(s: str):
    """Parse comma/space-separated label list into a set[str]."""
    if s is None:
        return set()
    s = str(s).strip()
    if not s:
        return set()
    parts = re.split(r"[\s,]+", s)
    return {p.strip() for p in parts if p and p.strip()}

def load_manifest(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            items.append(json.loads(ln))
    return items


def load_gmms(model_dir: Path):
    import pickle

    pkl = model_dir / "supervised_gmms.pkl"
    if not pkl.exists():
        raise SystemExit(f"supervised_gmms.pkl not found in: {model_dir}")
    with open(pkl, "rb") as f:
        gmms = pickle.load(f)
    return gmms


def score_all_labels(gmms: dict, vec: np.ndarray):
    """Return list[(label, score)] sorted by score descending."""
    scores = []
    x = vec.reshape(1, -1)
    for lab, gmm in gmms.items():
        # sklearn GaussianMixture.score returns average log-likelihood for samples
        s = float(gmm.score(x))
        scores.append((lab, s))
    scores.sort(key=lambda t: t[1], reverse=True)
    return scores


def report_per_label(conf: dict, supports: dict):
    # conf: dict[gt][pred] -> count
    labs = sorted(supports.keys())
    lines = []
    lines.append("per_label_accuracy:")
    for gt in labs:
        sup = supports[gt]
        if sup <= 0:
            continue
        row = conf.get(gt, {})
        ok = row.get(gt, 0)
        acc = ok / sup
        # most confused-to label
        conf_to = sorted([(p, c) for p, c in row.items() if p != gt], key=lambda x: -x[1])[:1]
        if conf_to:
            p, c = conf_to[0]
            lines.append(f"  - {gt:12s} acc={acc:6.2%}  support={sup:5d}  top_confuse={p} ({c})")
        else:
            lines.append(f"  - {gt:12s} acc={acc:6.2%}  support={sup:5d}")
    return "\n".join(lines)


def report_confusion(conf: dict, labels: list[str], max_cols: int = 60):
    # Create a TSV confusion matrix with header
    # rows: gt, cols: pred
    header = ["gt\\pred"] + labels
    out_lines = ["\t".join(header)]
    for gt in labels:
        row = [gt]
        gt_row = conf.get(gt, {})
        for pr in labels:
            row.append(str(int(gt_row.get(pr, 0))))
        out_lines.append("\t".join(row))

    txt = "\n".join(out_lines)
    # Avoid printing extremely wide lines in some terminals by warning only
    if len(header) > max_cols:
        txt = "[WARN] Too many labels to print confusion matrix nicely.\n" + txt
    return txt


def _fmt_eta(elapsed_s: float, done: int, total: int) -> str:
    if done <= 0 or total <= 0:
        return "?:??"
    rate = done / max(1e-9, elapsed_s)
    remain = max(0, total - done)
    eta = remain / max(1e-9, rate)
    return time.strftime("%H:%M:%S", time.gmtime(eta))


def _cache_file_for(wav_path: str, sr: int, cache_dir: Path) -> Path:
    """
    Cache key includes:
    - absolute path
    - sr
    - file size + mtime_ns (so changed wavs invalidate cache automatically)
    """
    p = Path(wav_path)
    ap = str(p.resolve())
    try:
        st = p.stat()
        size = st.st_size
        mtime = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
    except Exception:
        size, mtime = 0, 0

    h = hashlib.sha1(f"{ap}|sr={sr}".encode("utf-8")).hexdigest()
    sub = h[:2]
    cache_sub = cache_dir / sub
    cache_sub.mkdir(parents=True, exist_ok=True)
    name = f"{h}_sr{sr}_s{size}_m{mtime}.npy"
    return cache_sub / name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, required=True, help="manifest.jsonl")
    p.add_argument("--level", choices=["utt", "spk"], default="utt", help="evaluation level")
    p.add_argument(
        "--spk_mode",
        choices=["vote", "embed"],
        default="embed",
        help="speaker-level strategy: vote=vote over utterance predictions; embed=average embeddings then predict once (recommended)",
    )
    p.add_argument("--max_utts_per_spk", type=int, default=999999, help="cap utts per speaker (speaker-level)")
    p.add_argument("--model_dir", type=str, default="./data/dialect_model", help="directory containing supervised_gmms.pkl")
    p.add_argument("--sr", type=int, default=16000, help="target sample rate for feature extraction (should match training)")
    p.add_argument("--topk", type=int, default=1, help="top-k accuracy (k>=1). Only affects printed accuracy.")
    p.add_argument("--report", choices=["summary", "per_label", "confusion"], default="summary", help="extra reports")
    p.add_argument(
        "--label_include",
        type=str,
        default="",
        help="Optional: comma/space-separated dialect labels to KEEP (after OOV filtering). Empty => keep all.",
    )
    p.add_argument(
        "--label_exclude",
        type=str,
        default="",
        help="Optional: comma/space-separated dialect labels to DROP (after OOV filtering). Useful to exclude noisy labels.",
    )

    p.add_argument(
        "--label_subset",
        type=str,
        default="all",
        choices=["all", "top10"],
        help="Evaluate only a subset of labels. top10 picks the 10 most frequent labels (by speaker count) within the in-model portion of this manifest.",
    )
    p.add_argument(
        "--topn",
        type=int,
        default=0,
        help="Evaluate only top-N labels by speaker count within the in-model portion of this manifest. Overrides --label_subset when > 0.",
    )
    # Progress + cache
    p.add_argument("--log_every", type=int, default=10, help="print progress every N speakers (spk) or N utterances (utt)")
    p.add_argument(
        "--cache_dir",
        type=str,
        default="./data/mfcc_cache",
        help="directory for MFCC-stat cache (.npy). Set to '' to disable.",
    )
    p.add_argument(
        "--no_cache",
        action="store_true",
        help="disable MFCC-stat caching even if --cache_dir is set",
    )
    args = p.parse_args()

    man = Path(args.manifest)
    items = load_manifest(man)

    # keep only labeled items
    items = [it for it in items if ("dialect" in it and not is_unknown(it["dialect"]))]

    if not items:
        raise SystemExit("No labeled items in manifest (dialect missing or 'unknown').")

    model_dir = Path(args.model_dir)
    gmms = load_gmms(model_dir)
    model_labels = sorted(list(gmms.keys()))
    model_label_set = set(model_labels)

    # OOV labels in manifest
    m_counts = Counter(it["dialect"] for it in items)
    oov = [(lab, cnt) for lab, cnt in m_counts.items() if lab not in model_label_set]
    if oov:
        oov_sorted = sorted(oov, key=lambda x: -x[1])
        oov_utts = sum(v for _, v in oov_sorted)
        print(f"[INFO] OOV labels in this manifest (not in model): {oov_sorted}  (oov_utts={oov_utts}/{len(items)})")

    # filter OOV so accuracy is meaningful for trained labels
    items_in = [it for it in items if it["dialect"] in model_label_set]
    if len(items_in) != len(items):
        print(f"[INFO] Evaluating on in-model items only: {len(items_in)}/{len(items)}")
    items = items_in

    # Optional: include/exclude specific labels (after OOV filtering, before label_subset/topN).
    include_set = parse_label_list(args.label_include)
    exclude_set = parse_label_list(args.label_exclude)
    if include_set:
        before = len(items)
        items = [it for it in items if it.get("dialect") in include_set]
        print(f"[INFO] label_include: kept {len(items)}/{before} items (labels={sorted(include_set)})")
    if exclude_set:
        before = len(items)
        items = [it for it in items if it.get("dialect") not in exclude_set]
        print(f"[INFO] label_exclude: kept {len(items)}/{before} items (excluded={sorted(exclude_set)})")


    # Optional: evaluate on a subset of labels (e.g., top10) within in-model items.
    topn = args.topn if (args.topn and args.topn > 0) else (10 if args.label_subset == "top10" else 0)
    if topn and items:
        spk2lab_eval = {}
        for it in items:
            spk = _speaker_key(it)
            if spk:
                spk2lab_eval[spk] = it["dialect"]
        counts = Counter(spk2lab_eval.values())
        keep_labels = [lab for lab, _ in counts.most_common(min(topn, len(counts)))]
        keep_set = set(keep_labels)
        items = [it for it in items if it["dialect"] in keep_set]
        # recompute speakers after filtering
        spk2lab_eval2 = {}
        for it in items:
            spk = _speaker_key(it)
            if spk:
                spk2lab_eval2[spk] = it["dialect"]
        print(f"[INFO] label_subset enabled: evaluating top{len(keep_labels)} labels by speaker count.")
        print(f"[INFO] eval_labels={keep_labels}")
        print(f"[INFO] After label_subset: items={len(items)} speakers={len(spk2lab_eval2)}")

    if not items:
        raise SystemExit("After filtering OOV labels / label_subset, no items left to evaluate.")

    k = max(1, int(args.topk))
    log_every = max(1, int(args.log_every))

    # Cache setup
    use_cache = (not args.no_cache) and bool(str(args.cache_dir).strip())
    cache_dir = Path(args.cache_dir) if use_cache else None
    cache_hits = 0
    cache_miss = 0

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] MFCC cache enabled: {cache_dir}")
    else:
        print("[INFO] MFCC cache disabled.")

    def get_vec(it: dict) -> np.ndarray:
        nonlocal cache_hits, cache_miss
        wav = it["wav_path"]
        if not use_cache:
            return wav_to_mfcc_stats(wav, sr=args.sr)

        cp = _cache_file_for(wav, args.sr, cache_dir)
        if cp.exists():
            try:
                v = np.load(cp)
                cache_hits += 1
                return v
            except Exception:
                # broken cache -> recompute
                pass

        v = wav_to_mfcc_stats(wav, sr=args.sr)
        cache_miss += 1
        # Save as float32 to reduce disk size
        try:
            tmp = cp.with_suffix(".tmp.npy")
            np.save(tmp, v.astype(np.float32))
            os.replace(tmp, cp)
        except Exception:
            # cache is best-effort
            pass
        return v

    # Confusion bookkeeping
    conf = defaultdict(lambda: defaultdict(int))
    supports = Counter()

    t0 = time.perf_counter()

    # Utterance-level
    if args.level == "utt":
        n = 0
        ok1 = 0
        okk = 0
        total = len(items)

        for i, it in enumerate(items, 1):
            vec = get_vec(it)
            scores = score_all_labels(gmms, vec)
            pred1 = scores[0][0]
            topk_labels = [lab for lab, _ in scores[:k]]
            gt = it["dialect"]

            ok1 += int(pred1 == gt)
            okk += int(gt in topk_labels)
            n += 1

            conf[gt][pred1] += 1
            supports[gt] += 1

            if (i % log_every) == 0 or i == total:
                elapsed = time.perf_counter() - t0
                eta = _fmt_eta(elapsed, i, total)
                print(f"[PROG] {i:6d}/{total} utt | acc={ok1/max(1,n):.4f} | elapsed={elapsed:0.1f}s eta={eta} | cache hit/miss={cache_hits}/{cache_miss}")

        print(f"utt_accuracy={ok1/n:.4f}  (ok={ok1}, n={n})")
        if k > 1:
            print(f"utt_top{k}_accuracy={okk/n:.4f}  (ok={okk}, n={n})")

        if args.report == "per_label":
            print(report_per_label(conf, supports))
        elif args.report == "confusion":
            labels = [lab for lab, _ in supports.most_common()]
            print(report_confusion(conf, labels))

        if use_cache:
            print(f"[INFO] Cache stats: hit={cache_hits} miss={cache_miss} hit_rate={cache_hits/max(1,(cache_hits+cache_miss)):.2%}")
        return

    # Speaker-level
    spk2items = defaultdict(list)
    for it in items:
        spk = _speaker_key(it)
        if not spk:
            raise SystemExit("Manifest item missing speaker key (speaker/speaker_id). Please regenerate manifest.")
        spk2items[spk].append(it)

    speakers = list(spk2items.items())
    total_spk = len(speakers)

    n = 0
    ok1 = 0
    okk = 0

    for si, (spk, its) in enumerate(speakers, 1):
        its = its[: args.max_utts_per_spk]
        gt = its[0]["dialect"]

        if args.spk_mode == "vote":
            votes = []
            for it in its:
                vec = get_vec(it)
                scores = score_all_labels(gmms, vec)
                votes.append(scores[0][0])
            pred1 = Counter(votes).most_common(1)[0][0]
            # Approx top-k under vote mode by most-voted labels
            topk_labels = [lab for lab, _ in Counter(votes).most_common(k)]
        else:
            feats = [get_vec(it) for it in its]
            spk_vec = np.mean(np.stack(feats, axis=0), axis=0)
            scores = score_all_labels(gmms, spk_vec)
            pred1 = scores[0][0]
            topk_labels = [lab for lab, _ in scores[:k]]

        ok1 += int(pred1 == gt)
        okk += int(gt in topk_labels)
        n += 1

        conf[gt][pred1] += 1
        supports[gt] += 1

        if (si % log_every) == 0 or si == total_spk:
            elapsed = time.perf_counter() - t0
            eta = _fmt_eta(elapsed, si, total_spk)
            print(f"[PROG] {si:6d}/{total_spk} spk | acc={ok1/max(1,n):.4f} | last_spk={spk} (utts={len(its)}) | elapsed={elapsed:0.1f}s eta={eta} | cache hit/miss={cache_hits}/{cache_miss}")

    mode = args.spk_mode
    print(f"spk_accuracy={ok1/n:.4f}  (ok={ok1}, n={n}, speakers={len(spk2items)})  mode={mode}")
    if k > 1:
        print(f"spk_top{k}_accuracy={okk/n:.4f}  (ok={okk}, n={n}, speakers={len(spk2items)})  mode={mode}")

    if args.report == "per_label":
        print(report_per_label(conf, supports))
    elif args.report == "confusion":
        labels = [lab for lab, _ in supports.most_common()]
        print(report_confusion(conf, labels))

    if use_cache:
        print(f"[INFO] Cache stats: hit={cache_hits} miss={cache_miss} hit_rate={cache_hits/max(1,(cache_hits+cache_miss)):.2%}")


if __name__ == "__main__":
    main()
