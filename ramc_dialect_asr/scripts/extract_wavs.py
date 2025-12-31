import argparse
import tarfile
from pathlib import Path

def extract_one(tar_path: Path, out_dir: Path):
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(out_dir)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True, help="Dataset root, e.g. ./data/AISHELL-1")
    p.add_argument("--wav_dir", type=str, default="data_aishell/wav", help="Relative wav archive dir")
    p.add_argument("--out", type=str, default=None, help="Extract output directory (default: same as wav_dir)")
    args = p.parse_args()

    root = Path(args.root)
    wav_dir = root / args.wav_dir
    if not wav_dir.exists():
        raise SystemExit(f"Not found: {wav_dir}")

    out_dir = Path(args.out) if args.out else wav_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tars = sorted(wav_dir.glob("*.tar.gz"))
    if not tars:
        raise SystemExit(f"No .tar.gz found in {wav_dir}. Did you download 'data_aishell/wav/**'?")

    for i, tar_path in enumerate(tars, 1):
        print(f"[{i}/{len(tars)}] Extracting {tar_path.name} ...")
        extract_one(tar_path, out_dir)

    print(f"[OK] Extracted into: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
