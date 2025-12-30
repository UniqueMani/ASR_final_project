import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True, help="Output directory, e.g. ./data/AISHELL-1")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("Please install: pip install -U huggingface_hub")

    snapshot_download(
        repo_id="AISHELL/AISHELL-1",
        repo_type="dataset",
        local_dir=str(out),
        allow_patterns=["data_aishell/**", "resource_aishell/**"],
    )
    print(f"[OK] Downloaded into: {out.resolve()}")

if __name__ == "__main__":
    main()
