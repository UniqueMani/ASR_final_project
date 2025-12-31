# scripts/filter_spkinfo.py
from __future__ import annotations
from pathlib import Path
import argparse, re

def read_tsv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        rows = []
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            cols = line.split("\t")
            # 容错：列数不齐就跳过
            if len(cols) != len(header):
                continue
            rows.append(dict(zip(header, cols)))
    return header, rows

def write_tsv(path: Path, header: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r.get(k, "") for k in header) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="RAMC_small 根目录，例如 ./data/RAMC_small")
    ap.add_argument("--splits", type=str, default="dev", help="逗号分隔：dev,train,test")
    ap.add_argument("--spkinfo", type=str, required=True, help="原始 SPKINFO.txt 路径")
    ap.add_argument("--out", type=str, default="", help="输出 SPKINFO（默认写到 <data_root>/metadata/SPKINFO.subsampled.txt）")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    spkinfo_path = Path(args.spkinfo)
    out_path = Path(args.out) if args.out else (data_root / "metadata" / "SPKINFO.subsampled.txt")

    header, rows = read_tsv(spkinfo_path)

    # 从 RAMC_small/<split>/ 下面的 speaker 文件夹收集 SPKID
    spk_pat = re.compile(r"^\d+_\d+$")
    kept_spk = set()
    for sp in [s.strip() for s in args.splits.split(",") if s.strip()]:
        sp_dir = data_root / sp
        if not sp_dir.exists():
            continue
        for p in sp_dir.iterdir():
            if p.is_dir() and spk_pat.match(p.name):
                kept_spk.add(p.name)

    filtered = [r for r in rows if r.get("SPKID") in kept_spk]

    write_tsv(out_path, header, filtered)
    print(f"[OK] speakers in data: {len(kept_spk)}")
    print(f"[OK] rows kept in SPKINFO: {len(filtered)}")
    print(f"[OK] wrote: {out_path}")

if __name__ == "__main__":
    main()
