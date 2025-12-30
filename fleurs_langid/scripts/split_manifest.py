# scripts/split_manifest.py
import json, argparse, random
from pathlib import Path
from collections import defaultdict

def read_jsonl(p):
    rows=[]
    with open(p,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: rows.append(json.loads(line))
    return rows

def write_jsonl(rows, p):
    with open(p,"w",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")

def guess_group_key(r):
    # 1) speaker字段（如果你的manifest里有）
    for k in ("speaker", "speaker_id", "spk", "client_id"):
        if k in r: return str(r[k])
    # 2) 否则用文件名前缀做近似分组（至少能避免同一文件系列混到两边）
    wav = str(r.get("wav",""))
    name = Path(wav).stem
    return name.split("_")[0]  # 你也可以改成更合适的规则

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in_manifest", required=True)
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_dev", required=True)
    ap.add_argument("--dev_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args=ap.parse_args()

    rows = read_jsonl(args.in_manifest)
    groups = defaultdict(list)
    for r in rows:
        groups[guess_group_key(r)].append(r)

    keys = list(groups.keys())
    rnd = random.Random(args.seed)
    rnd.shuffle(keys)

    n_dev = int(len(keys)*args.dev_ratio)
    dev_keys = set(keys[:n_dev])

    train_rows, dev_rows = [], []
    for k, rs in groups.items():
        (dev_rows if k in dev_keys else train_rows).extend(rs)

    write_jsonl(train_rows, args.out_train)
    write_jsonl(dev_rows, args.out_dev)
    print(f"train={len(train_rows)} dev={len(dev_rows)} groups={len(keys)} dev_groups={n_dev}")

if __name__=="__main__":
    main()
