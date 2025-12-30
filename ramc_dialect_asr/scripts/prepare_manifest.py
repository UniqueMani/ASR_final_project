import argparse, json, re
from pathlib import Path

UTT_RE = re.compile(r'^(?P<utt>[A-Z0-9]+)\s+(?P<txt>.+)$')

def read_transcripts(transcript_path: Path):
    mapping = {}
    with transcript_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = UTT_RE.match(line)
            if not m:
                continue
            mapping[m.group("utt")] = m.group("txt").strip()
    return mapping

def speaker_from_utt(utt_id: str) -> str:
    m = re.search(r"S(\d{4})", utt_id)
    return m.group(1) if m else "????"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True, help="Dataset root, e.g. ./data/AISHELL-1")
    p.add_argument("--out", type=str, required=True, help="Output manifest jsonl, e.g. ./data/manifest.jsonl")
    args = p.parse_args()

    root = Path(args.root)
    transcript_path = root / "data_aishell" / "transcript" / "aishell_transcript_v0.8.txt"
    if not transcript_path.exists():
        raise SystemExit(f"Transcript not found: {transcript_path}")

    wav_root = root / "data_aishell" / "wav"
    if not wav_root.exists():
        raise SystemExit(f"Wav dir not found: {wav_root}")

    transcripts = read_transcripts(transcript_path)
    wavs = list(wav_root.rglob("*.wav"))
    if not wavs:
        raise SystemExit(
            "No .wav found. If you only have *.tar.gz, run:\n"
            "  python scripts/extract_wavs.py --root <root>"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with out.open("w", encoding="utf-8") as fw:
        for wav in wavs:
            utt_id = wav.stem
            if utt_id not in transcripts:
                continue
            spk = speaker_from_utt(utt_id)
            rec = {
                "utt_id": utt_id,
                "speaker_id": spk,
                "wav_path": str(wav.resolve()),
                "text": transcripts[utt_id],
            }
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[OK] Wrote {n_written} items -> {out.resolve()}")

if __name__ == "__main__":
    main()
