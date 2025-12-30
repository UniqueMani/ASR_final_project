import os
import json
import re
from pathlib import Path
from typing import Optional, Dict

import numpy as np


class BaseASREngine:
    """ASR interface.

    This project supports:
    - mock: show ground-truth text progressively (for demo/streaming UX)
    - whisper: real ASR via faster-whisper (baseline)
    """

    name = "base"

    def set_current_context(self, original_filename: str):
        """Optional hook for streaming (to know which transcript to reveal)."""

    def transcribe_file(self, wav_path: str) -> str:
        raise NotImplementedError

    def transcribe_pcm(self, pcm: np.ndarray, sr: int) -> str:
        return ""


class MockTranscriptEngine(BaseASREngine):
    """A lightweight engine for demos.

    It loads (wav_filename -> text) from one of:
    1) env MOCK_TRANS_TSV (a TSV with header: UtteranceID\tSpeakerID\tTranscription)
    2) env MOCK_MANIFEST_JSONL (jsonl with fields including wav_path + text)
    3) ./data/manifest.jsonl (same as 2)

    Then, for an uploaded file, it tries to match by its **original filename**.
    """

    name = "mock"

    def __init__(self):
        self.map: Dict[str, str] = {}
        self._current_full = ""
        self._current_progress = 0.0
        self._current_filename: Optional[str] = None

        base = Path(__file__).resolve().parent.parent.parent
        data_dir = base / "data"

        trans_tsv = os.environ.get("MOCK_TRANS_TSV")
        manifest_jsonl = os.environ.get("MOCK_MANIFEST_JSONL")

        if trans_tsv:
            self._load_trans_tsv(Path(trans_tsv))
        elif manifest_jsonl:
            self._load_manifest(Path(manifest_jsonl))
        else:
            default_manifest = data_dir / "manifest.jsonl"
            if default_manifest.exists():
                self._load_manifest(default_manifest)

    def _load_trans_tsv(self, path: Path):
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            header = f.readline()
            if not header:
                return
            for line in f:
                line = line.strip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                wav_name = parts[0].strip()
                text = parts[2].strip()
                if wav_name and text:
                    self.map[wav_name] = text

    def _load_manifest(self, path: Path):
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                wav_path = rec.get("wav_path")
                text = rec.get("text")
                if not wav_path or not text:
                    continue
                wav_name = Path(wav_path).name
                if wav_name not in self.map:
                    self.map[wav_name] = text

    def set_current_context(self, original_filename: str):
        self._current_filename = original_filename
        self._current_full = self.map.get(original_filename, "")
        self._current_progress = 0.0

    def _guess_original_filename_from_tmp(self, wav_path: str) -> str:
        # Our server stores uploads as: upload_<timestamp>_<original_filename>
        name = Path(wav_path).name
        m = re.match(r"^upload_\d+_(.+)$", name)
        return m.group(1) if m else name

    def transcribe_file(self, wav_path: str) -> str:
        fn = self._current_filename or self._guess_original_filename_from_tmp(wav_path)
        if fn in self.map:
            return self.map[fn]
        return (
            "[mock ASR] Transcript not found for this file. "
            "Tip: run prepare_manifest_*.py to create ./data/manifest.jsonl, "
            "or set MOCK_TRANS_TSV / MOCK_MANIFEST_JSONL."
        )

    def transcribe_pcm(self, pcm: np.ndarray, sr: int) -> str:
        if not self._current_full:
            return ""
        self._current_progress += len(pcm) / sr
        # Roughly reveal ~4 CJK chars per second
        n = min(len(self._current_full), int(self._current_progress * 4))
        return self._current_full[:n]


class FasterWhisperEngine(BaseASREngine):
    name = "whisper"

    def __init__(self, model_size: str = "small", device: str = "auto"):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_size, device=device, compute_type="int8")
        self.lang = "zh"

    def transcribe_file(self, wav_path: str) -> str:
        segments, _info = self.model.transcribe(
            wav_path,
            language=self.lang,
            beam_size=5,
            vad_filter=True,
        )
        return "".join(seg.text for seg in segments).strip()

    def transcribe_pcm(self, pcm: np.ndarray, sr: int) -> str:
        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, pcm, sr)
            return self.transcribe_file(tmp.name)


def make_asr_engine() -> BaseASREngine:
    engine = os.environ.get("ASR_ENGINE", "mock").lower().strip()
    if engine == "whisper":
        model_size = os.environ.get("WHISPER_MODEL", "small")
        device = os.environ.get("WHISPER_DEVICE", "auto")
        return FasterWhisperEngine(model_size=model_size, device=device)
    return MockTranscriptEngine()
