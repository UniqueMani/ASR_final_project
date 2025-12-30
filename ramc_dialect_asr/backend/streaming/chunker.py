import numpy as np
import soundfile as sf

def iter_wav_chunks(wav_path: str, chunk_sec: float = 4.0, overlap_sec: float = 1.0):
    y, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y[:, 0]
    total = len(y)
    chunk = int(chunk_sec * sr)
    hop = max(1, int((chunk_sec - overlap_sec) * sr))

    t0 = 0
    while t0 < total:
        t1 = min(total, t0 + chunk)
        pcm = y[t0:t1].astype(np.float32)
        yield {"pcm": pcm, "sr": sr, "t0": t0 / sr, "t1": t1 / sr}
        if t1 >= total:
            break
        t0 += hop
