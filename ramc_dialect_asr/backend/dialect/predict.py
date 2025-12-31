import numpy as np
from pathlib import Path

from backend.dialect.features import pcm_to_mfcc_stats

class DialectPredictor:
    def __init__(self):
        self.data_dir = Path(__file__).resolve().parent.parent.parent / "data"
        self.model_dir = self.data_dir / "dialect_model"
        self.mode = "none"
        self.centers = None
        self.gmms = None

        if (self.model_dir / "supervised_gmms.pkl").exists():
            import pickle
            with open(self.model_dir / "supervised_gmms.pkl", "rb") as f:
                self.gmms = pickle.load(f)
            self.mode = "supervised"
        elif (self.model_dir / "cluster_centers.npy").exists():
            self.centers = np.load(self.model_dir / "cluster_centers.npy")
            self.mode = "cluster"

    def _predict_vec(self, vec: np.ndarray) -> dict:
        if self.mode == "supervised" and self.gmms:
            scores = {lab: float(gmm.score(vec.reshape(1, -1))) for lab, gmm in self.gmms.items()}
            best_lab = max(scores, key=scores.get)
            sorted_scores = sorted(scores.values(), reverse=True)
            margin = (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
            conf = float(1.0 / (1.0 + np.exp(-margin)))
            return {"label": best_lab, "confidence": conf}
        if self.mode == "cluster" and self.centers is not None:
            d = np.sum((self.centers - vec.reshape(1, -1)) ** 2, axis=1)
            k = int(np.argmin(d))
            d_sorted = np.sort(d)
            ratio = float(d_sorted[0] / (d_sorted[1] + 1e-6)) if len(d_sorted) > 1 else 0.0
            conf = float(1.0 - min(1.0, ratio))
            return {"label": f"cluster_{k}", "confidence": conf}
        return {"label": "unknown", "confidence": 0.0}

    def predict_chunk(self, pcm: np.ndarray, sr: int) -> dict:
        vec = pcm_to_mfcc_stats(pcm, sr=sr)
        return self._predict_vec(vec)

    def predict_file(self, wav_path: str, max_sec: float = 6.0) -> dict:
        import soundfile as sf
        y, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y[:, 0]
        n = int(sr * max_sec)
        y = y[:n]
        vec = pcm_to_mfcc_stats(y, sr=sr)
        return self._predict_vec(vec)
