from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import pickle
import numpy as np


@dataclass
class GaussDiagModel:
    labels: List[str]
    means: np.ndarray  # (K, D)
    vars: np.ndarray   # (K, D) diagonal variances
    priors: np.ndarray # (K,)
    reg: float = 1e-4

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return unnormalized log-probabilities (log-likelihood + log-prior) for each class.
        X: (N, D)
        """
        X = np.asarray(X, dtype=np.float32)
        means = self.means.astype(np.float32)
        vars_ = np.maximum(self.vars.astype(np.float32), self.reg)

        # log N(x | mu, var_diag)
        # ll = -0.5 * sum( log(2pi*var) + (x-mu)^2/var )
        # Compute for all classes using broadcasting:
        diff = X[:, None, :] - means[None, :, :]
        ll = -0.5 * (np.sum(np.log(2.0 * np.pi * vars_)[None, :, :], axis=-1) + np.sum((diff * diff) / vars_[None, :, :], axis=-1))
        ll += np.log(np.maximum(self.priors, 1e-12))[None, :]
        return ll

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (pred_idx, pred_label)."""
        ll = self.predict_proba(X)
        idx = np.argmax(ll, axis=1)
        labels = np.array([self.labels[i] for i in idx], dtype=object)
        return idx, labels

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "gauss_diag.pkl", "wb") as f:
            pickle.dump(self, f)
        (out_dir / "labels.json").write_text(json.dumps(self.labels, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load(model_dir: str | Path) -> "GaussDiagModel":
        model_dir = Path(model_dir)
        with open(model_dir / "gauss_diag.pkl", "rb") as f:
            m = pickle.load(f)
        return m


def train_gauss_diag(X: np.ndarray, y: List[str], reg: float = 1e-4, prior: str = "uniform", var_floor: float | None = None) -> GaussDiagModel:
    if var_floor is not None:
        reg = float(var_floor)
    X = np.asarray(X, dtype=np.float32)
    labels = sorted(list(set(y)))
    lab2i = {l:i for i,l in enumerate(labels)}
    yi = np.array([lab2i[t] for t in y], dtype=np.int64)
    K = len(labels)
    D = X.shape[1]

    means = np.zeros((K, D), dtype=np.float32)
    vars_ = np.zeros((K, D), dtype=np.float32)
    counts = np.zeros((K,), dtype=np.int64)

    for k in range(K):
        Xk = X[yi == k]
        counts[k] = Xk.shape[0]
        means[k] = Xk.mean(axis=0)
        # diag var
        vars_[k] = Xk.var(axis=0) + reg

    if prior == "empirical":
        priors = counts / np.maximum(counts.sum(), 1)
    else:
        priors = np.ones((K,), dtype=np.float32) / float(K)

    return GaussDiagModel(labels=labels, means=means, vars=vars_, priors=priors.astype(np.float32), reg=reg)