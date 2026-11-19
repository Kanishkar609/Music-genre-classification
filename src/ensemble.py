"""Pick ensemble members from *validation* scores (never from the test set).

The three weakest CNNs (separable / attention / baseline) drag a uniform average
below the best single model, so the ensemble keeps the top-K models and weights
them by their validation accuracy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CNN_NAMES, MODELS_DIR, REPORTS_DIR

CLASSIC_NAMES = ["random_forest", "svm", "hist_gb"]
DEFAULT_TOP_K = 4


def _read(path: Path, key: str) -> float | None:
    if not path.exists():
        return None
    try:
        return float(json.loads(path.read_text()).get(key, 0.0))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def val_scores() -> dict[str, float]:
    """Validation accuracy (CNN) / CV accuracy (classic) for every trained model."""
    scores: dict[str, float] = {}
    for name in CNN_NAMES:
        if (MODELS_DIR / f"cnn_{name}.pt").exists():
            value = _read(REPORTS_DIR / f"cnn_{name}.json", "best_val_acc")
            if value is not None:
                scores[f"cnn_{name}"] = value
    for name in CLASSIC_NAMES:
        if (MODELS_DIR / f"classic_{name}.joblib").exists():
            value = _read(REPORTS_DIR / f"classic_{name}.json", "cv_mean")
            if value is not None:
                scores[f"classic_{name}"] = value
    return scores


def ensemble_weights(top_k: int = DEFAULT_TOP_K) -> dict[str, float]:
    """Normalised weights over the best `top_k` models. Empty if no reports exist."""
    scores = val_scores()
    if not scores:
        return {}
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
    total = sum(max(0.0, s) for _, s in ranked) or 1.0
    return {name: max(0.0, s) / total for name, s in ranked}


if __name__ == "__main__":
    weights = ensemble_weights()
    if not weights:
        print("No trained models with reports found.")
    for name, weight in sorted(weights.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<24} {weight:.3f}")
