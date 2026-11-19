"""Inference: audio file -> genre probabilities.

CLI:
    python src/predict.py song.mp3
    python src/predict.py song.mp3 --model cnn_resnet
    python src/predict.py song.mp3 --model ensemble --full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnn_models import build_model
from config import CNN_NAMES, GENRES, MODELS_DIR, SEGMENTS_PER_TRACK, TRACK_SECONDS
from ensemble import ensemble_weights
from features import extract_classic, load_audio, segments_from_audio

CLASSIC_NAMES = ["random_forest", "svm", "hist_gb"]


# --------------------------------------------------------------------------- #
# model loading
# --------------------------------------------------------------------------- #
def available_models() -> list[str]:
    out = [f"cnn_{n}" for n in CNN_NAMES if (MODELS_DIR / f"cnn_{n}.pt").exists()]
    out += [f"classic_{n}" for n in CLASSIC_NAMES if (MODELS_DIR / f"classic_{n}.joblib").exists()]
    return out


def load_classic(name: str):
    return joblib.load(MODELS_DIR / f"classic_{name}.joblib")


def load_cnn(name: str):
    ckpt = torch.load(MODELS_DIR / f"cnn_{name}.pt", map_location="cpu", weights_only=False)
    model = build_model(ckpt.get("arch", name), ckpt.get("num_classes", len(GENRES)))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, float(ckpt["mean"]), float(ckpt["std"])


# --------------------------------------------------------------------------- #
# prediction
# --------------------------------------------------------------------------- #
def predict_classic(audio: tuple[np.ndarray, int], artifact: dict) -> np.ndarray:
    y, sr = audio
    feats = extract_classic(y, sr)
    row = np.array([[feats[c] for c in artifact["features"]]], dtype=np.float32)
    return artifact["pipeline"].predict_proba(row)[0]


def predict_cnn(segments: np.ndarray, model, mean: float, std: float) -> np.ndarray:
    x = segments.astype(np.float32) / 255.0
    x = (x - mean) / std
    tensor = torch.from_numpy(x).unsqueeze(1)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).numpy()
    return probs.mean(axis=0)          # average over the track's segments


def _predict_named(name: str, audio, segments, cache: dict) -> np.ndarray:
    family, _, short = name.partition("_")
    if family == "cnn":
        key = f"cnn:{short}"
        if key not in cache:
            cache[key] = load_cnn(short)
        model, mean, std = cache[key]
        return predict_cnn(segments, model, mean, std)
    if family == "classic":
        key = f"classic:{short}"
        if key not in cache:
            cache[key] = load_classic(short)
        return predict_classic(audio, cache[key])
    raise ValueError(f"Unknown model '{name}'")


def predict_file(path: str | Path, model: str = "ensemble", top: int = 3,
                 full_track: bool = False, max_windows: int = SEGMENTS_PER_TRACK,
                 cache: dict | None = None) -> dict:
    """Predict a single audio file.

    model: "ensemble" (val-weighted top models), "cnn", "classic", or one exact
           artifact name such as "cnn_resnet" / "classic_svm".
    """
    cache = cache or {}
    audio = load_audio(path, duration=None if full_track else TRACK_SECONDS)
    segments = segments_from_audio(audio[0], audio[1], max_windows)

    available = available_models()
    if not available:
        raise SystemExit("No trained models found - run the training scripts first.")

    if model == "ensemble":
        weights = ensemble_weights()
        weights = {n: w for n, w in weights.items() if n in available}
        if not weights:                      # no reports -> uniform over everything
            weights = {n: 1.0 for n in available}
    elif model == "cnn":
        weights = {n: 1.0 for n in available if n.startswith("cnn_")}
    elif model == "classic":
        weights = {n: 1.0 for n in available if n.startswith("classic_")}
    elif model in available:
        weights = {model: 1.0}
    else:
        raise ValueError(f"Unknown model '{model}'. Available: {['ensemble', 'cnn', 'classic', *available]}")

    total = sum(weights.values()) or 1.0
    probs_by_model: dict[str, np.ndarray] = {}
    blended = np.zeros(len(GENRES))
    for name, weight in weights.items():
        probs = _predict_named(name, audio, segments, cache)
        probs_by_model[name] = probs
        blended += (weight / total) * probs

    order = np.argsort(blended)[::-1]
    return {
        "file": Path(path).name,
        "model": model,
        "genre": GENRES[order[0]],
        "confidence": float(blended[order[0]]),
        "top": [{"genre": GENRES[i], "probability": float(blended[i])}
                for i in order[:top]],
        "probabilities": {g: float(p) for g, p in zip(GENRES, blended)},
        "per_model": {name: {g: float(p) for g, p in zip(GENRES, probs)}
                      for name, probs in probs_by_model.items()},
        "weights": {name: float(w / total) for name, w in weights.items()},
        "n_models": len(weights),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--model", default="ensemble")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--full", action="store_true", help="sample windows across the whole song")
    args = ap.parse_args()

    result = predict_file(args.audio, model=args.model, top=args.top,
                          full_track=args.full)
    print(f"\n{result['file']}  ->  {result['genre'].upper()}  "
          f"({result['confidence'] * 100:.1f}%)   [{result['n_models']} model(s)]")
    for item in result["top"]:
        bar = "#" * int(item["probability"] * 40)
        print(f"  {item['genre']:<10} {item['probability'] * 100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
