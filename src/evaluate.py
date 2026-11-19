"""Compare every trained model, build confusion matrices and ensembles.

All metrics are **track-level**: the 10 segment predictions of a CNN are averaged
into one probability vector per song, so CNN and classic models are directly
comparable.

    python src/evaluate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CNN_NAMES, GENRES, MODELS_DIR, REPORTS_DIR
from ensemble import ensemble_weights

CLASSIC_NAMES = ["random_forest", "svm", "hist_gb"]


def load_run(prefix: str, name: str):
    path = MODELS_DIR / f"probs_{prefix}_{name}.npy"
    if not path.exists():
        return None
    probs = np.load(path)
    tracks = np.load(MODELS_DIR / f"probs_{prefix}_{name}_tracks.npy",
                     allow_pickle=True).astype(str)
    y = np.load(MODELS_DIR / f"probs_{prefix}_{name}_y.npy")
    seg_acc = None
    if prefix == "cnn":
        seg_acc = float(accuracy_score(y, probs.argmax(1)))
        # average the 10 segments of each track -> one row per song
        uniq = list(dict.fromkeys(tracks.tolist()))
        agg = np.stack([probs[tracks == t].mean(axis=0) for t in uniq])
        y = np.asarray([y[tracks == t][0] for t in uniq])
        tracks = np.asarray(uniq)
        probs = agg
    return {"name": f"{prefix}_{name}", "probs": probs, "tracks": tracks,
            "y": y, "seg_acc": seg_acc}


def plot_confusion(name: str, y: np.ndarray, pred: np.ndarray) -> None:
    cm = confusion_matrix(y, pred, labels=range(len(GENRES)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(8.2, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                xticklabels=GENRES, yticklabels=GENRES, ax=ax, annot_kws={"size": 8})
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{name} - row-normalised confusion matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / f"confusion_{name}.png", dpi=130)
    plt.close(fig)


def align(runs: list[dict], weights: list[float] | None = None
          ) -> tuple[np.ndarray, np.ndarray]:
    """Blend track-level probabilities, aligning rows by track name."""
    ref = runs[0]
    order = list(ref["tracks"])
    stacked = np.zeros((len(runs), len(order), len(GENRES)))
    for i, run in enumerate(runs):
        lookup = {t: j for j, t in enumerate(run["tracks"])}
        stacked[i] = run["probs"][[lookup[t] for t in order]]
    w = np.ones(len(runs)) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    return np.tensordot(w, stacked, axes=1), ref["y"]


def score(probs: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    pred = probs.argmax(1)
    return float(accuracy_score(y, pred)), float(f1_score(y, pred, average="macro"))


def main() -> None:
    runs = [r for r in (load_run("cnn", n) for n in CNN_NAMES) if r]
    runs += [r for r in (load_run("classic", n) for n in CLASSIC_NAMES) if r]
    if not runs:
        raise SystemExit("No predictions found - train some models first.")

    rows = []
    for run in runs:
        acc, f1 = score(run["probs"], run["y"])
        plot_confusion(run["name"], run["y"], run["probs"].argmax(1))
        rows.append({"name": run["name"], "test_acc": acc, "test_f1": f1,
                     "segment_acc": run["seg_acc"],
                     "kind": run["name"].split("_")[0]})

    ensembles = [
        ("ensemble_cnn6", [r for r in runs if r["name"].startswith("cnn_")]),
        ("ensemble_classic3", [r for r in runs if r["name"].startswith("classic_")]),
        ("ensemble_all", runs),
    ]
    for name, members in ensembles:
        if len(members) < 2:
            continue
        probs, y = align(members)
        acc, f1 = score(probs, y)
        plot_confusion(name, y, probs.argmax(1))
        rows.append({"name": name, "test_acc": acc, "test_f1": f1,
                     "segment_acc": None, "kind": "ensemble", "members": len(members)})

    # val-weighted ensemble over the top-K models (same recipe the app uses)
    weights = ensemble_weights()
    members = [r for r in runs if r["name"] in weights]
    if len(members) >= 2:
        probs, y = align(members, [weights[r["name"]] for r in members])
        acc, f1 = score(probs, y)
        plot_confusion("ensemble_weighted", y, probs.argmax(1))
        rows.append({"name": "ensemble_weighted", "test_acc": acc, "test_f1": f1,
                     "segment_acc": None, "kind": "ensemble", "members": len(members),
                     "weights": {r["name"]: round(weights[r["name"]], 3) for r in members}})

    rows.sort(key=lambda r: -r["test_acc"])
    print("\n" + "=" * 70)
    print(f"  {'model':<24}{'track acc':>11}{'macro F1':>10}{'seg acc':>10}")
    print("=" * 70)
    for r in rows:
        seg = f"{r['segment_acc']:.4f}" if r["segment_acc"] is not None else "-"
        print(f"  {r['name']:<24}{r['test_acc']:>11.4f}{r['test_f1']:>10.4f}{seg:>10}")

    fig, ax = plt.subplots(figsize=(10, max(4, 0.38 * len(rows) + 1.5)))
    names = [r["name"] for r in rows][::-1]
    accs = [r["test_acc"] for r in rows][::-1]
    colors = ["#2a9d8f" if r["kind"] == "ensemble" else
              ("#457b9d" if r["kind"] == "cnn" else "#e9c46a") for r in rows][::-1]
    ax.barh(names, accs, color=colors)
    for i, v in enumerate(accs):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("track-level test accuracy")
    ax.set_title("Music genre classification - model comparison")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "comparison.png", dpi=140)
    plt.close(fig)

    (REPORTS_DIR / "metrics.json").write_text(json.dumps(rows, indent=2))
    print(f"\nWrote metrics.json, comparison.png and confusion_*.png to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
