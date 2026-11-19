"""Track-level stratified train / val / test split.

Splitting by *track* (not by segment) is essential: GTZAN segments from the same
song are highly correlated, so a random segment split leaks information and
inflates accuracy by 10+ points.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CACHE_DIR, GENRES, SPLITS_FILE


def build_splits(val_size: float = 0.15, test_size: float = 0.15,
                 seed: int = 42) -> dict[str, list[str]]:
    tracks, labels = [], []
    for idx, genre in enumerate(GENRES):
        folder = CACHE_DIR.parent / "data" / "genres" / genre
        for path in sorted(p for p in folder.glob("*.wav") if not p.name.startswith("._")):
            tracks.append(path.stem)
            labels.append(idx)
    if not tracks:
        raise SystemExit("No tracks found - run download_data.py first.")

    labels = np.asarray(labels)
    train_val, test, _, _ = train_test_split(
        tracks, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    labels_train_val = np.asarray([GENRES.index(Path(t).name.split(".")[0]) for t in train_val])
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1.0 - test_size),
        stratify=labels_train_val,
        random_state=seed,
    )
    splits = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
    SPLITS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_FILE.write_text(json.dumps(splits, indent=2))
    print(f"train={len(train)}  val={len(val)}  test={len(test)}  -> {SPLITS_FILE}")
    return splits


def load_splits(rebuild: bool = False) -> dict[str, list[str]]:
    if SPLITS_FILE.exists() and not rebuild:
        return json.loads(SPLITS_FILE.read_text())
    return build_splits()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    load_splits(rebuild=args.rebuild or not SPLITS_FILE.exists())
