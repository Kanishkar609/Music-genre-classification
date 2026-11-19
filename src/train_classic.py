"""Train classic ML baselines (RandomForest / SVM / HistGradientBoosting).

    python src/train_classic.py
    python src/train_classic.py --cv 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASSIC_FEATURES, GENRES, MODELS_DIR, REPORTS_DIR, SEED
from make_splits import load_splits


def make_models() -> dict[str, Pipeline]:
    return {
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=600, max_depth=None, min_samples_leaf=1,
                max_features="sqrt", n_jobs=-1, random_state=SEED)),
        ]),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(C=10.0, gamma="scale", kernel="rbf",
                        probability=True, random_state=SEED)),
        ]),
        "hist_gb": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                early_stopping=True, random_state=SEED)),
        ]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", type=int, default=5, help="folds for train-set CV (0 to skip)")
    args = ap.parse_args()

    if not CLASSIC_FEATURES.exists():
        raise SystemExit("Features missing - run: python src/features.py --classic")

    df = pd.read_parquet(CLASSIC_FEATURES)
    feature_cols = [c for c in df.columns if c not in ("track", "genre", "label")]
    splits = load_splits()

    def subset(names):
        part = df[df["track"].isin(names)]
        return part[feature_cols].to_numpy(np.float32), part["label"].to_numpy()

    X_train, y_train = subset(splits["train"])
    X_val, y_val = subset(splits["val"])
    X_test, y_test = subset(splits["test"])
    print(f"features: {len(feature_cols)}  "
          f"train {X_train.shape[0]} / val {X_val.shape[0]} / test {X_test.shape[0]}")

    results = []
    for name, pipe in make_models().items():
        start = time.time()
        print(f"\n--- {name} ---")
        if args.cv > 0:
            cv = cross_val_score(
                pipe, X_train, y_train,
                cv=StratifiedKFold(args.cv, shuffle=True, random_state=SEED),
                scoring="accuracy", n_jobs=-1,
            )
            print(f"  {args.cv}-fold CV on train: {cv.mean():.4f} +/- {cv.std():.4f}")
        else:
            cv = np.array([np.nan])

        pipe.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, pipe.predict(X_val))
        probs = pipe.predict_proba(X_test)
        pred = probs.argmax(1)
        test_acc = accuracy_score(y_test, pred)
        test_f1 = f1_score(y_test, pred, average="macro")
        elapsed = time.time() - start
        print(f"  val acc {val_acc:.4f}  |  test acc {test_acc:.4f}  macro-F1 {test_f1:.4f}"
              f"  ({elapsed:.1f}s)")
        print(classification_report(y_test, pred, target_names=GENRES, digits=3, zero_division=0))

        joblib.dump({"pipeline": pipe, "features": feature_cols},
                    MODELS_DIR / f"classic_{name}.joblib")
        np.save(MODELS_DIR / f"probs_classic_{name}.npy", probs)
        np.save(MODELS_DIR / f"probs_classic_{name}_y.npy", y_test)
        np.save(MODELS_DIR / f"probs_classic_{name}_tracks.npy",
                df[df["track"].isin(splits["test"])]["track"].to_numpy())

        result = {
            "name": f"classic_{name}", "cv_mean": float(np.nanmean(cv)),
            "cv_std": float(np.nanstd(cv)), "val_acc": float(val_acc),
            "test_acc": float(test_acc), "test_f1": float(test_f1),
            "seconds": round(elapsed, 1),
        }
        results.append(result)
        (REPORTS_DIR / f"classic_{name}.json").write_text(json.dumps(result, indent=2))

    summary = sorted(results, key=lambda r: -r["test_acc"])
    print("\n" + "=" * 62 + "\n  CLASSIC SUMMARY\n" + "=" * 62)
    for r in summary:
        print(f"  {r['name']:<20} cv {r['cv_mean']:.4f}  test {r['test_acc']:.4f}  "
              f"f1 {r['test_f1']:.4f}")
    (REPORTS_DIR / "classic_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
