"""Train the six CNN models on cached mel-spectrogram segments.

Examples:
    python src/train_cnn.py --quick                  # 5 epochs, 40% of data
    python src/train_cnn.py                          # all six, 20 epochs
    python src/train_cnn.py --models baseline,resnet # a subset
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnn_models import build_model, count_params
from config import (
    BATCH_SIZE, CNN_NAMES, DEFAULT_EPOCHS, GENRES, MEL_CACHE, MEL_LABELS,
    MEL_TRACKS, MODELS_DIR, PATIENCE, REPORTS_DIR, SEED,
)
from make_splits import load_splits


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class MelSegments(Dataset):
    def __init__(self, arr, labels, indices, mean: float, std: float, train: bool):
        self.arr = arr
        self.labels = labels
        self.indices = indices
        self.mean = mean
        self.std = std
        self.train = train

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        x = self.arr[idx].astype(np.float32)
        x = (x / 255.0 - self.mean) / self.std
        x = torch.from_numpy(x).unsqueeze(0)
        y = int(self.labels[idx])
        if self.train:
            x = self._augment(x)
        return x, y

    @staticmethod
    def _augment(x: torch.Tensor) -> torch.Tensor:
        # SpecAugment-lite: two frequency masks + two time masks + gain jitter
        _, h, w = x.shape
        for _ in range(2):
            f = torch.randint(0, max(2, h // 8), (1,)).item()
            f0 = torch.randint(0, h - f, (1,)).item()
            x[:, f0:f0 + f, :] = 0.0
            t = torch.randint(0, max(2, w // 8), (1,)).item()
            t0 = torch.randint(0, w - t, (1,)).item()
            x[:, :, t0:t0 + t] = 0.0
        if torch.rand(1).item() < 0.5:
            x = x * (0.9 + 0.2 * torch.rand(1).item())
        return x


def _indices_for(tracks: np.ndarray, split_tracks: list[str]) -> np.ndarray:
    wanted = set(split_tracks)
    return np.flatnonzero(np.isin(tracks, list(wanted)))


# --------------------------------------------------------------------------- #
# train / eval loops
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, preds, targets = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(y)
        preds.append(logits.argmax(1).cpu())
        targets.append(y.cpu())
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    return (
        total_loss / len(loader.dataset),
        accuracy_score(targets, preds),
        f1_score(targets, preds, average="macro"),
    )


@torch.no_grad()
def predict_proba(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, targets = [], []
    for x, y in loader:
        logits = model(x.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def train_one(name: str, args, arr, labels, tracks, splits) -> dict:
    device = torch.device("cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_idx = _indices_for(tracks, splits["train"])
    val_idx = _indices_for(tracks, splits["val"])
    test_idx = _indices_for(tracks, splits["test"])
    if args.quick:
        rng = np.random.default_rng(SEED)
        train_idx = rng.choice(train_idx, size=int(len(train_idx) * 0.4), replace=False)

    sample = arr[train_idx[: min(2000, len(train_idx))]].astype(np.float32) / 255.0
    mean, std = float(sample.mean()), float(sample.std() + 1e-6)

    dl = dict(num_workers=0, pin_memory=False)
    train_loader = DataLoader(
        MelSegments(arr, labels, train_idx, mean, std, True),
        batch_size=args.batch_size, shuffle=True, drop_last=False, **dl)
    val_loader = DataLoader(
        MelSegments(arr, labels, val_idx, mean, std, False),
        batch_size=args.batch_size * 2, shuffle=False, **dl)
    test_loader = DataLoader(
        MelSegments(arr, labels, test_idx, mean, std, False),
        batch_size=args.batch_size * 2, shuffle=False, **dl)

    model = build_model(name, num_classes=len(GENRES)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\n{'=' * 62}\n  {name.upper()}  ({count_params(model):,} params, "
          f"{len(train_idx)} train / {len(val_idx)} val segments)\n{'=' * 62}")

    best_val, best_state, bad_epochs = -1.0, None, 0
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc, va_f1 = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "val_f1": va_f1})
        print(f"  epoch {epoch:2d}/{args.epochs}  train loss {tr_loss:.3f} acc {tr_acc:.3f}"
              f"  |  val loss {va_loss:.3f} acc {va_acc:.3f} f1 {va_f1:.3f}")
        if va_acc > best_val:
            best_val, bad_epochs = va_acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"  early stopping (no val improvement for {args.patience} epochs)")
                break

    model.load_state_dict(best_state)
    test_probs, test_targets = predict_proba(model, test_loader, device)
    test_acc = accuracy_score(test_targets, test_probs.argmax(1))
    test_f1 = f1_score(test_targets, test_probs.argmax(1), average="macro")
    elapsed = time.time() - start
    print(f"  -> test acc {test_acc:.4f}  macro-F1 {test_f1:.4f}  ({elapsed / 60:.1f} min)")

    ckpt = MODELS_DIR / f"cnn_{name}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "mean": mean, "std": std,
        "arch": name, "num_classes": len(GENRES),
        "test_acc": test_acc, "test_f1": test_f1,
    }, ckpt)
    np.save(MODELS_DIR / f"probs_cnn_{name}.npy", test_probs)
    np.save(MODELS_DIR / f"probs_cnn_{name}_y.npy", test_targets)
    np.save(MODELS_DIR / f"probs_cnn_{name}_tracks.npy", tracks[test_idx])

    result = {
        "name": name, "params": count_params(model),
        "best_val_acc": best_val, "test_acc": test_acc, "test_f1": test_f1,
        "minutes": round(elapsed / 60, 2), "epochs_run": len(history),
        "history": history,
    }
    (REPORTS_DIR / f"cnn_{name}.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all", help="comma list or 'all'")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--quick", action="store_true", help="fewer epochs, 40% of train data")
    args = ap.parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 5)

    threads = max(1, min(8, (os.cpu_count() or 4)))
    torch.set_num_threads(threads)
    print(f"torch {torch.__version__} on CPU, {threads} threads")

    if not MEL_CACHE.exists():
        raise SystemExit("Mel cache missing - run: python src/features.py --mel")

    arr = np.load(MEL_CACHE, mmap_mode="r")
    labels = np.load(MEL_LABELS)
    tracks = np.load(MEL_TRACKS, allow_pickle=True)
    splits = load_splits()
    print(f"mel cache {arr.shape}, {len(GENRES)} classes, tracks "
          f"{len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")

    names = CNN_NAMES if args.models == "all" else [m.strip() for m in args.models.split(",")]
    results = []
    for name in names:
        results.append(train_one(name, args, arr, labels, tracks, splits))

    summary = sorted(results, key=lambda r: -r["test_acc"])
    print("\n" + "=" * 62 + "\n  CNN SUMMARY\n" + "=" * 62)
    for r in summary:
        print(f"  {r['name']:<11} params {r['params']:>8,}  "
              f"test acc {r['test_acc']:.4f}  f1 {r['test_f1']:.4f}")
    (REPORTS_DIR / "cnn_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
