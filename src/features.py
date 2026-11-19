"""Audio loading, classic feature extraction and mel-spectrogram caching.

Build the caches with:
    python src/features.py --classic     # ~5-10 min for the full dataset
    python src/features.py --mel         # ~3-5 min for the full dataset
    python src/features.py --all
"""
from __future__ import annotations

import argparse
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CACHE_DIR, CLASSIC_FEATURES, DB_FLOOR, GENRES, GENRES_DIR, HOP_LENGTH,
    MEL_CACHE, MEL_LABELS, MEL_TRACKS, N_FFT, N_MELS, SAMPLE_RATE,
    SEG_FRAMES, SEGMENTS_PER_TRACK, TRACK_SECONDS,
)

warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------------- #
# audio
# --------------------------------------------------------------------------- #
def load_audio(path: str | Path, duration: float | None = TRACK_SECONDS,
               offset: float = 0.0) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True,
                         duration=duration, offset=offset)
    if y.size == 0:
        raise ValueError(f"No audio decoded from {path}")
    return y, sr


def list_wavs(folder: Path) -> list[Path]:
    """Wav files in a folder, ignoring macOS AppleDouble junk (._name.wav)."""
    return sorted(p for p in folder.glob("*.wav") if not p.name.startswith("._"))


def list_tracks() -> list[tuple[Path, str, int]]:
    """Return (path, genre, label) for every track on disk."""
    tracks = []
    for idx, genre in enumerate(GENRES):
        for path in list_wavs(GENRES_DIR / genre):
            tracks.append((path, genre, idx))
    if not tracks:
        raise SystemExit(f"No wav files under {GENRES_DIR}. Run download_data.py first.")
    return tracks


# --------------------------------------------------------------------------- #
# classic features
# --------------------------------------------------------------------------- #
def _stats(x: np.ndarray, prefix: str) -> dict[str, float]:
    x = np.atleast_2d(x)
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
    }


def sanitize_data(verbose: bool = True) -> list[Path]:
    """Move unreadable wav files out of the dataset (GTZAN has a few known-bad ones)."""
    corrupt_dir = GENRES_DIR.parent / "corrupt"
    bad: list[Path] = []
    for path, _, _ in list_tracks():
        try:
            with sf.SoundFile(str(path)) as f:
                f.seek(max(0, f.frames - 2000))
                f.read(min(2000, f.frames), dtype="float32", always_2d=False)
        except Exception:  # noqa: BLE001
            bad.append(path)
    for path in bad:
        corrupt_dir.mkdir(parents=True, exist_ok=True)
        target = corrupt_dir / f"{path.parent.name}__{path.name}"
        try:
            path.replace(target)
        except OSError:
            pass
    if verbose:
        if bad:
            print(f"Quarantined {len(bad)} unreadable file(s) -> {corrupt_dir}: "
                  + ", ".join(p.name for p in bad))
        else:
            print("All audio files decoded cleanly.")
    return bad


def extract_classic(y: np.ndarray, sr: int) -> dict[str, float]:
    """Aggregate spectral / timbral descriptors into one vector per track."""
    feats: dict[str, float] = {}
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    mfcc = librosa.feature.mfcc(S=S_db, n_mfcc=20)
    for name, mat in (
        ("mfcc", mfcc),
        ("mfcc_d1", librosa.feature.delta(mfcc)),
        ("mfcc_d2", librosa.feature.delta(mfcc, order=2)),
    ):
        for i in range(mat.shape[0]):
            feats[f"{name}_{i}_mean"] = float(np.mean(mat[i]))
            feats[f"{name}_{i}_std"] = float(np.std(mat[i]))

    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    for i in range(chroma.shape[0]):
        feats[f"chroma_{i}_mean"] = float(np.mean(chroma[i]))
        feats[f"chroma_{i}_std"] = float(np.std(chroma[i]))

    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    for i in range(contrast.shape[0]):
        feats[f"contrast_{i}_mean"] = float(np.mean(contrast[i]))
        feats[f"contrast_{i}_std"] = float(np.std(contrast[i]))

    tonnetz = librosa.feature.tonnetz(chroma=chroma, sr=sr)
    for i in range(tonnetz.shape[0]):
        feats[f"tonnetz_{i}_mean"] = float(np.mean(tonnetz[i]))
        feats[f"tonnetz_{i}_std"] = float(np.std(tonnetz[i]))

    for name, arr in (
        ("centroid", librosa.feature.spectral_centroid(S=S, sr=sr)),
        ("bandwidth", librosa.feature.spectral_bandwidth(S=S, sr=sr)),
        ("rolloff", librosa.feature.spectral_rolloff(S=S, sr=sr)),
        ("flatness", librosa.feature.spectral_flatness(S=S)),
        ("zcr", librosa.feature.zero_crossing_rate(y)),
        ("rms", librosa.feature.rms(y=y)),
    ):
        feats.update(_stats(arr, name))

    tempo = librosa.beat.beat_track(y=y, sr=sr)[0]
    feats["tempo"] = float(np.atleast_1d(tempo)[0])
    onset = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP_LENGTH)
    feats["onset_rate"] = float(len(onset) / (len(y) / sr))
    return feats


def build_classic_cache(force: bool = False) -> Path:
    if CLASSIC_FEATURES.exists() and not force:
        print(f"Classic features already cached at {CLASSIC_FEATURES}")
        return CLASSIC_FEATURES
    sanitize_data()
    tracks = list_tracks()
    print(f"Extracting classic features for {len(tracks)} tracks ...")

    def work(item):
        path, genre, label = item
        try:
            y, sr = load_audio(path)
            row = {"track": path.stem, "genre": genre, "label": label}
            row.update(extract_classic(y, sr))
            return row
        except Exception as exc:  # noqa: BLE001
            print(f"  ! skipped {path.name}: {exc}")
            return None

    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in tqdm(pool.map(work, tracks), total=len(tracks), ncols=80):
            if row is not None:
                rows.append(row)
    df = pd.DataFrame(rows)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLASSIC_FEATURES, index=False)
    print(f"Saved {df.shape[0]} x {df.shape[1]} -> {CLASSIC_FEATURES}")
    return CLASSIC_FEATURES


# --------------------------------------------------------------------------- #
# mel spectrogram segments
# --------------------------------------------------------------------------- #
def _quantized_mel(y: np.ndarray, sr: int) -> np.ndarray:
    """dB mel spectrogram mapped to uint8 in [0, 255]."""
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmax=sr / 2,
    )
    S_db = librosa.power_to_db(S, ref=1.0, top_db=None)
    q = np.clip((S_db + DB_FLOOR) / DB_FLOOR, 0.0, 1.0) * 255.0
    return q.astype(np.uint8)


def mel_segments(y: np.ndarray, sr: int) -> np.ndarray:
    """(SEGMENTS_PER_TRACK, N_MELS, SEG_FRAMES) for a 30 s training track."""
    q = _quantized_mel(y, sr)
    segs = []
    for i in range(SEGMENTS_PER_TRACK):
        start = i * SEG_FRAMES
        seg = q[:, start:start + SEG_FRAMES]
        if seg.shape[1] < SEG_FRAMES:                      # pad short tracks
            seg = np.pad(seg, ((0, 0), (0, SEG_FRAMES - seg.shape[1])))
        segs.append(seg)
    return np.stack(segs)


def segments_from_audio(y: np.ndarray, sr: int,
                        max_windows: int = SEGMENTS_PER_TRACK) -> np.ndarray:
    """Segments spread evenly over an arbitrary-length track (for inference)."""
    q = _quantized_mel(y, sr)
    n = max(1, q.shape[1] // SEG_FRAMES)
    idx = np.linspace(0, n - 1, min(n, max_windows)).round().astype(int) if n > max_windows \
        else np.arange(n)
    segs = []
    for i in idx:
        seg = q[:, i * SEG_FRAMES:(i + 1) * SEG_FRAMES]
        if seg.shape[1] < SEG_FRAMES:
            seg = np.pad(seg, ((0, 0), (0, SEG_FRAMES - seg.shape[1])))
        segs.append(seg)
    return np.stack(segs)


def track_mel_from_file(path: str | Path, max_windows: int = SEGMENTS_PER_TRACK,
                        full_track: bool = False) -> np.ndarray:
    """Segments for inference: first 30s by default, or windows spread over a song."""
    y, sr = load_audio(path, duration=None if full_track else TRACK_SECONDS)
    return segments_from_audio(y, sr, max_windows)


def build_mel_cache(force: bool = False) -> Path:
    if MEL_CACHE.exists() and not force:
        print(f"Mel cache already present at {MEL_CACHE}")
        return MEL_CACHE
    sanitize_data()
    tracks = list_tracks()
    print(f"Building mel-spectrogram cache for {len(tracks)} tracks ...")

    def work(item):
        path, genre, label = item
        try:
            y, sr = load_audio(path)
            return mel_segments(y, sr), path.stem, label
        except Exception as exc:  # noqa: BLE001
            print(f"  ! skipped {path.name}: {exc}")
            return None

    data, names, labels = [], [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in tqdm(pool.map(work, tracks), total=len(tracks), ncols=80):
            if result is None:
                continue
            segs, name, label = result
            data.append(segs)
            names.extend([name] * len(segs))
            labels.extend([label] * len(segs))

    arr = np.concatenate(data, axis=0)
    np.save(MEL_CACHE, arr)
    np.save(MEL_LABELS, np.asarray(labels, dtype=np.int64))
    np.save(MEL_TRACKS, np.asarray(names))
    print(f"Saved mel cache {arr.shape} ({arr.nbytes / 1e6:.0f} MB) -> {MEL_CACHE}")
    return MEL_CACHE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classic", action="store_true")
    ap.add_argument("--mel", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not (args.classic or args.mel or args.all):
        ap.error("choose --classic, --mel or --all")
    if args.classic or args.all:
        build_classic_cache(args.force)
    if args.mel or args.all:
        build_mel_cache(args.force)


if __name__ == "__main__":
    main()
