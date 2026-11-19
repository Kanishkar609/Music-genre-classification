"""Download and extract the GTZAN dataset.

Usage:
    python src/download_data.py            # full 1000-track dataset (~1.2 GB)
    python src/download_data.py --sample   # 10 tracks per genre (100 tracks)
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, GENRES, GENRES_DIR

URL = "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz"
ARCHIVE = DATA_DIR / "gtzan.tar.gz"


def _progress(blocks: int, block_size: int, total: int) -> None:
    if total <= 0:
        return
    done = min(blocks * block_size, total)
    pct = 100 * done / total
    mb = done / 1e6
    total_mb = total / 1e6
    bar = "#" * int(pct / 2.5)
    print(f"\r  [{bar:<40}] {pct:5.1f}%  {mb:7.1f}/{total_mb:.1f} MB", end="", flush=True)


def download(force: bool = False) -> Path:
    if ARCHIVE.exists() and not force:
        size = ARCHIVE.stat().st_size
        if size > 1_000_000_000:
            print(f"Archive already present ({size / 1e9:.2f} GB) - skipping download.")
            return ARCHIVE
        print("Existing archive looks incomplete, re-downloading.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = ARCHIVE.with_suffix(".part")
    print(f"Downloading GTZAN from HuggingFace ...\n  {URL}")
    start = time.time()
    urllib.request.urlretrieve(URL, tmp, reporthook=_progress)
    print(f"\n  done in {time.time() - start:.0f}s")
    tmp.replace(ARCHIVE)
    return ARCHIVE


def extract(archive: Path, force: bool = False) -> None:
    if GENRES_DIR.exists() and any(GENRES_DIR.iterdir()) and not force:
        print("Extracted genres folder already exists - skipping extraction.")
        return
    if GENRES_DIR.exists():
        shutil.rmtree(GENRES_DIR)
    print("Extracting archive (this takes a minute or two) ...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(DATA_DIR)
    print(f"  extracted to {GENRES_DIR}")


def verify(sample: bool = False) -> None:
    counts = {}
    for genre in GENRES:
        folder = GENRES_DIR / genre
        files = sorted(p for p in folder.glob("*.wav") if not p.name.startswith("._")) if folder.exists() else []
        counts[genre] = len(files)
    print("\nFiles per genre:")
    for genre in GENRES:
        print(f"  {genre:<10} {counts[genre]:>4}")
    total = sum(counts.values())
    missing = [g for g, c in counts.items() if c == 0]
    if missing:
        raise SystemExit(f"Missing genres: {missing}. Extraction failed?")
    print(f"  {'TOTAL':<10} {total:>4}")
    if not sample and total < 1000:
        raise SystemExit("Expected 1000 tracks. Delete data/genres and re-run.")


def make_sample(per_genre: int = 10) -> None:
    """Keep only the first `per_genre` tracks of each genre (fast smoke test)."""
    print(f"\nBuilding sample subset ({per_genre} per genre) ...")
    for genre in GENRES:
        folder = GENRES_DIR / genre
        files = sorted(p for p in folder.glob("*.wav") if not p.name.startswith("._"))
        for extra in files[per_genre:]:
            extra.unlink()
    verify(sample=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="keep 10 tracks per genre only")
    ap.add_argument("--per-genre", type=int, default=10)
    ap.add_argument("--force", action="store_true", help="re-download / re-extract")
    args = ap.parse_args()

    archive = download(force=args.force)
    extract(archive, force=args.force)
    verify()
    if args.sample:
        make_sample(args.per_genre)


if __name__ == "__main__":
    main()
