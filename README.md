# Music Genre Classification

Classify music by genre from an audio file. **Six CNN architectures** on mel-spectrograms
plus three classic ML baselines, trained on the **GTZAN** dataset (10 genres, 1000 tracks),
with a Streamlit web app for drag-and-drop prediction.

## Genres

blues · classical · country · disco · hiphop · jazz · metal · pop · reggae · rock

## How it works

```
audio ──► 3 s segments ──► mel spectrogram (128×128) ──► 6 CNNs ──► weighted ensemble
       └─► librosa descriptors (MFCC/chroma/tonnetz/…) ──► RF / SVM / HistGB ──┘
```

| Model | Idea | Params |
|---|---|---|
| `baseline` | 3 conv blocks + GAP | 94 k |
| `deep` | 4 blocks, 32→256 channels | 423 k |
| `vgg` | double 3×3 conv per block | 288 k |
| `resnet` | residual blocks | 1.21 M |
| `separable` | depthwise-separable convs | 49 k |
| `attention` | squeeze-excitation + attention pooling | 100 k |

Plus `random_forest`, `svm` and `hist_gb` on 184 hand-crafted features, so the report can
compare learned representations against feature engineering.

**Splitting is done by track, not by segment.** Segments from one song are highly
correlated; splitting them randomly leaks the track across train/test and inflates
accuracy by 10+ points. All reported numbers are **track-level**: the 10 segment
predictions of a CNN are averaged into one prediction per song.

## Results (GTZAN, 150 held-out tracks)

| model | track acc | macro F1 | segment acc |
|---|---|---|---|
| `cnn_resnet` | **0.860** | 0.860 | 0.791 |
| `ensemble` (val-weighted top-4) | **0.860** | 0.855 | – |
| `cnn_vgg` | 0.847 | 0.841 | 0.787 |
| `ensemble_all` (uniform, 9 models) | 0.833 | 0.826 | – |
| `cnn_deep` | 0.813 | 0.801 | 0.741 |
| `ensemble_cnn6` (uniform, 6 CNNs) | 0.807 | 0.793 | – |
| `ensemble_classic3` | 0.767 | 0.763 | – |
| `classic_svm` | 0.753 | 0.744 | – |
| `classic_hist_gb` | 0.753 | 0.754 | – |
| `cnn_baseline` | 0.733 | 0.715 | 0.644 |
| `classic_random_forest` | 0.707 | 0.697 | – |
| `cnn_attention` | 0.680 | 0.664 | 0.633 |
| `cnn_separable` | 0.653 | 0.633 | 0.592 |

A uniform average over all six CNNs is *worse* than the best single CNN, because the three
small models drag it down. `src/ensemble.py` therefore keeps the top-4 models by
**validation** accuracy and weights them by that accuracy (never by test accuracy):

```
cnn_vgg 25% · cnn_resnet 25% · classic_svm 25% · cnn_deep 25%
```

Training all six CNNs takes ~2–3 h on an 8-core laptop CPU; `--quick` gives a ~5 min run.

## Setup

```bash
python -m pip install -r requirements.txt
# CPU-only PyTorch (much smaller than the default CUDA wheel):
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Run the pipeline

```bash
# 1. data (1.2 GB download, extracts to data/genres/)
python src/download_data.py

# 2. features (cached to cache/, corrupt files are quarantined automatically)
python src/features.py --all
python src/make_splits.py

# 3. models
python src/train_classic.py
python src/train_cnn.py                       # all six CNNs, 20 epochs
python src/train_cnn.py --models resnet --epochs 30
python src/train_cnn.py --quick               # 5 epochs, 40% of data

# 4. compare + reports
python src/evaluate.py    # -> reports/metrics.json, comparison.png, confusion_*.png

# 5. try it
python src/predict.py path/to/song.mp3 --full
streamlit run app/streamlit_app.py
```

Fast smoke test before committing to the full run:

```bash
python src/download_data.py --sample      # 10 tracks per genre
python src/features.py --all
python src/train_cnn.py --quick
```

## Project layout

```
src/
  config.py         paths + audio constants
  download_data.py  GTZAN download / extract / verify
  features.py       184 classic features + mel cache + corrupt-file quarantine
  make_splits.py    track-level stratified 70/15/15 split
  cnn_models.py     the six CNN architectures
  train_classic.py  RF / SVM / HistGB with 5-fold CV
  train_cnn.py      shared training loop, early stopping, checkpoints
  ensemble.py       validation-weighted ensemble selection
  evaluate.py       track-level metrics, confusion matrices, comparison chart
  predict.py        audio file -> genre probabilities
app/
  streamlit_app.py  upload UI
cache/              feature + mel caches (regenerated, ~165 MB)
models/             .joblib / .pt checkpoints + test predictions
reports/            metrics.json, comparison.png, confusion_*.png
```

## Notes

- GTZAN ships one corrupt file (`jazz.00054.wav`); `features.py` detects unreadable audio,
  moves it to `data/corrupt/`, and continues. Final dataset: 999 tracks.
- MP3 decoding uses libsndfile (bundled with `soundfile` ≥ 0.12). If a file fails, convert
  it: `ffmpeg -i in.m4a -ar 22050 out.wav`.
- The app analyses the first 30 s by default (matching training). Enable *Analyse the whole
  track* to sample windows across the song instead.
- GTZAN is small and has known label noise / artist repetition; treat the numbers as a
  benchmark, not a production claim.
