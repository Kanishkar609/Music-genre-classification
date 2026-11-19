"""Streamlit demo: upload a song, get its genre.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import GENRES, N_FFT, HOP_LENGTH, REPORTS_DIR, SAMPLE_RATE  # noqa: E402
from features import load_audio  # noqa: E402
from predict import available_models, predict_file  # noqa: E402

st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵", layout="wide")

MODEL_CHOICES = ["ensemble", "cnn", "classic"] + available_models()


@st.cache_resource
def model_cache() -> dict:
    return {}


@st.cache_data(show_spinner=False)
def cached_predict(path: str, model: str, full_track: bool, windows: int) -> dict:
    return predict_file(path, model=model, top=10, full_track=full_track,
                        max_windows=windows, cache=model_cache())


@st.cache_data(show_spinner=False)
def audio_figures(path: str):
    y, sr = load_audio(path)
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.6))
    librosa.display.waveshow(y, sr=sr, ax=axes[0], color="#2a9d8f")
    axes[0].set_title("waveform")
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT,
                                         hop_length=HOP_LENGTH, n_mels=128)
    librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), sr=sr,
                             hop_length=HOP_LENGTH, x_axis="time", y_axis="mel",
                             ax=axes[1], cmap="magma")
    axes[1].set_title("mel spectrogram")
    plt.tight_layout()
    return fig


def model_scores() -> pd.DataFrame:
    metrics = REPORTS_DIR / "metrics.json"
    if metrics.exists():
        rows = json.loads(metrics.read_text())
        return pd.DataFrame([
            {"model": r["name"], "test_acc": r["test_acc"], "macro_f1": r["test_f1"]}
            for r in rows
        ]).sort_values("test_acc", ascending=False)
    rows = []
    for path in sorted(REPORTS_DIR.glob("*_summary.json")):
        for item in json.loads(path.read_text()):
            rows.append({"model": item["name"], "test_acc": item["test_acc"],
                         "macro_f1": item["test_f1"]})
    return pd.DataFrame(rows).sort_values("test_acc", ascending=False) if rows else pd.DataFrame()


# --------------------------------------------------------------------------- #
st.title("🎵 Music Genre Classifier")
st.caption("Six CNNs on mel-spectrograms + three classic ML baselines, trained on GTZAN.")

with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", MODEL_CHOICES, index=0)
    full_track = st.toggle("Analyse the whole track", value=False,
                           help="Off = first 30 s (fast). On = windows spread across the song.")
    windows = st.slider("Windows to average", 1, 30, 10, disabled=not full_track)
    st.divider()
    st.subheader("Model leaderboard")
    board = model_scores()
    if board.empty:
        st.info("Train the models to populate the leaderboard.")
    else:
        st.dataframe(board, hide_index=True, width="stretch")

uploaded = st.file_uploader("Drop a song", type=["wav", "mp3", "flac", "ogg", "m4a"])

if uploaded is None:
    st.info("Upload an audio file to classify it. WAV, MP3, FLAC and OGG work out of the box.")
    st.stop()

suffix = Path(uploaded.name).suffix or ".wav"
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
tmp.write(uploaded.getbuffer())
tmp.close()

try:
    with st.spinner("Analysing audio ..."):
        result = cached_predict(tmp.name, model, full_track, windows)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not process this file: {exc}")
    st.stop()

left, right = st.columns([1, 1.4])
with left:
    st.metric("Predicted genre", result["genre"].capitalize(),
              f"{result['confidence'] * 100:.1f}% confidence")
    st.caption(f"{result['file']} · {result['n_models']} model(s) · `{result['model']}`")
    if result.get("weights") and len(result["weights"]) > 1:
        st.caption("ensemble weights: " + ", ".join(
            f"{name.replace('cnn_', '').replace('classic_', '')} {w:.0%}"
            for name, w in sorted(result["weights"].items(), key=lambda kv: -kv[1])))
    st.subheader("Top 3")
    for item in result["top"][:3]:
        st.write(f"**{item['genre'].capitalize()}** — {item['probability'] * 100:.1f}%")
        st.progress(min(1.0, item["probability"]))
    st.subheader("All genres")
    dist = pd.DataFrame({"probability": pd.Series(result["probabilities"])}).sort_values(
        "probability", ascending=False)
    st.bar_chart(dist, height=260)

with right:
    st.pyplot(audio_figures(tmp.name), width="stretch")

with st.expander("Per-model predictions"):
    rows = []
    for name, probs in result["per_model"].items():
        best = max(probs, key=probs.get)
        rows.append({"model": name, "prediction": best,
                     "confidence": probs[best], **probs})
    st.dataframe(pd.DataFrame(rows).sort_values("confidence", ascending=False),
                 hide_index=True, width="stretch")
