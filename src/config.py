"""Project-wide constants and paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
GENRES_DIR = DATA_DIR / "genres"
CACHE_DIR = ROOT / "cache"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]

# --- audio / representation ---
SAMPLE_RATE = 22050
TRACK_SECONDS = 30.0
SEGMENT_SECONDS = 3.0
SEGMENTS_PER_TRACK = 10          # 10 x 3s = the 30s GTZAN track
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
SEG_FRAMES = 128                 # 3s @ hop 512 ~= 129 frames, trimmed to 128
DB_FLOOR = 80.0                  # top_db used for uint8 quantization

# --- training ---
SEED = 42
BATCH_SIZE = 64
DEFAULT_EPOCHS = 20
PATIENCE = 5

# six CNN architectures, see src/cnn_models.py
CNN_NAMES = ["baseline", "deep", "vgg", "resnet", "separable", "attention"]

# --- artifacts ---
CLASSIC_FEATURES = CACHE_DIR / "classic_features.parquet"
MEL_CACHE = CACHE_DIR / "mel_segments.npy"
MEL_LABELS = CACHE_DIR / "mel_labels.npy"
MEL_TRACKS = CACHE_DIR / "mel_tracks.npy"
SPLITS_FILE = CACHE_DIR / "splits.json"
METRICS_FILE = REPORTS_DIR / "metrics.json"

for _d in (DATA_DIR, CACHE_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
