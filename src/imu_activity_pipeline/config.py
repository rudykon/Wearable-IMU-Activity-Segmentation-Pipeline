"""Central configuration for data paths, model paths, labels, and hyperparameters.

Purpose:
    Provides one place for sensor settings, activity labels, window sizes,
    post-processing thresholds, training parameters, and environment-variable
    overrides.
Inputs:
    Reads optional `HLS_HAR_*` environment variables to override data, model, and
    annotation locations without changing code.
Outputs:
    Exposes constants used by data loading, training, inference, evaluation, and
    experiment scripts.
"""
import os
import sys

# Paths: handle normal Python runs and PyInstaller-frozen executables.
# In source checkouts, this module lives under src/imu_activity_pipeline/ while
# data/, saved_models/, and generated outputs stay at the repository root.
if getattr(sys, 'frozen', False):
    BUNDLE_DIR = sys._MEIPASS          # bundled assets (models)
    RUNTIME_DIR = os.path.dirname(sys.executable)  # Directory that holds runtime data and outputs.
else:
    PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(os.path.dirname(PACKAGE_DIR))
    BUNDLE_DIR = BASE_DIR
    RUNTIME_DIR = BASE_DIR

BASE_DIR = RUNTIME_DIR


def _first_existing(*paths):
    """Return the first existing path, or the first candidate as a stable default."""
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]


# Unified data root. The PhysioNet-style release keeps all signals, annotations,
# split files, and metadata under one top-level data/ directory.
DATA_ROOT = os.getenv("HLS_HAR_DATA_ROOT", os.path.join(RUNTIME_DIR, "data"))
SIGNALS_DIR = os.path.join(DATA_ROOT, "signals")
ANNOTATIONS_DIR = os.path.join(DATA_ROOT, "annotations")
SPLITS_DIR = os.path.join(DATA_ROOT, "splits")
METADATA_DIR = os.path.join(DATA_ROOT, "metadata")

TRAIN_DATA_DIR = os.getenv("HLS_HAR_TRAIN_DATA_DIR", os.path.join(SIGNALS_DIR, "train"))
INTERNAL_EVAL_DATA_DIR = os.getenv(
    "HLS_HAR_INTERNAL_EVAL_DATA_DIR",
    os.path.join(SIGNALS_DIR, "internal_eval"),
)
EXTERNAL_TEST_DATA_DIR = os.getenv(
    "HLS_HAR_EXTERNAL_TEST_DATA_DIR",
    os.path.join(SIGNALS_DIR, "external_test"),
)

TRAIN_ANNOTATIONS_FILE = os.getenv(
    "HLS_HAR_TRAIN_ANNOTATIONS_FILE",
    os.path.join(ANNOTATIONS_DIR, "train_annotations.csv"),
)
INTERNAL_EVAL_GOLD_FILE = os.getenv(
    "HLS_HAR_INTERNAL_EVAL_GOLD_FILE",
    os.path.join(ANNOTATIONS_DIR, "internal_eval_annotations.csv"),
)
EXTERNAL_TEST_GOLD_FILE = os.getenv(
    "HLS_HAR_EXTERNAL_TEST_GOLD_FILE",
    os.path.join(ANNOTATIONS_DIR, "external_test_annotations.csv"),
)
ALL_ANNOTATIONS_FILE = os.getenv(
    "HLS_HAR_ALL_ANNOTATIONS_FILE",
    os.path.join(ANNOTATIONS_DIR, "all_annotations.csv"),
)

# Canonical data-release split names. Evaluation and diagnostic scripts should
# use these names rather than generic "test" or "val" labels.
SPLIT_NAMES = ("train", "internal_eval", "external_test")
SPLIT_DATA_DIRS = {
    "train": TRAIN_DATA_DIR,
    "internal_eval": INTERNAL_EVAL_DATA_DIR,
    "external_test": EXTERNAL_TEST_DATA_DIR,
}
SPLIT_ANNOTATION_FILES = {
    "train": TRAIN_ANNOTATIONS_FILE,
    "internal_eval": INTERNAL_EVAL_GOLD_FILE,
    "external_test": EXTERNAL_TEST_GOLD_FILE,
}
DEFAULT_INFERENCE_SPLIT = os.getenv("HLS_HAR_INFERENCE_SPLIT", "external_test")
DEFAULT_EVALUATION_SPLIT = os.getenv("HLS_HAR_EVALUATION_SPLIT", "external_test")

RAW_INDEX_FILE = os.getenv("HLS_HAR_RAW_INDEX_FILE", os.path.join(DATA_ROOT, "raw", "raw.csv"))
RAW_ZIP_DIR = os.getenv("HLS_HAR_RAW_ZIP_DIR", os.path.join(DATA_ROOT, "raw"))

MODEL_DIR = os.getenv("HLS_HAR_MODEL_DIR", os.path.join(BUNDLE_DIR, "saved_models"))
os.makedirs(MODEL_DIR, exist_ok=True)

# Sensor parameters
ACC_GYRO_RATE = 100  # Hz
PPG_RATE = 25  # Hz

# Column names in sensor data
COLUMNS = [
    'ACC_TIME', 'PPG_TIME', 'GYRO_TIME',
    'PPG1', 'PPG2', 'PPG3', 'PPG4', 'PPG5', 'PPG6', 'PPG7', 'PPG8',
    'PPG9', 'PPG10', 'PPG11', 'PPG12', 'PPG13', 'PPG14', 'PPG15', 'PPG16',
    'PPG17', 'PPG18', 'PPG19', 'PPG20', 'PPG21', 'PPG22', 'PPG23', 'PPG24',
    'ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z'
]

PPG_COLUMNS = [f'PPG{i}' for i in range(1, 25)]

# Activity classes
ACTIVITIES = ['羽毛球', '跳绳', '飞鸟', '跑步', '乒乓球']
ACTIVITY_TO_IDX = {a: i for i, a in enumerate(ACTIVITIES)}
IDX_TO_ACTIVITY = {i: a for i, a in enumerate(ACTIVITIES)}
NUM_CLASSES = len(ACTIVITIES)

# Window parameters
WINDOW_SIZE_SEC = 3  # seconds
WINDOW_STEP_SEC = 1  # seconds
WINDOW_SIZE = WINDOW_SIZE_SEC * ACC_GYRO_RATE  # 300 samples
WINDOW_STEP = WINDOW_STEP_SEC * ACC_GYRO_RATE  # 100 samples

# 5-second window for multi-scale ensemble
WINDOW_SIZE_5S_SEC = 5
WINDOW_SIZE_5S = WINDOW_SIZE_5S_SEC * ACC_GYRO_RATE  # 500 samples
WINDOW_STEP_5S = WINDOW_STEP_SEC * ACC_GYRO_RATE  # 100 samples (same step)

# 8-second window for extended multi-scale ensemble
WINDOW_SIZE_8S_SEC = 8
WINDOW_SIZE_8S = WINDOW_SIZE_8S_SEC * ACC_GYRO_RATE  # 800 samples
WINDOW_STEP_8S = WINDOW_STEP_SEC * ACC_GYRO_RATE  # 100 samples (same step)

# All window configs for training: (window_size_sec, window_size_samples, window_step_samples, name_suffix)
WINDOW_CONFIGS = [
    (WINDOW_SIZE_SEC, WINDOW_SIZE, WINDOW_STEP, "3s"),
    (WINDOW_SIZE_5S_SEC, WINDOW_SIZE_5S, WINDOW_STEP_5S, "5s"),
    (WINDOW_SIZE_8S_SEC, WINDOW_SIZE_8S, WINDOW_STEP_8S, "8s"),
]

# Stage 1: activity/background detection.
STAGE1_THRESHOLD = 0.5
STAGE1_SMOOTH_WINDOW = 5  # seconds
STAGE1_RECALL_TARGET = 0.95
STAGE1_FOCAL_ALPHA = 0.75
STAGE1_FOCAL_GAMMA = 2.0

# Stage 2: activity classification.
STAGE2_WINDOW_SEC = 3
STAGE2_STEP_SEC = 1
STAGE2_FOCAL_GAMMA = 2.0
STAGE2_TRIPLET_MARGIN = 1.0

# Stage 3: temporal boundary refinement.
BOUNDARY_SEARCH_RANGE = 15  # seconds

# Post-processing and segment filtering.
MIN_ACTIVITY_DURATION = 300  # seconds (5 min, conservative)
SHORT_GAP_THRESHOLD = 60  # seconds
MIN_SEGMENT_FOR_OUTPUT = 120  # seconds (2 min minimum for output)

# Viterbi decoding parameters.
VITERBI_SELF_TRANSITION = 0.90
VITERBI_CROSS_TRANSITION = 0.01
VITERBI_BG_TO_ACT = 0.02
VITERBI_ACT_TO_BG = 0.05

# Butterworth low-pass filter.
BUTTERWORTH_ORDER = 4
BUTTERWORTH_CUTOFF = 40  # Hz, lowpass cutoff for ACC/GYRO

# Training-time data augmentation.
AUG_TIME_STRETCH_RANGE = (0.9, 1.1)
AUG_AMPLITUDE_SCALE_RANGE = (0.8, 1.2)
AUG_NOISE_SNR_DB = 20
AUG_MIXUP_ALPHA = 0.3

# Ensemble training configuration.
ENSEMBLE_NUM_MODELS = 3  # 3 models with different seeds
ENSEMBLE_SEEDS = [42, 123, 456]

# Training
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS_STAGE1 = 30
NUM_EPOCHS_STAGE2 = int(os.getenv("NUM_EPOCHS_STAGE2", "100"))
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", "30"))
MIN_EPOCHS_BEFORE_EARLY_STOP = int(os.getenv("MIN_EPOCHS_BEFORE_EARLY_STOP", "40"))
DEVICE = "cuda"
NUM_WORKERS = 4
VAL_SPLIT = 0.2
LABEL_SMOOTHING = 0.1
