"""Data loading, preprocessing, feature extraction, and augmentation utilities.

Purpose:
    Converts raw tab-separated sensor files and annotation tables into clean IMU
    arrays, sliding windows, labels, normalization statistics, and optional
    augmented training samples.
Inputs:
    Reads signal text files with accelerometer/gyroscope columns and annotation
    CSV/XLSX files with `user_id`, `category`, `start`, and `end` fields.
Outputs:
    Returns NumPy arrays and pandas DataFrames consumed by training, inference,
    evaluation, and experiment scripts.
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from .config import *


# ==================== Signal Preprocessing =======================================

def butterworth_filter(data: np.ndarray, cutoff: float = BUTTERWORTH_CUTOFF,
                       fs: float = ACC_GYRO_RATE, order: int = BUTTERWORTH_ORDER) -> np.ndarray:
    """Apply a Butterworth low-pass filter for IMU denoising.

    Args:
        data: (N, channels) signal data
        cutoff: lowpass cutoff frequency in Hz
        fs: sampling rate
        order: filter order
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1.0:
        return data  # cutoff too high, skip filter

    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = np.copy(data)
    for ch in range(data.shape[1]):
        if len(data[:, ch]) > 3 * max(len(a), len(b)):
            filtered[:, ch] = filtfilt(b, a, data[:, ch])
    return filtered


# ==================== Data Loading ================================================

def load_sensor_data(file_path: str, apply_filter: bool = True) -> Optional[np.ndarray]:
    """Load sensor data from a text file. Returns array with columns:
    [ACC_TIME, ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z]
    Handles files where GYRO_Z may be named 'GYRO_' and files with repeated headers.
    Applies Butterworth filtering when requested.
    """
    try:
        with open(file_path, 'r') as f:
            header_line = f.readline().strip()
        if 'ACC_TIME' not in header_line:
            return None

        columns = header_line.split('\t')
        columns = [c.strip() for c in columns]
        if 'GYRO_Z' not in columns and 'GYRO_' in columns:
            columns[columns.index('GYRO_')] = 'GYRO_Z'

        df = pd.read_csv(file_path, sep='\t', names=columns, header=0,
                         on_bad_lines='skip', low_memory=False)

        needed = ['ACC_TIME', 'ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z']
        df = df[needed]
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna().reset_index(drop=True)

        df['ACC_TIME'] = df['ACC_TIME'].astype(np.int64)
        for col in needed[1:]:
            df[col] = df[col].astype(np.float32)

        df = df[df['ACC_TIME'] > 0].reset_index(drop=True)
        df = df.sort_values('ACC_TIME').reset_index(drop=True)

        data = df.values  # shape: (N, 7)

        # Smooth the IMU channels while keeping the timestamp unchanged.
        if apply_filter and len(data) > 100:
            data[:, 1:] = butterworth_filter(data[:, 1:])

        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def load_gold_labels(gold_path: str) -> pd.DataFrame:
    """Load gold standard labels from CSV or Excel."""
    ext = os.path.splitext(gold_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(gold_path, encoding="utf-8-sig")
    else:
        df = pd.read_excel(gold_path)

    required = ['user_id', 'category', 'start', 'end']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required annotation columns in {gold_path}: {missing}")

    df['user_id'] = df['user_id'].astype(str)
    df['start'] = df['start'].astype(np.int64)
    df['end'] = df['end'].astype(np.int64)
    return df


# ==================== Hand-Crafted Features ======================================

def compute_features_for_window(window: np.ndarray) -> np.ndarray:
    """Compute hand-crafted features for a single window of IMU data.

    Input: window shape (W, 6) - [ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z]
    Output: feature vector (62-d)
    """
    features = []

    # Per-channel statistical features (6 channels x 8 features = 48)
    for ch in range(6):
        signal = window[:, ch]
        features.append(np.mean(signal))
        features.append(np.std(signal))
        features.append(np.min(signal))
        features.append(np.max(signal))
        features.append(np.median(signal))
        features.append(np.mean(signal ** 2))  # Energy
        zcr = np.sum(np.abs(np.diff(np.sign(signal - np.mean(signal))))) / (2 * len(signal))
        features.append(zcr)  # Zero crossing rate
        features.append(np.max(signal) - np.min(signal))  # Peak-to-peak

    acc = window[:, :3]
    gyro = window[:, 3:]

    # ACC/GYRO magnitude (6 features)
    acc_mag = np.sqrt(np.sum(acc ** 2, axis=1))
    features.extend([np.mean(acc_mag), np.std(acc_mag), np.max(acc_mag)])
    gyro_mag = np.sqrt(np.sum(gyro ** 2, axis=1))
    features.extend([np.mean(gyro_mag), np.std(gyro_mag), np.max(gyro_mag)])

    # Signal Magnitude Area captures overall movement intensity.
    sma_acc = np.mean(np.sum(np.abs(acc), axis=1))
    sma_gyro = np.mean(np.sum(np.abs(gyro), axis=1))
    features.extend([sma_acc, sma_gyro])

    # Axis correlations (3 features)
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        if np.std(acc[:, i]) > 0 and np.std(acc[:, j]) > 0:
            features.append(np.corrcoef(acc[:, i], acc[:, j])[0, 1])
        else:
            features.append(0.0)

    # Frequency-domain features from acceleration magnitude.
    fft_vals = np.abs(np.fft.rfft(acc_mag))
    fft_freqs = np.fft.rfftfreq(len(acc_mag), d=1.0 / ACC_GYRO_RATE)
    if len(fft_vals) > 1:
        dominant_idx = np.argmax(fft_vals[1:]) + 1
        features.append(fft_freqs[dominant_idx])  # Dominant frequency
        features.append(fft_vals[dominant_idx])  # Dominant power
        # Spectral energy bands capture activity-specific frequency content.
        features.append(np.sum(fft_vals[(fft_freqs >= 0.5) & (fft_freqs < 3)]))  # jump rope range
        features.append(np.sum(fft_vals[(fft_freqs >= 3) & (fft_freqs < 10)]))   # mid freq
        features.append(np.sum(fft_vals[(fft_freqs >= 10) & (fft_freqs < 25)]))  # high freq
    else:
        features.extend([0.0] * 5)

    return np.array(features, dtype=np.float32)


HANDCRAFT_DIM = 62  # Number of hand-crafted features


# ==================== Data Augmentation ==========================================

def augment_amplitude_scale(window: np.ndarray) -> np.ndarray:
    """Scale amplitudes to simulate subject-level sensor-intensity variation."""
    scale = np.random.uniform(*AUG_AMPLITUDE_SCALE_RANGE)
    return window * scale


def augment_gaussian_noise(window: np.ndarray) -> np.ndarray:
    """Add Gaussian noise at the configured signal-to-noise ratio."""
    signal_power = np.mean(window ** 2, axis=0)
    snr_linear = 10 ** (AUG_NOISE_SNR_DB / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.randn(*window.shape) * np.sqrt(noise_power + 1e-10)
    return window + noise.astype(np.float32)


def augment_time_shift(window: np.ndarray, max_shift: int = 30) -> np.ndarray:
    """Randomly shift the window along the time axis."""
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(window, shift, axis=0)


def augment_time_warp(window: np.ndarray) -> np.ndarray:
    """Time stretch/compress by 0.9x-1.1x then resample back to original length.
    Uses scipy.interpolate.interp1d on the time axis.
    """
    orig_len = window.shape[0]
    stretch_factor = np.random.uniform(*AUG_TIME_STRETCH_RANGE)
    stretched_len = int(orig_len * stretch_factor)
    if stretched_len < 2:
        return window

    # Create interpolation function for each channel
    x_orig = np.linspace(0, 1, orig_len)
    x_stretched = np.linspace(0, 1, stretched_len)

    # Stretch the signal
    warped = np.zeros((stretched_len, window.shape[1]), dtype=np.float32)
    for ch in range(window.shape[1]):
        f = interp1d(x_orig, window[:, ch], kind='linear', fill_value='extrapolate')
        warped[:, ch] = f(x_stretched)

    # Resample back to original length
    x_warped = np.linspace(0, 1, stretched_len)
    x_target = np.linspace(0, 1, orig_len)
    result = np.zeros_like(window)
    for ch in range(window.shape[1]):
        f = interp1d(x_warped, warped[:, ch], kind='linear', fill_value='extrapolate')
        result[:, ch] = f(x_target)

    return result.astype(np.float32)


def apply_augmentation(window: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Apply random augmentation to a window.
    Each augmentation applied with probability p.
    """
    aug = window.copy()
    if np.random.random() < p:
        aug = augment_amplitude_scale(aug)
    if np.random.random() < p:
        aug = augment_gaussian_noise(aug)
    if np.random.random() < p * 0.5:  # less frequent
        aug = augment_time_shift(aug)
    if np.random.random() < p * 0.5:
        aug = augment_time_warp(aug)
    return aug


# ==================== Window Creation ============================================

def create_windows(data: np.ndarray, window_size: int, step_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding windows from sensor data.

    Input: data shape (N, 7) - [ACC_TIME, ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z]
    Returns: (timestamps, imu_windows)
        timestamps: center timestamp for each window
        imu_windows: shape (num_windows, window_size, 6)
    """
    N = len(data)
    timestamps = []
    windows = []

    for start in range(0, N - window_size + 1, step_size):
        end = start + window_size
        window_data = data[start:end, 1:]  # exclude timestamp column
        center_time = data[start + window_size // 2, 0]
        timestamps.append(center_time)
        windows.append(window_data)

    if not windows:
        return np.array([]), np.array([])

    return np.array(timestamps), np.stack(windows)


def normalize_imu(windows: np.ndarray,
                  mean: Optional[np.ndarray] = None,
                  std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize IMU windows using train-set channel statistics."""
    if mean is None:
        flat = windows.reshape(-1, windows.shape[-1])
        mean = np.mean(flat, axis=0)
        std = np.std(flat, axis=0) + 1e-8

    normalized = (windows - mean) / std
    return normalized, mean, std


def assign_window_labels(timestamps: np.ndarray,
                         labels: pd.DataFrame,
                         user_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """Assign labels to windows based on timestamps.

    Returns:
        binary_labels: 0 (background) or 1 (activity)
        class_labels: -1 (background) or 0-4 (activity class)
    """
    user_labels = labels[labels['user_id'] == user_id]

    binary = np.zeros(len(timestamps), dtype=np.int64)
    classes = np.full(len(timestamps), -1, dtype=np.int64)

    for _, row in user_labels.iterrows():
        start_ts = row['start']
        end_ts = row['end']
        cls_idx = ACTIVITY_TO_IDX[row['category']]
        mask = (timestamps >= start_ts) & (timestamps <= end_ts)
        binary[mask] = 1
        classes[mask] = cls_idx

    return binary, classes
