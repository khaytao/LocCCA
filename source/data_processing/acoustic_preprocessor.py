import scipy
import numpy as np


def preprocess_audio_array(x1: np.ndarray, x2:np.ndarray, fs: int, n_fft: int = 1024, hop_length: int = 512, window: str = "hann"):
    N = min(len(x1), len(x2))
    f, t, z1 = scipy.signal.stft(x1[:N], fs=fs, nperseg=n_fft, noverlap=n_fft - hop_length, window=window)
    _, _, z2 = scipy.signal.stft(x2[:N], fs=fs, nperseg=n_fft, noverlap=n_fft - hop_length, window=window)

    prp = np.angle(z2) - np.angle(z1)

    mean_amplitude = (np.abs(z1) + np.abs(z2)) / 2

    return mean_amplitude.T, prp.T  # Assume Data is N x d
