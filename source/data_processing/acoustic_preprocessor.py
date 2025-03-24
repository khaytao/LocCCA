import scipy
import numpy as np
import librosa


# def preprocess_audio_array(x1: np.ndarray, x2:np.ndarray, fs: int, n_fft: int = 1024, hop_length: int = 512, window: str = "hann"):
#     N = min(len(x1), len(x2))
#     f, t, z1 = scipy.signal.stft(x1[:N], fs=fs, nperseg=n_fft, noverlap=n_fft - hop_length, window=window)
#     _, _, z2 = scipy.signal.stft(x2[:N], fs=fs, nperseg=n_fft, noverlap=n_fft - hop_length, window=window)

#     prp = np.angle(z2) - np.angle(z1)

#     mean_amplitude = (np.abs(z1) + np.abs(z2)) / 2

#     return mean_amplitude.T, prp.T  # Assume Data is N x d

def preprocess_audio_array(x: np.ndarray, fs: int, n_fft: int = 1024, hop_length: int = 512, window: str = "hann"):
    z = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window)

    # Split into even (left) and odd (right) channels
    z1 = z[::2, :, :]  # Even channels (left mics)
    z2 = z[1::2, :, :]  # Odd channels (right mics)

    prp = np.angle(z2) - np.angle(z1)

    mean_amplitude = (np.abs(z1) + np.abs(z2)) / 2

    # # Reshape prp and mean_amplitude to combine channels and frequency bins
    C, F, T = prp.shape  # channels x frequencies x time
    # prp_reshaped = np.transpose(prp, (2, 0, 1)).reshape(T, C * F)
    # mean_amplitude_reshaped = np.transpose(mean_amplitude, (2, 0, 1)).reshape(T, C * F)
    #
    # Update variables to return
    prp = np.transpose(prp, (2, 0, 1))
    mean_amplitude = np.transpose(mean_amplitude, (2, 0, 1))

    return mean_amplitude, prp  # First Dimension is samples. Each sample is the time index. Each vector is a 2d image, of frequency f from mic array c
