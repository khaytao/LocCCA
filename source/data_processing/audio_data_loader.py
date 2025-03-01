import pyroomacoustics as pra
import os
from scipy.io import wavfile
import librosa

# Works, but data is too noisy
# def download_google_speech(basedir):
#     dataset = pra.datasets.GoogleSpeechCommands(download=True, subset=10, seed=0)
#
#     print(dataset)
#     dataset.head(n=10)
#     print("All sounds in the dataset:")
#     print(dataset.classes)
#


def load_all_wavs_in_dir(data_path, fs=16000):
    signals = []
    filenames = []
    for f in os.listdir(data_path):
        if f.lower().endswith(".wav"):
            # fs, x = wavfile.read(os.path.join(data_path, f))
            x, _ = librosa.load(os.path.join(data_path, f), sr=fs)
            signals.append(x)
            filenames.append(f)

    return signals, filenames


if __name__ == '__main__':
    # download_google_speech(".")
    load_all_wavs_in_dir(r"C:\Projects\School\unsupervised_project\LocCCA\data\TIMIT\sample_dataset")
