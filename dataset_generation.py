import pyroomacoustics as pra
import numpy as np
from tqdm import tqdm
from scipy.io.wavfile import write
import csv

from source.data_processing.location_sampler import LocationSampler
from source.data_processing.audio_data_loader import load_all_wavs_in_dir
from source.data_processing.acoustic_preprocessor import preprocess_audio_array


def save_dataset_to_file(dataset, filename):
    """
    Save a dataset to a CSV file.

    Parameters:
    dataset (tuple of tuples): The dataset to save, where the outer tuple represents samples and the inner tuples represent features.
    filename (str): The name of the file to save the dataset to.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for sample in dataset:
            writer.writerow(sample)


np.random.seed(0)
DATA_LOCATION = r"data/TIMIT/sample_dataset"

# Define room dimensions
room_dim = [6, 6, 6]
height = 1.8
fs = 16000
T60 = 0.2

mic_position = (0.5, 3)
d = 0.1

num_speakers = 100


def get_d_vector(coordinates, room_center=(3, 3)):
    x, y = np.abs(coordinates[0] - room_center[0]), np.abs(coordinates[1] - room_center[1])
    if y > x:
        return np.array([1, 0, 0])
    else:
        return np.array([0, 1, 0])


# Define absorption for walls to achieve T60 of 0.2 seconds
absorption, max_order = pra.inverse_sabine(rt60=T60, room_dim=room_dim)

# Create the room
room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(absorption), max_order=max_order)

# add microphones


array_distance = 0.2
mic_height = 1.8

mic_array_location = np.array([mic_position[0], mic_position[1], mic_height])
mic_1_pos = mic_array_location - (get_d_vector(mic_position) * array_distance)
mic_2_pos = mic_array_location + (get_d_vector(mic_position) * array_distance)
mic_positions = np.c_[mic_1_pos, mic_2_pos]
room.add_microphone_array(pra.MicrophoneArray(mic_positions, room.fs))

location_sampler = LocationSampler(room_dim[0], room_dim[1])

speaker_locations = location_sampler.sample_uniform(num_speakers)
signals, filenames = load_all_wavs_in_dir(DATA_LOCATION)

num_signals = len(signals)
X_angle = []
X_amplitude = []
Y_list = []
for i, speaker_location in enumerate(tqdm(speaker_locations)):
    room.add_source(position=np.array([speaker_location[0], speaker_location[1], height]), signal=signals[i % num_signals])

    room.simulate()

    x1 = room.mic_array.signals[0,:]
    x2 = room.mic_array.signals[1,:]
    # write(f"data/generated/{filenames[i % num_signals]}_{speaker_location[0]}_{speaker_location[1]}_left_mic.wav", fs, x1)
    # write(f"data/generated/{filenames[i % num_signals]}_{speaker_location[0]}_{speaker_location[1]}_right_mic.wav", fs, x2)

    mean_amplitude, prp = preprocess_audio_array(x1, x2, fs)

    p = np.tile([speaker_location[0], speaker_location[1]], [prp.shape[0], 1])

    X_angle.append(prp)
    X_amplitude.append(mean_amplitude)
    Y_list.append(p)
    room.sources = []

combined_audio_feature_list = [np.concatenate((a, b), axis=1) for a, b in zip(X_amplitude, X_angle)]
X = np.concatenate(combined_audio_feature_list, axis=0)
Y = np.concatenate(Y_list, axis=0)

save_dataset_to_file(X, "X_train.csv")
save_dataset_to_file(Y, "Y_train.csv")
