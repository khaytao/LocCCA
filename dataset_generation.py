import pyroomacoustics as pra
import numpy as np
from tqdm import tqdm
from scipy.io.wavfile import write
import csv

from source.data_processing.location_sampler import LocationSampler
from source.data_processing.audio_data_loader import load_all_wavs_in_dir
from source.data_processing.acoustic_preprocessor import preprocess_audio_array


def get_candidates(room_dim, resolution):
    """
    Generate a grid of candidate points by splitting each room dimension into equal segments.
    
    Parameters:
    room_dim (list): List containing room dimensions [x, y, z]
    resolution (int): Number of segments to split each axis into
    
    Returns:
    numpy.ndarray: Array of shape (resolution^2, 2) containing all candidate (x,y) coordinates
    """
    # Create evenly spaced points along each axis
    x_points = np.linspace(0, room_dim[0], resolution)
    y_points = np.linspace(0, room_dim[1], resolution)
    
    # Create a mesh grid of all combinations
    X, Y = np.meshgrid(x_points, y_points)
    
    # Stack and reshape to get array of (x,y) coordinates
    candidates = np.column_stack((X.flatten(), Y.flatten()))
    
    return candidates



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
# DATA_LOCATION = r"data/TIMIT/sample_dataset"
DATA_LOCATION = r"data/sine"

# Define room dimensions
room_dim = [6, 6, 6]
height = 1.8
fs = 16000
T60 = 0.2

mic_position = (0.5, 3)
d = 0.1

num_speakers = 100
DATASET_NAME = "toy"
ROOM_RESOLUTION = 10


def get_d_vector(coordinates, room_center=(3, 3)):
    x, y = np.abs(coordinates[0] - room_center[0]), np.abs(coordinates[1] - room_center[1])
    if y > x:
        return np.array([1, 0, 0])
    else:
        return np.array([0, 1, 0])


# Define absorption for walls to achieve T60 of 0.2 seconds
absorption, max_order = pra.inverse_sabine(rt60=T60, room_dim=room_dim)

# Create the room
# room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(absorption), max_order=max_order)

# add microphones


array_distance = 0.2
mic_height = 1.8

mic_array_location = np.array([mic_position[0], mic_position[1], mic_height])
mic_1_pos = mic_array_location - (get_d_vector(mic_position) * array_distance)
mic_2_pos = mic_array_location + (get_d_vector(mic_position) * array_distance)
mic_positions = np.c_[mic_1_pos, mic_2_pos]
# room.add_microphone_array(pra.MicrophoneArray(mic_positions, room.fs))

location_sampler = LocationSampler(room_dim[0], room_dim[1])

speaker_locations = location_sampler.sample_uniform(num_speakers)
signals, filenames = load_all_wavs_in_dir(DATA_LOCATION)

num_signals = len(signals)
X_angle = []
X_amplitude = []
Y_list = []

p = [0, 0, 45, 45]
A = [2, 4, 2, 4]
# Convert polar coordinates (amplitude, angle) to cartesian coordinates relative to mic_position
cartesian_coords = np.zeros((len(p), 2))
for i in range(len(p)):
    # Convert angle to radians
    theta = np.radians(p[i])
    # Convert polar to cartesian coordinates and offset by mic_position
    cartesian_coords[i,0] = mic_positions[0,0] + A[i] * np.cos(theta)
    cartesian_coords[i,1] = mic_positions[1,0] + A[i] * np.sin(theta)


sp = cartesian_coords
# speaker_locations = sp #todo remove
for i, speaker_location in enumerate(tqdm(speaker_locations)):
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(absorption), max_order=max_order)
    room.add_microphone_array(pra.MicrophoneArray(mic_positions, room.fs))

    room.add_source(position=np.array([speaker_location[0], speaker_location[1], height]), signal=signals[i % num_signals])

    room.simulate()

    x1 = room.mic_array.signals[0,:]
    x2 = room.mic_array.signals[1,:]
    write(f"data/generated/{filenames[i % num_signals]}_{speaker_location[0]}_{speaker_location[1]}_left_mic.wav", fs, x1)
    write(f"data/generated/{filenames[i % num_signals]}_{speaker_location[0]}_{speaker_location[1]}_right_mic.wav", fs, x2)

    mean_amplitude, prp = preprocess_audio_array(x1, x2, fs)

    p = np.tile([speaker_location[0], speaker_location[1]], [prp.shape[0], 1])

    X_angle.append(prp)
    X_amplitude.append(mean_amplitude)
    Y_list.append(p)
    # room.sources.pop()

combined_audio_feature_list = [np.concatenate((a, b), axis=1) for a, b in zip(X_amplitude, X_angle)]
X = np.concatenate(combined_audio_feature_list, axis=0)
Y = np.concatenate(Y_list, axis=0)

candidates = get_candidates(room_dim, ROOM_RESOLUTION)

save_dataset_to_file(candidates, f"candidates_{DATASET_NAME}.csv")

# train_test_string = "train" if TRAIN else "test"
save_dataset_to_file(X, f"X_{DATASET_NAME}.csv")
save_dataset_to_file(Y, f"Y_{DATASET_NAME}.csv")