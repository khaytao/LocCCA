import pyroomacoustics as pra
import numpy as np
from source.data_processing.location_sampler import LocationSampler
from source.data_processing.audio_data_loader import load_all_wavs_in_dir
from tqdm import tqdm
from scipy.io.wavfile import write


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
for i, speaker_location in enumerate(tqdm(speaker_locations)):
    room.add_source(position=np.array([speaker_location[0], speaker_location[1], height]), signal=signals[i % num_signals])

    room.simulate()

    write(f"data/generated/{filenames[i]}_{speaker_location[0]}_{speaker_location[1]}_left_mic.wav", fs, room.mic_array.signals[0,:])
    write(f"data/generated/{filenames[i]}_{speaker_location[0]}_{speaker_location[1]}_right_mic.wav", fs, room.mic_array.signals[1,:])