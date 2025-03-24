import pyroomacoustics as pra
import numpy as np
from tqdm import tqdm
from scipy.io.wavfile import write
import csv
import pandas as pd
import matplotlib.pyplot as plt
import os

from source.data_processing.location_sampler import LocationSampler
from source.data_processing.audio_data_loader import load_all_wavs_in_dir
from source.data_processing.acoustic_preprocessor import preprocess_audio_array

DEBUG = True
DELETE_DIR = True  # if true, generated data will overwrite the previous under the same data_name

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


def get_d_vector(coordinates, room_center=(3, 3)):
    x, y = np.abs(coordinates[0] - room_center[0]), np.abs(coordinates[1] - room_center[1])
    if y > x:
        return np.array([1, 0, 0])
    else:
        return np.array([0, 1, 0])


def main(args, mic_locations):
    # Define absorption for walls to achieve T60 of 0.2 seconds
    absorption, max_order = pra.inverse_sabine(rt60=args.t60, room_dim=args.room_dim)

    array_distance = args.array_distance
    mic_height = args.mic_height

    mic_positions = []
    for i in range(len(mic_locations)):
        mic_array_location = np.array([mic_locations[i][0], mic_locations[i][1], mic_height])
        mic_1_pos = mic_array_location - (get_d_vector(mic_array_location) * array_distance)
        mic_2_pos = mic_array_location + (get_d_vector(mic_array_location) * array_distance)
        mic_positions.extend([mic_1_pos, mic_2_pos])
    mic_positions = np.array(mic_positions).T  # Convert to 2xN array

    # Debug flag for visualization

    if DEBUG:
        # Plot microphone positions
        plt.figure()
        plt.scatter(mic_positions[0, :], mic_positions[1, :], c='red', marker='^', s=100, label='Microphones')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('Microphone Array Positions')
        plt.grid(True)
        plt.legend()
        plt.xlim(0, args.room_dim[0])
        plt.ylim(0, args.room_dim[1])
        plt.show()

    location_sampler = LocationSampler(args.room_dim[0], args.room_dim[1])
    if args.distribution == 'uniform':
        speaker_locations = location_sampler.sample_uniform(args.num_speakers)
    elif args.distribution == 'gaussian':
        speaker_locations = location_sampler.sample_gaussians(args.centroids, args.std, args.num_speakers)
    else:
        raise ValueError("Distribution must be either 'uniform' or 'gaussian'")
    speaker_locations = location_sampler.sample_uniform(args.num_speakers)
    signals, filenames = load_all_wavs_in_dir(args.data_location)

    num_signals = len(signals)
    X_angle = []
    X_amplitude = []
    Y_list = []

    recording_df_columns = ['filename', 'x', 'y', 'mic_index', 'mic_location_x', 'mic_location_y']
    recordings_info_rows = []
    for i, speaker_location in enumerate(tqdm(speaker_locations)):
        room = pra.ShoeBox(args.room_dim, fs=args.fs, materials=pra.Material(absorption), max_order=max_order)
        room.add_microphone_array(pra.MicrophoneArray(mic_positions, room.fs))

        room.add_source(position=np.array([speaker_location[0], speaker_location[1], args.height]),
                        signal=signals[i % num_signals])

        room.simulate()

        savedir = f"data/generated/{args.dataset_name}/"
        if os.path.exists(savedir):
            for file in os.listdir(savedir):
                file_path = os.path.join(savedir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error: {e}")
        os.makedirs(savedir, exist_ok=True)
        for mic_idx in range(room.mic_array.signals.shape[0]):
            # Even indices are left mics, odd indices are right mics
            mic_type = "left_mic" if mic_idx % 2 == 0 else "right_mic"
            mic_signal = room.mic_array.signals[mic_idx, :]
            save_name = f"{filenames[i % num_signals]}_{speaker_location[0]}_{speaker_location[1]}_{mic_idx // 2}_{mic_type}.wav"


            write(f"{savedir}/{save_name}",
                  args.fs, mic_signal)
            recordings_info_rows.append({
                'filename': save_name,
                'x': speaker_location[0],
                'y': speaker_location[1],
                'mic_index': mic_idx,
                'mic_location_x': mic_positions[0, mic_idx],
                'mic_location_y': mic_positions[1, mic_idx]
            })

        recordings_info = pd.DataFrame(recordings_info_rows, columns=recording_df_columns)
        # Add hyperlinks to wav files
        recordings_info['filename'] = recordings_info['filename'].apply(
            lambda x: f'=HYPERLINK("{x}","{x}")'
        )
        while True:
            try:
                recordings_info.to_excel(f"{savedir}/recordings_info.xlsx", index=False)
                break
            except PermissionError:
                input("Excel file is open. Please close it and press Enter to try again...")
        
        x = room.mic_array.signals
        mean_amplitude, prp = preprocess_audio_array(x, args.fs)
        #
        # p = np.tile([speaker_location[0], speaker_location[1]], [prp.shape[0], 1])
        #
        # X_angle.append(prp)
        # X_amplitude.append(mean_amplitude)
        # Y_list.append(p)

    # combined_audio_feature_list = [np.concatenate((a, b), axis=1) for a, b in zip(X_amplitude, X_angle)]
    # X = np.concatenate(combined_audio_feature_list, axis=0)
    # Y = np.concatenate(Y_list, axis=0)
    #
    # candidates = get_candidates(args.room_dim, args.room_resolution)
    #
    # save_dataset_to_file(candidates, f"candidates_{args.dataset_name}.csv")
    # save_dataset_to_file(X, f"X_{args.dataset_name}.csv")
    # save_dataset_to_file(Y, f"Y_{args.dataset_name}.csv")
    return


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Generate acoustic dataset')

    # Room configuration
    parser.add_argument('--room_dim', nargs=3, type=float, default=[6.0, 6.0, 6.0],
                        help='Room dimensions [x, y, z] in meters')
    parser.add_argument('--t60', type=float, default=0.2,
                        help='Reverberation time T60 in seconds')
    parser.add_argument('--fs', type=int, default=16000,
                        help='Sampling frequency in Hz')

    # Microphone configuration  

    # parser.add_argument('--mic_position', nargs=2, type=float, default=[0.5, 3.0],
    #                   help='Microphone array center position [x, y]')
    parser.add_argument('--mic_height', type=float, default=1.8,
                        help='Height of microphones in meters')
    parser.add_argument('--array_distance', type=float, default=0.2,
                        help='Distance between microphones in array')

    # Speaker/source configuration
    parser.add_argument('--height', type=float, default=1.8,
                        help='Height of sound sources in meters')
    parser.add_argument('--num_speakers', type=int, default=100,
                        help='Number of speaker positions to simulate')

    parser.add_argument('--distribution', type=str, default='uniform',
                        help='Distribution of speaker positions')
    parser.add_argument('--centroids', nargs=2, type=float, default=[3, 3],
                        help='Centroids of Gaussian distributions')
    parser.add_argument('--std', type=float, default=0.5,
                        help='Standard deviation of Gaussian distributions')

    # Dataset configuration
    parser.add_argument('--data_location', type=str, default='data\TIMIT\sample_dataset',
                        help='Directory containing input audio files')
    parser.add_argument('--dataset_name', type=str, default='TIMIT_sample',
                        help='Name for the output dataset files')
    parser.add_argument('--room_resolution', type=int, default=10,
                        help='Resolution for candidate position grid')

    args = parser.parse_args()

    with open('config/mic_location.json', 'r') as f:
        mic_locations = json.load(f)
    np.random.seed(0)
    main(args, mic_locations)
