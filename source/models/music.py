import numpy as np
import torch
import pandas as pd
from scipy.io import wavfile
import os
from tqdm import tqdm

from speechbrain.processing.multi_mic import Music
from speechbrain.processing.features import STFT
from speechbrain.processing.multi_mic import Covariance
from speechbrain.processing.multi_mic import SrpPhat

from .model_base import LocalizationModel


class SoundLocalization(LocalizationModel):
    """
    Implementation of sound source localization using SpeechBrain.
    Supports both MUSIC and SRP-PHAT algorithms.
    """

    def __init__(self, fs=16000, speaker_hight=0, room_size=(6, 6, 6), algorithm='music', device='cpu'):
        """
        Initialize the localization model.
        
        Args:
            fs (int): Sample rate for STFT
            speaker_hight (float): Height of the speakers
            algorithm (str): Algorithm to use - either 'music' or 'srp-phat'
            device (str): Device to run model on - 'cpu' or 'cuda'
        """
        self.fs = fs
        self.stft = STFT(sample_rate=fs)
        self.sp_model = None
        self.cov = Covariance()
        self.speaker_hight = speaker_hight
        self.room_size = room_size
        self.algorithm = algorithm.lower()
        self.device = device
        self.mic_positions = None
    
    def get_mic_positions(self):
        return self.mic_positions
    
    def preprocess(self, data_dir: str):
        """
        Preprocess audio data from directory.
        
        Args:
            data_dir (str): Path to directory containing audio files
            
        Returns:
            tuple: (preprocessed_data, locations)
                - preprocessed_data: Audio data ready for processing
                - locations: Nx2 array of [x,y] speaker locations
        """

        # Read the metadata Excel file
        info_df = pd.read_excel(os.path.join(data_dir, "recordings_info.xlsx"))

        # Filter for scenario 0
        scenario_0_df = info_df[info_df['scenario_index'] == 0]

        # Extract mic locations and create array with height
        mic_x = scenario_0_df['mic_location_x'].values
        mic_y = scenario_0_df['mic_location_y'].values
        mic_z = np.ones(len(scenario_0_df)) * self.speaker_hight
        mic_array = np.stack([mic_x, mic_y, mic_z])
        # Normalize mic positions using same function as speaker locations
        norm_x, norm_y = self.normalize_coordinates(mic_array[0], mic_array[1])
        mic_array[0] = norm_x
        mic_array[1] = norm_y
        mic_positions = torch.from_numpy(mic_array).T.float()  # Transpose to get Nx3 shape

        # Initialize lists to store data
        audio_data = []
        locations = []

        # Group by scenario index to process each scenario
        for scenario_idx, scenario_group in info_df.groupby('scenario_index'):
            # Sort by mic_index to ensure consistent ordering
            scenario_group = scenario_group.sort_values('mic_index')

            # Load audio for each mic
            scenario_signals = []
            for mic_idx, row in scenario_group.iterrows():
                wav_path = os.path.join(data_dir, row['filename'])
                _, signal = wavfile.read(wav_path)
                scenario_signals.append(torch.from_numpy(signal))

            # Pad/trim signals to match longest signal in scenario
            max_length = max(len(sig) for sig in scenario_signals)
            padded_signals = []
            for signal in scenario_signals:
                if len(signal) < max_length:
                    # Pad shorter signals with zeros
                    padded_signal = torch.zeros(max_length, dtype=torch.float32)
                    padded_signal[:len(signal)] = signal
                    padded_signals.append(padded_signal)
                else:
                    # Trim longer signals
                    padded_signals.append(signal[:max_length])

            # Stack signals for this scenario
            scenario_tensor = torch.stack(padded_signals, dim=-1)  # Shape: [samples, mics]
            audio_data.append(scenario_tensor.float())

            # Store location (only need one row since x,y is same for all mics in scenario)
            # Get original coordinates
            orig_x = scenario_group.iloc[0]['x']
            orig_y = scenario_group.iloc[0]['y']

            # norm_x, norm_y = self.normalize_coordinates(orig_x, orig_y)

            locations.append(torch.tensor([orig_x, orig_y], dtype=torch.float32))

        # Convert lists to tensors
        locations = torch.stack(locations)

        # Store original room size for inverse transform
        self._room_size = torch.tensor(self.room_size[:2], dtype=torch.float32)

        # Initialize the spatial processing model based on algorithm choice
        if self.algorithm == 'music':
            self.sp_model = Music(mics=mic_positions)
        elif self.algorithm == 'srp-phat':
            self.sp_model = SrpPhat(mics=mic_positions)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}. Use 'music' or 'srp-phat'.")

        denorm_x, denorm_y = self.denormalize_coordinates(mic_positions[:,0], mic_positions[:,1])
        self.mic_positions = torch.column_stack((denorm_x, denorm_y))
        return audio_data, locations

    def process(self, data):
        """
        Apply localization algorithm to localize sound sources.
        
        Args:
            data: List of audio tensors, each of shape (channels, samples)
            
        Returns:
            List of estimated DOAs (direction of arrival) for detected sources
        """
        results = []
        for audio in tqdm(data):
            if not isinstance(audio, torch.Tensor):
                audio = torch.from_numpy(audio)

            # Add batch dimension
            audio = audio.unsqueeze(0)  # Shape: (1, channels, samples)

            # Move to device
            audio = audio.to(self.device)

            # Compute spatial spectrum
            Xs = self.stft(audio)
            XXs = self.cov(Xs)
            doa = self.sp_model(XXs)

            # Move to CPU and drop third element
            doa = doa.cpu().squeeze()[:,:2]  # Keep only x,y coordinates
            
            # Average across K predictions to get single (x,y) point
            avg_doa = doa.mean(dim=0)
            
            # Denormalize coordinates
            denorm_x, denorm_y = self.denormalize_coordinates(avg_doa[0], avg_doa[1])
            doa = torch.tensor([denorm_x, denorm_y], dtype=torch.float32)
            
            results.append(doa)

        return results

    def normalize_coordinates(self, x, y):
        """
        Normalize coordinates to [-1, 1] range relative to room center.
        
        Args:
            x: Original x coordinate
            y: Original y coordinate
            
        Returns:
            Tuple of (normalized_x, normalized_y)
        """
        # Shift coordinates to be relative to room center
        centered_x = x - (self.room_size[0] / 2)
        centered_y = y - (self.room_size[1] / 2)

        # Normalize to [-1, 1] range
        norm_x = centered_x / (self.room_size[0] / 2)
        norm_y = centered_y / (self.room_size[1] / 2)

        return norm_x, norm_y

    def denormalize_coordinates(self, norm_x, norm_y):
        """
        Denormalize coordinates from [-1, 1] range to original room size.
        
        Args:
            norm_x: Normalized x coordinate 
            norm_y: Normalized y coordinate
            
        Returns:
            Tuple of (denormalized_x, denormalized_y)
        """
        # Shift coordinates to be relative to room center   
        denorm_x = norm_x * (self.room_size[0] / 2) + (self.room_size[0] / 2)
        denorm_y = norm_y * (self.room_size[1] / 2) + (self.room_size[1] / 2)

        return denorm_x, denorm_y


if __name__ == '__main__':
    from eval_model import evaluate_localization

    path = r"../../data/generated/TIMIT_sample"

    model = SoundLocalization(algorithm='srp-phat')

    X, y = model.preprocess(path)

    predictions = model.process(X)

    # Average over N dimension for each prediction
    avg_predictions = [p.mean(dim=1).squeeze() for p in predictions]

    # Create scatter plot
    import matplotlib.pyplot as plt

    # Extract x,y coordinates (ignoring last dimension)
    xs = [p[0].item() for p in avg_predictions]
    ys = [p[1].item() for p in avg_predictions]

    # Denormalize coordinates
    # denorm_xs = [model.denormalize_coordinates(x, y)[0] for x, y in zip(xs, ys)]
    # denorm_ys = [model.denormalize_coordinates(x, y)[1] for x, y in zip(xs, ys)]
    denorm_xs = xs
    denorm_ys = ys

    plt.figure()
    # Plot predictions
    plt.scatter(denorm_xs, denorm_ys, c='blue', marker='o', label='Predictions')

    # Plot ground truth
    true_xs = y[:, 0].numpy()
    true_ys = y[:, 1].numpy()
    plt.scatter(true_xs, true_ys, c='red', marker='^', label='Ground Truth')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Predictions vs Ground Truth')
    plt.grid(True)
    plt.legend()
    plt.show()

    permitted_distance_meters = 0.1
    threshold = permitted_distance_meters / 3  # normalized
    num_failures, mse = evaluate_localization(np.array(avg_predictions)[:, :2], np.array(y), threshold)
    print(f"Number of failures: {num_failures}")
    print(f"MSE: {mse}")
