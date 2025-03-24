import pyroomacoustics as pra
import numpy as np
from pyroom_loc import localize_sound
from source.data_processing.audio_data_loader import load_all_wavs_in_dir

def main(args):
    
    signals, filenames = load_all_wavs_in_dir(args.data_location)
    for signal in signals:
        angle = localize_sound(signal, args.mic_position)
        print(f"Estimated direction: {angle:.1f} degrees")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate acoustic dataset')

    # Room configuration
    parser.add_argument('--room_dim', nargs=3, type=float, default=[6.0, 6.0, 6.0],
                        help='Room dimensions [x, y, z] in meters')
    parser.add_argument('--t60', type=float, default=0.2,
                        help='Reverberation time T60 in seconds')
    parser.add_argument('--fs', type=int, default=16000,
                        help='Sampling frequency in Hz')

    # Microphone configuration  
    parser.add_argument('--mic_position', nargs=2, type=float, default=[0.5, 3.0],
                        help='Microphone array center position [x, y]')
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

    np.random.seed(0)
    main(args)
