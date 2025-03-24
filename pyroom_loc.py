import numpy as np
import pyroomacoustics as pra


def localize_sound(audio, mic_array_geometry, nfft=256, fs=16000):
    """
    Localize sound source using pyroomacoustics DOA algorithms
    
    Args:
        audio (ndarray): Audio signal array
        mic_array_geometry (ndarray): Microphone array geometry, shape (2,M) or (3,M)
        nfft (int): FFT size for MUSIC algorithm
        fs (int): Sampling frequency in Hz
        
    Returns:
        float: Estimated direction of arrival in degrees
    """
    # Convert to numpy array if needed
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio)

    if not isinstance(mic_array_geometry, np.ndarray):
        mic_array_geometry = np.array(mic_array_geometry)
    
    # Convert 1D array to column array if needed
    if len(mic_array_geometry.shape) == 1:
        mic_array_geometry = mic_array_geometry.reshape(-1, 1)
        
    # Assert mic array has correct dimensions
    assert mic_array_geometry.shape[0] in [2, 3], "Microphone array geometry must have 2 or 3 spatial dimensions"
   
    # Make sure audio is 2D array
    # if len(audio.shape) == 1:
    #     audio = np.expand_dims(audio, axis=1)
    
    # Create DOA object
    doa = pra.doa.MUSIC(mic_array_geometry, fs, nfft=nfft)
    
    # Perform localization
    doa.locate_sources(audio.T)
    
    # Get DOA estimates in degrees
    angles = doa.azimuth_recon * 180 / np.pi
    
    return angles[0] # Return first angle estimate

# Example usage:
# mic_array = np.array([[0, 0.05, 0.1], [0, 0, 0]])  # 3 mics in line
# angle = localize_sound(audio_array, mic_array)
# print(f"Estimated direction: {angle:.1f} degrees")
