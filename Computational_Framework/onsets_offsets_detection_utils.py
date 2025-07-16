import os
import numpy as np
import librosa as lb
import pandas as pd
import madmom
import glob
from scipy.io import wavfile
from scipy.signal import medfilt
from visualization import plot_signal
import matplotlib.pyplot as plt


MIN_DURATION =0.02 #seconds
MAX_DURATION = 0.54
AV_DURATION = 0.19


def median_filter(signal, kernel_size=11):
    """
    Apply a 1D median filter to a multi-channel signal.

    Parameters:
    - signal: The input signal (either a 1D or 2D NumPy array).
    - kernel_size: The size of the median filter kernel (must be an odd integer).

    Returns:
    - Filtered signal: A NumPy array with the same shape as the input signal.
    """
    if signal.ndim == 1:  # Mono signal (1D)
        return medfilt(signal, kernel_size=kernel_size)
    else:  # Multi-channel signal (2D or more)
        return np.array([medfilt(channel, kernel_size=kernel_size) for channel in signal])


    

def global_shift_correction(predicted_onsets, shift):
    '''subtract shift second to all the predicted onsets.
    Args:
        predicted_onsets (list): List of predicted onsets.
        shift (float): Global shift in seconds.
    Returns:
        list: Corrected predicted onsets.
    '''
    # compute global shift
    corrected_predicted_onsets = []
    for po in predicted_onsets:
        #subtract a global shift of 0.01 ms or more  to all the predicted onsets
        if po - shift > 0: # to avoid negative onsets
            corrected_predicted_onsets.append(po - shift)
        else:
            continue

    return np.array(corrected_predicted_onsets)


def normalise_audio(file_name):
    '''Normalise the audio file to have a maximum amplitude of 1.
    Args:
        file_name (str): Path to the audio file.
    Returns:
        np.array: Normalised audio signal.
    '''
    # Read the audio file
    sample_rate, audio_data = wavfile.read(file_name)

    # Compute the normalization factor (Xmax)
    max_amplitude = np.max(np.abs(audio_data))

    # Normalize the audio
    normalized_audio = audio_data / max_amplitude

    return normalized_audio, sample_rate


def double_onset_correction(onsets_predicted, correction= 0.020):
    '''Correct double onsets by removing onsets which are less than a given threshold in time.
    Args:
        onsets_predicted (list): List of predicted onsets.
        gt_onsets (list): List of ground truth onsets.
        correction (float): Threshold in seconds.
    Returns:
        list: Corrected predicted onsets.
    '''    
    # Calculate interonsets difference
    #gt_onsets = np.array(gt_onsets, dtype=float)

    # Calculate the difference between consecutive onsets
    differences = np.diff(onsets_predicted)

    # Create a list to add the filtered onset and add a first value

    filtered_onsets = [onsets_predicted[0]]  #Add the first onset

    # Subtract all the onsets which are less than fixed threshold in time
    for i, diff in enumerate(differences):
      if diff >= correction:
      # keep the onset if the difference is more than the given selected time
        filtered_onsets.append(onsets_predicted[i + 1])
        #print the number of onsets predicted after correction
    return np.array(filtered_onsets)




def filter_calls_within_experiment(onsets, offsets, start_exp, end_exp):
    """
    Filters onsets and offsets between start and end of experiment, ensuring paired consistency.
    
    Parameters:
        onsets (list or np.array): List of onset times in seconds.
        offsets (list or np.array): List of offset times in seconds.
        start_exp (float): Start time of the experiment in seconds.
        end_exp (float): End time of the experiment in seconds.

    Returns:
        np.array: Filtered onsets.
        np.array: Filtered offsets.
    """
    filtered_onsets = []
    filtered_offsets = []
    
    for onset, offset in zip(onsets, offsets):
        # Check if both onset and offset are within the experimental window
        if start_exp <= onset <= end_exp and start_exp <= offset <= end_exp:
            filtered_onsets.append(onset)
            filtered_offsets.append(offset)
    
    return np.array(filtered_onsets), np.array(filtered_offsets)
