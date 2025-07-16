import numpy as np
import os
import pandas as pd
import librosa as lb
import utils as ut
from typing import List, Optional
import soundfile as sf
import evaluation as my_eval
from matplotlib import pyplot as plt
from torch import Tensor
import torch as tr

def visualise_spectrogram_and_harmonics(spec, F0, F1, F2, sr, hop_length, chick_id):
        # Plot the linear spectrogram
    plt.figure(figsize=(13, 8))
    lb.display.specshow(lb.amplitude_to_db(spec, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', cmap='viridis', alpha=0.75)
    times = lb.times_like(F0, sr=sr, hop_length=hop_length)

    plt.plot(times, F0, label='Pitch (Hz)', color='red')
    # plt.plot(f0_pyin_lb_nan_zeros, label='Pitch (Hz)', color='red')

    plt.plot(times, F1, label='F1 (Hz)', color='blue')
    plt.plot(times, F2, label='F2 (Hz)', color='green')
    # plt.colorbar(format='%+2.0f dB')
    plt.title('Linear Spectrogram and F0, F1, F2')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(chick_id + '_spectrogram_F0_F1_F2.png'))
    
    return


def plot_scalogram(scalogram: Tensor,
                   sr: float,
                   y_coords: List[float],
                   title: str = "scalogram",
                   hop_len: int = 1,
                   cmap: str = "magma",
                   vmax: Optional[float] = None,
                   save_path: Optional[str] = None,
                   x_label: str = "Time (seconds)",
                   y_label: str = "Frequency (Hz)") -> None:
    """
    Plots a scalogram of the provided data.

    The scalogram is a visual representation of the wavelet transform of a signal over time.
    This function uses matplotlib and librosa to create the plot.

    Parameters:
        scalogram (T): The scalogram data to be plotted.
        sr (float): The sample rate of the audio signal.
        y_coords (List[float]): The y-coordinates for the scalogram plot.
        title (str, optional): The title of the plot. Defaults to "scalogram".
        hop_len (int, optional): The hop length for the time axis (or T). Defaults to 1.
        cmap (str, optional): The colormap to use for the plot. Defaults to "magma".
        vmax (Optional[float], optional): The maximum value for the colorbar. If None, the colorbar scales with the data. Defaults to None.
        save_path (Optional[str], optional): The path to save the plot. If None, the plot is not saved. Defaults to None.
        x_label (str, optional): The label for the x-axis. Defaults to "Time (seconds)".
        y_label (str, optional): The label for the y-axis. Defaults to "Frequency (Hz)".
    """
    assert scalogram.ndim == 2
    assert scalogram.size(0) == len(y_coords)
    x_coords = lb.times_like(scalogram.size(1), sr=sr, hop_length=hop_len)

    plt.figure(figsize=(10, 5))
    lb.display.specshow(scalogram.numpy(),
                             sr=sr,
                             x_axis="time",
                             x_coords=x_coords,
                             y_axis="cqt_hz",
                             y_coords=np.array(y_coords),
                             cmap=cmap,
                             vmax=vmax)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    if len(y_coords) < 12:
        ax = plt.gca()
        ax.set_yticks(y_coords)
    plt.minorticks_off()
    plt.title(title, fontsize=16)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()





def visualise_spectrogram_and_spectral_centroid(lin_spec, spectral_centroid_feature_calls, sr, hop_length, chick_id):
    # Plot the linear spectrogram
    plt.figure(figsize=(13, 8))
    lb.display.specshow(lb.amplitude_to_db(lin_spec, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', cmap='viridis', alpha=0.75)
    times = lb.times_like(spectral_centroid_feature_calls[0], sr=sr, hop_length=hop_length)
    plt.plot(times, spectral_centroid_feature_calls[0], label='Spectral Centroid', color='green')
    # plt.colorbar(format='%+2.0f dB')
    plt.title('Linear Spectrogram and Spectral Centroid')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    plt.savefig(chick_id + '_spectrogram_spectral_centroid.png')
    
    return  


def visualise_spectrogram_and_RMS(lin_spec, rms_features_calls, sr, hop_length, chick_id):
    # Plot the linear spectrogram
    fig, ax = plt.subplots(nrows=2, sharex=True)
    rms_y_times = lb.times_like(rms_features_calls, sr=sr, hop_length=hop_length)
    ax[0].semilogy(rms_y_times, rms_features_calls, label='RMS Energy')
    ax[0].set(xticks=[])
    ax[0].legend()
    ax[0].label_outer()
    lb.display.specshow(lb.amplitude_to_db(lin_spec, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', cmap='viridis', alpha=0.75)
    ax[1].set(xlabel='Time (s)', ylabel='Frequency')
    plt.show()
    plt.savefig(os.path.join(chick_id + '_spectrogram_RMS.png'))
    return

def visualise_spectrogram_and_envelope(lin_spec, envelope_features_calls, sr, hop_length, chick_id):
    # Plot the linear spectrogram
    fig, ax = plt.subplots(nrows=2, sharex=True)
    envelope_y_times = lb.times_like(envelope_features_calls, sr=sr, hop_length=hop_length)
    ax[0].semilogy(envelope_y_times, envelope_features_calls, label='Envelope')
    ax[0].set(xticks=[])
    ax[0].legend()
    ax[0].label_outer()
    lb.display.specshow(lb.amplitude_to_db(lin_spec, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', cmap='viridis', alpha=0.75)
    ax[1].set(xlabel='Time (s)', ylabel='Frequency')
    plt.show()
    plt.savefig(chick_id + '_spectrogram_envelope.png')
    return