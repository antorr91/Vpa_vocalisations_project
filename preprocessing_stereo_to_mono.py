# Convert multichannel WAV files to mono by averaging across all channels
# (same approach as Audacity's "Stereo Track to Mono" option).

import os
import glob
import librosa
import numpy as np
import soundfile as sf
import librosa.display
import matplotlib.pyplot as plt

# Set your paths here
input_folder = "path/to/multichannel/wav/files"
output_folder = "path/to/mono/wav/files"
sample_rate = 44100

# Frequency range of interest for the spectrogram (used only if plots are enabled)
low_freq = 1000
high_freq = 13500

os.makedirs(output_folder, exist_ok=True)

for file_path in glob.glob(os.path.join(input_folder, "*.wav")):
    file_name = os.path.basename(file_path)
    print(f"Processing {file_name}")

    # Load multichannel audio
    y, sr = librosa.load(file_path, sr=sample_rate, mono=False)

    # # Optional: save spectrogram of each input channel for visual QC
    # if y.ndim > 1:
    #     num_channels = y.shape[0]
    #     fig, axes = plt.subplots(num_channels, 1, figsize=(10, 4 * num_channels))
    #     for i in range(num_channels):
    #         D = librosa.amplitude_to_db(np.abs(librosa.stft(y[i])), ref=np.max)
    #         librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=axes[i])
    #         axes[i].set_title(f"Channel {i + 1}")
    #         axes[i].set_ylim([low_freq, high_freq])
    #         axes[i].set_xlabel("Time (s)")
    #         axes[i].set_ylabel("Frequency (Hz)")
    #     plt.tight_layout()
    #     spec_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + "_spectrogram.png")
    #     plt.savefig(spec_path)
    #     plt.close()

    # Convert to mono by averaging across channels
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    # Save mono WAV
    output_path = os.path.join(output_folder, file_name)
    sf.write(output_path, y, sr)
    print(f"Saved: {output_path}")
