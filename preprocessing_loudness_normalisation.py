# Loudness normalisation for WAV audio files.
# Applies max-amplitude normalisation to all WAV files in an input folder
# and saves them to an output folder. Step 0 of the chick vocalisation pipeline.

import os
import numpy as np
from scipy.io import wavfile


# Set your paths here
input_folder = "path/to/raw/wav/files"
output_folder = "path/to/normalised/wav/files"


def normalise_and_save_audio(input_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    sample_rate, audio_data = wavfile.read(input_path)

    max_amplitude = np.max(np.abs(audio_data))
    if max_amplitude == 0:
        print(f"Skipping silent file: {input_path}")
        return

    normalised_audio = audio_data / max_amplitude

    output_path = os.path.join(output_folder, os.path.basename(input_path))
    wavfile.write(output_path, sample_rate, normalised_audio)
    print(f"Normalised audio saved to {output_path}")


if __name__ == "__main__":
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".wav"):
            normalise_and_save_audio(os.path.join(input_folder, filename), output_folder)
