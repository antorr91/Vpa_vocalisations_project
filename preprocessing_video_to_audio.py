# Video to audio (mono WAV) extraction.
#
# For each video file (.MTS, .mp4) or audio file (.wav) in the input
# directory, extracts the audio track with ffmpeg at the specified
# sample rate and saves it as a mono WAV file. If the source has
# multiple channels, they are averaged into a single mono channel.
#
# This is the entry point of the pipeline when starting from video
# recordings (e.g., from a webcam). The output of this step can then
# be passed to loudness_normalisation.py.

import os
import glob

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import ffmpeg


# Set your paths and parameters here
input_directory = "path/to/video/files"
output_directory = "path/to/audio/files"
file_types = [".MTS", ".mp4", ".wav"]
sample_rate = 44100


def convert_video_to_audio(input_file, output_file, sample_rate=sample_rate):
    # Extract the audio track from the video using ffmpeg
    ffmpeg.input(input_file).output(
        output_file, acodec="pcm_s16le", ar=sample_rate
    ).run(overwrite_output=True, quiet=True)

    # Reload the extracted audio (preserve all channels)
    y, sr = librosa.load(output_file, sr=sample_rate, mono=False)

    # Convert to mono by averaging across channels
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    # Save as mono WAV
    sf.write(output_file, y, sample_rate)


def process_directory(input_dir, output_dir, file_types=file_types,
                      sample_rate=sample_rate):
    os.makedirs(output_dir, exist_ok=True)

    files = []
    for ext in file_types:
        files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))

    for file in tqdm(files, desc="Converting files"):
        output_filename = os.path.splitext(os.path.basename(file))[0] + ".wav"
        output_file = os.path.join(output_dir, output_filename)
        convert_video_to_audio(file, output_file, sample_rate)

    print(f"Conversion completed. Processed {len(files)} files.")


if __name__ == "__main__":
    process_directory(input_directory, output_directory, file_types, sample_rate)
