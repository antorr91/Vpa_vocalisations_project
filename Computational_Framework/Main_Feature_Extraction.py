# Libraries used for the function of feature extraction: numpy, os, pandas, librosa, matplotlib, scipy, soundfile
import numpy as np
import os
import pandas as pd
import librosa as lb
import matplotlib.pyplot as plt
import utils as ut
import scipy.signal as signal
from scipy.signal import hilbert
import scipy.stats as stats
import soundfile as sf
from tqdm import tqdm
import visualisation_features as vf
from visualisation_features import visualise_spectrogram_and_harmonics, visualise_spectrogram_and_spectral_centroid, visualise_spectrogram_and_RMS
from update_features_csv import add_recording_callid_columns
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from feature_extraction_functions import ( spectral_centroid, rms_features, compute_envelope_features, compute_f0_features)


frame_length = 2048
hop_length = 512
win_length = frame_length // 2
n_fft = 2048*2





def process_single_file(file_path, save_folder):
    """Process a single audio file and extract features"""
    try:
        # Get the reference onsets and offsets        
        onsets = ut.get_reference_onsets(file_path.replace('.wav', '.txt'))
        offsets = ut.get_reference_offsets(file_path.replace('.wav', '.txt'))

        chick_id = os.path.basename(file_path)[:-4]
        save_features_file = 'features_data_' + chick_id + '.csv'   
        save_features_file = os.path.join(save_folder, save_features_file)

        features_data = pd.DataFrame()
        features_data['onsets_sec']= onsets
        features_data['offsets_sec']=offsets

        ##### 1- Load audio file
        audio_y, sr = lb.load(file_path, sr=44100)
        audio_fy = ut.bp_filter(audio_y, sr, lowcut=2000, highcut=15000)

        onsets_sec = onsets
        offsets_sec = offsets
        sr = 44100
        pyin_fmin_hz = 2000
        pyin_fmax_hz = 12500
        pyin_beta = (0.10, 0.10)
        pyin_ths = 100
        pyin_resolution = 0.02

        events = list(zip(onsets, offsets))
        durations = ut.calculate_durations(events)
        features_data['Duration_call'] = durations
        features_data.to_csv(save_features_file, index=False)

        # Compute F0 features
        f0_features_calls, F0_wholesignal, features_data = compute_f0_features(
            audio_fy, features_data, sr, hop_length, frame_length, n_fft, 
            pyin_fmin_hz, pyin_fmax_hz, pyin_beta, pyin_ths, pyin_resolution
        )
        features_data.to_csv(save_features_file, index=False)

        # Compute the spectral centroid
        spectral_centroid_feature_calls, features_data = spectral_centroid(
            audio_fy, features_data, sr, frame_length, hop_length
        )
        features_data.to_csv(save_features_file, index=False)

        # Compute the RMS
        rms_features_calls, features_data = rms_features(
            audio_fy, features_data, sr, frame_length, hop_length
        )
        features_data.to_csv(save_features_file, index=False)

        # Compute the envelope features
        envelope_features_calls, features_data = compute_envelope_features(
            audio_fy, features_data, sr
        )
        features_data.to_csv(save_features_file, index=False)

        print(f'Features extracted successfully for file: {file_path}')
        return True, file_path
        
    except Exception as e:
        print(f'Error processing file {file_path}: {str(e)}')
        return False, file_path

def parallel_feature_extraction_multiprocessing(files_folder, save_folder, n_processes=None):
    """
    Process files using multiprocessing.Pool
    """
    if n_processes is None:
        n_processes = cpu_count()
    
    print(f"Using {n_processes} processes")
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    list_files = [os.path.join(files_folder, file) for file in os.listdir(files_folder) if file.endswith('.wav')]
    
    # Create a partial function with save_folder fixed
    from functools import partial
    process_func = partial(process_single_file, save_folder=save_folder)
    
    with Pool(processes=n_processes) as pool:
        # Use imap for progress tracking
        results = list(tqdm(pool.imap(process_func, list_files), total=len(list_files)))
    
    # Check results
    successful = sum(1 for success, _ in results if success)
    failed = len(results) - successful
    
    print(f"Processed {successful} files successfully, {failed} failed")
    
    # Add recording and call id columns
    add_recording_callid_columns(save_folder)
    print('All features extracted successfully')

def parallel_feature_extraction_concurrent(files_folder, save_folder, max_workers=None):
    """
    Process files using concurrent.futures.ProcessPoolExecutor
    Provides better error handling and progress tracking
    """
    if max_workers is None:
        max_workers = cpu_count()
    
    print(f"Using {max_workers} workers")
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    list_files = [os.path.join(files_folder, file) for file in os.listdir(files_folder) if file.endswith('.wav')]
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_file, file_path, save_folder): file_path 
            for file_path in list_files
        }
        
        # Process completed tasks with progress bar
        successful = 0
        failed = 0
        
        for future in tqdm(as_completed(future_to_file), total=len(list_files)):
            file_path = future_to_file[future]
            try:
                success, processed_file = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f'Error processing {file_path}: {str(e)}')
                failed += 1
    
    print(f"Processed {successful} files successfully, {failed} failed")
    
    # Add recording and call id columns
    add_recording_callid_columns(save_folder)
    print('All features extracted successfully')

def parallel_feature_extraction_batched(files_folder, save_folder, batch_size=4, n_processes=None):
    """
    Process files in batches to balance memory usage and parallelization
    """
    if n_processes is None:
        n_processes = cpu_count()
    
    print(f"Using {n_processes} processes with batch size {batch_size}")
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    list_files = [os.path.join(files_folder, file) for file in os.listdir(files_folder) if file.endswith('.wav')]
    
    # Process in batches
    from functools import partial
    process_func = partial(process_single_file, save_folder=save_folder)
    
    successful = 0
    failed = 0
    
    for i in range(0, len(list_files), batch_size):
        batch = list_files[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(list_files) + batch_size - 1)//batch_size}")
        
        with Pool(processes=min(n_processes, len(batch))) as pool:
            results = pool.map(process_func, batch)
        
        # Count results
        batch_successful = sum(1 for success, _ in results if success)
        batch_failed = len(results) - batch_successful
        
        successful += batch_successful
        failed += batch_failed
        
        print(f"Batch completed: {batch_successful} successful, {batch_failed} failed")
    
    print(f"Total: {successful} files processed successfully, {failed} failed")
    
    # Add recording and call id columns
    add_recording_callid_columns(save_folder)
    print('All features extracted successfully')

if __name__ == '__main__':
    files_folder = 'C:\\Users\\anton\\VPA_vocalisations_project\\Automatic_calls_detected_070'
    save_folder = 'C:\\Users\\anton\\VPA_vocalisations_project\\Results_features_extraction_automatic_070'

    # Define your input and output directories here 
    # in files_folder you should have the audio files (.wav) to process & onsets/offsets files (.txt)
    files_folder = 'path/to/your/input_data'
    save_folder = 'path/to/your/output_results'

    
    # Choose one of the parallel processing methods:
    
    # Method 1: Custom number of processes [ 1 process is sequential, >1 is parallel]
    parallel_feature_extraction_multiprocessing(files_folder, save_folder, n_processes=1)
    
    # Method 2: Concurrent futures (better error handling)
    # parallel_feature_extraction_concurrent(files_folder, save_folder)
    
    # Method 3: Batched processing (if you have memory constraints)
    # parallel_feature_extraction_batched(files_folder, save_folder, batch_size=4)
    
