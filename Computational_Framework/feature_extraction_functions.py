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
# import joint_time_frequency_scattering_coefficients as jtfs  # for analyse higher order dependencies  ( time / frequncy coefficients)
from update_features_csv import add_recording_callid_columns

# this prevent possible leakage and stabilise pyin and pytorch in case of parallelised processing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"



frame_length = 2048
hop_length = 512
win_length = frame_length // 2
n_fft = 2048*2
np.random.seed(42)




def spectral_centroid(audio_fy, features_data, sr, frame_length, hop_length):

    ''' Compute the mean of the spectral centroid for each call in the audio file
    returns: return the mean of the spectral centroid for each call in the audio file
    '''
    onsets_sec = features_data.onsets_sec
    offsets_sec = features_data.offsets_sec

   
    Spectral_Centroid_mean = [] 
    Spectral_Centroid_std = []
    call_numbers = []
    # Compute the spectrogram of the entire audio file
    # set a minimum threshold for the spectrogram   
    #lin_spec= np.abs(lb.stft(y=audio_fy, n_fft=frame_length, hop_length=hop_length))
    
    mel_spec = lb.feature.melspectrogram(y=audio_fy, sr=sr, n_fft=frame_length, 
                                        hop_length=hop_length, n_mels=128, fmin=2000, fmax=12600)

        
    # Extract the spectrogram calls from the audio file
    calls_s_files = ut.segment_spectrogram(spectrogram= mel_spec, onsets=onsets_sec, offsets=offsets_sec, sr=sr)
      
    for i, call_s in enumerate(calls_s_files):
        call_numbers.append(i)
        # Compute the spectral centroid and then extract the mean
        spectral_centroid_call = lb.feature.spectral_centroid(S=call_s, sr=sr, n_fft=frame_length, hop_length=hop_length)
        # visual_spectral_centroid.append(spectral_centroid_call)
        # replace nans with zeros
        spectral_centroid_call_without_nans = spectral_centroid_call[~np.isnan(spectral_centroid_call)]
        if len(spectral_centroid_call_without_nans) == 0:
            Spectral_Centroid_mean.append(np.nan)
            Spectral_Centroid_std.append(np.nan)
            continue 
        else:     
        # compute the mean of the spectral centroid
            mean_spectral_centroid = np.mean(spectral_centroid_call_without_nans)
            st_dev_spectral_centroid = np.std(spectral_centroid_call_without_nans)
            Spectral_Centroid_mean.append(mean_spectral_centroid)
            Spectral_Centroid_std.append(st_dev_spectral_centroid)
    
    spectral_centroid_feature_calls = pd.DataFrame()
    spectral_centroid_feature_calls['Spectral Centroid Mean'] = Spectral_Centroid_mean
    spectral_centroid_feature_calls['Spectral Centroid Std'] = Spectral_Centroid_std

    #plot_spectral_centroid = pd.DataFrame(visual_spectral_centroid)

    features_data['Spectral Centroid Mean'] = Spectral_Centroid_mean
    features_data['Spectral Centroid Std'] = Spectral_Centroid_std

    return spectral_centroid_call, features_data




def rms_features(audio_fy, features_data, sr, frame_length, hop_length):
    ''' Compute the mean and the standard deviation of the RMS of each call in the audio file'''

    onsets_sec = features_data.onsets_sec
    offsets_sec = features_data.offsets_sec


    call_numbers = []
    RMS_mean = []
    RMS_std = []

    # Extract the waveform segments from the audio file
    calls_wave_file = ut.get_calls_waveform(audio_fy, onsets_sec, offsets_sec, sr)

    for i, call in enumerate(calls_wave_file):
        call_numbers.append(i)        
        # Compute the RMS of each call
        rms_call = lb.feature.rms(y=call, frame_length=frame_length, hop_length=hop_length)
        rms_call_without_nans = rms_call[~np.isnan(rms_call)]
        # check if the call is empty
        if len(rms_call_without_nans) == 0:
            RMS_mean.append(np.nan)
            RMS_std.append(np.nan)
            
        else:

            mean_rms = np.mean(rms_call_without_nans)
            st_dev_rms = np.std(rms_call_without_nans)
            RMS_mean.append(mean_rms)
            RMS_std.append(st_dev_rms)

    rms_features_calls = pd.DataFrame()
    rms_features_calls['RMS Mean'] = RMS_mean
    rms_features_calls['RMS Std'] = RMS_std

    features_data['RMS Mean'] = RMS_mean
    features_data['RMS Std'] = RMS_std

    return rms_features_calls, features_data


def compute_envelope_features(audio_fy,features_data, sr= 44100):
    ''' Compute the envelope of each call in the audio file and extract the following features'''

    onsets_sec = features_data.onsets_sec
    offsets_sec = features_data.offsets_sec
    
    Slopes = []
    Attack_magnitude = []
    Attack_time = []
    call_numbers = []


    # Extract the waveform segments from the audio file
    calls_wave_file = ut.get_calls_waveform(audio_fy, onsets_sec, offsets_sec, sr)
    
    for i, call in enumerate(calls_wave_file):
        call_numbers.append(i) 
        # Compute the analytic signal
        analytic_signal = hilbert(call)
        # Compute the envelope
        envelope = np.abs(analytic_signal)
        # Compute the onset index
        onset_in_time= 0
        onset = lb.time_to_frames(onset_in_time, sr=sr)
        # Compute the peak index
        peak_index = np.argmax(envelope)
        # Compute the difference of magnitude between onset and peak
        attack_magnitude = envelope[peak_index] - envelope[onset]
        # Compute the time difference between onset and peak
        peak_index_in_time = lb.frames_to_time(peak_index, sr=sr)
        attack_time= peak_index_in_time - onset_in_time
        # Compute the rise time (time difference between onset and maximum peak)
       
        # Compute the slope of the envelope
        slope = attack_magnitude/ attack_time
       

        Slopes.append(slope)
        Attack_magnitude.append(attack_magnitude)
        Attack_time.append(attack_time)

    envelope_features_calls = pd.DataFrame()
    envelope_features_calls['Slope'] = Slopes
    envelope_features_calls['Attack_magnitude'] = Attack_magnitude
    envelope_features_calls['Attack_time'] = Attack_time

    features_data['Slope'] = Slopes
    features_data['Attack_magnitude'] = Attack_magnitude
    features_data['Attack_time'] = Attack_time

    return envelope_features_calls, features_data







def compute_f0_features(audio, features_data ,sr, hop_length, frame_length, n_fft, pyin_fmin_hz, pyin_fmax_hz, pyin_beta, pyin_ths, pyin_resolution):

    ''' run the pitch estimation using PYIN and compute the F0 features.
    F0 features computed: 
    -Mean, 
    -Standard Deviation, 
    -Skewness, 
    -Kurtosis, 
    -Bandwidth, 
    -1st order difference of the F0
    -Slope of the F0
    -Magnitude of the F0
    
    Then, compute the F1 and F2 harmonics of the F0 and compute the mean of the magnitudes of F1 and F2.
    
    Returns: return dataframe containing features for each call in recording
    
    '''
    onsets_sec = features_data.onsets_sec
    offsets_sec = features_data.offsets_sec
    # ## 3- Estimate pitch using PYIN    
    f0_pyin_lb, _, _ = lb.pyin(audio, sr=sr, frame_length=frame_length, hop_length=hop_length, 
                               fmin=pyin_fmin_hz, fmax=pyin_fmax_hz, n_thresholds=pyin_ths, beta_parameters=pyin_beta,
                               resolution= pyin_resolution)  # f0 is in hertz!! per frames

    # # compute spectrogram
    S = np.abs(lb.stft(y=audio, n_fft=frame_length, hop_length=hop_length))

    # segment spectrogram into calls
    calls_S = ut.segment_spectrogram(S, onsets_sec, offsets_sec, sr=sr)

    # Compute for each calls statistics over the F0: Mean, Standard Deviation, Skewness, Kurtosis
    f0_calls = ut.get_calls_F0(f0_pyin_lb, onsets_sec, offsets_sec, sr, hop_length, n_fft)

    call_numbers = []
    F0_means = []
    F0_stds = []
    F0_skewnesses = []
    F0_kurtosises = []
    F0_bandwidths = []
    F0_fst_order_diffs_means = []
    F0_slopes = []
    F0_mag_means = []  
    
    F1_mag_means = []
    F2_mag_means = []
    F1_F0_ratios_mag_means = []
    F2_F0_ratios_mag_means = []
    

    discarded_calls = 0
    # compute the statistics for each call 
    #for i,  f0_call in enumerate(f0_calls):
    for i, (f0_call, call_s) in enumerate(zip(f0_calls, calls_S)):
        call_numbers.append(i)
        # remove nans from the f0 call
        f0_call_without_nans = f0_call[~np.isnan(f0_call)]
        # check if the call is empty 
        if len(f0_call_without_nans) == 0: #or len(f0_call_without_nans) < 0.7 * len(f0_call):
            discarded_calls += 1
            F0_means.append(np.nan)
            F0_stds.append(np.nan)
            F0_skewnesses.append(np.nan)
            F0_kurtosises.append(np.nan)
            F0_bandwidths.append(np.nan)
            F0_fst_order_diffs_means.append(np.nan)
            F0_slopes.append(np.nan)
            F0_mag_means.append(np.nan)

            F1_mag_means.append(np.nan)
            F2_mag_means.append(np.nan)
            F1_F0_ratios_mag_means.append(np.nan)
            F2_F0_ratios_mag_means.append(np.nan)
        
        else:
            # compute the statistics
            f0_call_mean = f0_call_without_nans.mean()
            f0_call_std = f0_call_without_nans.std()
            f0_call_skewness = stats.skew(f0_call_without_nans)
            f0_call_kurtosis = stats.kurtosis(f0_call_without_nans)
            # Calculate the minimum and maximum frequencies within the F0 range
            min_f0 = np.min(f0_call_without_nans)
            max_f0 = np.max(f0_call_without_nans)
            # Compute the bandwidth of F0
            f0_bandwidth = max_f0 - min_f0
            # compute the 1st derivative of the F0
            f0_fst_order_diff_mean = np.diff(f0_call_without_nans).mean()
            # compute the slope of the F0
            # Compute the onset index
            onset_in_time= 0
            onset = lb.time_to_frames(onset_in_time, sr=sr)
            # Compute the peak index
            peak_index = np.argmax(f0_call_without_nans)
            # Compute the difference of magnitude between onset and peak
            attack_fr_hz = f0_call_without_nans[peak_index] - f0_call_without_nans[onset]
            # Compute the time difference between onset and peak
            peak_index_in_time = lb.frames_to_time(peak_index, sr=sr)
            attack_time_f0_hz= peak_index_in_time - onset_in_time
            # Compute the slope of the  f0
            # F0_slope = attack_fr_hz/ attack_time_f0_hz

            if attack_time_f0_hz == 0:
                F0_slope = 0  # if the time difference is 0 
            elif np.isnan(attack_fr_hz) or attack_time_f0_hz < 0:
                F0_slope = np.nan  # if the time difference is negative or the attack frequency is nan
            else:
                F0_slope = attack_fr_hz / attack_time_f0_hz

            # compute F1 and F2 
            F1_Hz_withoutNans = f0_call_without_nans * 2
            F2_Hz_withoutNans = f0_call_without_nans * 3
            # compute the mean of the harmonics
            F1_Hz_mean = np.mean(F1_Hz_withoutNans)
            F2_Hz_mean = np.mean(F2_Hz_withoutNans)

    

            # compute the magnitudes of F0, F1, and F2
            F0_mag = []
            F1_mag = []
            F2_mag = []
            call_s_without_nans = []
            for time_frame, freqhz in enumerate(f0_call):
                if np.isnan(freqhz):
                     # skip this time frame if either the frequency bin or the call frame contains NaN
                    continue
                else:
                    f0_bin = int( np.floor(freqhz * n_fft / sr))
                    f1_bin = int(np.floor(freqhz * 2 * n_fft / sr))
                    f2_bin =int(np.floor(freqhz * 3 * n_fft / sr))

                    # compute the magnitudes and append to the corresponding lists
                    F0_mag.append(call_s[f0_bin, time_frame])
                    try: 
                        F1_mag.append(call_s[f1_bin, time_frame])
                    except IndexError:
                        # if out of  boundaries, append 0
                        F1_mag.append(0)
                    try:
                        F2_mag.append(call_s[f2_bin, time_frame])
                    except IndexError:
                        # if out of  boundaries, append 0
                        F2_mag.append(0)

            # remove nans from the magnitudes
            F1_mag_without_nans = [mag for mag in F1_mag if not np.isnan(mag)]
            F2_mag_without_nans = [mag for mag in F2_mag if not np.isnan(mag)]

            F0_mag_mean = np.mean(F0_mag)

            if F1_mag_without_nans == []:
               F1_mag_mean= np.nan
            else:  
                F1_mag_mean = np.mean(F1_mag_without_nans)
            # Compute the mean of the magnitudes of F1 and F2

            if F2_mag_without_nans == []:
                F2_mag_mean = np.nan
            else:
                F2_mag_mean = np.mean(F2_mag_without_nans)

            # Compute the F1/F0 and F2/F0 ratios
            # F1/F0 ratio is computed as the mean of the F1 magnitudes divided
            if np.isnan(F1_mag_mean):
                F1_F0_ratios_magnitude = np.nan
            else:
                F1_F0_ratios_magnitude = [F1 / F0 for F1, F0 in zip(F1_mag, F0_mag) if F0 > 0]
                
                F1_F0_ratios_magnitude_mean = (np.mean(F1_F0_ratios_magnitude) if F1_F0_ratios_magnitude else np.nan)

            if np.isnan(F2_mag_mean):
                F2_F0_ratios_magnitude = np.nan 
            else:
                F2_F0_ratios_magnitude = [F2 / F0 for F2, F0 in zip(F2_mag, F0_mag) if F0 > 0]
                F2_F0_ratios_magnitude_mean = (np.mean(F2_F0_ratios_magnitude) if F2_F0_ratios_magnitude else np.nan)

        
            # F0 features computed for the study
            F0_means.append(f0_call_mean)
            F0_stds.append(f0_call_std)
            F0_skewnesses.append(f0_call_skewness)
            F0_kurtosises.append(f0_call_kurtosis)
            F0_bandwidths.append(f0_bandwidth)
            F0_fst_order_diffs_means.append(f0_fst_order_diff_mean)
            F0_slopes.append(F0_slope)
            F0_mag_means.append(F0_mag_mean)

            # Features computed for the harmonics
            F1_mag_means.append(F1_mag_mean)
            F2_mag_means.append(F2_mag_mean) 
            F1_F0_ratios_mag_means.append(F1_F0_ratios_magnitude_mean)
            F2_F0_ratios_mag_means.append(F2_F0_ratios_magnitude_mean)

       
    #  visualise F0, F1 and F2 on top of the spectrogram
    # visualise_spectrogram_and_harmonics(S, f0_pyin_lb, F1_Hz, F2_Hz, sr, hop_length)

    f0_features_calls = pd.DataFrame()
    f0_features_calls['Call Number'] = call_numbers
    f0_features_calls['F0 Mean'] = F0_means
    f0_features_calls['F0 Std'] = F0_stds
    f0_features_calls['F0 Skewness'] = F0_skewnesses
    f0_features_calls['F0 Kurtosis'] = F0_kurtosises
    f0_features_calls['F0 Bandwidth'] = F0_bandwidths
    f0_features_calls['F0 1st Order Diff'] = F0_fst_order_diffs_means
    f0_features_calls['F0 Slope'] = F0_slopes
    f0_features_calls['F0 Mag Mean'] = F0_mag_means

    # Actual F1 and F2 values
    f0_features_calls['F1 Mag Mean'] = F1_mag_means
    f0_features_calls['F2 Mag Mean'] = F2_mag_means
    f0_features_calls['F1-F0 Ratio'] = F1_F0_ratios_mag_means
    f0_features_calls['F2-F0 Ratio'] = F2_F0_ratios_mag_means

        
    features_data['Call Number'] = call_numbers
    features_data['F0 Mean'] = F0_means
    features_data['F0 Std'] = F0_stds
    features_data['F0 Skewness'] = F0_skewnesses
    features_data['F0 Kurtosis'] = F0_kurtosises
    features_data['F0 Bandwidth'] = F0_bandwidths
    features_data['F0 1st Order Diff'] = F0_fst_order_diffs_means
    features_data['F0 Slope'] = F0_slopes
    features_data['F0 Mag Mean'] = F0_mag_means

    features_data['F1 Mag Mean'] = F1_mag_means
    features_data['F2 Mag Mean'] = F2_mag_means
    features_data['F1-F0 Ratio'] = F1_F0_ratios_mag_means
    features_data['F2-F0 Ratio'] = F2_F0_ratios_mag_means


    print(f"Discarded calls: {discarded_calls}")

    return f0_features_calls, f0_pyin_lb, features_data



if __name__ == '__main__':



    files_folder = 'C:\\Users\\audio_data_and_onsets_offsets_txts_files'



    list_files = [os.path.join(files_folder, file) for file in os.listdir(files_folder) if file.endswith('.wav')]
    for file in tqdm(list_files):
        # Get the reference onsets and offsets        
        onsets = ut.get_reference_onsets(file.replace('.wav', '.txt'))
        offsets = ut.get_reference_offsets(file.replace('.wav', '.txt'))

        chick_id = os.path.basename(file)[:-4]


        save_features_file = 'features_data_' + chick_id + '.csv'   


        # Define the save folder for the features : this is the folder where the features will be saved in csv format
        save_folder = 'C:\\Users\\save_folder\\results_features_extraction_highpass_12600'        
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_features_file = os.path.join(save_folder, save_features_file)


        features_data = pd.DataFrame()
        features_data['onsets_sec']= onsets
        features_data['offsets_sec']=offsets


        ##### 1- Load audio file
        audio_y, sr = lb.load(file, sr=44100) # ADD duration to 800 to avoid hours for file 45 + rechange the highcut of bp to 12600 for the features study
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
        f0_features_calls, F0_wholesignal, features_data = compute_f0_features(audio_fy, features_data,sr, hop_length, frame_length, n_fft, pyin_fmin_hz, pyin_fmax_hz, pyin_beta, pyin_ths, pyin_resolution)
        
        # # save locally features_data to a csv file
        features_data.to_csv(save_features_file, index=False)

        # to visualise the F0, F1 and F2 harmonics, uncomment the following lines       
        # F1_Hz = F0_wholesignal*2
        # F2_Hz = F0_wholesignal*3
        # # # Visualise the spectrogram and the features computed (F0 stats, F0/F1 ratio, F0/F2 ratio, spectral centroid stats, RMS stats, envelope stats)
        # S = np.abs(lb.stft(y=audio_fy, n_fft=frame_length, hop_length=hop_length))

        # visualise_spectrogram_and_harmonics(S, F0_wholesignal, F1_Hz, F2_Hz, sr, hop_length, chick_id)

        #Compute the spectral centroid
        spectral_centroid_feature_calls, features_data = spectral_centroid(audio_fy, features_data, sr, frame_length, hop_length)
        # # save locally features_data to a csv file
        features_data.to_csv(save_features_file, index=False)

        # # Compute the RMS
        rms_features_calls, features_data = rms_features(audio_fy, features_data, sr, frame_length, hop_length)
        # # save locally features_data to a csv file
        features_data.to_csv(save_features_file, index=False)


        # # Compute the envelope features
        envelope_features_calls, features_data = compute_envelope_features(audio_fy, features_data, sr)
        # # save locally features_data to a csv file
        features_data.to_csv(save_features_file, index=False)


        # features_data, features_jtfs =jtfs.jtfs_coefficients(file, sr, min_freq_hz=2000, max_freq_hz=12600, max_dur_s=0.500, features_data=features_data)

        # # save locally features_data to a csv file
        # features_data.to_csv(save_features_file, index=False)

        # # Visualise the spectrogram and the features computed (F0 stats, F0/F1 ratio, F0/F2 ratio, spectral centroid stats, RMS stats, envelope stats)
        # S = np.abs(lb.stft(y=audio_fy, n_fft=frame_length, hop_length=hop_length))

        # visualise_spectrogram_and_harmonics(S, F0_wholesignal, F1_Hz, F2_Hz, sr, hop_length, chick_id)
        
        # visualise_spectrogram_and_spectral_centroid(S, plot_spectral_centroid, sr, hop_length, chick_id)
        
        # visualise_spectrogram_and_RMS(S, rms_features_calls, sr, hop_length, chick_id)


        # add the recording and call id columns to the features_data
        print('Features extracted successfully for file: ', file)
        
    add_recording_callid_columns(save_folder)
    print('Features extracted successfully for all files')