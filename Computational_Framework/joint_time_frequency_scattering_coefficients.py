import math
from typing import List, Optional
import numpy as np
import torchaudio
from kymatio.torch import Scattering1D, TimeFrequencyScattering
from matplotlib import pyplot as plt
from torch import Tensor
import torch as tr
from visualisation_features import plot_scalogram
import utils as ut
import os
from tqdm import tqdm
import pandas as pd
import librosa as lb
import evaluation as my_eval



def jtfs_coefficients(file: None,  sr: int, min_freq_hz: int, max_freq_hz: int, max_dur_s: float, features_data) -> None:

    audio_y, sr = torchaudio.load(file)
                          
    onsets_sec = features_data.onsets_sec
    offsets_sec = features_data.offsets_sec


    features_jtfs = pd.DataFrame()  
    Avg_up = []
    Std_up = []
    Avg_down = []
    Std_down = []
    Avg_flat = []
    Std_flat = []

    call_numbers = []
    # Extract the waveform segments from the audio file
    calls_wave_file = ut.segment_calls(audio_y, sr, onsets_sec, offsets_sec)

    for i, call in enumerate(calls_wave_file):
        call_numbers.append(i)
        audio = call
            
        assert audio.size(0) == 1
        audio = audio.view(1, 1, -1)
        n_samples = audio.size(-1)

        # # print(f"n_samples = {n_samples}, sr = {sr} Hz")
        # J = math.ceil(math.log2(sr / min_freq_hz)) 
      
        # T = 2 ** 11   #T = 2^12 = 4096 samples but T = 2^11 = 2048 
        # print(f"max_dur_s in samples = {max_dur_s * sr}")
        # print(f"J1: {J}, T: {T}")
        # # Increasing these numbers will make computation time increase exponentially
        # # Q1 = 24  # This is the number of filters per octave, increase for better frequency resolution
        # # Q2 = 6 # second order number of filters per octave ( along time axis)
        # # J_fr = 4 # number of the octaves for the last (third filter bank)
        # # Q_fr = 2 
        # # F = 1

        # Q1 = 32  # Increased from 24 for better frequency resolution

        # Q2 = 8  # Increased from 6 for better time-frequency resolution

        # J_fr = 4  # This can remain the same

        # Q_fr = 2  # This can remain the same

        # F = 1  # This can remain the same

        J= 12 
        T = 2 ** 11
        Q1 = 32
        Q2 = 8
        J_fr = 4
        Q_fr = 2
        F = 1
        


        print(f"audio.shape before padding = {audio.shape}")
        if n_samples < T:
            padding_n = (T - n_samples) // 2
            if n_samples % 2 == 1:
                padding = (padding_n, padding_n + 1)
            else:
                padding = (padding_n, padding_n)
            audio = tr.nn.functional.pad(audio, padding, mode="constant", value=0)
            print(f"audio.shape after padding = {audio.shape}")
            assert audio.size(-1) == T
            n_samples = audio.size(-1)

       # For plotting scalograms
        scat_1d_u1 = Scattering1D(shape=(n_samples,), J=J, Q=(Q1, 1), T=1, max_order=1)
        meta = scat_1d_u1.meta()
        order_1_indices_u1 = []
        filter_freqs_u1 = []
        for idx, order in enumerate(meta["order"]):
            if order == 1:
                freq = meta["xi"][idx, 0] * sr
                if min_freq_hz <= freq <= max_freq_hz:
                    order_1_indices_u1.append(idx)
                    filter_freqs_u1.append(freq)
        print(f"filter_freqs_u1 = {filter_freqs_u1}")
        u1 = scat_1d_u1(audio)
        print(f"u1.shape = {u1.shape}")
        u1 = u1[0, 0, order_1_indices_u1, :]
        plot_scalogram(u1,
                    sr=sr,
                    y_coords=filter_freqs_u1,
                    title=f"Scalogram U1 (J = {J}, Q = {Q1}, sr = {sr} Hz)")
        plt.show()



        jtfs = TimeFrequencyScattering(
            shape=(n_samples,),
            J=J,
            Q=(Q1, Q2),
            Q_fr=Q_fr,
            J_fr=J_fr,
            T=T,
            F=F,
            format="time",  # If you use a 2D CNN this needs to change to "joint"
        )
        meta = jtfs.meta()


        Sx = jtfs(audio)  # (batch_size, n_channels, n_paths, time_dim)
        print(f"Sx.shape = {Sx.shape}")
        n_paths = Sx.size(2)
        energies = tr.sum(Sx ** 2, dim=-1).squeeze()
        assert energies.ndim == 1



        o1 = []
        o2 = []
        relevant_path_indices = []
        for idx in range(n_paths):
            order = meta["order"][idx]
            spin = meta["spin"][idx]
            energy = energies[idx].item()
            o1_cf_hz = meta["xi"][idx, 0] * sr
            # Filter out paths we don't care about
            if min_freq_hz <= o1_cf_hz <= max_freq_hz:
                relevant_path_indices.append(idx)
                t_cf = meta["xi"][idx, 1]
                fr_cf = meta["xi_fr"][idx]
                if order == 1:
                    o1.append({
                        "o1_cf_hz": o1_cf_hz,
                        "t_cf": t_cf,
                        "fr_cf": fr_cf,
                        "spin": spin,
                        "energy": energy,
                    })
                elif order == 2:
                    o2.append({
                        "o1_cf_hz": o1_cf_hz,
                        "t_cf": t_cf,
                        "fr_cf": fr_cf,
                        "spin": spin,
                        "energy": energy,
                    })
        o1 = sorted(o1, key=lambda x: x["energy"], reverse=True)
        o2 = sorted(o2, key=lambda x: x["energy"], reverse=True)
        print(f"len(o1) = {len(o1)}, len(o2) = {len(o2)}")
        o2_up_energies = []
        o2_down_energies = []
        o2_flat_energies = []
        for x in o2:
            if x["spin"] == 1:
                o2_up_energies.append(x["energy"])
            elif x["spin"] == -1:
                o2_down_energies.append(x["energy"])
            elif x["spin"] == 0:
                o2_flat_energies.append(x["energy"])
        mean_up = np.mean(o2_up_energies)
        std_up = np.std(o2_up_energies)

        mean_down = np.mean(o2_down_energies)
        std_down = np.std(o2_down_energies)

        mean_flat = np.mean(o2_flat_energies)
        std_flat = np.std(o2_flat_energies)
     
        # print(f"avg o2_up_energies: {np.mean(o2_up_energies):.2E}, std o2_up_energies: {np.std(o2_up_energies):.2E}")
        # print(f"avg o2_down_energies: {np.mean(o2_down_energies):.2E}, std o2_down_energies: {np.std(o2_down_energies):.2E}")
        # print(f"avg o2_flat_energies: {np.mean(o2_flat_energies):.2E}, std o2_flat_energies: {np.std(o2_flat_energies):.2E}")

        features = Sx[:, :, relevant_path_indices, :]
    
        Avg_up.append(mean_up)
        Std_up.append(std_up)
        Avg_down.append(mean_down)
        Std_down.append(std_down)
        Avg_flat.append(mean_flat)
        Std_flat.append(std_flat)

    features_jtfs['call_number'] = call_numbers

    features_jtfs['Avg_up'] = Avg_up
    features_jtfs['Std_up'] = Std_up

    features_jtfs['Avg_down'] = Avg_down
    features_jtfs['Std_down'] = Std_down

    features_jtfs['Avg_flat'] = Avg_flat
    features_jtfs['Std_flat'] = Std_flat


    #features_data['call_number'] = call_numbers

    features_data['Avg_up'] = Avg_up
    features_data['Std_up'] = Std_up

    features_data['Avg_down'] = Avg_down
    features_data['Std_down'] = Std_down

    features_data['Avg_flat'] = Avg_flat
    features_data['Std_flat'] = Std_flat

        
    return features_data , features_jtfs



if __name__ == "__main__":
    


    files_folder = 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\calls_envelope_test'
    
    list_files = [os.path.join(files_folder, file) for file in os.listdir(files_folder) if file.endswith('.wav')]

    for file in tqdm(list_files):
        # Get the reference onsets and offsets        
        # onsets = my_eval.get_reference_onsets(file.replace('.wav', '.txt'))
        # offsets = my_eval.get_reference_offsets(file.replace('.wav', '.txt'))

        chick_id = os.path.basename(file)[:-4]

        # save 

        save_features_file = 'features_data_' + chick_id + '.csv'   

        save_folder= 'C:\\Users\\anton\\Chicks_Onset_Detection_project\\Results_features\\_jtfs_coefficients_T2_11_Q1_32_Q2_8_J_fr_4_Q_fr_FR_10000'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_features_file = os.path.join(save_folder, save_features_file)

        

            #Get the reference onsets and offsets        
        onsets = my_eval.get_reference_onsets(file.replace('.wav', '.txt'))
        offsets = my_eval.get_reference_offsets(file.replace('.wav', '.txt'))

        features_data = pd.DataFrame()
        features_data['onsets_sec'] = onsets
        features_data['offsets_sec'] = offsets


        sr = 44100
        min_freq_hz = 2000  # This is the lowest frequency we care about in Hz
        max_freq_hz = 12500  # This is the highest frequency we care about in Hz
        max_dur_s = 0.500  # This is the longest duration we care about in seconds
      
        # audio, sr = torchaudio.load("C:/Users/anton/Chicks_Onset_Detection_project/call_jtfs/chick21_d0_contact_call.wav")
        #audio, sr = torchaudio.load("C:/Users/anton/Chicks_Onset_Detection_project/call_jtfs/chick21_d0_pleasure_call.wav")
        #audio, sr = torchaudio.load("C:/Users/anton/Chicks_Onset_Detection_project/call_jtfs/chick21_d0_distress_call.wav")4

        features_jtfs, features_data= jtfs_coefficients(file= file, sr=sr, min_freq_hz=2000, max_freq_hz=10000, max_dur_s=0.500, features_data=features_data)


        # plot_scalogram(u1, sr=sr, y_coords=filter_freqs_u1, title=f"Scalogram U1 (J = {J}, Q = {Q1}, sr = {sr} Hz)")

        # features_data.to_csv(save_features_file, index=False)

        features_jtfs.to_csv(save_features_file, index=False)

