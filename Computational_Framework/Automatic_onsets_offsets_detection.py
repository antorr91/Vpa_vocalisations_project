import os
import glob
import onset_detection_algorithms as onset_detectors
import offset_detection as off_detection
from tqdm import tqdm
import pandas as pd
import json
from visualization import visualize_activation_and_gt
from save_results import save_detected_calls_in_csv # optional can depend on the visualisation method ( example csv for Sonic Visualzer)
import onsets_offsets_detection_utils as onset_utils
from onsets_offsets_detection_utils import  high_frequency_content, filter_calls_within_experiment, offset_detection_first_order 

# Parameters for the onset detection functions [ best and attuned to chicks vocalisation]
HFC_parameters = {'hop_length': 441, 'sr':44100, 'spec_num_bands':15, 'spec_fmin': 2500, 'spec_fmax': 5000, 'spec_fref': 2800,
                  'pp_threshold':  1.8, 'pp_pre_avg':25, 'pp_post_avg':1, 'pp_pre_max':3, 'pp_post_max':2,'global shift': 0.050, 'double_onset_correction': 0.1}

                  

# ###############################
audio_folder = 'C:\\Users\\anton\\Data_recording'

metadata = pd.read_csv("c:\\Users\\Documents\\Data\\metadata_exp.csv")


save_results_path = r'C:\\Users\\Documents\\Calls_detected_path'

if not os.path.exists(save_results_path):
    os.makedirs(save_results_path)

save_json_path = r'C:\\Users\\Documents\\Calls_detected_path\\Automated_extraction_calls_numbers'
if not os.path.exists(save_results_path):
    os.makedirs(save_results_path)

list_files = glob.glob(os.path.join(audio_folder, "*.wav"))

for file in tqdm(list_files):
    chick = os.path.basename(file)[:-4]  # Nome base senza estensione

    filtered_metadata = metadata[metadata['Filename'] == chick]

    if filtered_metadata.empty:
        print(f"File {chick} not found in metadata. Skipping...")
        continue

    exp_start = metadata[metadata['Filename'] == chick]['Start_exp_sec'].values[0]
    exp_end = metadata[metadata['Filename'] == chick]['End_exp_sec'].values[0]
    sex = metadata[metadata['Filename'] == chick]['Sex'].iloc[0]
    group = metadata[metadata['Filename'] == chick]['Group'].iloc[0]

    ##  High Frequency Content Onset Detection
        # Onset detection function for High frequency content that give back the onsets in seconds and the frames of the function
    hfc_pred_scnd,  HFC_activation_frames = onset_detectors.high_frequency_content(file, visualise_activation=True)

    hfc_pred_scnd= onset_utils.global_shift_correction(hfc_pred_scnd, 00.070)
    
    hfc_pred_scnd= onset_utils.double_onset_correction(hfc_pred_scnd, correction= 0.1)
    
    onsets_hfc= len(hfc_pred_scnd)


    # Compute offsets using the first-order difference method
    predicted_offsets = off_detection.offset_detection_first_order(file, hfc_pred_scnd)
    
    # Count total calls detected with HFC
    total_numb_calls_detected_with_hfc = len(hfc_pred_scnd)

    # # Save summary statistics in a JSON file
    calls_detected_hfc = {
        'audiofilename': chick,
        'Algorithm': 'High Frequency Content',
        'Number of calls': total_numb_calls_detected_with_hfc,
        'Group': group,
        'Sex': sex
    }
    json_path = os.path.join(save_json_path, f"{chick}_calls_detected.json")
    with open(json_path, 'w') as fp:
        json.dump(calls_detected_hfc, fp)



    # Optional: this function allow to select the start of experiment and and of the experiment in second when longer recording have been done
    hfc_pred_scnd, predicted_offsets = filter_calls_within_experiment( hfc_pred_scnd, predicted_offsets, exp_start, exp_end) 

    # Save detailed onsets and offsets in a .txt file
    detail_path = os.path.join(save_results_path, f"{chick}.txt")
    with open(detail_path, 'w') as file_out:
        for onset, offset in zip(hfc_pred_scnd, predicted_offsets):
            file_out.write(f"{onset:.4f}\t{offset:.4f}\n")

    print(f"Processed {file} and saved results to:\n{json_path} (Summary)\n{detail_path} (Detailed)")

print("Process completed.")