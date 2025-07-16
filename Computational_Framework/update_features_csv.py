import os
import pandas as pd
import glob

def add_recording_callid_columns(csv_directory):
    """
    Adds 'recording' and 'call_id' columns to each CSV file in the given directory.
    
    Args:
    csv_directory (str): Path to the directory containing CSV files.
    """
    # Find all CSV files in the directory that match the pattern
    csv_files = glob.glob(os.path.join(csv_directory, 'features_data_*.csv'))
    
    if not csv_files:
        print("No CSV files found in the directory.")
        return
    
    for csv_file in csv_files:
        # Extract the recording name from the CSV filename
        base_name = os.path.basename(csv_file)
        recording_name = base_name.replace('features_data_', '').replace('.csv', '')

        # Load the CSV file
        df = pd.read_csv(csv_file)

        # Add the 'recording' column
        df['recording'] = recording_name

        # Add the 'call_id' column using 'Call Number'
        if 'Call Number' in df.columns:
            df['call_id'] = df['recording'] + '_call_' + df['Call Number'].astype(str)
        else:
            print(f"Warning: 'Call Number' column missing in {csv_file}. Skipping...")
            continue

        # Save the updated DataFrame back to the CSV file
        df.to_csv(csv_file, index=False)
        
        print(f"Updated CSV: {csv_file} with 'recording' and 'call_id' columns.")

# Example usage
if __name__ == "__main__":
    # Directory containing your CSV files
    csv_directory = 'C:\\Users\\anton\\VPA_vocalisations_project\\Results_features_extraction'
    
    # Add the new columns to each CSV
    add_recording_callid_columns(csv_directory)
