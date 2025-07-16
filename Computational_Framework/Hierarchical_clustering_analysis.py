import os
import glob
import pandas as pd
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
import librosa as lb
import librosa.display
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from clustering_utils import ( plot_umap_3d, plot_umap_2d, get_random_samples, plot_audio_segments, plot_dendrogram, 
                              statistical_report, create_statistical_report_with_radar_plots, plot_and_save_audio_segments)


from identikit_bioacoustic import call_spectrogram_generator


# features_path = 'C:\\Users\\anton\\VPA_vocalisations_project\\Features\\VPA'
features_path = 'data/features'            # Folder with extracted feature CSV files
metadata_path = 'data/metadata.csv'        # Path to your metadata file
audio_path = 'data/audio_files'            # Folder with the audio (.wav) files


def load_audio_data(audio_dir):
    audio_data = {}
    for fname in os.listdir(audio_dir):
        if fname.lower().endswith('.wav'):
            key = os.path.splitext(fname)[0]  # senza estensione
            path = os.path.join(audio_dir, fname)
            y, sr = lb.load(path, sr=None)
            audio_data[key] = (y, sr)
    return audio_data

# uso:
audio_data = load_audio_data(audio_path)



# # Path to save the results
clusterings_results_path = 'data/clusterings_results'  # Folder to save clustering results
# Create the results directory if it doesn't exist
if not os.path.exists(clusterings_results_path):
    os.makedirs(clusterings_results_path)


distance_model = pd.DataFrame(columns=['distance_to_closest_cluster', 'children'])
# Get a list of all CSV files in the directory
list_files = glob.glob(os.path.join(features_path, '*.csv'))

all_data = pd.concat([pd.read_csv(f) for f in list_files], ignore_index=True)
metadata = pd.read_csv(metadata_path)

# Drop NaN values
all_data = all_data.dropna()

# scale data with StandardScaler on used features only
scaler = StandardScaler()
features = all_data.drop(['Call Number', 'onsets_sec', 'offsets_sec','recording', 'call_id'], axis=1)
features_scaled = scaler.fit_transform(features)

# Here you should define the number of clusters you want to use based on the grid search findings
n_clusters = 3

agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', compute_distances=True)

cluster_membership = agg.fit_predict(features_scaled)

# Assign cluster memberships
all_data['cluster_membership'] = cluster_membership


linkage_matrix = linkage(features_scaled, method='ward')


# Compute cluster centers or representative points
cluster_centers = np.array([features_scaled[all_data['cluster_membership'] == i].mean(axis=0) for i in range(n_clusters)])

# Calculate distances of all points to their cluster centers ( this will give you an ide about the compactness of clusters, prototype calls, and outliers -farest points in the cluster)
distances_to_centers = cdist(features_scaled, cluster_centers, 'euclidean')
# Save the distances to the DataFrame
all_data['distance_to_center'] = [distances_to_centers[i, cluster] for i, cluster in enumerate(cluster_membership)]


# Save all_data with cluster membership and distances
all_data.to_csv(os.path.join(clusterings_results_path, f'hierarchical_clustering_{n_clusters}_distance_membership.csv'), index=False)


# Sort the DataFrame by cluster membership and distance to center
all_data_sorted = all_data.sort_values(by=['cluster_membership', 'distance_to_center'])

# Define percentiles for each cluster
# You can adjust these percentiles based on your analysis needs and obsserve statistical distribution of features based on the distance to center
percentiles = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

# Create a new column for percentiles
all_data_sorted['percentile'] = None

# Iterate through each cluster and apply percentiles
# This will assign percentiles based on the distance to center for each cluster
for cluster in range(n_clusters):
    cluster_mask = all_data_sorted['cluster_membership'] == cluster
    cluster_distances = all_data_sorted.loc[cluster_mask, 'distance_to_center']
    
    # check if all distances are the same
    if cluster_distances.nunique() < len(percentiles):  
        continue  

    # Apply percentiles
    all_data_sorted.loc[cluster_mask, 'percentile'] = pd.qcut(
        cluster_distances, len(percentiles), labels=percentiles, duplicates='drop'
    )

# Export the sorted data with percentiles
output_path = os.path.join(clusterings_results_path, f'hierarchical_clustering_{n_clusters}_percentiles.csv')
all_data_sorted.to_csv(output_path, index=False)
print(f"File saved: {output_path}")


# optionally, you can save percentiles as strings for better readability
all_data_sorted['percentile'] = all_data_sorted['percentile'].astype(str)

# Extract and observe the representative calls for each cluster ( plot spectrograms and save audio segments)
for cluster in range(n_clusters):
    # Select calls in the 5th percentile for each cluster
    calls_5th = all_data_sorted[
        (all_data_sorted['cluster_membership'] == cluster) &
        (all_data_sorted['percentile'] == '5')
    ].reset_index(drop=True)
    
    # take only the first 10 calls in the 5th percentile
    calls_5th = calls_5th.iloc[:10]
    
    # Check if there are enough calls in the 5th percentile
    output_folder = os.path.join(clusterings_results_path, f'cluster_{cluster}_5th_percentile_top10')
    os.makedirs(output_folder, exist_ok=True)
    
    # Salva anche il CSV per riferimento
    calls_5th.to_csv(os.path.join(output_folder, f'calls_5th_percentile_cluster_{cluster}_top10.csv'), index=False)
    
    # Plotta e salva gli spettrogrammi
    call_spectrogram_generator(calls_5th, audio_data, output_folder)


print("Clustering and audio segment processing completed successfully.")
# Plot UMAP in 3D
plot_umap_3d(features_scaled, cluster_membership, n_clusters, clusterings_results_path)

# Plot UMAP in 2D
plot_umap_2d(features_scaled, cluster_membership, n_clusters, clusterings_results_path)




# # code to observe the outliers in the UMAP embedding to customise based on the points images in the UMAP plot ( x dimensions: umap1, y dimensions: umap2)

# all_data['UMAP1'] = embedding[:, 0]
# all_data['UMAP2'] = embedding[:, 1]


# # Esempio di filtro per outlier spaziali (adatta i valori in base alla tua UMAP)
# umap_outliers = all_data[(all_data['UMAP1'] > 0) & (all_data['UMAP2'] < 2)]

# # Esporta CSV per controllarli
# umap_outliers.to_csv(os.path.join(clusterings_results_path, 'umap_outliers.csv'), index=False)


# # plot outliers in 10 samples per cycle
# for i in range(0, len(umap_outliers), 10):
#     sample = umap_outliers.iloc[i:i+10]
#     plot_and_save_audio_segments(
#         representative_calls=sample,
#         audio_path=audio_path,
#         save_path=clusterings_results_path,
#         cluster_label=f'outliers_a{i//10 + 1}'
#     )


# Plot the dendrogram and get the cluster memberships
membership = plot_dendrogram(agg, num_clusters=n_clusters)
if membership is not None:
    print(membership)


# Get 5 random samples for each cluster ( you can change the number of samples as needed)
random_samples = get_random_samples(all_data, 'cluster_membership', num_samples=5)

# Plot the audio segments and save audio files
plot_and_save_audio_segments(random_samples, audio_path, clusterings_results_path, 'cluster_membership')

# Plot the audio segments
plot_audio_segments(random_samples, audio_path, clusterings_results_path, 'cluster_membership')

# Get the statistical report
stats = statistical_report(all_data, n_clusters, metadata, clusterings_results_path)
print(stats)

# as a radar plot for a summary visualisation of all the features
radar= statistical_report_df = create_statistical_report_with_radar_plots(all_data, n_clusters, metadata, clusterings_results_path)
















