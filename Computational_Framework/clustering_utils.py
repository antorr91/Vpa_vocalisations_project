import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from scipy.io import wavfile
import librosa as lb
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import skfuzzy as fuzz
from math import pi
import seaborn as sns
import itertools
import soundfile as sf
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




def split_data_recordings(all_data, test_size_per_group=2, group_col='recording'):
    """
    Split data into train and test sets based on a grouping column (e.g., recording).
    Ensures that no overlap exists between the train and test sets based on the grouping.

    Parameters:
    - all_data: DataFrame containing all data.
    - test_size_per_group: Number of groups (e.g., recordings) to include in the test set.
    - group_col: Column to group by (e.g., 'recording').

    Returns:
    - train_data: Training dataset.
    - test_data: Test dataset.
    """

    # change the code to have in the training set the recqording chick32_d0, chick34_d0, chick41_d0, chick85_d0, chick87_d0, chick89_d0, chick91_d0
    # then in the test set the reaiming recordings



    unique_groups = all_data[group_col].unique()
    test_groups = np.random.choice(unique_groups, test_size_per_group, replace=False)
    
    test_data = all_data[all_data[group_col].isin(test_groups)]
    train_data = all_data[~all_data[group_col].isin(test_groups)]
    
    return train_data, test_data




def plot_dendrogram(model, num_clusters=None, **kwargs):
    """
    Plot a dendrogram for hierarchical clustering.

    Args:
        model: The hierarchical clustering model (e.g., from scikit-learn).
        num_clusters (int): The number of clusters desired. If provided, a threshold line will be drawn on the dendrogram to indicate where to cut it.
        **kwargs: Additional keyword arguments to pass to the dendrogram function.

    Returns:
        threshold (float): The distance threshold at which to cut the dendrogram.
        linkage_matrix (numpy.ndarray): The linkage matrix used to construct the dendrogram.
        counts (numpy.ndarray): Counts of samples under each node in the dendrogram.
        n_samples (int): Total number of samples.
        labels (numpy.ndarray): Labels assigned to each sample by the clustering model.
    """
    
    # Create linkage matrix and then plot the dendrogram
    counts = np.zeros(model.children_.shape[0])  # Initialize counts of samples under each node
    n_samples = len(model.labels_)  # Total number of samples
    
    # Iterate over merges to calculate counts
    for i, merge in enumerate(model.children_):
        current_count = 0
        # Iterate over children of merge
        for child_idx in merge:
            if (child_idx < n_samples):
                current_count += 1  # Leaf node
            else:
                current_count += counts[child_idx - n_samples]  # Non-leaf node
        counts[i] = current_count  # Update counts
        
    # Construct the linkage matrix
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    
    # Debug: print the shape and content of the linkage matrix
    print(f"linkage_matrix shape: {linkage_matrix.shape}")
    print(f"linkage_matrix: {linkage_matrix}")

    # Compute the threshold for the dendrogram
    if num_clusters is not None:
        distances = linkage_matrix[:, 2]
        sorted_distances = np.sort(distances)
        threshold = sorted_distances[-num_clusters + 1]
    else:
        threshold = None

    # Plot the dendrogram with color_threshold
    dendrogram(linkage_matrix, color_threshold=threshold, **kwargs, leaf_rotation=90., leaf_font_size=5.0, truncate_mode='level', p=4, above_threshold_color='darkblue')

    # Plot the threshold line if num_clusters is specified
    if num_clusters is not None:
        plt.axhline(y=threshold, color='crimson', linestyle='--', label=f'{num_clusters} clusters', linewidth=1.5)
        plt.legend()
    
    plt.xlabel('Sample index or Cluster size', fontsize=16, fontfamily='Times New Roman')

    
    plt.ylabel('Euclidean distance (Ward’s linkage)', fontsize=16, fontfamily='Times New Roman')
    # plt.title('Hierarchical Clustering Dendrogram', fontsize=13, fontfamily='Palatino Linotype')
    # Save the dendrogram as an image in the save directory
    plt.tight_layout()
    plt.savefig('Hierarchical_Clustering_Dendrogram.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    return threshold, linkage_matrix, counts, n_samples, model.labels_


# Define the function to find the elbow point
def find_elbow_point(scores):
    n_points = len(scores)
    all_coord = np.vstack((range(n_points), scores)).T
    first_point = all_coord[0]
    line_vec = all_coord[-1] - all_coord[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    vec_from_first = all_coord - first_point
    scalar_product = np.sum(vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    best_index = np.argmax(dist_to_line)
    return best_index + 2  



# Function to get 5 random samples for each cluster
def get_random_samples(df, cluster_col, num_samples=5):
    random_samples = {}
    for cluster in df[cluster_col].unique():
        cluster_df = df[df[cluster_col] == cluster]
        if len(cluster_df) >= num_samples:
            random_samples[cluster] = cluster_df.sample(num_samples)
        else:
            random_samples[cluster] = cluster_df
    return random_samples




def segment_spectrogram(spectrogram, onsets, offsets, sr=44100):
    # Initialize lists to store spectrogram slices
    calls_S = []
    # Loop through each onset and offset pair
    for onset, offset in zip(onsets, offsets):
        # Convert time (in seconds) to sample indices
        onset_frames = lb.time_to_frames(onset, sr=sr)
        offset_frames = lb.time_to_frames(offset, sr=sr)

        call_spec = spectrogram[:, onset_frames: offset_frames]
        # call_audio = audio_data[onset_frames: offset_frames]

        # Append the scaled log-spectrogram slice to the calls list
        calls_S.append(call_spec)
        # calls_audio.append(call_audio)
    
    return calls_S

         

# Function to extract and plot audio segments
def plot_audio_segments(samples_dict, audio_path, clusterings_results_path, cluster_membership_label):
    for cluster, samples in samples_dict.items():
        fig, axes = plt.subplots(1, len(samples), figsize=(2 * len(samples), 2))
        fig.suptitle(f'Cluster {cluster} Audio Segments')
        if len(samples) == 1:
            axes = [axes]

        for idx, (i, sample) in enumerate(samples.iterrows()):
            audio_file = os.path.join(audio_path, sample['recording'] + '.wav')
            if os.path.exists(audio_file):
                # Load the audio file with librosa
                data, sr = lb.load(audio_file, sr=44100)
                
                # Compute the mel spectrogram
                S = lb.feature.melspectrogram(y=data, sr=sr, n_mels=128, fmin=2000, fmax=10000)
                log_S = lb.power_to_db(S, ref=np.max)

                # Segment the spectrogram
                calls_S = segment_spectrogram(log_S, [sample['onsets_sec']], [sample['offsets_sec']], sr=sr)
                call_S = calls_S[0]

                # Convert onset seconds with decimals to readable format
                # onset_sec = sample['onsets_sec']
                # if onset_sec < 60:
                #     onset_time = f"{onset_sec:.2f} sec"
                # else:
                #     minutes = int(onset_sec // 60)
                #     seconds = onset_sec % 60
                #     onset_time = f"{minutes} min & {seconds:.2f} sec"

                # Plot the audio segment
                img= axes[idx].imshow(call_S, aspect='auto', origin='lower', cmap='magma')
                axes[idx].set_title(f'Call {idx + 1} of {sample["recording"]} \n cluster {cluster}', fontsize=6)

                axes[idx].set_xlabel('Time', fontsize=5)
                axes[idx].set_ylabel('Frequency', fontsize=5)
                fig.colorbar(img, ax=axes[idx])
            else:
                print(f'Audio file {audio_file} not found')

        # Save the plot
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename = f'cluster_{cluster}_{cluster_membership_label}.png'
        plt.savefig(os.path.join(clusterings_results_path, plot_filename))





def plot_and_save_audio_segments(representative_calls, audio_path, save_path, cluster_label):
    """
    Extract, plot, and save audio segments for representative calls of a cluster.
    
    Args:
    representative_calls (DataFrame): DataFrame containing representative calls for a cluster
    audio_path (str): Path to the directory containing audio files
    save_path (str): Path to save the results
    cluster_label (str): Label for the cluster
    """
    fig, axes = plt.subplots(1, len(representative_calls), figsize=(2 * len(representative_calls), 2))
    fig.suptitle(f'{cluster_label} Audio Segments')
    if len(representative_calls) == 1:
        axes = [axes]

    cluster_audio_dir = os.path.join(save_path, 'audio')
    os.makedirs(cluster_audio_dir, exist_ok=True)
    
    # make rapresentative_calls a dataframe
    representative_calls = pd.DataFrame(representative_calls)

    for idx, (_, call) in enumerate(representative_calls.iterrows()):
        audio_file = os.path.join(audio_path, call['recording'] + '.wav')
        if os.path.exists(audio_file):
            data, sr = lb.load(audio_file, sr=44100)
            S = lb.feature.melspectrogram(y=data, sr=sr, n_mels=128, fmin=2000, fmax=10000)
            log_S = lb.power_to_db(S, ref=np.max)

            # Use your existing segment_spectrogram function
            calls_S = segment_spectrogram(log_S, [call['onsets_sec']], [call['offsets_sec']], sr=sr)
            call_S = calls_S[0]

            # Extract audio segment
            onset_samples = int(call['onsets_sec'] * sr)
            offset_samples = int(call['offsets_sec'] * sr)
            call_audio = data[onset_samples:offset_samples]

            img = axes[idx].imshow(call_S, aspect='auto', origin='lower', cmap='magma')
            axes[idx].set_title(f'Call {idx + 1} of {call["call_id"]} \n {cluster_label}', fontsize=6)
            axes[idx].set_xlabel('Time', fontsize=5)
            axes[idx].set_ylabel('Frequency', fontsize=5)
            fig.colorbar(img, ax=axes[idx])

            # Save the audio file
        #     audio_filename = os.path.join(cluster_audio_dir, f'call_{idx + 1}_{call["call_id"]}.wav')
        #     sf.write(audio_filename, call_audio, sr)
        # else:
        #     print(f'Audio file {audio_file} not found')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = f'{cluster_label}.png'
    plt.savefig(os.path.join(save_path, plot_filename))
    plt.close(fig)





def silhouette_visualizer(data, n_clusters, title, method='gmm', **kwargs):
    """
    Visualizza il coefficiente silhouette per vari metodi di clustering.
    
    Args:
        data (array-like): Dati da clusterizzare.
        n_clusters (int): Numero di cluster desiderati.
        title (str): Titolo del grafico.
        method (str): Metodo di clustering da usare ('fcm', 'gmm', 'dbscan', 'agglomerative').
        **kwargs: Parametri aggiuntivi per il metodo di clustering scelto.
    """
    # Esegui il clustering in base al metodo specificato
    if method == 'fcm':
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            data.T, n_clusters, 2, error=0.005, maxiter=1000, init=None
        )
        cluster_labels = np.argmax(u, axis=0)
    elif method == 'gmm':
        model = GaussianMixture(n_components=n_clusters, **kwargs)
        cluster_labels = model.fit_predict(data)
    elif method == 'dbscan':
        model = DBSCAN(**kwargs)
        cluster_labels = model.fit_predict(data)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    elif method == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
        cluster_labels = model.fit_predict(data)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Compute the silhouette score
    silhouette_avg = silhouette_score(data, cluster_labels)
    sample_silhouette_values = silhouette_samples(data, cluster_labels)
    
    print(f"Average silhouette score: {silhouette_avg}")

    # Create the silhouette plot
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(9, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        # Usa colori più chiari
        color = sns.color_palette("pastel", n_colors=n_clusters)[i]
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.set_title(title)
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks(np.arange(-0.1, 1.1, 0.2))

    plt.tight_layout()
    plt.show()


def statistical_report(all_data, n_clusters, metadata, output_folder, include_noise=True):
    # os.makedirs(output_folder, exist_ok=True)

    # Drop unnecessary columns
    all_data = all_data.drop(['recording', 'Call Number', 'onsets_sec', 'offsets_sec', 'call_id'], axis=1)

    # Select numeric columns, excluding 'cluster_membership'
    numeric_cols = all_data.select_dtypes(include=['number']).columns
    numeric_cols = numeric_cols.drop('cluster_membership', errors='ignore')

    # Compute mean values per cluster
    statistical_report_df = all_data.groupby('cluster_membership')[numeric_cols].mean().reset_index()

    # Add number of samples per cluster
    statistical_report_df['n_samples'] = all_data['cluster_membership'].value_counts().sort_index().values

    # Save CSV and LaTeX report
    csv_file_path = os.path.join(output_folder, 'statistical_report.csv')
    statistical_report_df.to_csv(csv_file_path, index=False)
    
    latex_file_path = os.path.join(output_folder, 'statistical_report.tex')
    statistical_report_df.to_latex(latex_file_path, index=False)

    # Identify clusters (handle -1 for DBSCAN)
    if include_noise:
        valid_clusters = sorted(all_data['cluster_membership'].unique())  # Include -1
    else:
        valid_clusters = sorted([c for c in all_data['cluster_membership'].unique() if c != -1])

    if not valid_clusters:
        print("Nessun cluster valido trovato. Interrompo la generazione dei grafici.")
        return statistical_report_df  

    # Define colors (add gray for noise if included)
    colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
    color_map = {k: colors[i % len(colors)] for i, k in enumerate(valid_clusters)}
    if include_noise and -1 in valid_clusters:
        color_map[-1] = 'gray'  # Assign gray to noise

    # Define plotting parameters
    features = list(numeric_cols)  
    num_features = len(features)
    features_per_plot = 4  
    num_plots = (num_features + features_per_plot - 1) // features_per_plot  

    for i in range(num_plots):
        start_idx = i * features_per_plot
        end_idx = min(start_idx + features_per_plot, num_features)
        plot_features = features[start_idx:end_idx]

        fig, axs = plt.subplots(1, len(plot_features), figsize=(20, 5))

        if len(plot_features) == 1:
            axs = [axs]  # Ensure axs is iterable even for one subplot

        for j, feature in enumerate(plot_features):
            data = [all_data[all_data['cluster_membership'] == k][feature] for k in valid_clusters]

            # Boxplot con cluster validi (include noise se attivo)
            bplot = axs[j].boxplot(data, patch_artist=True, showfliers=False, positions=range(len(valid_clusters)))

            for patch, k in zip(bplot['boxes'], valid_clusters):
                patch.set_alpha(0.3)
                patch.set_facecolor(color_map[k])

            # Scatter overlay: ben allineato con i boxplot
            for idx, cluster in enumerate(valid_clusters):
                cluster_data = all_data[all_data['cluster_membership'] == cluster]
                axs[j].scatter([idx] * len(cluster_data), cluster_data[feature], 
                               alpha=0.3, color=color_map[cluster], edgecolor=color_map[cluster], s=13)

            axs[j].set_title(f'{feature} per cluster', fontsize=10)
            axs[j].set_xlabel('Cluster', fontsize=9)
            axs[j].set_ylabel('Value', fontsize=9)
            axs[j].set_xticks(range(len(valid_clusters)))
            axs[j].set_xticklabels([str(k) for k in valid_clusters])

        plt.tight_layout()
        plot_file_path = os.path.join(output_folder, f'statistical_report_part_{i+1}.png')
        plt.savefig(plot_file_path)
        plt.close()

    return statistical_report_df

    # # To plot all features in a single plot ( suggested only if  feature are less than 10)
    # fig, axs = plt.subplots(nrows=6, ncols=5, figsize=(40, 25))  # Create a grid of subplots

    # # Flatten the axes array for easier indexing
    # axs = axs.flatten()

    # for j, feature in enumerate(features):
    #     data = [all_data[all_data['cluster_membership'] == k][feature] for k in range(n_clusters)]
        
    #     bplot = axs[j].boxplot(data, patch_artist=True, showfliers= True)
    #     for patch, k in zip(bplot['boxes'], range(n_clusters)):
    #         patch.set_facecolor(color_map[k])
    #         patch.set_alpha(0.1)

    #         # Scatterplot overlay
    #     for cluster in range(n_clusters):
    #         cluster_data = all_data[all_data['cluster_membership'] == cluster]
    #         axs[j].scatter([cluster + 1] * len(cluster_data), cluster_data[feature], 
    #                         alpha=0.3, c=color_map[cluster], edgecolor='k', s=13, label=f'Cluster {cluster}')
            
    #         axs[j].set_title(f'{feature}', size=7)
    #         axs[j].set_xlabel('Clusters', size=5)
    #         axs[j].set_ylabel('Value', size=5)

    
    # # remove unused axes if there are fewer features than subplots
    # for j in range(len(features), len(axs)):
    #     fig.delaxes(axs[j])

    # plt.tight_layout()

    # # Save and export the plot
    # plot_file_path = os.path.join(output_folder, 'statistical_report_all_features.png')
    # plt.savefig(plot_file_path)
    # plt.show()

    # return statistical_report_df



def create_statistical_report_with_radar_plot(all_data, n_clusters, metadata, output_folder):
    """
    Create a statistical report with radar plots for each cluster.
    """
    
    # List of features to analyse
    features_to_analyze = [
        'Duration_call', 'F0 Mean', 'F0 Std', 'F0 Skewness', 'F0 Kurtosis', 
        'F0 Bandwidth', 'F0 1st Order Diff', 'F0 Slope', 'F0 Mag Mean',
        'F1 Mag Mean', 'F2 Mag Mean', 'F1-F0 Ratio', 'F2-F0 Ratio',
        'Spectral Centroid Mean', 'Spectral Centroid Std', 'RMS Mean', 
        'RMS Std', 'Slope', 'Attack_magnitude', 'Attack_time'
    ]
    
    # Create a copy of the data to avoid modifying the original DataFrame
    data_copy = all_data.copy()
    
    # Rename columns to match the features to analyse
    data_copy = data_copy.rename(columns={
        'Duration_call': 'Duration call',
        'Attack_time': 'Attack time',
        'Attack_magnitude': 'Attack magnitude'
    })
    
    # update the features to analyse with the renamed columns
    features_to_analyze = [
        'Duration call', 'F0 Mean', 'F0 Std', 'F0 Skewness', 'F0 Kurtosis', 
        'F0 Bandwidth', 'F0 1st Order Diff', 'F0 Slope', 'F0 Mag Mean',
        'F1 Mag Mean', 'F2 Mag Mean', 'F1-F0 Ratio', 'F2-F0 Ratio',
        'Spectral Centroid Mean', 'Spectral Centroid Std', 'RMS Mean', 
        'RMS Std', 'Slope', 'Attack magnitude', 'Attack time'
    ]
    
    # check if all features to analyze are present in the data
    missing_features = [f for f in features_to_analyze if f not in data_copy.columns]
    if missing_features:
        print(f"Warning: The following features are missing from the data and will be excluded from the analysis: {missing_features}")
        features_to_analyze = [f for f in features_to_analyze if f in data_copy.columns]
    
    print(f"Features utilizzate per l'analisi: {features_to_analyze}")
    print(f"Numero di features: {len(features_to_analyze)}")
    
    # Separate the cluster membership column ( if not this will results in unreadable radar plots)
    cluster_membership_col = data_copy['cluster_membership']
    features_data = data_copy[features_to_analyze]
    
    # scale the features data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_data)
    
    # Convert scaled features back to a DataFrame
    scaled_features_df = pd.DataFrame(scaled_features, columns=features_to_analyze)
    
    # Add the cluster membership column to the scaled features DataFrame
    scaled_features_df['cluster_membership'] = cluster_membership_col.values
    
    # Compute the statistical report for scaled features
    statistical_report_df = scaled_features_df.groupby('cluster_membership')[features_to_analyze].mean().reset_index()
    
    # Add number of samples per cluster
    n_samples = scaled_features_df['cluster_membership'].value_counts().sort_index()
    statistical_report_df['n_samples'] = n_samples.values
    
    # Save the statistical report to CSV
    csv_file_path = os.path.join(output_folder, 'statistical_report_scaled.csv')
    statistical_report_df.to_csv(csv_file_path, index=False)
    
    # Save the original statistics for comparison
    original_stats = data_copy.groupby('cluster_membership')[features_to_analyze].agg(['mean', 'std']).reset_index()
    original_stats_file = os.path.join(output_folder, 'statistical_report_original.csv')
    original_stats.to_csv(original_stats_file, index=False)
    
    # Convert the statistical report to LaTeX format
    latex_file_path = os.path.join(output_folder, 'statistical_report.tex')
    statistical_report_df.to_latex(latex_file_path, index=False)
    
    # color palette for radar plots
    colors = ['blue', 'mediumseagreen','red', '#FF8C00', '#9932CC', '#8B4513', '#FF69B4', '#708090', '#808000', '#00CED1']
    
    # Creta radar plots for each cluster
    create_individual_radar_plots(statistical_report_df, features_to_analyze, colors, output_folder)
    
    # Create a comparative radar plot with all clusters
    create_comparative_radar_plot(statistical_report_df, features_to_analyze, colors, output_folder)
    
    # PRInt the statistical report
    print(f"Cluster statistics (scaled data):")
    for i, row in statistical_report_df.iterrows():
        cluster_id = int(row["cluster_membership"])
        n_samples = row["n_samples"]
        print(f"Cluster {cluster_id}: {n_samples} samples")
        print(f"  Mean values range: {row[features_to_analyze].min():.3f} to {row[features_to_analyze].max():.3f}")
        print(f"  Standard deviation of means: {row[features_to_analyze].std():.3f}")
    
    return statistical_report_df






def create_individual_radar_plots(statistical_report_df, features_to_analyze, colors, output_folder):
    """Crea radar plot singoli ad alta qualità per ogni cluster"""
    
    num_features = len(features_to_analyze)
    angles = [n / float(num_features) * 2 * pi for n in range(num_features)]
    angles += angles[:1]
    
    # Create output folder for individual plots
    single_plots_folder = os.path.join(output_folder, 'individual_radar_plots')
    if not os.path.exists(single_plots_folder):
        os.makedirs(single_plots_folder)
    
    # Compare global min and max for consistent y-limits
    # Collect all values from the features to analyze
    all_values = []
    for _, row in statistical_report_df.iterrows():
        all_values.extend(row[features_to_analyze].values)
    
    global_min = min(all_values)
    global_max = max(all_values)
    y_limit_min = global_min - 0.3
    y_limit_max = global_max + 0.3
    
    for i, row in statistical_report_df.iterrows():
        # Configura matplotlib per alta qualità
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 14
        
        # Crea figura singola grande
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
        
        # Prepare the values for the radar plot
        values = list(row[features_to_analyze].values)
        values += values[:1]  # close the circle
        
        color = colors[i % len(colors)]
        cluster_id = int(row["cluster_membership"])
        n_samples = row["n_samples"]
        
        # set axes
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        
        # labels for the axes
        ax.set_xticklabels(features_to_analyze, color='black', size=19, 
                          fontname='Times New Roman')
        
        # Plot radar
        ax.plot(angles, values, color=color, linewidth=3, linestyle='solid', alpha=0.9)
        ax.fill(angles, values, color=color, alpha=0.3)
        
        # Title section
        ax.set_title(f'Cluster {cluster_id}\n({n_samples} samples)', 
                    size=30,  color='black', y=1.08, fontname='Times New Roman', weight='bold')
        
        # set limits for the radial axis
        ax.set_ylim(y_limit_min, y_limit_max)
        
        # trasparent grid level
        ax.grid(True, alpha=0.3)
        
        # Add radial ticks  
        radial_ticks = np.linspace(y_limit_min, y_limit_max, 5)
        ax.set_yticks(radial_ticks)
        ax.set_yticklabels([f'{tick:.1f}' for tick in radial_ticks], 
                          size=10, color='gray')
        
        # Add a reference line for zero if applicable
        if y_limit_min < 0 < y_limit_max:
            zero_line = [0] * (num_features + 1)
            ax.plot(angles, zero_line, color='black', linewidth=1, linestyle='--', alpha=0.5)
        
        # Save the radar plot as PNG and PDF
        plt.tight_layout()
        plot_file_path = os.path.join(single_plots_folder, f'cluster_{cluster_id}_radar_plot.png')
        plt.savefig(plot_file_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        # Save as PDF
        plot_file_path_pdf = os.path.join(single_plots_folder, f'cluster_{cluster_id}_radar_plot.pdf')
        plt.savefig(plot_file_path_pdf, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        plt.close()
        
        print(f"Radar plot per Cluster {cluster_id} salvato in: {plot_file_path}")


        

def create_comparative_radar_plot(statistical_report_df, features_to_analyze, colors, output_folder):
    """Create a comparative radar plot for all clusters"""
    
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    
    num_features = len(features_to_analyze)
    angles = [n / float(num_features) * 2 * pi for n in range(num_features)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(14, 12), subplot_kw=dict(polar=True))
    
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features_to_analyze, color='black', size=14, 
                      fontname='Times New Roman')
    
    # Compare global min and max for consistent y-limits
    all_values = []
    for _, row in statistical_report_df.iterrows():
        all_values.extend(row[features_to_analyze].values)
    
    global_min = min(all_values)
    global_max = max(all_values)
    y_limit_min = global_min - 0.3
    y_limit_max = global_max + 0.3
    
    # Plot each cluster
    for i, row in statistical_report_df.iterrows():
        values = list(row[features_to_analyze].values)
        values += values[:1]  # Chiudi il cerchio
        
        color = colors[i % len(colors)]
        cluster_id = int(row["cluster_membership"])
        n_samples = row["n_samples"]
        
        ax.plot(angles, values, color=color, linewidth=3, linestyle='solid', 
                label=f'Cluster {cluster_id} (n={n_samples})', alpha=0.9)
        ax.fill(angles, values, color=color, alpha=0.2)
    
    # Title & layout
    ax.set_title('Comparison of All Clusters', size=20, color='black', y=1.08, 
                fontname='Times New Roman', weight='bold')
    
    # Legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0), fontsize=12, 
             frameon=True, fancybox=True, shadow=True)
    
    # set limits for the radial axis
    ax.set_ylim(y_limit_min, y_limit_max)
    ax.grid(True, alpha=0.3)
    
    # Tick radiali
    radial_ticks = np.linspace(y_limit_min, y_limit_max, 5)
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([f'{tick:.1f}' for tick in radial_ticks], 
                      size=10, color='gray')
    
    # Line for zero if applicable
    if y_limit_min < 0 < y_limit_max:
        zero_line = [0] * (num_features + 1)
        ax.plot(angles, zero_line, color='black', linewidth=1, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save the comparative radar plot as PNG and PDF
    plot_file_path_png = os.path.join(output_folder, 'radar_plot_comparison_hq.png')
    plot_file_path_pdf = os.path.join(output_folder, 'radar_plot_comparison_hq.pdf')
    
    plt.savefig(plot_file_path_png, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.savefig(plot_file_path_pdf, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    
    plt.close()
    
    print(f"Radar plot comparativo salvato in: {plot_file_path_png}")
    print(f"Radar plot comparativo PDF salvato in: {plot_file_path_pdf}")

def create_statistical_report_with_radar_plots(all_data, n_clusters, metadata, output_folder):
    
    # Lista delle 20 features che vuoi analizzare
    features_to_analyze = [
        'Duration_call', 'F0 Mean', 'F0 Std', 'F0 Skewness', 'F0 Kurtosis', 
        'F0 Bandwidth', 'F0 1st Order Diff', 'F0 Slope', 'F0 Mag Mean',
        'F1 Mag Mean', 'F2 Mag Mean', 'F1-F0 Ratio', 'F2-F0 Ratio',
        'Spectral Centroid Mean', 'Spectral Centroid Std', 'RMS Mean', 
        'RMS Std', 'Slope', 'Attack_magnitude', 'Attack_time'
    ]
    
    # Crea una copia dei dati per non modificare l'originale
    data_copy = all_data.copy()
    
    # Rinomina colonne specifiche per rimuovere gli underscore
    data_copy = data_copy.rename(columns={
        'Duration_call': 'Duration call',
        'Attack_time': 'Attack time',
        'Attack_magnitude': 'Attack magnitude'
    })
    
    # Aggiorna la lista delle features con i nuovi nomi
    features_to_analyze = [
        'Duration call', 'F0 Mean', 'F0 Std', 'F0 Skewness', 'F0 Kurtosis', 
        'F0 Bandwidth', 'F0 1st Order Diff', 'F0 Slope', 'F0 Mag Mean',
        'F1 Mag Mean', 'F2 Mag Mean', 'F1-F0 Ratio', 'F2-F0 Ratio',
        'Spectral Centroid Mean', 'Spectral Centroid Std', 'RMS Mean', 
        'RMS Std', 'Slope', 'Attack magnitude', 'Attack time'
    ]
    
    # Verifica che tutte le features esistano nei dati
    missing_features = [f for f in features_to_analyze if f not in data_copy.columns]
    if missing_features:
        print(f"Attenzione: le seguenti features non sono state trovate nei dati: {missing_features}")
        features_to_analyze = [f for f in features_to_analyze if f in data_copy.columns]
    
    print(f"Features utilizzate per l'analisi: {features_to_analyze}")
    print(f"Numero di features: {len(features_to_analyze)}")
    
    # Separare la colonna 'cluster_membership' dalle features
    cluster_membership_col = data_copy['cluster_membership']
    features_data = data_copy[features_to_analyze]
    
    # Applicare lo scaling alle features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_data)
    
    # Convertire l'array scalato in DataFrame
    scaled_features_df = pd.DataFrame(scaled_features, columns=features_to_analyze)
    
    # Riaggiungere la colonna 'cluster_membership'
    scaled_features_df['cluster_membership'] = cluster_membership_col.values
    
    # Calcola le statistiche sui dati SCALATI per il radar plot
    statistical_report_df = scaled_features_df.groupby('cluster_membership')[features_to_analyze].mean().reset_index()
    
    # Aggiungi il numero di campioni in ogni cluster
    n_samples = scaled_features_df['cluster_membership'].value_counts().sort_index()
    statistical_report_df['n_samples'] = n_samples.values
    
    # Salva il report statistico su file CSV (dati scalati)
    csv_file_path = os.path.join(output_folder, 'statistical_report_scaled.csv')
    statistical_report_df.to_csv(csv_file_path, index=False)
    
    # Salva anche il report con dati originali per interpretazione
    original_stats = data_copy.groupby('cluster_membership')[features_to_analyze].agg(['mean', 'std']).reset_index()
    original_stats_file = os.path.join(output_folder, 'statistical_report_original.csv')
    original_stats.to_csv(original_stats_file, index=False)
    
    # Converti ed esporta il report statistico in LaTeX
    latex_file_path = os.path.join(output_folder, 'statistical_report.tex')
    statistical_report_df.to_latex(latex_file_path, index=False)
    
    # Creare radar plot per visualizzare le variazioni delle feature per cluster
    num_features = len(features_to_analyze)
    num_clusters = len(statistical_report_df)
    colors = ['green', 'red', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Funzione per creare un singolo radar plot
    def create_radar_plot(data, title, color, ax):
        categories = list(data.keys())
        values = list(data.values())
        values += values[:1]  # Chiudi il cerchio
        
        angles = [n / float(num_features) * 2 * pi for n in range(num_features)]
        angles += angles[:1]
        
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, color='black', size=6.8, fontname='Times New Roman')
        
        ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
        ax.fill(angles, values, color=color, alpha=0.4)
        
        ax.set_title(title, size=15, color='black', y=1.1, fontname='Times New Roman')
        
        # Aggiungi griglia radiale
        ax.grid(True)
        
        # Imposta i limiti dell'asse radiale per una migliore visualizzazione
        min_val = min(values[:-1])
        max_val = max(values[:-1])
        ax.set_ylim(min_val - 0.5, max_val + 0.5)
    
    # Creare una griglia di radar plot
    num_cols = min(3, num_clusters)
    num_rows = int(np.ceil(num_clusters / num_cols))
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5), subplot_kw=dict(polar=True))
    
    if num_clusters == 1:
        axs = [axs]
    elif num_rows == 1:
        axs = axs if num_clusters > 1 else [axs]
    else:
        axs = axs.flatten()
    
    for i, row in statistical_report_df.iterrows():
        data = row[features_to_analyze].to_dict()
        color = colors[i % len(colors)]
        
        create_radar_plot(data, f'Cluster {int(row["cluster_membership"])}', color, axs[i])
    
    # Rimuovi assi non utilizzati
    if num_clusters < len(axs):
        for j in range(num_clusters, len(axs)):
            fig.delaxes(axs[j])
    
    plt.tight_layout()
    plot_file_path = os.path.join(output_folder, 'radar_plots_clusters.png')
    plt.savefig(plot_file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a comparative radar plot with all clusters
    create_comparative_radar_plot(statistical_report_df, features_to_analyze, colors, output_folder)
    
    # print the statistical report
    print(f"Cluster statistics (scaled data):")
    for i, row in statistical_report_df.iterrows():
        cluster_id = int(row["cluster_membership"])
        n_samples = row["n_samples"]
        print(f"Cluster {cluster_id}: {n_samples} samples")
        print(f"  Mean values range: {row[features_to_analyze].min():.3f} to {row[features_to_analyze].max():.3f}")
        print(f"  Standard deviation of means: {row[features_to_analyze].std():.3f}")
    
    return statistical_report_df

def create_comparative_radar_plot(statistical_report_df, features_to_analyze, colors, output_folder):
    """Crea un radar plot comparativo con tutti i cluster sovrapposti"""
    
    num_features = len(features_to_analyze)
    angles = [n / float(num_features) * 2 * pi for n in range(num_features)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features_to_analyze, color='black', size=10, fontname='Times New Roman')
    
    # Plotta ogni cluster
    for i, row in statistical_report_df.iterrows():
        values = list(row[features_to_analyze].values)
        values += values[:1]  # Chiudi il cerchio
        
        color = colors[i % len(colors)]
        cluster_id = int(row["cluster_membership"])
        
        ax.plot(angles, values, color=color, linewidth=2, linestyle='solid', 
                label=f'Cluster {cluster_id}')
        ax.fill(angles, values, color=color, alpha=0.2)
    
    ax.set_title('Confronto tra tutti i cluster', size=16, color='black', y=1.1, fontname='Times New Roman')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plot_file_path = os.path.join(output_folder, 'radar_plot_comparison.png')
    plt.savefig(plot_file_path, dpi=300, bbox_inches='tight')
    plt.close()




def radarplot_individual(all_data, output_folder):
    # Rimuovere colonne non necessarie
    all_data = all_data.drop(['Call Number', 'onsets_sec', 'offsets_sec', 'call_id'], axis=1)

    # Apply scaling to the features
    scaled_features = StandardScaler().fit_transform(all_data.drop(['recording'], axis=1))

    # Convert the scaled features to a DataFrame
    scaled_features_df = pd.DataFrame(scaled_features, columns=all_data.columns[:-1])

    # add the 'recording' column back to the DataFrame
    scaled_features_df['recording'] = all_data['recording'].values

    # group by 'recording' and calculate the mean for each feature
    statistical_report_df = scaled_features_df.groupby('recording').mean().reset_index()

    # Add the number of samples for each recording
    n_samples = scaled_features_df['recording'].value_counts().sort_index()
    statistical_report_df['n_samples'] = n_samples.values

    # Save and export
    csv_file_path = os.path.join(output_folder, 'statistical_report.csv')
    statistical_report_df.to_csv(csv_file_path, index=False)

    # Convert and export the statistical report to LaTeX
    latex_file_path = os.path.join(output_folder, 'statistical_report.tex')
    statistical_report_df.to_latex(latex_file_path, index=False)

    # Prepare features for radar plot
    features = list(all_data.columns[:-1])  # take off 'recording'
    num_features = len(features)
    num_files = len(statistical_report_df)

    custom_colors = [
        "steelblue", "darkcyan", "mediumseagreen", 
        "indianred", "goldenrod", "orchid", 
        "lightskyblue", "limegreen", "tomato", 
        "mediumslateblue", "darkolivegreen", "cornflowerblue"
    ]

    # Function to create a radar plot for each recording
    def create_radar_plot(data, title, color, ax):
        categories = list(data.keys())
        values = list(data.values())
        values += values[:1]  # Close the circle

        angles = [n / float(num_features) * 2 * pi for n in range(num_features)]
        angles += angles[:1]

        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, color='dimgrey', size=5, fontweight='bold')
        
        ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
        ax.fill(angles, values, color=color, alpha=0.3)

        ax.set_title(title, size=12, color=color, y=1.1)

    # Create a grid of radar plots
    num_cols = 3
    num_rows = int(np.ceil(num_files / num_cols))
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 9, num_rows * 9), subplot_kw=dict(polar=True))

    axs = axs.flatten()

    for i, row in statistical_report_df.iterrows():
        data = row[features].to_dict()
        color = custom_colors[i % len(custom_colors)]
        
        # Check if the axis exists before plotting
        if i < len(axs) and axs[i] is not None:
            create_radar_plot(data, f'Recording of {row["recording"]}', color, axs[i])

    # Remove unused axes
    for j in range(len(statistical_report_df), len(axs)):
        if axs[j] is not None:
            fig.delaxes(axs[j])

    plt.tight_layout()
    plot_file_path = os.path.join(output_folder, 'radar_plots_recordings.png')
    plt.savefig(plot_file_path)
    # plt.show()

    # Add support for multiple files if there are more than 6 clusters to plot
    if num_files > 6:
        for k in range(2):
            fig, axs = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(polar=True))
            axs = axs.flatten()
            
            start_idx = k * 6
            end_idx = min(start_idx + 6, num_files)
            
            for idx in range(start_idx, end_idx):
                data = statistical_report_df.iloc[idx][features].to_dict()
                color = custom_colors[idx % len(custom_colors)]
                
                create_radar_plot(data, f'Recording of {statistical_report_df.iloc[idx]["recording"]}', color, axs[idx - start_idx])
            
            plt.tight_layout()
            plot_file_path = os.path.join(output_folder, f'radar_plots_recordings_{k + 1}.png')
            plt.savefig(plot_file_path)

    return statistical_report_df



def plot_and_save_extreme_calls(audio_data, audio_path, clusterings_results_path):
    """
    Extract, plot, and save spectrograms and audio of selected calls.

    Args:
    audio_data (pd.DataFrame): DataFrame containing audio metadata and cluster probabilities
    audio_path (str): Path to the directory containing audio files
    clusterings_results_path (str): Path to save the results
    """
    for idx, sample in audio_data.iterrows():
        try:
            audio_file = os.path.join(audio_path, sample['call_id'].split('_')[0] + '_d0.wav')

            if os.path.exists(audio_file):
                data, sr = lb.load(audio_file, sr=44100)

                # Extract the specific call
                onset_samples = int(sample['onsets_sec'] * sr)
                offset_samples = int(sample['offsets_sec'] * sr)
                call_audio = data[onset_samples:offset_samples]

                # Compute mel spectrogram
                S = lb.feature.melspectrogram(y=call_audio, sr=sr, n_mels=128, fmin=2000, fmax=10000)
                log_S = lb.power_to_db(S, ref=np.max)

                # Plot the spectrogram
                fig, ax = plt.subplots(figsize=(10, 5))
                img = ax.imshow(log_S, aspect='auto', origin='lower', cmap='magma')
                ax.set_title(f'{sample["call_id"]}_clustered in:_{sample["cluster_membership"]}', fontsize=14)
                ax.set_xlabel('Time (s)', fontsize=12)
                ax.set_ylabel('Frequency (Hz)', fontsize=12)
                fig.colorbar(img, ax=ax, format='%+2.0f dB')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                # Save the individual spectrogram
                spectrogram_filename = os.path.join(clusterings_results_path, f'{sample["call_id"]}_clustered_in_{sample["cluster_membership"]}_as_{sample["point_type"]}.png')
                plt.savefig(spectrogram_filename)
                plt.close(fig)  # Close the figure to free up memory

                # Save the audio
                audio_filename = os.path.join(clusterings_results_path, f'top_call_{sample["call_id"]}_clustered_in_{sample["cluster_membership"]}_as_{sample["point_type"]}.wav')
                sf.write(audio_filename, call_audio, sr)

                print(f"Processed call {sample['call_id']}")
            else:
                print(f'Audio file {audio_file} not found')

        except Exception as e:
            print(f"Error processing call {sample['call_id']}: {str(e)}")

    print(f"Processed all {len(audio_data)} calls.")


    # Function to get representative calls by percentile
def get_representative_calls_by_percentile(cluster_data, percentiles, n_calls=10):
    total_calls = len(cluster_data)
    percentile_indices = [int(np.percentile(range(total_calls), p)) for p in percentiles]

    representative_calls = []
    for idx in percentile_indices:
        start_idx = max(0, idx - n_calls//2)
        end_idx = min(total_calls, idx + n_calls//2)
        representative_calls.append(cluster_data.iloc[start_idx:end_idx])
    
    return representative_calls



def get_representative_calls_by_threshold(cluster_data, cluster_num, audio_path, results_path, n_calls=25):
    """
    Select representative calls for a given cluster, and determine a threshold based on the distance to the cluster center.

    Parameters:
    - cluster_data: DataFrame with all data related to the current cluster.
    - cluster_num: The cluster number being analyzed.
    - audio_path: Path to the directory containing the audio files.
    - results_path: Path to the directory where results will be saved.
    - n_calls: Number of calls to select per batch (default is 25).

    Returns:
    - A threshold rank and distance for the cluster, if found.
    """
    # Ensure results path exists
    cluster_results_path = os.path.join(results_path, f'cluster_{cluster_num}')
    os.makedirs(cluster_results_path, exist_ok=True)
    
    # Sort data by distance to center (from closest to farthest)
    cluster_data = cluster_data.sort_values('distance_to_center', ascending=False)
    start_rank = 0
    threshold_found = False
    
    while not threshold_found:
        # Select the next batch of calls
        representative_calls = cluster_data.iloc[start_rank:start_rank + n_calls]
        
        if representative_calls.empty:
            print(f"Reached the end of calls in cluster {cluster_num} without finding a threshold.")
            break
        
        # Save path for the current batch
        save_path = os.path.join(cluster_results_path, f'rank_{start_rank + 1}_{start_rank + len(representative_calls)}')
        os.makedirs(save_path, exist_ok=True)
        
        # Save representative calls
        representative_calls.to_csv(os.path.join(save_path, f'representative_calls_rank_{start_rank + 1}_{start_rank + len(representative_calls)}.csv'), index=False)

        # Plot and save audio segments
        plot_and_save_audio_segments(representative_calls, audio_path, save_path, f'cluster_{cluster_num}')
        
        print(f"\nAnalysing Cluster {cluster_num}, Calls {start_rank + 1} to {start_rank + len(representative_calls)}:")
        print(representative_calls[['recording', 'call_id', 'distance_to_center']])
        print("\nPlease analyse these calls.")
        response = input("Have you found the threshold in this batch? (yes/no): ")
    
        if response.lower() == 'yes':
            threshold_rank = int(input("Enter the rank number where you found the threshold: "))
            threshold_distance = cluster_data.iloc[threshold_rank - 1]['distance_to_center']
            print(f"Threshold found for cluster {cluster_num} at rank {threshold_rank}, distance {threshold_distance:.4f}")
            threshold_found = True
        else:
            start_rank += len(representative_calls)
    
    # After finding threshold, save it
    if threshold_found:
        threshold_path = os.path.join(cluster_results_path, 'threshold.txt')
        with open(threshold_path, 'w') as f:
            f.write(f"Threshold for cluster {cluster_num}: rank {threshold_rank}\n")
            f.write(f"This corresponds to a distance_to_center of {threshold_distance:.4f}\n")
        return threshold_rank, threshold_distance
    else:
        return None, None




def plot_and_save_single_audio_segment(representative_calls, audio_path, save_path, cluster_label):
    """
    Extract, plot, and save audio segments for representative calls of a cluster.
    
    Args:
    representative_calls (DataFrame): DataFrame containing representative calls for a cluster
    audio_path (str): Path to the directory containing audio files
    save_path (str): Path to save the results
    cluster_label (str): Label for the cluster
    """
    # Create directory for saving audio files and plots
    cluster_audio_dir = os.path.join(save_path, 'audio')
    os.makedirs(cluster_audio_dir, exist_ok=True)
    
    for idx, (_, call) in enumerate(representative_calls.iterrows()):
        audio_file = os.path.join(audio_path, call['recording'] + '.wav')
        if os.path.exists(audio_file):
            data, sr = lb.load(audio_file, sr=44100)
            S = lb.feature.melspectrogram(y=data, sr=sr, n_mels=128, fmin=2000, fmax=10000)
            log_S = lb.power_to_db(S, ref=np.max)

            # Use your existing segment_spectrogram function
            calls_S = segment_spectrogram(log_S, [call['onsets_sec']], [call['offsets_sec']], sr=sr)
            call_S = calls_S[0]

            # Extract audio segment
            onset_samples = int(call['onsets_sec'] * sr)
            offset_samples = int(call['offsets_sec'] * sr)
            call_audio = data[onset_samples:offset_samples]

            # Plot the spectrogram of the individual call
            plt.figure(figsize=(5, 2))
            plt.imshow(call_S, aspect='auto', origin='lower', cmap='magma')
            plt.title(f'Call {idx + 1} of {call["recording"]} \n {cluster_label}', fontsize=8)
            plt.xlabel('Time', fontsize=7)
            plt.ylabel('Frequency', fontsize=7)
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()

            # Save the spectrogram plot
            plot_filename = f'call_{idx + 1}_{call["recording"]}_{cluster_label}.png'
            plt.savefig(os.path.join(cluster_audio_dir, plot_filename))
            plt.close()

            # Save the audio file
            audio_filename = os.path.join(cluster_audio_dir, f'call_{idx + 1}_{call["recording"]}_{cluster_label}.wav')
            sf.write(audio_filename, call_audio, sr)

            print(f'Saved audio and plot for call {idx + 1} of {call["recording"]}')
        else:
            print(f'Audio file {audio_file} not found')

    print(f'All representative calls for {cluster_label} have been processed and saved.')




def save_jittered_boxplots(all_data, n_clusters, features, percentiles, clusterings_results_path):
    """
    Generate and save jittered boxplots of features for each cluster across percentiles.

    Parameters:
    - all_data: pd.DataFrame, dataset containing features, cluster memberships, and percentiles.
    - n_clusters: int, number of clusters.
    - features: list of str, feature names to plot.
    - percentiles: list of numeric, percentiles to group by.
    - clusterings_results_path: str, path to save plots.
    """
    for cluster in range(n_clusters):
        cluster_data = all_data[all_data['cluster_membership'] == cluster]

        for feature in features:
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='percentile', y=feature, data=cluster_data, color='lightblue')
            sns.stripplot(x='percentile', y=feature, data=cluster_data, color='black', size=3, jitter=True, alpha=0.7)

            avg_distances = cluster_data.groupby('percentile')['distance_to_center'].mean().round(2)
            labels = [f'{percentile} ({avg})' for percentile, avg in zip(percentiles, avg_distances)]
            plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45, ha='right')

            plt.title(f'Cluster {cluster} - {feature} Distribution Across Percentiles')
            plt.xlabel('Percentile (Avg. Distance)')
            plt.ylabel('Feature Value')
            plt.tight_layout()

            save_path = os.path.join(clusterings_results_path, f'cluster_{cluster}_{feature}_jittered_boxplot.png')
            plt.savefig(save_path)
            plt.close()
    
    print("Jittered boxplots for each feature across different percentiles saved.")



def save_representative_calls(all_data, n_clusters, percentiles, get_representative_calls_by_percentile, 
                              plot_and_save_audio_segments, audio_path, clusterings_results_path):
    """
    Extract and save representative calls for each cluster at specified percentiles.

    Parameters:
    - all_data: pd.DataFrame, dataset containing features, cluster memberships, and distances to cluster centers.
    - n_clusters: int, number of clusters.
    - percentiles: list of numeric, percentiles to extract calls.
    - get_representative_calls_by_percentile: callable, function to compute calls at percentiles.
    - plot_and_save_audio_segments: callable, function to plot and save audio segments.
    - audio_path: str, path to the audio data.
    - clusterings_results_path: str, path to save results.
    """
    for cluster in range(n_clusters):
        cluster_data = all_data[all_data['cluster_membership'] == cluster]
        cluster_data = cluster_data.sort_values('distance_to_center', ascending=True)

        calls_at_percentiles = get_representative_calls_by_percentile(cluster_data, percentiles)

        for percentile, calls in zip(percentiles, calls_at_percentiles):
            save_path = os.path.join(clusterings_results_path, f'cluster_{cluster}_percentile_{percentile}')
            os.makedirs(save_path, exist_ok=True)

            calls.to_csv(os.path.join(save_path, f'representative_calls_cluster_{cluster}_percentile_{percentile}.csv'), index=False)
            plot_and_save_audio_segments(calls, audio_path, save_path, f'cluster_{cluster}_percentile_{percentile}')

            print(f"\nRepresentative calls for cluster {cluster} at percentile {percentile}:")
            print(calls[['recording', 'call_id', 'distance_to_center', 'cluster_membership']])
    
    print("Selection of calls at specified percentiles completed.")



def plot_umap_2d(features_scaled, cluster_membership, n_clusters, clusterings_results_path, n_neighbors=20, n_components=2, min_dist=0.9, random_state=42):
    """
    Perform UMAP dimensionality reduction and plot a 2D scatter plot of clustered data.

    Parameters:
    - features_scaled: np.array or pd.DataFrame, scaled feature set
    - cluster_membership: np.array, array of cluster labels for each data point
    - n_clusters: int, number of clusters
    - clusterings_results_path: str, path to save the plot
    - n_neighbors: int, number of neighbors for UMAP (default: 20)
    - n_components: int, number of UMAP components (default: 2 for 2D)
    - min_dist: float, minimum distance for UMAP (default: 0.7)
    - random_state: int, random seed for reproducibility (default: 42)

    Returns:
    - None (saves the plot and displays it)
    """
    # UMAP embedding
    umap_reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist, random_state=random_state)
    standard_embedding = umap_reducer.fit_transform(features_scaled)
    
    # Plot UMAP in 2D
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Custom colors
    custom_colors = ['indianred', 'dodgerblue', 'green', 'lightcoral', 'cyan', 'mediumslateblue']
    
    for cluster_id in range(n_clusters):
        cluster_points = standard_embedding[cluster_membership == cluster_id]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=custom_colors[cluster_id % len(custom_colors)], 
                   s=17, label=f'Cluster {cluster_id}', alpha=0.7)
    
    # ax.set_title("UMAP projection of chicks' calls clustered with Hierarchical Clustering", fontsize=14)
    ax.set_xlabel("UMAP Dimension 1", fontsize=16, family='Times New Roman')
    ax.set_ylabel("UMAP Dimension 2", fontsize=16, family='Times New Roman')
    # add legend
    ax.legend(title="Clusters", loc='lower left', fontsize=11, title_fontsize='12', frameon=True, edgecolor='black')

    # put the legend bottom left


    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(clusterings_results_path, f'hierarchical_clustering_{n_clusters}_umap_2d.png')
    # plt.savefig(save_path)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"UMAP 2D plot saved at: {save_path}")
    
    plt.show()

    return standard_embedding  # Return the UMAP embedding for further analysis if needed


def plot_umap_3d(features_scaled, cluster_membership, n_clusters, clusterings_results_path, n_neighbors=20, n_components=3, min_dist=0.7, random_state=42):
    """
    Perform UMAP dimensionality reduction and plot a 3D scatter plot of clustered data.

    Parameters:
    - features_scaled: np.array or pd.DataFrame, scaled feature set
    - cluster_membership: np.array, array of cluster labels for each data point
    - n_clusters: int, number of clusters
    - clusterings_results_path: str, path to save the plot
    - n_neighbors: int, number of neighbors for UMAP (default: 20)
    - n_components: int, number of UMAP components (default: 3)
    - min_dist: float, minimum distance for UMAP (default: 0.7)
    - random_state: int, random seed for reproducibility (default: 42)

    Returns:
    - None (saves the plot and displays it)
    """
    # UMAP embedding
    umap_reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist, random_state=random_state)
    standard_embedding = umap_reducer.fit_transform(features_scaled)
    
    # Plot UMAP in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Custom colors
    custom_colors = ['dodgerblue','indianred', 'green', 'lightcoral', 'cyan', 'mediumslateblue']
    
    for cluster_id in range(n_clusters):
        cluster_points = standard_embedding[cluster_membership == cluster_id]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], 
                   c=custom_colors[cluster_id % len(custom_colors)], 
                   s=17, label=f'Cluster {cluster_id}', alpha=0.7)
    
    ax.set_title("UMAP projection of chicks' calls clustered with Hierarchical Clustering", fontsize=14)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_zlabel("UMAP Dimension 3")
    ax.legend(title="Clusters", loc='upper right')
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(clusterings_results_path, f'hierarchical_clustering_{n_clusters}_umap_3d.png')
    plt.savefig(save_path)
    print(f"UMAP 3D plot saved at: {save_path}")
    
    plt.show()




def plot_cluster_distributions(all_data_sorted, n_clusters):
    """
    Plots the distribution of distances to the cluster center for each cluster separately.

    Parameters:
    - all_data_sorted: pd.DataFrame containing the cluster memberships and distances.
    - n_clusters: int, number of clusters.
    """
    plt.figure(figsize=(15, 5 * n_clusters))

    for cluster in range(n_clusters):
        plt.subplot(n_clusters, 1, cluster + 1)
        cluster_data = all_data_sorted[all_data_sorted['cluster_membership'] == cluster]
        sns.histplot(cluster_data['distance_to_center'], kde=True, bins=30, color=f"C{cluster}")
        plt.xlabel("Distance to cluster center")
        plt.ylabel("Frequency")
        plt.title(f"Cluster {cluster} - Distance to center distribution")

    plt.tight_layout()
    plt.show()
