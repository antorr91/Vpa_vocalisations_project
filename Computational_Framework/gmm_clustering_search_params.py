import os
import glob
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from kneed import KneeLocator
# from gap_statistic import OptimalK
from clustering_utils import find_elbow_point
import itertools



features_path = 'data/features/ctrl'               # Feature CSV files for "CTRL" group or  'data/features/vpa' for "VPA" group
audio_path = 'data/audio_files'                    # Folder with the audio (.wav) files
metadata_path = 'data/metadata.csv'               # Metadata file (CSV)
clusterings_results_path = 'results/clustering/precomputed_hierarchical_ctrl'  # Where clustering results will be saved
if not os.path.exists(clusterings_results_path):
    os.makedirs(clusterings_results_path)


# Get a list of all CSV files in the directory
list_files = glob.glob(os.path.join(features_path, '*.csv'))


# Read all CSV files and concatenate them into a single DataFrame
all_data = pd.concat([pd.read_csv(f) for f in list_files], ignore_index=True)
metadata = pd.read_csv(metadata_path)

# Drop NaN values
all_data = all_data.dropna()

# Scale data with StandardScaler on used features only
scaler = StandardScaler()
features = all_data.drop(['recording','Call Number', 'onsets_sec', 'offsets_sec', 'call_id'], axis=1)
features_scaled = scaler.fit_transform(features)


# 1) Gaussian Mixture Model (GMM) clustering
n_max_components = 11
components= range(2, n_max_components)

covarience_type =  'full'
n_init= 5
max_iter=1000
toll = 1e-3
random_state=42
init_params = 'kmeans'


# Dictionary to store evaluation metrics for each number of components
gmm_cluster_evaluation_per_number_clusters = {
    n_comp: {'silhouette_score': 0, 'calinski_harabasz_score': 0, 'wcss': 999, 'bic': 999, 'aic': 999, 'number_of_clusters': 0} for n_comp in components
}


for n_comp in components:
    gmm = GaussianMixture(n_components=n_comp, covariance_type=covarience_type, n_init=n_init, max_iter=max_iter, tol=toll, random_state=random_state, init_params=init_params) 
    cluster_membership = gmm.fit_predict(features_scaled)
 
    number_of_clusters = len(set(cluster_membership)) 
    # save the number of clusters
    gmm_cluster_evaluation_per_number_clusters[n_comp]['number_of_cluster'] = number_of_clusters

    # Compute the centroid of each cluster (using GMM's means_)
    cluster_centers = gmm.means_
    # Compute WCSS
    wcss = np.sum((np.linalg.norm(features_scaled - cluster_centers[cluster_membership], axis=1) ** 2))
   


    gmm_cluster_evaluation_per_number_clusters[n_comp]['silhouette_score'] = silhouette_score(features_scaled, cluster_membership)
    gmm_cluster_evaluation_per_number_clusters[n_comp]['calinski_harabasz_score'] = calinski_harabasz_score(features_scaled, cluster_membership)
    gmm_cluster_evaluation_per_number_clusters[n_comp]['wcss'] = wcss
    # gmm_cluster_evaluation_per_number_clusters[n_comp]['wcss'] = np.sum((np.linalg.norm(features_scaled - cluster_centers[cluster_membership], axis=1) ** 2))
    gmm_cluster_evaluation_per_number_clusters[n_comp]['bic'] = gmm.bic(features_scaled)
    gmm_cluster_evaluation_per_number_clusters[n_comp]['aic'] = gmm.aic(features_scaled)

    print('Number of clusters:', number_of_clusters) 
    print('Silhouette score:', gmm_cluster_evaluation_per_number_clusters[n_comp]['silhouette_score'])
    print('Calinski Harabasz score:', gmm_cluster_evaluation_per_number_clusters[n_comp]['calinski_harabasz_score'])
    print('WCSS:', gmm_cluster_evaluation_per_number_clusters[n_comp]['wcss'])    
    print('BIC:', gmm_cluster_evaluation_per_number_clusters[n_comp]['bic'])
    print('AIC:', gmm_cluster_evaluation_per_number_clusters[n_comp]['aic'])
    print()

# Save results in the Results_Clustering folder
gmm_cluster_evaluation_per_number_clusters_df = pd.DataFrame(gmm_cluster_evaluation_per_number_clusters).T
gmm_cluster_evaluation_per_number_clusters_df.to_csv(os.path.join(clusterings_results_path, 'gmm_cluster_evaluation_per_number_clusters.csv'), index=False)
gmm_cluster_evaluation_per_number_clusters_df.to_latex(os.path.join(clusterings_results_path, 'gmm_cluster_evaluation_per_number_clusters.tex'))

# Find the vector of silhouette scores, calinski harabasz scores, wcss, bic, and aic across the number of clusters
wcss_vector_across_n_clusters = [result['wcss'] for result in gmm_cluster_evaluation_per_number_clusters.values()]
bic_vector_across_n_clusters = [result['bic'] for result in gmm_cluster_evaluation_per_number_clusters.values()]
aic_vector_across_n_clusters = [result['aic'] for result in gmm_cluster_evaluation_per_number_clusters.values()]



# fIND THE ELBOW POINTS
wcss_elbow = KneeLocator(range(2, len(wcss_vector_across_n_clusters)+2), wcss_vector_across_n_clusters, curve='convex', direction='decreasing', interp_method='interp1d', online=False)
bic_elbow = KneeLocator(range(2, len(bic_vector_across_n_clusters)+2), bic_vector_across_n_clusters, curve='convex', direction='decreasing', interp_method='polynomial', online=False)
aic_elbow = KneeLocator(range(2, len(bic_vector_across_n_clusters)+2), aic_vector_across_n_clusters,curve='convex', direction='decreasing', interp_method='polynomial', online=False)

best_n_clusters_by_wcss_elbow = wcss_elbow.elbow
best_n_clusters_by_bic_elbow = bic_elbow.elbow
best_n_clusters_by_aic_elbow = aic_elbow.elbow

# # optimal_k = OptimalK(parallel_backend='joblib')
# n_clusters_optimal_k = optimal_k(features_scaled, cluster_array=np.arange(2, n_max_components))
# print('Optimal number of clusters for k:', n_clusters_optimal_k)


plt.figure(figsize=(25, 6))

# Plot Silhouette score
plt.subplot(1, 2, 1)
# plot number of clusters vs silhouette score
# save the list of number of clusters and silhouette scores
number_of_clusters = [gmm_cluster_evaluation_per_number_clusters[n_comp]['number_of_cluster'] for n_comp in components]
silhouette_scores = [gmm_cluster_evaluation_per_number_clusters[n_comp]['silhouette_score'] for n_comp in components]

plt.plot(number_of_clusters, silhouette_scores, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.title('Silhouette score per number of clusters')

# Plot Calinski Harabasz score
plt.subplot(1, 2, 2)
calinski_harabasz_scores = [gmm_cluster_evaluation_per_number_clusters[n_comp]['calinski_harabasz_score'] for n_comp in gmm_cluster_evaluation_per_number_clusters.keys()]
plt.plot(number_of_clusters, calinski_harabasz_scores, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Calinski Harabasz score')
plt.title('Calinski Harabasz score per number of clusters')

plt.tight_layout()
plt.savefig(os.path.join(clusterings_results_path, 'gmm_cluster_evaluation_per_number_clusters_silhouette_calinski.png'))
plt.show()
# Create a new figure for BIC, AIC, and WCSS
plt.figure(figsize=(37, 8))

# plot the BIC, AIC, and WCSS per number of clusters
plt.subplot(1, 3, 1)
wcss_values = [gmm_cluster_evaluation_per_number_clusters[n_comp]['wcss'] for n_comp in gmm_cluster_evaluation_per_number_clusters.keys()]
plt.plot(number_of_clusters, wcss_values, 'bx-')
plt.axvline(x=best_n_clusters_by_wcss_elbow, color='r', linestyle='--', label=f'Elbow point: {best_n_clusters_by_wcss_elbow}')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('WCSS per number of clusters')
plt.legend()

plt.subplot(1, 3, 2)
bic_values = [gmm_cluster_evaluation_per_number_clusters[n_comp]['bic'] for n_comp in gmm_cluster_evaluation_per_number_clusters.keys()]
plt.plot(number_of_clusters, bic_values, 'bx-')
plt.axvline(x=best_n_clusters_by_bic_elbow, color='r', linestyle='--', label=f'Elbow point: {best_n_clusters_by_bic_elbow}')
plt.xlabel('Number of clusters')
plt.ylabel('BIC')
plt.title('BIC per number of clusters')
plt.legend()

# Plot AIC
plt.subplot(1, 3, 3)
aic_values = [gmm_cluster_evaluation_per_number_clusters[n_comp]['aic'] for n_comp in gmm_cluster_evaluation_per_number_clusters.keys()]
plt.plot(number_of_clusters, aic_values, 'bx-')
# signal the optimal number of clusters with a red circle
plt.axvline(x=best_n_clusters_by_aic_elbow, color='r', linestyle='--', label= f'Elbow point: {best_n_clusters_by_aic_elbow}')
plt.xlabel('Number of clusters')
plt.ylabel('AIC')
plt.title('AIC per number of clusters')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(clusterings_results_path, 'gmm_cluster_evaluation_per_number_clusters_bic_aic_wcss.png'))
plt.show()
print('Gaussian Mixture Model (GMM) clustering done')

