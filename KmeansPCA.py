import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('macosx')
import seaborn as sns

sns.set()
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# from PreProcessing import unique_objects_data
from test_pp import *

unique_objects_data = fin_unique_objects_data
# c2_df = c2_df.reset_index(drop=True)
# c3_df = c3_df.reset_index(drop=True)
# c4_df = c4_df.reset_index(drop=True)
# c5_df = c5_df.reset_index(drop=True)
# c6_df = c6_df.reset_index(drop=True)
#
# print(unique_objects_data['Center X'].sub(415.93).abs().idxmin())
# print(unique_objects_data.loc[447])
# print(c2_df['AreaShape_Center_X'].sub(415.93).abs().idxmin())
# print(c2_df.loc[880, ['AreaShape_Center_X', 'AreaShape_Center_Y', 'Intensity_MeanIntensity_phase_c2']])
# print(c3_df['AreaShape_Center_X'].sub(415.93).abs().idxmin())
# print(c3_df.loc[118, ['AreaShape_Center_X', 'AreaShape_Center_Y', 'Intensity_MeanIntensity_phase_c3']])
# print(c4_df['AreaShape_Center_X'].sub(415.93).abs().idxmin())
# print(c4_df.loc[142, ['AreaShape_Center_X', 'AreaShape_Center_Y', 'Intensity_MeanIntensity_phase_c4']])
# print(c5_df['AreaShape_Center_X'].sub(415.93).abs().idxmin())
# print(c5_df.loc[48, ['AreaShape_Center_X', 'AreaShape_Center_Y', 'Intensity_MeanIntensity_phase_c5']])
# print(c6_df['AreaShape_Center_X'].sub(415.93).abs().idxmin())
# print(c6_df.loc[11, ['AreaShape_Center_X', 'AreaShape_Center_Y', 'Intensity_MeanIntensity_phase_c6']])
#
# exit()

base_path = '/Users/aribakhan/Dropbox (MIT)/shared_Khan/final_data/'


def create_specific_points(data):
    irfp_ind1 = data['Center X'].sub(338.39).abs().idxmin()
    irfp_pca = data.loc[[irfp_ind1]].iloc[:, :4]

    mcherry_ind = data['Center X'].sub(415.93).abs().idxmin()
    mcherry_pca = data.loc[[mcherry_ind]].iloc[:, :4]

    plum_ind = data['Center X'].sub(570.62).abs().idxmin()
    plum_pca = data.loc[[plum_ind]].iloc[:, :4]

    orange_ind = data['Center X'].sub(647.94).abs().idxmin()
    orange_pca = data.loc[[orange_ind]].iloc[:, :4]

    gfp_ind = data['Center X'].sub(404.93).abs().idxmin()
    gfp_pca = data.loc[[gfp_ind]].iloc[:, :4]

    bfp_ind = data['Center X'].sub(380.03).abs().idxmin()
    bfp_pca = data.loc[[bfp_ind]].iloc[:, :4]

    return irfp_pca, mcherry_pca, plum_pca, orange_pca, gfp_pca, bfp_pca


raw_df = unique_objects_data.iloc[:, 7:12]
raw_filter_df = raw_df.loc[(raw_df != 0).any(axis=1)]

# preprocess data
scaler = StandardScaler()
pca = PCA(n_components=3)


# perform PCA


def pca_wrapper(specific_cells=False, split_channels=False, group=False):
    raw_std = scaler.fit_transform(raw_filter_df)
    scores_pca = pca.fit_transform(raw_std)
    # np.savetxt(base_path + "pca_vectors.csv", pca.components_.T, delimiter=",")
    # create dataframe of pca scores --> components
    pca_df = pd.DataFrame(data=scores_pca
                          , columns=['Comp 1', 'Comp 2', 'Comp 3'])
    # find the target dataframe where target is the channel w/ max value
    target_df = pd.DataFrame(data=raw_df.idxmax(axis=1)
                             , columns=['Target'])

    ax = plt.axes(projection="3d")
    ax.set_xlabel('Principal Component 1', fontsize=10)
    ax.set_ylabel('Principal Component 2', fontsize=10)
    ax.set_zlabel('Principal Component 3', fontsize=10)
    ax.set_xlim([-1.5, 1])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])

    if group:
        target_df.loc[(target_df['Target'] == 'Channel 2') | (target_df['Target'] == 'Channel 3') | (
                target_df['Target'] == 'Channel 4'), 'Target'] = 'Channel E'
        ax.set_title('Full Color 3 Component PCA w/ 3 Channels', fontsize=20)
        targets = ['Channel 2', 'Channel 3', 'Channel 4', 'Channel E']
        colors = ['darkviolet', 'r', 'y', 'k']
    else:
        ax.set_title('Full Color 3 Component PCA', fontsize=20)
        targets = ['Channel 2', 'Channel 3', 'Channel 4', 'Channel 5', 'Channel 6']
        colors = ['darkviolet', 'r', 'y', 'g', 'b']

    final_pca_df = pd.concat([pca_df, target_df], axis=1)
    for target, color in zip(targets, colors):
        indices = final_pca_df['Target'] == target
        ax.scatter3D(final_pca_df.loc[indices, 'Comp 1']
                     , final_pca_df.loc[indices, 'Comp 2']
                     , final_pca_df.loc[indices, 'Comp 3']
                     , c=color
                     , s=10
                     )
    ax2 = plt.axes(projection="3d")
    if split_channels:
        ax2.set_xlabel('Principal Component 1', fontsize=10)
        ax2.set_ylabel('Principal Component 2', fontsize=10)
        ax2.set_zlabel('Principal Component 3', fontsize=10)
        ax2.set_title('3 Component PCA w/ Channel 2, 3 and 4', fontsize=20)

        unique_objects_target_data = pd.concat([unique_objects_data, pd.DataFrame(data=raw_df.idxmax(axis=1)
                                                                                  , columns=['Target'])], axis=1)

        specific_raw_filter_df = unique_objects_target_data.loc[
            (unique_objects_target_data['Target'] == 'Channel 2') |
            (unique_objects_target_data['Target'] == 'Channel 3') |
            (unique_objects_target_data['Target'] == 'Channel 4')]

        # specific_raw_filter_df = specific_raw_filter_df_t.reset_index(drop=True)
        specific_raw_std = scaler.fit_transform(specific_raw_filter_df.iloc[:, 7:12])
        specific_scores_pca = pca.fit_transform(specific_raw_std)
        # np.savetxt(base_path + "pca_vectors.csv", pca.components_.T, delimiter=",")

        specific_pca_df = pd.DataFrame(data=specific_scores_pca
                                       , columns=['Comp 1', 'Comp 2', 'Comp 3'])

        specific_target_df = pd.DataFrame(data=specific_raw_filter_df.iloc[:, 7:12].idxmax(axis=1)
                                          , columns=['Target'])
        specific_target_df = specific_target_df.reset_index(drop=True)
        specific_x_y_df = specific_raw_filter_df.iloc[:, 2:4]
        specific_x_y_df = specific_x_y_df.reset_index(drop=True)
        specific_channel_vals_df = specific_raw_filter_df.iloc[:, 7:12]
        specific_channel_vals_df = specific_channel_vals_df.reset_index(drop=True)
        specific_image_df = specific_raw_filter_df.iloc[:, 0:1]
        specific_image_df = specific_image_df.reset_index(drop=True)

        final_specific_pca_df = pd.concat([specific_pca_df, specific_target_df, specific_x_y_df,
                                           specific_channel_vals_df, specific_image_df], axis=1)
        df = final_specific_pca_df.iloc[:, 0:4]

        specific_targets = targets[0:3]
        specific_colors = colors[0:3]
        for st, sc in zip(specific_targets, specific_colors):
            indices = df['Target'] == st
            ax2.scatter3D(df.loc[indices, 'Comp 1']
                          , df.loc[indices, 'Comp 2']
                          , df.loc[indices, 'Comp 3']
                          , c=sc
                          , s=10)

        ax2.legend(specific_targets)

    if specific_cells:
        irfp, mcherry, plum, orange, gfp, bfp = create_specific_points(final_specific_pca_df)

        pca_cell_df = pd.concat([irfp, mcherry, plum, orange])

        cells_name = ["irfp", "mcherry", "plum", "orange"]
        pca_cell_df['Target'] = cells_name
        colors_c = ['darkviolet', 'r', 'deeppink', 'orange']
        for t, c in zip(cells_name, colors_c):
            indices = pca_cell_df['Target'] == t
            ax2.scatter3D(pca_cell_df.loc[indices, 'Comp 1']
                          , pca_cell_df.loc[indices, 'Comp 2']
                          , pca_cell_df.loc[indices, 'Comp 3']
                          , c=c
                          , s=60
                          , marker='^')

    ax2.grid()
    plt.show()
    if split_channels:
        return specific_raw_filter_df, specific_scores_pca
    return raw_filter_df, scores_pca


raw_filter, scores_pca = pca_wrapper(specific_cells=True, split_channels=True)

# determine the number of clusters
# use within cluster sum squares method
# determine the number of clusters using elbow method
wcss = []
for i in range(1, 10):
    kmeans_pca = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans_pca.fit(scores_pca)
    wcss.append(kmeans_pca.inertia_)

plt.figure(figsize=(10, 8))
plt.plot(range(1, 10), wcss, marker='o', linestyle='--')
plt.xlabel("Num of Clusters")
plt.ylabel("Within Cluster Sum Squares")
plt.title("K-means w/ PCA Clustering")
plt.show()

# implement kmeans
kmeans_pca = KMeans(n_clusters=4, init='k-means++', random_state=42)
kmeans_pca.fit(scores_pca)
# create new data frame to store scores
seg_pca_kmeans = pd.concat([raw_filter.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)
seg_pca_kmeans.columns.values[-3:] = ["Comp 1", "Comp 2", "Comp 3"]
seg_pca_kmeans['Seg K-PCA'] = kmeans_pca.labels_
seg_pca_kmeans['Segment'] = seg_pca_kmeans['Seg K-PCA'].map({0: 'First', 1: 'Second', 2: "Third", 3: "Fourth"})

x_a = seg_pca_kmeans["Comp 2"]
y_a = seg_pca_kmeans["Comp 1"]
plt.figure(figsize=(10, 8))
np.savetxt(base_path + "centroids.csv", kmeans_pca.cluster_centers_.T, delimiter=",")

sns.scatterplot(x_a, y_a, hue=seg_pca_kmeans['Segment'], palette=['c', 'r', 'y', 'g'])
plt.title("Cluster by PCA Components")
plt.show()
