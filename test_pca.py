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
from PreProcessing import unique_objects_data

base_path = '/Users/aribakhan/Dropbox (MIT)/shared_Khan/final_data/'

ax = plt.axes(projection="3d")
ax.set_xlabel('Principal Component 1', fontsize=10)
ax.set_ylabel('Principal Component 2', fontsize=10)
ax.set_zlabel('Principal Component 3', fontsize=10)
ax.set_title('3 Component PCA w/ Channel 2, 3 and 4', fontsize=20)

specific_raw_filter_df = unique_objects_data.loc[(final_pca_df['Target'] == 'Channel 2') |
                                     (final_pca_df['Target'] == 'Channel 3') |
                                     (final_pca_df['Target'] == 'Channel 4')]
specific_raw_filter_df = specific_raw_filter_df.reset_index(drop=True)

scaler = StandardScaler()

scaler.fit(specific_raw_filter_df.iloc[:, 6:11])

specific_raw_std = scaler.transform(specific_raw_filter_df.iloc[:, 6:11])
pca.fit(specific_raw_std)
np.savetxt(base_path + "pca_vectors.csv", pca.components_.T, delimiter=",")
specific_scores_pca = pca.transform(specific_raw_std)
specific_pca_df = pd.DataFrame(data=specific_scores_pca
                               , columns=['Comp 1', 'Comp 2', 'Comp 3'])

specific_target_df = pd.DataFrame(data=specific_raw_filter_df.iloc[:, 6:11].idxmax(axis=1)
                                  , columns=['Target'])

final_specific_pca_df = pd.concat([specific_pca_df, specific_target_df], axis=1)

specific_targets = targets[0:3]
specific_colors = colors[0:3]
for st, sc in zip(specific_targets, specific_colors):
    indices = final_pca_df['Target'] == st
    ax.scatter3D(final_specific_pca_df.loc[indices, 'Comp 1']
                  , final_specific_pca_df.loc[indices, 'Comp 2']
                  , final_specific_pca_df.loc[indices, 'Comp 3']
                  , c=sc
                  , s=10)

ax2.legend(specific_targets)