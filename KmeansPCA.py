import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from Analysis import unique_objects_data

# print(list(unique_objects_data))

# preprocess data
unique_objects_feat = unique_objects_data[['Center X', 'Center Y', 'Intensity Sum', 'Weighted Sum']].copy()
scaler = StandardScaler()
unique_objects_feat_std = scaler.fit_transform(unique_objects_feat)
pca = PCA()
pca.fit(unique_objects_feat_std)

# choose the number of features
# rule of thumb: perserve 80% of variance
plt.figure(figsize=(10, 8))
plt.plot(range(1, 5), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
# plt.show()

# perform PCA
pca = PCA(n_components=3)
pca.fit(unique_objects_feat_std)
scores_pca = pca.transform(unique_objects_feat_std)

# determine the number of clusters
# use within cluster sum squares method
# determine the number of clusters using elbow method
wcss = []
for i in range(1, 11):
    kmeans_pca = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans_pca.fit(scores_pca)
    wcss.append(kmeans_pca.inertia_)

plt.figure(figsize=(10, 8))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.xlabel("Num of Clusters")
plt.ylabel("Within Cluster Sum Squares")
# plt.show()

# implement kmeans
kmeans_pca = KMeans(n_clusters=6, init='k-means++', random_state=42)
kmeans_pca.fit(scores_pca)

# create new data frame to store scores
seg_pca_kmeans = pd.concat([unique_objects_feat.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)
seg_pca_kmeans.columns.values[-3:] = ["Comp 1", "Comp 2", "Comp 3"]
seg_pca_kmeans['Segment Kmeans PCA'] = kmeans_pca.labels_
#print(seg_pca_kmeans)
seg_pca_kmeans['Segment'] = seg_pca_kmeans['Segment Kmeans PCA'].map(
    {0: 'first', 1: 'second', 2: "third", 3: "fourth", 4: "fifth", 5: "sixth"})

x_a = seg_pca_kmeans["Comp 2"]
y_a = seg_pca_kmeans["Comp 1"]
plt.figure(figsize=(10, 8))
sns.scatterplot(x_a, y_a, hue=seg_pca_kmeans['Segment'], palette=['m', 'r', 'y', 'g', 'b', 'c'])
plt.title("Cluster by PCA Components")
plt.show()


